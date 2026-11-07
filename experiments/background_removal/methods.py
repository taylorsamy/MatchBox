"""
Background removal methods under test.

Every method takes an RGB ``PIL.Image`` and returns an 8-bit ``L`` mask of the same size
(255 = robot, 0 = background), so they can be compared on equal terms and any winner can
be dropped into ``robot_photos.BackgroundRemover``.

ML models are the ONNX exports hosted by the rembg project, run directly through
onnxruntime with each model's published pre/post-processing (the same approach MatchBox
already uses for U²-Net).  Models are cached in ``$U2NET_HOME`` or ``~/.u2net``, shared
with MatchBox and rembg.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Callable, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

MODEL_HOST = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# onnxruntime threads per model; MatchBox uses 2 so OBS keeps the rest of the CPU
THREADS = 2
# onnxruntime's CPU memory arena: faster, but holds on to (much) more memory. Off = --no-arena
ARENA = True

FloatArray = npt.NDArray[np.float32]


def model_dir() -> Path:
    override = os.environ.get("U2NET_HOME")
    return Path(override) if override else Path.home() / ".u2net"


def fetch_model(filename: str, sha256: str, size: int, host: str = MODEL_HOST) -> Path:
    """Return the cached model path, downloading (and verifying) it first if needed."""
    path = model_dir() / filename
    if path.is_file() and path.stat().st_size == size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    digest = hashlib.sha256()
    print(f"  downloading {filename} ({size / 1e6:.0f} MB)...", flush=True)
    request = urllib.request.Request(host + filename, headers={"User-Agent": "MatchBox-benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:  # pyright: ignore[reportAny]
        while chunk := cast(bytes, response.read(1 << 20)):  # pyright: ignore[reportAny]
            _ = out.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        tmp.unlink()
        raise RuntimeError(f"{filename} failed its checksum")
    os.replace(tmp, path)
    return path


class Method:
    """Base class: a named way of turning a photo into a foreground mask."""

    name: str = ""
    family: str = ""  # "ml" | "classical" | "hybrid"
    description: str = ""
    license: str = ""
    download_bytes: int = 0

    def load(self) -> None:
        """Download / initialise (timed separately from per-image inference)."""

    def unload(self) -> None:
        """Free memory before the next method runs."""

    def predict(self, img: Image.Image) -> Image.Image:
        raise NotImplementedError


# --------------------------------------------------------------------------- ML models


class OnnxSalient(Method):
    """Salient-object / dichotomous segmentation model exported to ONNX."""

    family = "ml"

    def __init__(
        self, *, name: str, description: str, license: str, filename: str, sha256: str, download_bytes: int,
        input_size: int, mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD, sigmoid: bool = False, host: str = MODEL_HOST,
    ) -> None:
        self.name = name
        self.description = description
        self.license = license
        self.filename: str = filename
        self.sha256: str = sha256
        self.download_bytes = download_bytes
        self.input_size: int = input_size
        self.mean: tuple[float, float, float] = mean
        self.std: tuple[float, float, float] = std
        self.sigmoid: bool = sigmoid  # BiRefNet outputs logits
        self.host: str = host
        self._session: object | None = None

    def load(self) -> None:
        import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]

        path = fetch_model(self.filename, self.sha256, self.download_bytes, self.host)
        options = ort.SessionOptions()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        options.intra_op_num_threads = THREADS
        options.enable_cpu_mem_arena = ARENA
        self._session = ort.InferenceSession(  # pyright: ignore[reportUnknownMemberType]
            str(path), sess_options=options, providers=["CPUExecutionProvider"]  # pyright: ignore[reportUnknownArgumentType]
        )

    def unload(self) -> None:
        self._session = None

    def predict(self, img: Image.Image) -> Image.Image:
        if self._session is None:
            self.load()
        session = self._session
        size = (self.input_size, self.input_size)
        small = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        arr = arr / max(float(arr.max()), 1e-6)
        arr = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)

        run = cast(Callable[..., "list[object]"], getattr(session, "run"))
        input_name = cast(str, getattr(getattr(session, "get_inputs")()[0], "name"))  # pyright: ignore[reportAny]
        pred = cast(FloatArray, np.asarray(run(None, {input_name: tensor})[0], dtype=np.float32)[0, 0])
        if self.sigmoid:
            pred = 1.0 / (1.0 + np.exp(-pred))
        low, high = float(pred.min()), float(pred.max())
        pred = (pred - low) / max(high - low, 1e-8)
        return Image.fromarray((pred * 255.0).astype(np.uint8)).resize(img.size, Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------- classical


def _work_copy(img: Image.Image, max_edge: int) -> tuple[Image.Image, npt.NDArray[np.uint8]]:
    """Downscaled RGB copy (classical methods are slow at full size) and its BGR array for OpenCV."""
    work = img.convert("RGB")
    if max(work.size) > max_edge:
        work = work.copy()
        work.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    bgr = np.ascontiguousarray(np.asarray(work, dtype=np.uint8)[..., ::-1])
    return work, bgr


def _to_mask(binary: npt.NDArray[np.bool_], size: tuple[int, int]) -> Image.Image:
    """Binary work-size mask -> soft full-size mask (bilinear upscale gives anti-aliased edges)."""
    mask = Image.fromarray(binary.astype(np.uint8) * 255)
    return mask.resize(size, Image.Resampling.BILINEAR)


class GrabCut(Method):
    """OpenCV GrabCut initialised with a rectangle 5% in from the frame edges."""

    name = "grabcut"
    family = "classical"
    description = "OpenCV GrabCut, rectangle init (no model)"
    license = "Apache-2.0 (OpenCV)"

    def predict(self, img: Image.Image) -> Image.Image:
        import cv2

        cv2.setRNGSeed(10298)  # GrabCut/k-means are randomised: keep results independent of run order
        _, bgr = _work_copy(img, 800)
        h, w = bgr.shape[:2]
        m = 0.05
        rect = (int(w * m), int(h * m), int(w * (1 - 2 * m)), int(h * (1 - 2 * m)))
        gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        _ = cv2.grabCut(bgr, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        return _to_mask((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), img.size)


class GrabCutRefine(Method):
    """Use an ML mask as a trimap for GrabCut, which can open up holes the model filled in."""

    family = "hybrid"

    def __init__(self, base: Method) -> None:
        self.base: Method = base
        self.name = f"{base.name}+grabcut"
        self.description = f"{base.name} mask refined with OpenCV GrabCut"
        self.license = base.license
        self.download_bytes = base.download_bytes

    def load(self) -> None:
        self.base.load()

    def unload(self) -> None:
        self.base.unload()

    def predict(self, img: Image.Image) -> Image.Image:
        import cv2

        cv2.setRNGSeed(10298)  # GrabCut/k-means are randomised: keep results independent of run order
        base_mask = self.base.predict(img)
        work, bgr = _work_copy(img, 1000)
        soft = np.asarray(base_mask.resize(work.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        k = max(3, round(min(work.size) * 0.015)) | 1
        kernel = np.ones((k, k), np.uint8)
        sure_fg = cv2.erode((soft > 0.9).astype(np.uint8), kernel) > 0
        maybe = cv2.dilate((soft > 0.1).astype(np.uint8), kernel) > 0

        gc = np.full(soft.shape, cv2.GC_PR_BGD, np.uint8)
        gc[soft > 0.5] = cv2.GC_PR_FGD
        gc[sure_fg] = cv2.GC_FGD
        gc[~maybe] = cv2.GC_BGD
        if int(np.count_nonzero(gc == cv2.GC_BGD)) < 100 or int(np.count_nonzero(sure_fg)) < 100:
            return base_mask  # GrabCut needs both background and foreground samples
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        _ = cv2.grabCut(bgr, gc, (0, 0, 0, 0), bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)  # rect unused with a mask
        return _to_mask((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), img.size)


class BackdropKey(Method):
    """Key out the background colour sampled from the frame border (for a plain photo backdrop)."""

    name = "backdrop-key"
    family = "classical"
    description = "Colour key against the frame border (needs a plain backdrop; no model)"
    license = "n/a (this repo)"

    def predict(self, img: Image.Image) -> Image.Image:
        import cv2

        cv2.setRNGSeed(10298)  # GrabCut/k-means are randomised: keep results independent of run order
        _, bgr = _work_copy(img, 800)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        b = max(4, round(min(h, w) * 0.04))
        # Sides and top only: the robot usually stands on the bottom edge
        border = np.concatenate([lab[:b].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(border, 4, None, criteria, 3, cv2.KMEANS_PP_CENTERS)  # pyright: ignore[reportCallIssue, reportArgumentType]
        centers = np.asarray(centers, dtype=np.float32)
        dist = np.min(np.linalg.norm(lab[:, :, None, :] - centers[None, None], axis=3), axis=2)
        border_dist = np.min(np.linalg.norm(border[:, None, :] - centers[None], axis=2), axis=1)
        t0 = float(np.percentile(border_dist, 98)) + 4.0
        t1 = t0 * 2.0 + 6.0
        alpha = np.clip((dist - t0) / (t1 - t0), 0.0, 1.0)

        # Drop specks: keep connected regions at least 1% the size of the largest
        binary = (alpha > 0.5).astype(np.uint8)
        count, labels_mat, stats_mat, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        labels = np.asarray(labels_mat, dtype=np.intp)
        stats = np.asarray(stats_mat, dtype=np.int64)
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.zeros(count, np.bool_)
            keep[1:] = areas >= max(1, int(areas.max()) // 100)
            alpha = alpha * keep[labels]
        mask = Image.fromarray((alpha * 255).astype(np.uint8))
        return mask.resize(img.size, Image.Resampling.BILINEAR)


# --------------------------------------------------------------------------- post-processing


def keep_main_subject(mask: Image.Image, min_share: float = 0.15) -> Image.Image:
    """Drop blobs that are separate from, and much smaller than, the main subject (e.g. a loose ball on the floor).

    A blob is kept if it is the largest, overlaps the largest blob's bounding box, or is at least
    ``min_share`` of its area - so parts of the robot that came out disconnected survive.
    """
    import cv2

    alpha = np.asarray(mask, dtype=np.uint8)
    count, labels_mat, stats_mat, _ = cv2.connectedComponentsWithStats((alpha > 127).astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    labels = np.asarray(labels_mat, dtype=np.intp)
    stats = np.asarray(stats_mat, dtype=np.int64)
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    mx, my, mw, mh = (int(v) for v in stats[main, :4])
    keep = np.zeros(count, np.bool_)
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        overlaps = x < mx + mw and mx < x + w and y < my + mh and my < y + h
        keep[i] = i == main or overlaps or area >= min_share * float(areas.max())
    # The soft edge around a kept blob belongs to it: grow the kept area slightly before masking
    k = max(3, round(min(alpha.shape) * 0.01)) | 1
    kept = cv2.dilate(keep[labels].astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return Image.fromarray(np.where(kept, alpha, 0).astype(np.uint8))


# --------------------------------------------------------------------------- registry


def build_methods() -> dict[str, Method]:
    u2net = OnnxSalient(
        name="u2net", description="U²-Net, 320 px (current MatchBox default)", license="Apache-2.0",
        filename="u2net.onnx", download_bytes=175_997_641, input_size=320,
        sha256="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    )
    isnet = OnnxSalient(
        name="isnet", description="IS-Net general use (DIS), 1024 px", license="Apache-2.0",
        filename="isnet-general-use.onnx", download_bytes=178_648_008, input_size=1024,
        mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0),
        sha256="60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    )
    methods: list[Method] = [
        u2net,
        OnnxSalient(
            name="u2netp", description="U²-Net-P, 320 px (tiny)", license="Apache-2.0",
            filename="u2netp.onnx", download_bytes=4_574_861, input_size=320,
            sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        ),
        OnnxSalient(
            name="silueta", description="Silueta: U²-Net distilled by rembg, 320 px", license="unverified (rembg)",
            filename="silueta.onnx", download_bytes=44_173_029, input_size=320,
            sha256="75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb",
        ),
        isnet,
        OnnxSalient(
            name="birefnet-lite", description="BiRefNet general, Swin-T backbone, 1024 px", license="MIT",
            filename="BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", download_bytes=224_005_088, input_size=1024,
            sigmoid=True, sha256="5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
        ),
        OnnxSalient(
            name="birefnet", description="BiRefNet general, Swin-L backbone, 1024 px (heavy)", license="MIT",
            filename="BiRefNet-general-epoch_244.onnx", download_bytes=972_666_916, input_size=1024,
            sigmoid=True, sha256="58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
        ),
        OnnxSalient(
            name="birefnet-512", description="BiRefNet general, Swin-L backbone, 512 px export (from the BiRefNet authors)",
            license="MIT", filename="BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx",
            download_bytes=940_526_436, input_size=512, sigmoid=True,
            sha256="18b64a270d3abe7d36e665500c3f24136bcaf7616771b4cb21c6e8c57fec7237",
            host="https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/",
        ),
        GrabCut(),
        BackdropKey(),
        GrabCutRefine(u2net),
        GrabCutRefine(isnet),
    ]
    return {m.name: m for m in methods}


# Everything except the ~1 GB BiRefNet, which needed ~14 GB of RAM on CPU (opt in with --methods all or by name)
DEFAULT_METHODS = [
    "u2net", "u2netp", "silueta", "isnet", "birefnet-lite", "birefnet-512",
    "grabcut", "backdrop-key", "u2net+grabcut", "isnet+grabcut",
]
