"""
Robot photo processing for MatchBox.

Photos are uploaded from a phone/tablet through the web server (see ``web_photos/``).
Each upload is stored untouched, then a background worker:

1. fixes orientation (EXIF) and bounds the working size,
2. removes the background with a U²-Net salient-object model run directly through
   ``onnxruntime`` (no ``rembg`` dependency, so the standalone builds stay small),
3. crops to the robot and fits it onto a fixed-size transparent canvas, so every
   processed photo has identical dimensions.

Pillow, numpy and onnxruntime are imported lazily.  If they (or the model) are
unavailable the photo is still resized onto the canvas, just without the cut-out,
and the result is flagged so the operator can reprocess it later.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL import Image
    from matchbox import MatchBoxConfig

logger = logging.getLogger("matchbox")

TEAM_RE = re.compile(r"^[0-9]{1,6}$")
ALLOWED_FORMATS: dict[str, str] = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 100_000_000
WORK_MAX_EDGE = 3000  # longest edge of the working copy handed to the model
CANVAS_MARGIN = 0.04  # fraction of the canvas left empty around the robot
THUMB_WIDTH = 320
MODEL_RETRY_SECONDS = 300.0  # don't hammer a failing download on every upload
MIN_CANVAS = 200
MAX_CANVAS = 4000

PHOTO_DIRNAME = "robot_photos"


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str
    size: int


_REMBG_RELEASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"

# Both models share the same 320x320 pre/post-processing.  Files use rembg's names so a
# machine that already ran rembg reuses its ~/.u2net cache instead of downloading again.
MODEL_SPECS: dict[str, ModelSpec] = {
    "u2net": ModelSpec(
        filename="u2net.onnx",
        url=_REMBG_RELEASE + "u2net.onnx",
        sha256="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        size=175_997_641,
    ),
    "u2netp": ModelSpec(
        filename="u2netp.onnx",
        url=_REMBG_RELEASE + "u2netp.onnx",
        sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        size=4_574_861,
    ),
}
DEFAULT_MODEL = "u2net"


class PhotoRejected(ValueError):
    """The upload cannot be accepted (bad team number, not an image, too big...)."""


class BackgroundRemovalUnavailable(RuntimeError):
    """Background removal cannot run right now (missing libraries, no model, offline...)."""


def normalize_team(raw: str) -> str:
    """Validate a team number and strip leading zeros."""
    raw = raw.strip()
    if not TEAM_RE.match(raw) or int(raw) == 0:
        raise PhotoRejected("Team number must be 1-6 digits")
    return str(int(raw))


def get_model_dir() -> Path:
    override = os.environ.get("U2NET_HOME")
    return Path(override) if override else Path.home() / ".u2net"


def check_dependencies() -> str | None:
    """Return None if everything needed for image processing is importable, else why not."""
    missing: list[str] = []
    for module, package in (("PIL", "Pillow"), ("numpy", "numpy"), ("onnxruntime", "onnxruntime")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        return f"Missing Python packages: {', '.join(missing)} (pip install -r requirements.txt)"
    return None


# --------------------------------------------------------------------------- image helpers


def load_working_image(source: Path | bytes) -> Image.Image:
    """Open a photo, apply EXIF rotation, flatten transparency and bound its size."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    if img.width * img.height > MAX_PIXELS:
        raise PhotoRejected("Image is too large")
    if img.format == "JPEG":
        # Lets the JPEG decoder skip work: decode at reduced scale when the photo is huge.
        _ = img.draft("RGB", (WORK_MAX_EDGE, WORK_MAX_EDGE))
    _ = img.load()
    img = ImageOps.exif_transpose(img) or img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    else:
        img = img.convert("RGB")
    if max(img.size) > WORK_MAX_EDGE:
        img.thumbnail((WORK_MAX_EDGE, WORK_MAX_EDGE), Image.Resampling.LANCZOS)
    return img


def fit_to_canvas(cutout: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Crop to the visible subject and centre it, scaled to fit, on a transparent canvas."""
    from PIL import Image

    canvas_w, canvas_h = size
    rgba = cutout.convert("RGBA")
    visible = [255 if a > 10 else 0 for a in range(256)]  # alpha lookup table: ignore faint specks
    bbox = rgba.getchannel("A").point(visible).getbbox()
    subject = rgba.crop(bbox) if bbox else rgba

    avail_w = max(1, round(canvas_w * (1 - 2 * CANVAS_MARGIN)))
    avail_h = max(1, round(canvas_h * (1 - 2 * CANVAS_MARGIN)))
    scale = min(avail_w / subject.width, avail_h / subject.height)
    new_w = max(1, round(subject.width * scale))
    new_h = max(1, round(subject.height * scale))
    # Pillow resizes RGBA with premultiplied alpha, so edges don't pick up dark fringes.
    subject = subject.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(subject, ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2))  # no mask: copy RGBA verbatim
    return canvas


def _save_png_atomic(img: Image.Image, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, dest)


# --------------------------------------------------------------------------- background removal


class BackgroundRemover:
    """U²-Net / U²-Net-P segmentation through onnxruntime, with lazy model download."""

    _MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    _STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
    _SIZE: int = 320

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._sessions: dict[str, object] = {}
        self._failures: dict[str, tuple[float, str]] = {}

    def model_path(self, model: str) -> Path:
        return get_model_dir() / self._spec(model).filename

    def model_ready(self, model: str) -> bool:
        spec = self._spec(model)
        path = self.model_path(model)
        return path.is_file() and path.stat().st_size == spec.size

    def last_error(self, model: str) -> str | None:
        failure = self._failures.get(model)
        if failure and time.time() - failure[0] < MODEL_RETRY_SECONDS:
            return failure[1]
        return None

    @staticmethod
    def _spec(model: str) -> ModelSpec:
        return MODEL_SPECS.get(model) or MODEL_SPECS[DEFAULT_MODEL]

    def warm_up(self, model: str, allow_download: bool) -> None:
        """Load the model so the first photo isn't slow.  Never raises."""
        try:
            if allow_download or self.model_ready(model):
                _ = self._get_session(model, allow_download=allow_download)
        except BackgroundRemovalUnavailable as e:
            logger.warning(f"📷 Robot photo background removal unavailable: {e}")

    def _get_session(self, model: str, allow_download: bool = True) -> object:
        with self._lock:
            if model not in MODEL_SPECS:
                model = DEFAULT_MODEL
            session = self._sessions.get(model)
            if session is not None:
                return session

            failure = self._failures.get(model)
            if failure and time.time() - failure[0] < MODEL_RETRY_SECONDS:
                raise BackgroundRemovalUnavailable(failure[1])

            try:
                session = self._load_session(model, allow_download)
            except BackgroundRemovalUnavailable as e:
                if allow_download:
                    self._failures[model] = (time.time(), str(e))
                raise
            _ = self._failures.pop(model, None)
            self._sessions[model] = session
            return session

    def _load_session(self, model: str, allow_download: bool) -> object:
        problem = check_dependencies()
        if problem:
            raise BackgroundRemovalUnavailable(problem)
        import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]

        spec = self._spec(model)
        path = self.model_path(model)
        if not self.model_ready(model):
            if not allow_download:
                raise BackgroundRemovalUnavailable("Model not downloaded yet")
            self._download(spec, path)

        options = ort.SessionOptions()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        # OBS is usually encoding on this same machine: keep inference from taking every core.
        options.intra_op_num_threads = 2
        try:
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]  # pyright: ignore[reportUnknownArgumentType]
            )
        except Exception as e:
            # A corrupt file would otherwise fail forever; drop it so the next try re-downloads.
            try:
                path.unlink()
            except OSError:
                pass
            raise BackgroundRemovalUnavailable(f"Could not load model {spec.filename}: {e}") from e
        logger.info(f"📷 Loaded background removal model: {path}")
        return session

    @staticmethod
    def _download(spec: ModelSpec, dest: Path) -> None:
        logger.info(f"📷 Downloading background removal model {spec.filename} ({spec.size // (1024 * 1024)} MB)...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(spec.url, headers={"User-Agent": "MatchBox"})
            with urllib.request.urlopen(request, timeout=30) as response, open(tmp, "wb") as out:  # pyright: ignore[reportAny]
                received = 0
                next_report = 25
                while True:
                    chunk = cast(bytes, response.read(1024 * 1024))  # pyright: ignore[reportAny]
                    if not chunk:
                        break
                    _ = out.write(chunk)
                    _ = digest.update(chunk)
                    received += len(chunk)
                    percent = received * 100 // spec.size
                    if percent >= next_report:
                        logger.info(f"📷 Model download {min(percent, 100)}%")
                        next_report += 25
            if digest.hexdigest() != spec.sha256:
                raise BackgroundRemovalUnavailable(f"Downloaded {spec.filename} failed its checksum")
            os.replace(tmp, dest)
            logger.info(f"📷 Model saved to {dest}")
        except BackgroundRemovalUnavailable:
            raise
        except Exception as e:
            raise BackgroundRemovalUnavailable(
                f"Could not download {spec.filename} ({e}). Connect to the internet once, "
                + f"or place the file in {dest.parent}"
            ) from e
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def predict(self, img: Image.Image, model: str) -> Image.Image:
        """Return an 8-bit mask (255 = robot) the same size as ``img``."""
        session = self._get_session(model)  # raises BackgroundRemovalUnavailable (e.g. missing packages) first

        import numpy as np
        from PIL import Image

        small = img.convert("RGB").resize((self._SIZE, self._SIZE), Image.Resampling.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        arr = arr / max(float(arr.max()), 1e-6)
        arr = (arr - np.array(self._MEAN, dtype=np.float32)) / np.array(self._STD, dtype=np.float32)
        tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)

        run = cast("Callable[..., list[object]]", getattr(session, "run"))
        input_name = cast(str, getattr(getattr(session, "get_inputs")()[0], "name"))  # pyright: ignore[reportAny]
        outputs = run(None, {input_name: tensor})
        pred = cast("npt.NDArray[np.float32]", np.asarray(outputs[0], dtype=np.float32)[0, 0])
        low, high = float(pred.min()), float(pred.max())
        pred = (pred - low) / max(high - low, 1e-8)
        scaled = (pred * 255.0).astype(np.uint8)
        mask = Image.fromarray(scaled).resize(img.size, Image.Resampling.LANCZOS)

        # Stretch the middle of the range so faint haze disappears and solid areas are fully opaque.
        lut = [min(255, max(0, round((v - 24) * 255 / (255 - 48)))) for v in range(256)]
        return mask.point(lut)


# --------------------------------------------------------------------------- job handling


@dataclass
class PhotoJob:
    team: str
    original: Path
    state: str = "queued"  # queued | processing | done | error
    message: str = ""
    bg_removed: bool = False
    updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "team": self.team,
            "state": self.state,
            "message": self.message,
            "bg_removed": self.bg_removed,
            "updated": self.updated,
        }


class RobotPhotoProcessor:
    """Stores uploads and processes them one at a time on a background thread."""

    def __init__(self, base_dir: Callable[[], Path], config: MatchBoxConfig) -> None:
        self._base_dir: Callable[[], Path] = base_dir
        self._config: MatchBoxConfig = config
        self._remover: BackgroundRemover = BackgroundRemover()
        self._queue: queue.Queue[PhotoJob | None] = queue.Queue()
        self._jobs: dict[str, PhotoJob] = {}
        self._jobs_lock: threading.Lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # ---- paths

    @property
    def photo_dir(self) -> Path:
        return self._base_dir() / PHOTO_DIRNAME

    def _subdir(self, name: str) -> Path:
        path = self.photo_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def image_path(self, kind: str, team: str) -> Path | None:
        """Path of a servable image ('processed' or 'thumbs'), or None.  ``team`` is validated."""
        if kind not in ("processed", "thumbs") or not TEAM_RE.match(team):
            return None
        path = self.photo_dir / kind / f"{team}.png"
        return path if path.is_file() else None

    # ---- settings

    def _canvas_size(self) -> tuple[int, int]:
        def clamp(value: object, default: int) -> int:
            try:
                return max(MIN_CANVAS, min(MAX_CANVAS, int(cast(int, value))))
            except (TypeError, ValueError):
                return default

        return clamp(self._config.robot_photo_width, 1200), clamp(self._config.robot_photo_height, 900)

    def _model(self) -> str:
        model = self._config.robot_photo_model
        return model if model in MODEL_SPECS else DEFAULT_MODEL

    def capabilities(self) -> dict[str, object]:
        model = self._model()
        width, height = self._canvas_size()
        dependency_error = check_dependencies()
        return {
            "width": width,
            "height": height,
            "model": model,
            "dependencies_ok": dependency_error is None,
            "model_ready": dependency_error is None and self._remover.model_ready(model),
            # Why cut-outs aren't happening (shown to photographers), if they aren't
            "problem": dependency_error or self._remover.last_error(model),
        }

    def download_model(self) -> None:
        """Download/load the configured model now (blocking)."""
        self._remover.warm_up(self._model(), allow_download=True)

    def start_warmup(self) -> None:
        """Load an already-downloaded model in the background.  Never downloads."""
        model = self._model()
        _ = threading.Thread(target=self._remover.warm_up, args=(model, False), daemon=True).start()

    # ---- submitting work

    def submit(self, team_raw: str, data: bytes) -> PhotoJob:
        """Validate and store an upload, then queue it for processing."""
        from PIL import Image

        team = normalize_team(team_raw)
        if not data:
            raise PhotoRejected("Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise PhotoRejected("Photo is too large")

        try:
            probe = Image.open(io.BytesIO(data))
            image_format = probe.format or ""
            if probe.width * probe.height > MAX_PIXELS:
                raise PhotoRejected("Photo has too many pixels")
            probe.verify()
        except PhotoRejected:
            raise
        except Exception:
            raise PhotoRejected("That file is not a valid photo") from None
        if image_format not in ALLOWED_FORMATS:
            raise PhotoRejected("Photo must be a JPEG, PNG or WebP image")

        originals = self._subdir("originals")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        original = originals / f"{team}_{stamp}.{ALLOWED_FORMATS[image_format]}"
        counter = 1
        while original.exists():
            original = originals / f"{team}_{stamp}-{counter}.{ALLOWED_FORMATS[image_format]}"
            counter += 1
        tmp = original.with_name(original.name + ".tmp")
        _ = tmp.write_bytes(data)
        os.replace(tmp, original)

        logger.info(f"📷 Robot photo received for team {team} ({len(data) // 1024} KB)")
        return self._enqueue(team, original)

    def _enqueue(self, team: str, original: Path) -> PhotoJob:
        job = PhotoJob(team=team, original=original)
        with self._jobs_lock:
            self._jobs[team] = job
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="robot-photos", daemon=True)
                self._worker.start()
        self._queue.put(job)
        return job

    def reprocess_all(self) -> int:
        """Re-run every team's most recent original (after changing size/model settings)."""
        latest: dict[str, Path] = {}
        originals = self.photo_dir / "originals"
        if originals.is_dir():
            for path in sorted(originals.iterdir()):
                match = re.match(r"^([0-9]{1,6})_", path.name)
                if match and path.suffix.lower().lstrip(".") in ALLOWED_FORMATS.values():
                    latest[match.group(1)] = path  # sorted by name == sorted by timestamp
        for team, path in latest.items():
            _ = self._enqueue(team, path)
        return len(latest)

    def get_job(self, team: str) -> PhotoJob | None:
        with self._jobs_lock:
            return self._jobs.get(team)

    def status(self, team: str) -> dict[str, object] | None:
        """Processing state for one team: the live job if there is one, else what is on disk."""
        job = self.get_job(team)
        if job is not None:
            return job.to_dict()
        path = self.image_path("processed", team)
        if path is None:
            return None
        return {"team": team, "state": "done", "updated": path.stat().st_mtime}

    def stop(self) -> None:
        # Only signal a running worker; a stale sentinel would make a later worker exit immediately.
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)

    # ---- listing

    def list_photos(self) -> list[dict[str, object]]:
        """All teams that have a processed photo or a job in flight, by team number."""
        entries: dict[str, dict[str, object]] = {}
        processed = self.photo_dir / "processed"
        if processed.is_dir():
            for path in processed.glob("*.png"):
                if not TEAM_RE.match(path.stem):
                    continue
                meta: dict[str, object] = {}
                try:
                    meta = cast("dict[str, object]", json.loads(path.with_suffix(".json").read_text()))
                except (OSError, ValueError):
                    pass
                entries[path.stem] = {
                    "team": path.stem,
                    "state": "done",
                    "has_image": True,
                    "bg_removed": bool(meta.get("bg_removed", True)),
                    "message": str(meta.get("message", "")),
                    "updated": path.stat().st_mtime,
                }
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.state != "done":  # finished jobs are already represented by their files
                entries[job.team] = {**entries.get(job.team, {}), **job.to_dict()}
        return sorted(entries.values(), key=lambda e: int(cast(str, e["team"])))

    # ---- worker

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._process(job)
            except Exception as e:
                logger.error(f"📷 Failed to process robot photo for team {job.team}: {e}")
                job.state = "error"
                job.message = str(e) or type(e).__name__
                job.updated = time.time()

    def _process(self, job: PhotoJob) -> None:
        job.state = "processing"
        job.updated = time.time()
        started = time.time()

        img = load_working_image(job.original)
        cutout, removed, message = self._cut_out(img)
        canvas = fit_to_canvas(cutout, self._canvas_size())

        from PIL import Image

        processed_dir = self._subdir("processed")
        thumbs_dir = self._subdir("thumbs")
        thumb_h = round(THUMB_WIDTH * canvas.height / canvas.width)
        thumb = canvas.resize((THUMB_WIDTH, thumb_h), Image.Resampling.LANCZOS)
        _save_png_atomic(thumb, thumbs_dir / f"{job.team}.png")
        meta = {"bg_removed": removed, "message": message, "original": job.original.name}
        _ = (processed_dir / f"{job.team}.json").write_text(json.dumps(meta))
        _save_png_atomic(canvas, processed_dir / f"{job.team}.png")  # last: its mtime marks completion

        job.bg_removed = removed
        job.message = message
        job.state = "done"
        job.updated = time.time()
        logger.info(
            f"📷 Team {job.team} photo ready in {time.time() - started:.1f}s"
            + ("" if removed else f" (background NOT removed: {message})")
        )

    def _cut_out(self, img: Image.Image) -> tuple[Image.Image, bool, str]:
        """Return (RGBA image, background_removed, note)."""
        rgba = img.convert("RGBA")
        try:
            mask = self._remover.predict(img, self._model())
        except BackgroundRemovalUnavailable as e:
            return rgba, False, str(e)
        except Exception as e:
            logger.error(f"📷 Background removal failed: {e}")
            return rgba, False, f"Background removal failed: {e}"

        histogram = mask.histogram()
        coverage = sum(histogram[128:]) / max(1, mask.width * mask.height)
        if coverage < 0.005:
            return rgba, False, "No robot detected in the photo"
        rgba.putalpha(mask)
        return rgba, True, ""
