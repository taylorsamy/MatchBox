"""
Compare background removal methods on a set of photos.

    python experiments/background_removal/benchmark.py PHOTOS... [--methods u2net,isnet,...|all] [--out DIR]

PHOTOS are image files and/or folders.  If a ``masks/`` folder sits next to the photos (as
written by ``synthetic.py``) - or one is given with ``--masks`` - each photo is also scored
against its ground truth:

* IoU         overlap of predicted robot vs true robot (higher is better)
* MAE         mean alpha error over the whole image (lower is better)
* holes       how opaque the background seen *through* the robot came out (lower is better)
* thin        how much of the thin parts (rails, wires) survived (higher is better)
* shadow      how much of the floor shadow was kept (lower is better)

Photos go through MatchBox's own loader (EXIF rotation, size cap) and every mask gets the
same post-processing MatchBox applies, so the results reflect what would ship.  Timings use
the same 2 onnxruntime threads MatchBox uses.  Output (default ``results/`` next to this file):
``report.html`` (self-contained), ``results.json`` and full-size cut-outs in ``cutouts/``.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for robot_photos

import methods as methods_mod  # noqa: E402  # pyright: ignore[reportImplicitRelativeImport]  (run as a script)
from methods import DEFAULT_METHODS, Method, build_methods, keep_main_subject  # noqa: E402  # pyright: ignore[reportImplicitRelativeImport]
from robot_photos import load_working_image  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
REGIONS = ("holes", "thin", "shadow")
THUMB_W = 520
CUTOUT_MAX = 1600

# MatchBox's mask post-processing (robot_photos.BackgroundRemover.predict): drop faint haze, solidify the core
STRETCH_LUT = [min(255, max(0, round((v - 24) * 255 / (255 - 48)))) for v in range(256)]

Bool = npt.NDArray[np.bool_]
Float = npt.NDArray[np.float32]


def reset_peak_memory() -> None:
    """Linux: reset this process's peak-RSS counter, so each method's peak can be read separately."""
    try:
        _ = Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process since the last reset (Linux only), in MB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def find_photos(paths: list[Path]) -> list[Path]:
    photos: list[Path] = []
    for p in paths:
        if p.is_dir():
            photos.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.suffix.lower() in IMAGE_EXTS:
            photos.append(p)
    return photos


def load_truth(photo: Path, masks_dir: Path | None, size: tuple[int, int]) -> tuple[Float | None, dict[str, Bool]]:
    folder = masks_dir or photo.parent / "masks"
    gt_path = folder / f"{photo.stem}.png"
    if not gt_path.is_file():
        return None, {}

    def read(path: Path) -> Float:
        return np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    regions = {r: read(folder / f"{photo.stem}.{r}.png") > 0.5 for r in REGIONS if (folder / f"{photo.stem}.{r}.png").is_file()}
    return read(gt_path), regions


def score(pred: Float, gt: Float, regions: dict[str, Bool]) -> dict[str, float]:
    p, g = pred > 0.5, gt > 0.5
    union = int(np.count_nonzero(p | g))
    result = {
        "iou": float(np.count_nonzero(p & g)) / union if union else 1.0,
        "mae": float(np.mean(np.abs(pred - gt))),
    }
    for name, region in regions.items():
        if np.any(region):
            result[name] = float(np.mean(pred[region]))
    return result


# --------------------------------------------------------------------------- report helpers


def data_uri(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "WEBP":
        img.save(buf, "WEBP", quality=82, method=4)
    else:
        img.convert("RGB").save(buf, "JPEG", quality=82)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def thumb(img: Image.Image) -> Image.Image:
    h = round(img.height * THUMB_W / img.width)
    return img.resize((THUMB_W, h), Image.Resampling.LANCZOS)


def error_map(photo: Image.Image, pred: Float, gt: Float) -> Image.Image:
    """Dimmed photo; red = background kept as robot, blue = robot removed as background."""
    base = np.asarray(photo.convert("L").convert("RGB"), dtype=np.float32) * 0.35
    over = np.clip(pred - gt, 0, 1)[..., None]
    under = np.clip(gt - pred, 0, 1)[..., None]
    out = base * (1 - over - under) + over * np.array([255, 40, 40], np.float32) + under * np.array([40, 140, 255], np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


METRIC_INFO: list[tuple[str, str, bool]] = [  # key, label, higher_is_better
    ("iou", "IoU", True), ("mae", "MAE", False), ("holes", "Holes kept", False),
    ("thin", "Thin parts kept", True), ("shadow", "Shadow kept", False),
]


def fmt_metric(key: str, value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.3f}" if key in ("iou", "mae") else f"{value * 100:.0f}%"


def write_report(out: Path, runs: list[dict[str, object]], photos: list[dict[str, object]], env: dict[str, object]) -> None:
    summary_rows: list[str] = []
    means: dict[str, dict[str, float]] = {}
    for run in runs:
        name = cast(str, run["name"])
        means[name] = cast("dict[str, float]", run["mean_metrics"])
    best: dict[str, float] = {}
    for key, _, higher in METRIC_INFO:
        vals = [m[key] for m in means.values() if key in m]
        if vals:
            best[key] = max(vals) if higher else min(vals)
    times = [cast(float, r["mean_seconds"]) for r in runs if r.get("mean_seconds") is not None]
    fastest = min(times) if times else None

    for run in runs:
        name = cast(str, run["name"])
        m = means[name]
        cells = []
        for key, _, _ in METRIC_INFO:
            v = m.get(key)
            cls = ' class="best"' if v is not None and key in best and abs(v - best[key]) < 1e-9 else ""
            cells.append(f"<td{cls}>{fmt_metric(key, v)}</td>")
        secs = cast("float | None", run.get("mean_seconds"))
        t_cls = ' class="best"' if secs is not None and fastest is not None and abs(secs - fastest) < 1e-9 else ""
        error = f'<div class="err">{html.escape(cast(str, run["error"]))}</div>' if run.get("error") else ""
        peak = cast("float | None", run.get("peak_mb"))
        peak_txt = "–" if peak is None else f"{peak / 1024:.1f} GB"
        summary_rows.append(
            f"<tr><th>{html.escape(name)}<div class=\"desc\">{html.escape(cast(str, run['description']))}</div>{error}</th>"
            f"<td>{html.escape(cast(str, run['family']))}</td>"
            f"<td>{cast(int, run['download_bytes']) / 1e6:.0f} MB</td>"
            f"<td>{cast(float, run['load_seconds']):.1f} s</td>"
            f"<td>{peak_txt}</td>"
            f"<td{t_cls}>{'–' if secs is None else f'{secs:.2f} s'}</td>"
            + "".join(cells)
            + f"<td>{html.escape(cast(str, run['license']))}</td></tr>"
        )

    method_names = [cast(str, r["name"]) for r in runs]
    grid_rows: list[str] = []
    for photo in photos:
        cells = [f'<figure><img src="{photo["thumb"]}" alt=""><figcaption><b>{html.escape(cast(str, photo["name"]))}</b><br>original</figcaption></figure>']
        results = cast("dict[str, dict[str, object]]", photo["results"])
        for name in method_names:
            r = results.get(name)
            if not r:
                cells.append(f'<figure class="cut"><div class="missing">failed</div><figcaption>{html.escape(name)}</figcaption></figure>')
                continue
            metric_txt = ""
            mets = cast("dict[str, float]", r.get("metrics", {}))
            if mets:
                metric_txt = " · ".join(f"{label} {fmt_metric(k, mets.get(k))}" for k, label, _ in METRIC_INFO if k in mets)
            err_img = f'<img class="errmap" src="{r["error"]}" alt="">' if r.get("error") else ""
            cells.append(
                f'<figure class="cut"><div class="frame"><img class="cutout" src="{r["thumb"]}" alt="">{err_img}</div>'
                f'<figcaption><b>{html.escape(name)}</b> · {cast(float, r["seconds"]):.2f} s<br><span class="m">{metric_txt}</span></figcaption></figure>'
            )
        grid_rows.append(f'<div class="row">{"".join(cells)}</div>')

    metric_heads = "".join(f"<th>{label}</th>" for _, label, _ in METRIC_INFO)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Background Removal Comparison</title>
<style>
:root {{ --bg:#fafafa; --fg:#1d1d1f; --muted:#6e6e73; --line:#d9d9de; --card:#fff; --best:#1a7f37; --err:#c62828; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#18181a; --fg:#ececf0; --muted:#9a9aa2; --line:#36363c; --card:#222226; --best:#4cc26a; --err:#ff6b6b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:28px 0 10px; }}
.env {{ color:var(--muted); margin:0 0 18px; }}
.tablewrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; min-width:900px; }}
th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; vertical-align:top; }}
thead th {{ font-size:12px; color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--card); }}
tbody th {{ text-align:left; white-space:normal; min-width:220px; }}
td:last-child {{ text-align:left; color:var(--muted); }}
.desc {{ font-weight:400; color:var(--muted); font-size:12px; }}
.best {{ color:var(--best); font-weight:700; }}
.err {{ color:var(--err); font-weight:400; font-size:12px; white-space:normal; }}
.note {{ color:var(--muted); max-width:900px; }}
.controls {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:8px 0 14px; position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:2; }}
.controls button {{ border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:8px; padding:6px 12px; font:inherit; cursor:pointer; }}
.controls button.on {{ border-color:var(--fg); font-weight:600; }}
.grid {{ overflow-x:auto; }}
.row {{ display:flex; gap:10px; margin-bottom:14px; }}
figure {{ margin:0; flex:0 0 {THUMB_W // 2 + 40}px; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
figure img {{ width:100%; display:block; }}
.frame {{ position:relative; }}
.frame .errmap {{ position:absolute; inset:0; display:none; }}
body.showerr .frame .errmap {{ display:block; }}
figcaption {{ padding:6px 8px; font-size:12px; }}
figcaption .m {{ color:var(--muted); }}
.missing {{ aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; color:var(--err); }}
.cut .frame {{ background-color:#cfcfcf; background-image:linear-gradient(45deg,#a8a8a8 25%,transparent 25%,transparent 75%,#a8a8a8 75%),linear-gradient(45deg,#a8a8a8 25%,transparent 25%,transparent 75%,#a8a8a8 75%); background-size:16px 16px; background-position:0 0,8px 8px; }}
body.bg-white .cut .frame {{ background:#fff; }} body.bg-black .cut .frame {{ background:#000; }}
body.bg-green .cut .frame {{ background:#18c040; }} body.bg-magenta .cut .frame {{ background:#ff00ff; }}
</style></head>
<body>
<h1>Background removal comparison</h1>
<p class="env">{html.escape(str(env["photos"]))} photos · {html.escape(str(env["cpu"]))} · onnxruntime {html.escape(str(env["threads"]))} threads, memory arena {"on" if env.get("arena") else "off"} · {html.escape(str(env["date"]))}</p>
<div class="tablewrap"><table>
<thead><tr><th style="text-align:left">Method</th><th>Type</th><th>Download</th><th>Load</th><th>Peak RAM</th><th>Per photo</th>{metric_heads}<th>License</th></tr></thead>
<tbody>{"".join(summary_rows)}</tbody></table></div>
<p class="note">Scores are averages over the photos that have ground truth (synthetic scenes). "Holes kept" is how opaque the background seen through the robot came out, "Shadow kept" how much of the floor shadow survived: lower is better for both. Per-photo time is after a warm-up run. Peak RAM is the whole benchmark process (it includes the loaded photos) while that method ran. Best value per column in green.</p>
<h2>Cut-outs</h2>
<div class="controls">Background:
<button data-bg="checker" class="on">Checker</button><button data-bg="white">White</button><button data-bg="black">Black</button><button data-bg="green">Green</button><button data-bg="magenta">Magenta</button>
<span style="width:12px"></span><button id="errbtn">Show errors (red = kept background, blue = lost robot)</button></div>
<div class="grid">{"".join(grid_rows)}</div>
<script>
document.querySelectorAll('[data-bg]').forEach(function (b) {{
  b.addEventListener('click', function () {{
    document.body.className = document.body.className.replace(/\\bbg-\\w+/g, '').trim();
    if (b.dataset.bg !== 'checker') document.body.classList.add('bg-' + b.dataset.bg);
    document.querySelectorAll('[data-bg]').forEach(function (x) {{ x.classList.toggle('on', x === b); }});
  }});
}});
document.getElementById('errbtn').addEventListener('click', function () {{
  this.classList.toggle('on', document.body.classList.toggle('showerr'));
}});
</script>
</body></html>"""
    _ = (out / "report.html").write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("photos", nargs="+", type=Path, help="photo files and/or folders")
    _ = parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma separated, or 'all'")
    _ = parser.add_argument("--masks", type=Path, default=None, help="ground-truth folder (default: masks/ next to each photo)")
    _ = parser.add_argument("--out", type=Path, default=HERE / "results")
    _ = parser.add_argument("--threads", type=int, default=methods_mod.THREADS)
    _ = parser.add_argument("--no-stretch", action="store_true", help="skip MatchBox's mask post-processing")
    _ = parser.add_argument("--no-arena", action="store_true", help="disable onnxruntime's memory arena (less RAM, a bit slower)")
    _ = parser.add_argument("--keep-main", action="store_true", help="drop small blobs separate from the main subject (e.g. loose balls)")
    args = parser.parse_args()

    available = build_methods()
    wanted = list(available) if args.methods == "all" else [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in wanted if m not in available]
    if unknown:
        print(f"Unknown methods: {', '.join(unknown)}. Available: {', '.join(available)}")
        return 2
    methods_mod.THREADS = int(args.threads)
    methods_mod.ARENA = not bool(args.no_arena)

    photo_paths = find_photos(cast("list[Path]", args.photos))
    if not photo_paths:
        print("No photos found")
        return 2
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(photo_paths)} photos...")
    photos: list[dict[str, object]] = []
    for path in photo_paths:
        img = load_working_image(path)
        gt, regions = load_truth(path, cast("Path | None", args.masks), img.size)
        photos.append({"name": path.name, "stem": path.stem, "img": img, "gt": gt, "regions": regions,
                       "thumb": data_uri(thumb(img), "JPEG"), "results": {}})

    env: dict[str, object] = {
        "photos": len(photos),
        "cpu": f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores",
        "threads": methods_mod.THREADS,
        "arena": methods_mod.ARENA,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    runs: list[dict[str, object]] = []
    for name in wanted:
        method: Method = available[name]
        print(f"[{name}] {method.description}", flush=True)
        run: dict[str, object] = {"name": name, "family": method.family, "description": method.description,
                                  "license": method.license, "download_bytes": method.download_bytes,
                                  "load_seconds": 0.0, "mean_seconds": None, "mean_metrics": {}, "error": None,
                                  "peak_mb": None}
        runs.append(run)
        reset_peak_memory()
        try:
            t = time.perf_counter()
            method.load()
            run["load_seconds"] = time.perf_counter() - t
            _ = method.predict(cast(Image.Image, photos[0]["img"]))  # warm-up, not timed
        except Exception as e:
            run["error"] = f"{type(e).__name__}: {e}"
            print(f"  failed to load: {e}")
            traceback.print_exc()
            continue

        seconds: list[float] = []
        metric_lists: dict[str, list[float]] = {}
        for photo in photos:
            img = cast(Image.Image, photo["img"])
            try:
                t = time.perf_counter()
                mask = method.predict(img)
                dt = time.perf_counter() - t
            except Exception as e:
                print(f"  {photo['name']}: failed: {e}")
                continue
            if not args.no_stretch:
                mask = mask.point(STRETCH_LUT)
            if args.keep_main:
                mask = keep_main_subject(mask)
            seconds.append(dt)
            cutout = img.convert("RGBA")
            cutout.putalpha(mask)

            full = cutout.copy()
            full.thumbnail((CUTOUT_MAX, CUTOUT_MAX), Image.Resampling.LANCZOS)
            (out / "cutouts" / name).mkdir(parents=True, exist_ok=True)
            full.save(out / "cutouts" / name / f"{photo['stem']}.png")

            entry: dict[str, object] = {"seconds": dt, "thumb": data_uri(thumb(cutout), "WEBP")}
            gt = cast("Float | None", photo["gt"])
            if gt is not None:
                pred = np.asarray(mask, dtype=np.float32) / 255.0
                mets = score(pred, gt, cast("dict[str, Bool]", photo["regions"]))
                entry["metrics"] = mets
                for k, v in mets.items():
                    metric_lists.setdefault(k, []).append(v)
                entry["error"] = data_uri(thumb(error_map(img, pred, gt)), "JPEG")
            cast("dict[str, object]", photo["results"])[name] = entry
            print(f"  {photo['name']}: {dt:.2f} s" + (f"  IoU {cast('dict[str, float]', entry['metrics'])['iou']:.3f}" if "metrics" in entry else ""), flush=True)
        run["peak_mb"] = peak_memory_mb()
        method.unload()
        run["mean_seconds"] = float(np.mean(seconds)) if seconds else None
        run["mean_metrics"] = {k: float(np.mean(v)) for k, v in metric_lists.items()}
        # Save after every method: a method that runs out of memory gets the process killed,
        # which can't be caught, and shouldn't take the finished methods' results with it
        save(out, runs, photos, env)

    print(f"\nReport: {out / 'report.html'}")
    return 0


def save(out: Path, runs: list[dict[str, object]], photos: list[dict[str, object]], env: dict[str, object]) -> None:
    write_report(out, runs, photos, env)
    serializable = {
        "env": env,
        "methods": runs,
        "photos": [{"name": p["name"], "results": {m: {k: v for k, v in cast("dict[str, object]", r).items() if k in ("seconds", "metrics")}
                                                   for m, r in cast("dict[str, object]", p["results"]).items()}} for p in photos],
    }
    _ = (out / "results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
