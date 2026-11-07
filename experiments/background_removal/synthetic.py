"""
Synthetic robot photos with exact ground truth.

Draws an FTC-style robot (aluminium channel with lightening holes, mecanum wheels, an open
upright frame, linear slide rails, a claw, wiring and a team number sign) and composites it
onto pit / field / plain-backdrop scenes with a soft floor shadow, phone-like noise and JPEG
compression.  For every scene ``<name>`` it writes::

    <out>/<name>.jpg               the photo
    <out>/masks/<name>.png         ground-truth alpha (robot = 255)
    <out>/masks/<name>.holes.png   background seen *through* the robot (frame windows, channel holes)
    <out>/masks/<name>.thin.png    thin parts (slide rails, wires)
    <out>/masks/<name>.shadow.png  the robot's floor shadow (should end up transparent)

This is a stress test for structure (holes, thin parts, shadows), not a substitute for real
robot photos - run the benchmark on those as well.

    python experiments/background_removal/synthetic.py data/synthetic --count 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1600, 1200
SS = 2  # draw at 2x and downsample, for anti-aliased edges

Float = npt.NDArray[np.float32]
Rng = np.random.Generator

SILVER = (186, 190, 196, 255)
SILVER_HI = (232, 235, 240, 255)
SILVER_LO = (118, 122, 130, 255)
CLEAR = (0, 0, 0, 0)


def s(v: float) -> int:
    return int(round(v * SS))


def box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    return (s(x0), s(y0), s(x1), s(y1))


# --------------------------------------------------------------------------- robot


def _bar(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, holes: bool, rng: Rng) -> None:
    """Aluminium channel: silver bar with edge highlights and (optionally) a row of holes."""
    draw.rectangle(box(x0, y0, x1, y1), fill=SILVER)
    horizontal = (x1 - x0) >= (y1 - y0)
    edge = max(1.0, min(x1 - x0, y1 - y0) * 0.12)
    if horizontal:
        draw.rectangle(box(x0, y0, x1, y0 + edge), fill=SILVER_HI)
        draw.rectangle(box(x0, y1 - edge, x1, y1), fill=SILVER_LO)
    else:
        draw.rectangle(box(x0, y0, x0 + edge, y1), fill=SILVER_HI)
        draw.rectangle(box(x1 - edge, y0, x1, y1), fill=SILVER_LO)
    if not holes:
        return
    thick = min(x1 - x0, y1 - y0)
    d = thick * float(rng.uniform(0.38, 0.5))
    step = d * float(rng.uniform(1.5, 1.9))
    length = (x1 - x0) if horizontal else (y1 - y0)
    n = int((length - step) // step)
    start = ((length - n * step) / 2) + step / 2
    for i in range(n):
        t = start + i * step
        cx, cy = ((x0 + t, (y0 + y1) / 2) if horizontal else ((x0 + x1) / 2, y0 + t))
        draw.ellipse(box(cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2), fill=CLEAR)  # punched: see-through


def _wheel(width: float, height: float, rng: Rng) -> Image.Image:
    """Mecanum wheel seen from the front: dark hub with diagonal rollers."""
    tile = Image.new("RGBA", (s(width), s(height)), (38, 38, 42, 255))
    d = ImageDraw.Draw(tile)
    step = height / 7
    direction = 1 if rng.random() < 0.5 else -1
    for i in range(-3, 12):
        y = i * step
        d.polygon([(s(0), s(y)), (s(width), s(y + direction * width * 0.6)),
                   (s(width), s(y + direction * width * 0.6 + step * 0.45)), (s(0), s(y + step * 0.45))],
                  fill=(92, 94, 100, 255))
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile.width - 1, tile.height - 1), radius=s(width * 0.25), fill=255)
    tile.putalpha(mask)
    return tile


def draw_robot(rng: Rng) -> tuple[Image.Image, Image.Image, tuple[float, float, float, float]]:
    """Return (robot RGBA at final size, thin-parts mask at final size, (cx, floor_y, width, wheel_r))."""
    robot = Image.new("RGBA", (W * SS, H * SS), CLEAR)
    thin = Image.new("L", robot.size, 0)
    draw = ImageDraw.Draw(robot)
    thin_draw = ImageDraw.Draw(thin)

    rw = W * float(rng.uniform(0.42, 0.56))
    cx = W / 2 + W * float(rng.uniform(-0.05, 0.05))
    floor_y = H * float(rng.uniform(0.82, 0.88))
    wheel_r = rw * float(rng.uniform(0.085, 0.105))
    x0, x1 = cx - rw / 2, cx + rw / 2
    wheel_w = rw * 0.11
    chassis_h = rw * 0.075
    chassis_y1 = floor_y - wheel_r * 0.9
    chassis_y0 = chassis_y1 - chassis_h

    # Wheels
    for wx in (x0, x1 - wheel_w):
        tile = _wheel(wheel_w, wheel_r * 2, rng)
        robot.alpha_composite(tile, (s(wx), s(floor_y - wheel_r * 2)))

    # Chassis channel (with holes)
    _bar(draw, x0 + wheel_w * 0.55, chassis_y0, x1 - wheel_w * 0.55, chassis_y1, True, rng)

    # Open upright frame: four channels around a see-through window
    t = rw * 0.036
    fx0, fx1 = x0 + rw * 0.12, x1 - rw * 0.12
    frame_top = chassis_y0 - rw * float(rng.uniform(0.36, 0.5))
    _bar(draw, fx0, frame_top, fx0 + t, chassis_y0, True, rng)
    _bar(draw, fx1 - t, frame_top, fx1, chassis_y0, True, rng)
    _bar(draw, fx0, frame_top, fx1, frame_top + t, True, rng)

    # Cross bar + control hub inside the window
    win_h = chassis_y0 - frame_top
    hub_cy = frame_top + win_h * float(rng.uniform(0.55, 0.7))
    _bar(draw, fx0, hub_cy - t * 0.4, fx1, hub_cy + t * 0.4, False, rng)
    hub_w, hub_h = (fx1 - fx0) * 0.26, win_h * 0.2
    hub_x0 = cx - hub_w / 2 + rw * float(rng.uniform(-0.08, 0.08))
    draw.rounded_rectangle(box(hub_x0, hub_cy - hub_h / 2, hub_x0 + hub_w, hub_cy + hub_h / 2), radius=s(hub_h * 0.12), fill=(232, 112, 24, 255))
    draw.rectangle(box(hub_x0, hub_cy - hub_h / 2, hub_x0 + hub_w, hub_cy - hub_h / 2 + hub_h * 0.22), fill=(24, 24, 26, 255))

    # Team number sign (alliance colour)
    sign_w, sign_h = rw * 0.24, rw * 0.075
    sx = fx0 + t * 1.5
    sy = chassis_y0 - sign_h - t * 0.3
    colour = (196, 38, 38, 255) if rng.random() < 0.5 else (32, 72, 196, 255)
    draw.rectangle(box(sx, sy, sx + sign_w, sy + sign_h), fill=colour)
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.load_default(size=s(sign_h * 0.7))
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    draw.text((s(sx + sign_w * 0.08), s(sy + sign_h * 0.12)), str(int(rng.integers(1000, 30000))), fill=(250, 250, 250, 255), font=font)

    # Linear slide: two thin rails rising above the frame, cross member, claw
    rail_w = max(5.0, rw * 0.017)
    slide_half = rw * float(rng.uniform(0.12, 0.18))
    slide_top = frame_top - rw * float(rng.uniform(0.12, 0.3))
    for rx in (cx - slide_half, cx + slide_half - rail_w):
        draw.rectangle(box(rx, slide_top, rx + rail_w, chassis_y0), fill=SILVER)
        draw.rectangle(box(rx, slide_top, rx + rail_w * 0.25, chassis_y0), fill=SILVER_HI)
        thin_draw.rectangle(box(rx, slide_top, rx + rail_w, chassis_y0), fill=255)
    draw.rectangle(box(cx - slide_half, slide_top, cx + slide_half, slide_top + rail_w * 1.4), fill=SILVER_LO)
    thin_draw.rectangle(box(cx - slide_half, slide_top, cx + slide_half, slide_top + rail_w * 1.4), fill=255)
    claw_w = rw * 0.12
    cy0 = slide_top + rail_w * 1.4
    draw.rectangle(box(cx - claw_w / 2, cy0, cx + claw_w / 2, cy0 + rw * 0.04), fill=(52, 52, 58, 255))
    finger = rw * 0.085
    for side in (-1, 1):
        bx = cx + side * claw_w / 2
        draw.polygon([(s(bx), s(cy0 + rw * 0.04)), (s(bx - side * rail_w * 1.6), s(cy0 + rw * 0.04)),
                      (s(bx - side * claw_w * 0.35), s(cy0 + rw * 0.04 + finger)),
                      (s(bx - side * claw_w * 0.2), s(cy0 + rw * 0.04 + finger))], fill=(52, 52, 58, 255))

    # Wires from the hub to the motors / slide (thin, coloured)
    for _ in range(int(rng.integers(3, 6))):
        start = (hub_x0 + hub_w * float(rng.uniform(0.1, 0.9)), hub_cy + hub_h / 2)
        end = (float(rng.choice([x0 + wheel_w * 0.5, x1 - wheel_w * 0.5, cx - slide_half, cx + slide_half])),
               chassis_y0 - rw * 0.01)
        ctrl = (start[0] + float(rng.uniform(-0.15, 0.15)) * rw, (start[1] + end[1]) / 2 + float(rng.uniform(-0.05, 0.12)) * rw)
        pts = [((1 - u) ** 2 * start[0] + 2 * (1 - u) * u * ctrl[0] + u ** 2 * end[0],
                (1 - u) ** 2 * start[1] + 2 * (1 - u) * u * ctrl[1] + u ** 2 * end[1]) for u in np.linspace(0, 1, 40)]
        width = s(float(rng.uniform(2.5, 4.5)))
        colour = [(200, 30, 30, 255), (20, 20, 20, 255), (230, 200, 30, 255), (30, 80, 200, 255)][int(rng.integers(0, 4))]
        scaled = [(s(x), s(y)) for x, y in pts]
        draw.line(scaled, fill=colour, width=width, joint="curve")
        thin_draw.line(scaled, fill=255, width=width, joint="curve")

    # Lighting: brighter at the top, plus a little texture noise
    arr = np.asarray(robot, dtype=np.float32)
    shade = np.linspace(1.06, 0.84, arr.shape[0], dtype=np.float32)[:, None, None]
    arr[..., :3] = np.clip(arr[..., :3] * shade + rng.normal(0, 4, arr[..., :3].shape), 0, 255)
    robot = Image.fromarray(arr.astype(np.uint8), "RGBA").resize((W, H), Image.Resampling.LANCZOS)
    thin_final = thin.resize((W, H), Image.Resampling.LANCZOS)
    return robot, thin_final, (cx, floor_y, rw, wheel_r)


# --------------------------------------------------------------------------- backgrounds


def _noise(rng: Rng, sigma: float) -> Float:
    return rng.normal(0, sigma, (H, W, 1)).astype(np.float32)


def backdrop(rng: Rng, base: Float | None = None) -> Float:
    """Plain photo backdrop (grey/pastel by default)."""
    if base is None:
        base = np.array(rng.uniform(150, 215, 3), dtype=np.float32)
    top, bottom = base * 1.05, base * 0.86
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = top * (1 - y) + bottom * y + np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1 - 0.18 * np.clip(r - 0.4, 0, 1))[..., None]
    return img + _noise(rng, 2.5)


def greenscreen(rng: Rng) -> Float:
    """Chroma-green photo backdrop, as a photo booth could use."""
    return backdrop(rng, np.array([rng.uniform(40, 70), rng.uniform(150, 190), rng.uniform(60, 95)], dtype=np.float32))


def field(rng: Rng) -> Float:
    horizon = H * float(rng.uniform(0.4, 0.5))
    wall = Image.new("RGB", (W, H), (40, 40, 46))
    d = ImageDraw.Draw(wall)
    for _ in range(140):  # out-of-focus audience / lights
        x, y, rad = float(rng.uniform(0, W)), float(rng.uniform(0, horizon)), float(rng.uniform(15, 70))
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=tuple(int(v) for v in rng.uniform(30, 230, 3)))
    img = np.asarray(wall.filter(ImageFilter.GaussianBlur(22)), dtype=np.float32).copy()

    # Floor: grey foam tiles with perspective seams
    tile = float(rng.uniform(118, 140))
    floor = np.zeros((H, W, 3), np.float32) + tile
    floor_img = Image.fromarray(np.clip(floor, 0, 255).astype(np.uint8))
    fd = ImageDraw.Draw(floor_img)
    vx, vy = W / 2 + float(rng.uniform(-200, 200)), horizon - H * 0.9
    for k in range(-14, 15):
        bx = W / 2 + k * W * 0.16
        fd.line([(vx + (bx - vx) * ((horizon - vy) / (H - vy)), horizon), (bx, H)], fill=(96, 96, 100), width=3)
    for k in range(1, 9):
        y = horizon + (H - horizon) * (k / 8) ** 1.7
        fd.line([(0, y), (W, y)], fill=(98, 98, 102), width=3)
    floor = np.asarray(floor_img, dtype=np.float32) + _noise(rng, 6)
    rows = np.arange(H)[:, None, None] >= horizon
    img = np.where(rows, floor, img)

    # Polycarbonate perimeter with aluminium posts along the horizon
    band0 = horizon - H * 0.14
    band = (np.arange(H) >= band0) & (np.arange(H) < horizon)
    img[band] = img[band] * 0.75 + 255 * 0.25
    post_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pd = ImageDraw.Draw(post_img)
    for px in np.arange(float(rng.uniform(0, 300)), W, 330):
        pd.rectangle((px, band0, px + 16, horizon), fill=(170, 172, 178))
    pd.rectangle((0, band0 - 8, W, band0 + 6), fill=(170, 172, 178))
    return np.asarray(post_img, dtype=np.float32) + _noise(rng, 2.0)


def pit(rng: Rng) -> Float:
    horizon = H * float(rng.uniform(0.5, 0.62))
    wall_col = tuple(int(v) for v in rng.uniform(150, 210, 3))
    wall = Image.new("RGB", (W, H), wall_col)
    d = ImageDraw.Draw(wall)
    for shelf_y in np.linspace(H * 0.08, horizon - 40, 4):
        d.rectangle((0, shelf_y, W, shelf_y + 14), fill=(110, 84, 58))
        x = float(rng.uniform(-60, 0))
        while x < W:  # bins, boxes, spare parts on the shelves
            bw, bh = float(rng.uniform(60, 220)), float(rng.uniform(40, 120))
            colour = tuple(int(v) for v in rng.uniform(20, 240, 3))
            d.rectangle((x, shelf_y - bh, x + bw, shelf_y), fill=colour)
            if rng.random() < 0.35:  # metal extrusions as distractors
                d.rectangle((x + 8, shelf_y - bh - 30, x + 20, shelf_y - bh + 4), fill=(180, 182, 186))
            x += bw + float(rng.uniform(5, 50))
    d.rectangle((W * 0.15, H * 0.02, W * 0.55, H * 0.07), fill=(30, 60, 160))  # team banner
    img = np.asarray(wall.filter(ImageFilter.GaussianBlur(3.5)), dtype=np.float32).copy()
    table = np.array(rng.uniform(70, 150, 3), dtype=np.float32)
    rows = np.arange(H)[:, None, None] >= horizon
    img = np.where(rows, table + _noise(rng, 9), img)
    img[int(horizon):int(horizon) + 6] *= 0.6  # table edge
    return img + _noise(rng, 3.0)


BACKGROUNDS = {"pit": pit, "field": field, "backdrop": backdrop, "greenscreen": greenscreen}


# --------------------------------------------------------------------------- scenes


def holes_mask(alpha: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Background pixels fully enclosed by the robot (flood fill from outside)."""
    import cv2

    solid = (alpha > 127).astype(np.uint8)
    padded = np.pad(solid, 1)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    _ = cv2.floodFill(flood, ff_mask, (0, 0), 2)
    outside = flood[1:-1, 1:-1] == 2
    return ((~outside) & (solid == 0)).astype(np.uint8) * 255


def make_scene(kind: str, rng: Rng) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    robot, thin, (cx, floor_y, rw, wheel_r) = draw_robot(rng)
    bg = BACKGROUNDS[kind](rng)

    # Soft contact shadow under the robot
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).ellipse((cx - rw * 0.58, floor_y - wheel_r * 0.35, cx + rw * 0.58, floor_y + wheel_r * 0.45), fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(rw * 0.035))
    sh = np.asarray(shadow, dtype=np.float32)[..., None] / 255.0
    bg = bg * (1 - 0.5 * sh)

    rgba = np.asarray(robot, dtype=np.float32)
    a = rgba[..., 3:4] / 255.0
    photo = rgba[..., :3] * a + bg * (1 - a)
    photo_img = Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    noisy = np.asarray(photo_img, dtype=np.float32) + rng.normal(0, 2.5, (H, W, 3))
    photo_img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))

    alpha = np.asarray(robot.getchannel("A"))
    thin_vis = np.minimum(np.asarray(thin), alpha)
    shadow_only = ((sh[..., 0] > 0.3) & (alpha < 13)).astype(np.uint8) * 255
    return (photo_img, Image.fromarray(alpha), Image.fromarray(holes_mask(alpha)),
            Image.fromarray(thin_vis), Image.fromarray(shadow_only))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("out", type=Path)
    _ = parser.add_argument("--count", type=int, default=6)
    _ = parser.add_argument("--seed", type=int, default=10298)
    _ = parser.add_argument("--kinds", default="pit,field,backdrop", help=f"comma separated, from: {', '.join(BACKGROUNDS)}")
    args = parser.parse_args()
    out: Path = args.out
    masks = out / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    kinds = [k.strip() for k in str(args.kinds).split(",") if k.strip()]
    unknown = [k for k in kinds if k not in BACKGROUNDS]
    if unknown:
        parser.error(f"unknown kinds: {', '.join(unknown)}")
    for i in range(int(args.count)):
        kind = kinds[i % len(kinds)]
        rng = np.random.default_rng(int(args.seed) + i)
        photo, alpha, holes, thin, shadow = make_scene(kind, rng)
        name = f"synthetic_{i + 1:02d}_{kind}"
        photo.save(out / f"{name}.jpg", quality=88)
        alpha.save(masks / f"{name}.png")
        holes.save(masks / f"{name}.holes.png")
        thin.save(masks / f"{name}.thin.png")
        shadow.save(masks / f"{name}.shadow.png")
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
