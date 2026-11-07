# Background removal experiments

A test bench for comparing ways of cutting robots out of photos for the robot photo
feature (`robot_photos.py`). Nothing here is used by MatchBox at runtime.

## Quick start

```bash
pip install -r experiments/background_removal/requirements.txt

# 1. Optional: synthetic robot scenes with exact ground truth (holes, thin parts, shadows)
python experiments/background_removal/synthetic.py experiments/background_removal/data/synthetic

# 2. Run the methods on any mix of folders/files (real robot photos are the real test)
python experiments/background_removal/benchmark.py \
    experiments/background_removal/data/synthetic  path/to/robot/photos

# 3. Open experiments/background_removal/results/report.html
```

Useful options:

- `--methods u2net,birefnet-512`: run only some methods. `all` adds the ~1 GB, ~14 GB-RAM `birefnet`.
- `--masks DIR`: use your own ground truth. A mask is a PNG with the same name as the photo, white = robot.
- `--threads N`: onnxruntime threads. The default of 2 matches MatchBox.
- `--no-arena`: turn off onnxruntime's memory arena. This roughly halves peak RAM for the big models and costs 20–60% speed. A big model in MatchBox would run this way.
- `--keep-main`: drop blobs that are separate from, and much smaller than, the main subject, such as a loose ball on the floor.
- `--no-stretch`: compare raw model masks, without MatchBox's post-processing.

`data/` and `results/` are git-ignored. Models download on first use (SHA-256 verified) into `~/.u2net`, or `$U2NET_HOME`, the same cache MatchBox uses. On Linux the report also shows each method's peak RAM.

## Methods

| Name | Type | Download | Notes |
|---|---|---|---|
| `u2net` | ML | 176 MB | U²-Net at 320 px, the current MatchBox default |
| `u2netp` | ML | 4.6 MB | Tiny U²-Net, the current MatchBox "fast" option |
| `silueta` | ML | 44 MB | U²-Net distilled by rembg |
| `isnet` | ML | 179 MB | IS-Net (DIS) general use, 1024 px |
| `birefnet-lite` | ML | 224 MB | BiRefNet general, Swin-T backbone, 1024 px |
| `birefnet-512` | ML | 941 MB | BiRefNet general, Swin-L backbone, the BiRefNet authors' own 512 px export |
| `birefnet` | ML | 973 MB | BiRefNet general, Swin-L backbone, 1024 px. Only runs with `--methods all` or by name, and needed ~14 GB of RAM |
| `grabcut` | classical | – | OpenCV GrabCut from a rectangle 5% inside the frame |
| `backdrop-key` | classical | – | Colour key against the frame border. Only meant for a plain photo backdrop |
| `u2net+grabcut` | hybrid | 176 MB | U²-Net mask used as a GrabCut trimap, to open up holes |
| `isnet+grabcut` | hybrid | 179 MB | The same refinement applied to IS-Net |

The ML models are ONNX exports, from the rembg project's releases or (for `birefnet-512`) the [BiRefNet authors' releases](https://github.com/ZhengPeng7/BiRefNet/releases). They run directly through `onnxruntime` with each model's own pre/post-processing, as MatchBox already does for U²-Net.

BRIA RMBG 2.0 is not included. Its weights are licensed for non-commercial use only (CC BY-NC 4.0), which doesn't fit MatchBox's MIT license.

To add a method, subclass `Method` in `methods.py`, implement `predict(img) -> L mask`, and register it in `build_methods()`.

## Scores (photos with ground truth)

- **IoU / MAE**: overall mask quality.
- **Holes kept**: how opaque the background seen *through* the robot came out. Robot frames are full of lightening holes and open windows. Lower is better.
- **Thin parts kept**: how much of the slide rails and wires survived. Higher is better.
- **Shadow kept**: how much of the floor shadow was wrongly kept. Lower is better.

The synthetic scenes are a structural stress test, not a substitute for real photos. As the findings below show, they can be misleading on their own.

## Findings

Hardware: a 4-core x86 cloud VM, CPU only, onnxruntime 1.29 with 2 threads (MatchBox's setting). Timings and memory will differ on event laptops; compare methods relative to each other.

### Round 2: real robot photos (5 event photos) + synthetic scenes

The real photos covered:

- a robot on the pit floor with an open U-shaped chassis;
- a robot standing on a grey cart in the pits;
- a robot on the field next to loose balls;
- a robot seen through the field wall;
- two robots side by side.

There's no ground truth for these, so they were judged by eye from the cut-outs, including zoomed-in crops. The photos themselves aren't in the repo.

| Method | Download | Peak RAM¹ | Per photo¹ | Synthetic IoU ↑ | Real robot photos |
|---|---|---|---|---|---|
| `u2net` (current) | 176 MB | 0.9 GB | 0.6 s | 0.465 | Solid robots and usually no stray background. It **fills open frames with floor** (the U-chassis came out solid) and leaves soft, smeared edges |
| `isnet` | 179 MB | 1.3 GB | 1.7 s | 0.759 | **Disqualified.** It made the front half of the cart robot ~40% see-through (panel opacity 0.58–0.63, checked with both published normalisations), and it kept field tape lines |
| `birefnet-lite` | 224 MB | 6.9 GB | 18 s | 0.806 | Excellent: crisp edges and open frames. It kept the cart's handle |
| **`birefnet-512`** | 941 MB | **3.3 GB** | **7.4 s** | **0.824** | Excellent: as clean as BiRefNet-lite, slightly softer edges, and **the only method that left the cart out**. It kept one strip of field tape |

¹ With `--no-arena`. With the arena on, `birefnet-512` runs at 6.1 s but peaks at 4.5 GB, and `birefnet-lite` at ~11 s but grows to 12 GB over several photos.

All methods kept both robots in the two-robot photo. That's expected, since both are equally prominent, and only the photographer knows which one is meant. Every method also kept a loose ball on the floor beside one robot. `--keep-main` removed that ball and changed nothing else: there was no IoU change on any synthetic scene for any method.

**Takeaways**

1. **Real photos overturned the synthetic winner.** IS-Net looked like the clear upgrade on synthetic scenes but fails badly on a real robot. Don't adopt a model on synthetic scores alone.
2. **BiRefNet is clearly the best on real robots**: open frames, solid bodies, crisp edges.
3. **`birefnet-512` is the practical BiRefNet.** It scores best on synthetic scenes, needs half the RAM of `birefnet-lite`, and runs twice as fast. Its cost is a 941 MB one-time download, and about 3.3 GB of RAM while processing (~4.5 GB with the arena on).
4. **U²-Net remains the low-resource fallback.** It's fast and light and decent on real photos, but it fills see-through gaps in frames.
5. **Full BiRefNet at 1024 px is impractical**: it was killed after reaching ~14 GB of RAM.
6. **`--keep-main` is worth shipping.** MatchBox crops to the visible subject before resizing, so a stray blob far from the robot would otherwise shrink the robot on the final canvas.

### Round 1: synthetic scenes only

The first round used 6 synthetic scenes (2 pit, 2 field, 2 plain backdrop) plus non-robot real photos. These numbers were taken with the arena on.

| Method | Download | Per photo | IoU ↑ | MAE ↓ | Holes kept ↓ | Thin parts kept ↑ | Shadow kept ↓ |
|---|---|---|---|---|---|---|---|
| `u2net` (current default) | 176 MB | 0.54 s | 0.465 | 0.065 | 61% | 62% | 5% |
| `u2netp` | 5 MB | 0.25 s | 0.405 | 0.055 | 37% | 47% | 1% |
| `silueta` | 44 MB | 0.81 s | 0.448 | 0.067 | 61% | 58% | 5% |
| `isnet` | 179 MB | 1.40 s | 0.759 | 0.021 | 16% | 87% | 1% |
| `birefnet-lite` | 224 MB | 11.3 s | 0.806 | 0.016 | 13% | 98% | 0% |
| `grabcut` | – | 5.8 s | 0.207 | 0.278 | 34% | 96% | 47% |
| `backdrop-key` | – | 0.12 s | 0.173 | 0.062 | 1% | 22% | 21% |
| `u2net+grabcut` | 176 MB | 1.44 s | 0.501 | 0.057 | 51% | 63% | 9% |
| `isnet+grabcut` | 179 MB | 2.36 s | 0.738 | 0.022 | 16% | 84% | 1% |

On a chroma-green backdrop (`synthetic.py --kinds greenscreen --seed 500`, 3 scenes), the IoU scores were:

- `isnet` 0.939
- `birefnet-lite` 0.907
- `backdrop-key` 0.752, keeping 56% of the floor shadow
- `u2net` 0.476, keeping 84% of the holes

U²-Net's frame-filling is the model's own, not caused by busy backgrounds.

Ruled out in round 1:

- **GrabCut refinement** doesn't pay off: +0.04 IoU on U²-Net, −0.02 on IS-Net, and a second or more per photo.
- **Plain GrabCut** is far behind every model.
- **Colour keying** needs a coloured backdrop, because aluminium is grey. Even then it keeps shadows.
- **`u2netp` and `silueta`** are worse or slower than U²-Net.
- **More threads** don't change any masks; 4 threads instead of 2 is ~1.5× faster.

### Suggested next steps

1. **Make `birefnet-512` MatchBox's default**, if a 941 MB one-time download and ~3.3 GB of RAM while processing are acceptable on event laptops. Run it with the arena off and keep U²-Net as the low-resource option. Since processing takes ~7 s, let the photographer move on to the next robot while a photo is processed, instead of waiting on the result screen.
2. **Ship `--keep-main` style cleanup.** In MatchBox this needs either `opencv-python-headless` or a small connected-components pass on a downscaled mask.
3. **Optionally re-export BiRefNet-lite at 512–768 px** from its PyTorch weights (170 MB, MIT). That could get BiRefNet quality at a much smaller download and memory footprint. No such export is published.
