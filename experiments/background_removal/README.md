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

- `--methods u2net,isnet,birefnet-lite`: run only some methods. `all` adds the ~1 GB BiRefNet.
- `--masks DIR`: use your own ground truth. A mask is a PNG with the same name as the photo, white = robot.
- `--threads N`: onnxruntime threads. The default of 2 matches MatchBox.
- `--no-stretch`: compare raw model masks, without MatchBox's post-processing.

`data/` and `results/` are git-ignored. Models download on first use (SHA-256 verified) into `~/.u2net`, or `$U2NET_HOME`, the same cache MatchBox uses.

## Methods

| Name | Type | Download | Notes |
|---|---|---|---|
| `u2net` | ML | 176 MB | U²-Net at 320 px, the current MatchBox default |
| `u2netp` | ML | 4.6 MB | Tiny U²-Net, the current MatchBox "fast" option |
| `silueta` | ML | 44 MB | U²-Net distilled by rembg |
| `isnet` | ML | 179 MB | IS-Net (DIS) general use, 1024 px |
| `birefnet-lite` | ML | 224 MB | BiRefNet general, Swin-T backbone, 1024 px |
| `birefnet` | ML | 973 MB | BiRefNet general, Swin-L backbone, 1024 px. Only runs with `--methods all` or by name, and needed ~14 GB of RAM |
| `grabcut` | classical | – | OpenCV GrabCut from a rectangle 5% inside the frame |
| `backdrop-key` | classical | – | Colour key against the frame border. Only meant for a plain photo backdrop |
| `u2net+grabcut` | hybrid | 176 MB | U²-Net mask used as a GrabCut trimap, to open up holes |
| `isnet+grabcut` | hybrid | 179 MB | The same refinement applied to IS-Net |

All ML models are the ONNX exports hosted by the rembg project. They run directly through `onnxruntime` with each model's own pre/post-processing, as MatchBox already does for U²-Net.

BRIA RMBG 2.0 is not included. Its weights are licensed for non-commercial use only (CC BY-NC 4.0), which doesn't fit MatchBox's MIT license.

To add a method, subclass `Method` in `methods.py`, implement `predict(img) -> L mask`, and register it in `build_methods()`.

## Scores (photos with ground truth)

- **IoU / MAE**: overall mask quality.
- **Holes kept**: how opaque the background seen *through* the robot came out. Robot frames are full of lightening holes and open windows. Lower is better.
- **Thin parts kept**: how much of the slide rails and wires survived. Higher is better.
- **Shadow kept**: how much of the floor shadow was wrongly kept. Lower is better.

The synthetic scenes are a structural stress test, not a substitute for real photos. Use them to compare methods on holes, thin parts and shadows, and judge final quality on real robot photos.

## Findings (first run, 2026-11-07)

**Setup:**

- Hardware: 4-core x86 cloud VM, CPU only, onnxruntime 1.29 with 2 threads (MatchBox's setting).
- Scored photos: 6 synthetic robot scenes (2 pit, 2 field, 2 plain grey/pastel backdrop) with ground truth.
- Visual check: 5 real photos without ground truth (motorcycle ×2, coffee cup, rocket, cat).
- **No real robot photos were available**, so treat this as a strong hint, not a verdict.

| Method | Download | Per photo | IoU ↑ | MAE ↓ | Holes kept ↓ | Thin parts kept ↑ | Shadow kept ↓ |
|---|---|---|---|---|---|---|---|
| `u2net` (current default) | 176 MB | 0.54 s | 0.465 | 0.065 | 61% | 62% | 5% |
| `u2netp` | 5 MB | 0.25 s | 0.405 | 0.055 | 37% | 47% | 1% |
| `silueta` | 44 MB | 0.81 s | 0.448 | 0.067 | 61% | 58% | 5% |
| **`isnet`** | 179 MB | 1.40 s | 0.759 | 0.021 | 16% | 87% | 1% |
| **`birefnet-lite`** | 224 MB | 11.3 s | **0.806** | **0.016** | **13%** | **98%** | **0%** |
| `grabcut` | – | 5.8 s | 0.207 | 0.278 | 34% | 96% | 47% |
| `backdrop-key` | – | 0.12 s | 0.173 | 0.062 | 1% | 22% | 21% |
| `u2net+grabcut` | 176 MB | 1.44 s | 0.501 | 0.057 | 51% | 63% | 9% |
| `isnet+grabcut` | 179 MB | 2.36 s | 0.738 | 0.022 | 16% | 84% | 1% |
| `birefnet` (Swin-L) | 973 MB | – | – | – | – | – | – |

**Green-screen photo booth** (3 synthetic scenes on a chroma-green backdrop, `synthetic.py --kinds greenscreen --seed 500`):

| Method | Per photo | IoU | Holes kept | Shadow kept |
|---|---|---|---|---|
| `backdrop-key` | 0.15 s | 0.752 | 5% | 56% |
| `u2net` | 0.58 s | 0.476 | 84% | 0% |
| `isnet` | 1.57 s | **0.939** | **4%** | 0% |
| `birefnet-lite` | 11.6 s | 0.907 | 7% | 0% |

With 4 threads instead of 2, `isnet` takes 0.90 s per photo and `birefnet-lite` 7.1 s. The masks are identical either way.

**Takeaways**

1. **The current U²-Net default fills in robot frames.** 61% of the background seen through the robot comes out opaque, and 84% even on a perfect green screen. The problem is the model at 320 px, not busy backgrounds. It also loses tall thin parts like slides and claws (62% kept), and on the real motorcycle it clips the mirrors.
2. **IS-Net is the clear upgrade.** It is the same size to download (179 vs 176 MB) and fits the same pipeline (only the input size and normalisation differ). IoU goes from 0.47 to 0.76, holes kept from 61% to 16%, and thin parts from 62% to 87%. It costs about 1.4 s per photo, and it was the best method on the green screen.
3. **BiRefNet-lite is the most accurate on busy backgrounds**, with the cleanest real-photo cut-outs and 98% of thin parts kept. It is about 8× slower than IS-Net (11 s per photo, or 7 s with 4 threads). In one field scene it also kept part of the perimeter rail where it touched the robot's slide. It's an option for a "best quality, slower" setting.
4. **Full BiRefNet is impractical.** It used about 14 GB of RAM while loading and was killed.
5. **GrabCut refinement doesn't pay off.** It adds 0.04 IoU to U²-Net, loses 0.02 on IS-Net, and adds a second or more per photo. Plain GrabCut is far behind every model.
6. **A backdrop alone isn't enough.** Colour keying fails against grey or pastel backdrops because aluminium is grey (IoU 0.17). On a green screen it reaches 0.75 but keeps over half of the floor shadow, and IS-Net still does better on the same green screen.
7. **`u2netp` and `silueta` are not worth using.** `u2netp` has the lowest quality of the ML models, and `silueta` is slower than U²-Net without being better.

**Suggested next step:** check IS-Net on real robot photos. If it holds up, make it MatchBox's default, and consider BiRefNet-lite as an opt-in "best quality" setting.

**Caveats:**

- The synthetic robots are flat-shaded drawings. Real photos add reflections, motion blur, hands holding the robot and partly occluded robots.
- All timings come from one CPU type.
- The ONNX models have fixed input sizes (320 or 1024 px). A faster low-resolution BiRefNet would mean re-exporting it from PyTorch.
