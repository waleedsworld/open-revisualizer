# gen/: the generative Reality View

`train/` builds a per-scene Gaussian splat of the drive. This folder tests the other route: AI generation that
re-renders the drive from the virtual chase camera. Earlier free video generation (Wan 2.1 VACE, Lyra) drifted
(about 15-16 dB revisit consistency), so every mode here is **geometry-anchored**. A 3D render supplies the
layout and a one-step image-to-image (I2I) diffusion model only repaints it and fills holes.

The chase camera is the viewer's `reality` view, exactly as in `web/revisualizer.js` but without its smoothing
lerp: ego `p`, yaw from `tracks.json`; position `p - 10.5 fwd + 5.6 z`; look-at `p + 14 fwd + 0.6 z`; up +z;
vertical FOV 52 degrees; 960x540; 105 frames at 10 fps (`common.chase_camera`).

## Modes

| mode | what it does | output |
|---|---|---|
| **lift** | Per frame and causal: the current lidar sweep is coloured from the 7 ring cameras, plus the static points of the last 10 sweeps, a persistent ground texture and mono-depth fill. Then point splatting into the chase camera. No training. | `gen/raw_lift/` |
| **lift + I2I** ("live") | lift, then sky prior and pull-push hole fill, then Difix3D+ (one step, front cameras as reference), then the previous output warped in and blended | `web/scenes/av2-miami-turn/gen/live/` |
| **splat** | the per-scene StreetGS splat from `train/` (original `background.ply` + actors moved by `tracks.json` + `sky.ply`), rendered with gsplat | comparison only |
| **splat + I2I** | splat render, then Difix3D+ with the same reference and temporal blend. This is probably what Tesla does: a coarse 3D render, then a small per-frame I2I diffusion polish | `web/scenes/av2-miami-turn/gen/splat_i2i/` |

StreetCrafter was **not run**. See the verdict section.

### Lift details (`lift.py`, `ground.py`)
- Lidar points move to the world frame with the ego pose at the sweep. Returns from the ego car itself are dropped.
  Points inside AV2 3D boxes count as dynamic and come from the current sweep only. All other points are static and kept for 10 sweeps.
- Colour: each point is projected into all 7 cameras, using the ego pose interpolated to each camera's exposure
  time. A coarse z-buffer per camera (8 px cells, 3x3 min) rejects occluded points. The colour comes from the
  camera that sees the point closest to its optical axis. Points that land on a sky pixel (SegFormer) are never coloured from it.
  Images are decoded at half resolution (JPEG DCT scaling) in a prefetch thread.
- Ground: the lidar rings are 1-3 m apart near the car, which leaves stripes in a high view. A 1 m height map
  (lowest static return, pull-push filled) carries a world-fixed colour grid at 0.15 m. The grid is coloured from the cameras
  with an occlusion test against the non-ground lidar points. A cell keeps its sharpest view (closest camera) and is
  replaced only by a view that is not much blurrier, or after 30 frames.
- Densify: Depth Anything V2 Metric Outdoor Small runs on the 7 cameras (batched), with each camera scaled to the lidar by
  the median ratio. It is back-projected at a 3 px stride where no lidar point lands within 6 px, with sky excluded
  (SegFormer-B0 Cityscapes, mask dilated). This fills the tall buildings above the lidar's vertical field of view.
- Splatting: square splats sized by world footprint / depth (lidar: 0.6 % of range, clamped 0.12-0.5 m), int64
  `scatter_reduce(amin)` z-buffer, then a screen-space visibility filter that drops far points showing through gaps in near surfaces.
- Before I2I: holes whose ray points above the horizon get a zenith-to-horizon gradient from the cameras' sky pixels. All other
  holes get pull-push fill. The hole under and in front of the ego car is filled this way, and the viewer's car model covers most of it.

### I2I details (`repaint.py`)
- Difix3D+ `nvidia/difix_ref`, 1 step at t=199, fp16, run at 544x960 and resized to 960x540. The reference image is a squashed
  panorama: ring_front_left | the horizon band of ring_front_center | ring_front_right.
- Speed-ups that do not change the output: the VAE ships an **unmerged LoRA** ("vae_skip"). Merging it saves about 60 ms. `torch.compile`
  (inductor) on the UNet and the VAE encoder and decoder saves about another 80 ms.
- Temporal fix: the previous output is warped into the current frame through the render's depth (lift depth or
  splat expected depth). In static pixels where the depth agrees, the result is 0.35 x new + 0.65 x warped. Dynamic pixels (box points / actor
  splats) and holes are not blended. The diffusion step itself is deterministic here (one step from the encoded
  image), so a fixed noise seed is not needed.

## Measured on 1x RTX 4090 (Vast.ai, torch 2.4, fp16)

**Live lift + I2I, ms per frame (median of frames 5-104, `results/timing_live.json`)**

| stage | ms |
|---|---|
| upload 7 camera frames to GPU (half-res decode runs in a prefetch thread, not counted) | 4.7 |
| depth + sky nets (DA-V2 Small + SegFormer-B0, 7 cameras) | 52.0 |
| colour lidar sweep | 10.5 |
| ground height map + texture update | 15.2 |
| mono-depth back-projection | 16.7 |
| point splatting (about 1-2 M points) | 34.5 |
| sky prior + pull-push fill | 1.8 |
| Difix3D+ (544x960, compiled) | 194.6 |
| temporal warp + blend | 1.3 |
| **total** | **335.5 (about 3 fps)** |

The lift alone takes about 140 ms per frame, so even without the I2I step it is not real time. It is unoptimised PyTorch.

**Splat**: gsplat render of 1.5 M + actor Gaussians: 22 ms per frame. **Splat + I2I**: 196 ms I2I + 1.8 ms blend.

**I2I step alone** (`bench_i2i.py`, random input, 10 runs after warm-up, `results/bench_i2i.json`):

| model | 960x544 eager | 960x544 compiled | 512x288 eager | 512x288 compiled |
|---|---|---|---|---|
| Difix3D+ with reference (`difix_ref`, batch 2 through encoder and UNet) | 274 | 193 | 76 | 61 (16 fps) |
| Difix without reference (`difix`, SD-Turbo size, batch 1) | 132 | 98 | 50 | 37 (27 fps) |

Breakdown at 544x960 before compiling: VAE encoder 68 ms (x2 images), UNet 45 ms (x2), VAE decoder 220 ms with the
unmerged LoRA and 162 ms after merging. **The VAE decoder at full resolution is the bottleneck, not the UNet.**

**Can a small distilled I2I model reach 60 fps (16.7 ms)?** Not with the SD-Turbo-class Difix as it stands. The best measured is 37 ms (27 fps) for Difix without a reference at 512x288, compiled.
It would plausibly get there by stacking: (1) a tiny autoencoder (TAESD-class, about 1-2 ms at this size instead of 30-160 ms),
(2) a pruned or distilled UNet (SD-Turbo UNet at 64x36 latents is a few ms after TensorRT; StreamDiffusion reports
about 90 fps (about 11 ms) for one-step SD-Turbo img2img at 512x512 on a 4090 with TensorRT and TAESD), (3) dropping the reference
branch or caching its keys and values, because the reference halves the throughput, and (4) TensorRT fp16 or int8 instead of inductor.
pix2pix-turbo is the same SD-Turbo backbone with skip connections, so it has the same cost profile as Difix. Tesla's claim
(60 Hz on car hardware) implies a much smaller purpose-trained network, not an off-the-shelf SD derivative. I did not build or test a TensorRT path.

## Quality and consistency

Revisit metric (`metrics.py`): for frames t and t+k, static geometry-backed pixels of t+k are re-projected into t
with depth. Only pixels where both frames see the same surface are kept, and the colours are compared (PSNR, higher = more
consistent). Every method is scored on the same pixel set. The geometry used biases the score toward the method that produced it, so both are reported.

Using **lidar-lift geometry** (fair to the lift modes; `results/metrics_lidargeo.json`):

| | k=1 (0.1 s) | k=5 (0.5 s) | k=10 (1 s) |
|---|---|---|---|
| lift only | 23.9 | 21.5 | 20.3 |
| lift + I2I (live) | 30.5 | 23.1 | 21.2 |
| splat | 31.4 | 26.8 | 25.4 |
| splat + I2I | 28.5 | 24.8 | 23.6 |
| splat + I2I, no temporal blend | 25.9 | 23.3 | 22.4 |

Using **splat geometry** (fair to the splat modes; `results/metrics_splatgeo.json`): splat 37.5 / 31.9 / 29.2,
splat + I2I 35.3 / 28.4 / 26.1, splat + I2I without the blend 27.5 / 25.0 / 23.8.
Effect of the temporal blend on live (v1 run, `results/metrics_v1.json`): 21.6 / 19.9 / 19.1 without and 30.1 / 22.7 / 21.0 with.
For comparison, the earlier free video generation scored about 15-16 dB.

## Verdict

- **splat + I2I looks best**, as expected. Difix removes most splat artefacts: smeared road, floaters, blurry vertical
  surfaces and mushy cars become clean, sharp cars, lane lines, kerbs and facades. Sky and far buildings become plausible. It stays
  anchored: the layout never changes, only texture, so it is far above the 15-16 dB drift of free video generation. The cost is 2-3 dB of consistency compared with the bare splat.
  The visible flicker is fine texture (foliage, facade detail) shimmering from frame to frame. The warp-and-blend removes most of it (+2.5 dB at k=1, +1.2 dB at k=10).
  It needs a per-scene splat trained offline (55 min here), so it is a replay mode, not live.
- **lift + I2I (live)** works per frame with no training and no future frames, and it shows the *current* scene, including
  moving traffic exactly where the sensors saw it. Road, lane markings, parked cars and near facades look good. Weak points:
  (a) tree canopies and building tops above the lidar's field of view come out as fragments or blocks that Difix
  only partly heals. (b) Exposure seams between ring cameras show on the road. (c) Some objects get repainted
  (frame 14: a bus at the intersection turned into an awning and storefront). (d) The I2I output on lidar-sparse areas changes from frame to frame
  (light poles, horizon strip, the edges of moving cars). Revisit consistency is about 21 dB at 1 s, below the splat (25) and splat + I2I (23.6).
  At 336 ms per frame on a 4090 it is not live in the Tesla sense. The lift costs about 140 ms and the I2I about 195 ms.
- **The I2I step is the shared bottleneck.** About 195 ms at 960x540 and 61 ms at 512x288 with the reference-guided Difix.
  60 fps needs a much smaller model (see above).
- **StreetCrafter skipped**: it needs its own environment (SVD-based video diffusion), weights from Google Drive, and Waymo-format
  preprocessing with its own lidar-condition renderer. Realistically that is more than 1 hour of setup before the first frame. It was also trained
  on lateral shifts of road-level cameras, so a view 5.6 m up and 10.5 m back is far out of distribution. After the
  owner's update (Tesla is reportedly "coarse render, then small per-frame I2I"), splat + I2I was the higher priority and got the time instead.

## What I would do next
1. Live speed: a TAESD-style decoder and TensorRT for the I2I step, and fused CUDA point splatting (or render the lift
   points as tiny Gaussians with gsplat). Mono depth only every few frames.
2. Live quality: per-camera exposure gains before colouring; feed the lift + I2I output back as a persistent colour
   map (as the ground texture already does), so static areas are painted once and stop flickering; a small per-scene LoRA for Difix.
3. Hybrid: splat for the static world plus live lift for moving objects, then I2I. That gives the stability of the splat and live traffic.

## Commands (exactly what ran)

Box: Vast.ai 1x RTX 4090, image `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`, 70 GB. Code in `/root/gen/code` = this folder
+ `train/download_log.sh`, `train/preprocess_av2.py` and `web/scenes/av2-miami-turn/tracks.json`.
```bash
cd /root/gen
bash code/download_log.sh val 3de5b5d6-68c4-3c95-84ed-be7c83d829f8 data/raw
python code/preprocess_av2.py --log_dir data/raw/val/3de5b5d6-68c4-3c95-84ed-be7c83d829f8 --out data/processed --scene 0 --t0 5.0 --t1 15.5
python -m venv --system-site-packages venv && . venv/bin/activate
pip install diffusers==0.25.1 huggingface_hub==0.25.1 transformers==4.46.3 peft==0.13.2 accelerate einops gsplat==1.4.0 \
    numpy==1.26.4 pandas==2.2.3 pyarrow==17.0.0 opencv-python-headless==4.10.0.84 hf_transfer
pip uninstall -y torch   # gsplat/lpips pull a newer torch into the venv; keep the image's torch 2.4 (and remove venv nvidia-* cu13 libs)
git clone https://github.com/nv-tlabs/Difix3D.git
huggingface-cli download nvidia/difix_ref --local-dir weights/difix_ref
huggingface-cli download nvidia/difix --local-dir weights/difix
huggingface-cli download depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf --local-dir weights/da2s
huggingface-cli download nvidia/segformer-b0-finetuned-cityscapes-1024-1024 --local-dir weights/segb0
# original splat export (background.ply etc. from the train/ export tarball) in /root/gen/scene
cd code; export TORCHINDUCTOR_CACHE_DIR=/root/gen/inductor
L=/root/gen/data/raw/val/3de5b5d6-68c4-3c95-84ed-be7c83d829f8; P=/root/gen/data/processed/000
python run_live.py --proc $P --raw $L --tracks tracks.json --out /root/gen/out_live2 --mono            # live
python render_splat.py --scene /root/gen/scene --tracks tracks.json --out /root/gen/out_splat          # splat + depth
python run_i2i.py --frames /root/gen/out_splat --geo /root/gen/out_splat/geo --proc $P --raw $L --out /root/gen/out_splat_i2i
python run_i2i.py ... --out /root/gen/out_splat_i2i_notemp --alpha 1.0                                 # ablation
python metrics.py --geo /root/gen/out_live2/geo --seq lift=.../raw_lift live=.../live splat=/root/gen/out_splat splat_i2i=... --out metrics_lidargeo.json
python bench_i2i.py bench_i2i.json difix_ref; python bench_i2i.py bench_noref.json difix   # one model per process
```
On the Mac: `python3 gen/make_compare.py --front <out_live>/front --splat <out_splat> --splat_i2i <out_splat_i2i> --live <out_live2>/live --out out/compare.mp4`
(add `--lift <out_live2>/raw_lift` for the 6-panel version `out/compare_6panel.mp4`).

Gotchas: Difix needs diffusers 0.25.x (`PositionNet` import). `difix` and `difix_ref` both ship `unet_2d_condition.py` as remote
code and diffusers caches it by file name, so load only one per process, or delete `~/.cache/huggingface/modules/diffusers_modules/local`.
torchvision 0.19 `decode_jpeg` on CUDA is slow per image (about 18 ms each at 2048x1550), so PIL draft-mode decoding in a thread is used instead.

## Models and licences
- **Difix3D+** (nv-tlabs/Difix3D, HF `nvidia/difix_ref`, `nvidia/difix`): **NVIDIA non-commercial licence**. It is built on SD-Turbo
  (Stability AI community / non-commercial terms). The `gen/live` and `gen/splat_i2i` frames are therefore research / non-commercial only.
- Depth Anything V2 Metric Outdoor Small: Apache-2.0 (the Small model). SegFormer-B0 Cityscapes (NVIDIA): NVIDIA source-code licence, non-commercial.
- gsplat: Apache-2.0. Argoverse 2 data: CC BY-NC-SA 4.0, so every frame here is non-commercial and share-alike.

## Files
`common.py` (data and camera), `lift.py`, `ground.py`, `repaint.py`, `run_live.py`, `render_splat.py`, `run_i2i.py`,
`metrics.py`, `bench_i2i.py`, `make_compare.py`; `results/` (timings, metrics, benchmark JSON); `raw_lift/` (lift before
repaint, holes black); `SPEND.md`.
