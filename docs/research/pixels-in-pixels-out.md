# Pixels-in, pixels-out chase view: open options (2026-11-22)

Goal: a model that takes the 7 posed AV2 ring cameras (optionally lidar) and renders a virtual chase camera about 10 m behind and 5.6 m above the ego car, with no per-scene optimisation. We want something close to Tesla FSD v15 "Reality View": a small decoder on the driving network's latent plus a one-step image-to-image (I2I) diffusion pass at 60 fps that also fills unseen sides of objects.

How this was checked: GitHub numbers come from `gh api repos/...` on 2026-11-22. Weight availability and licences come from the Hugging Face API (`/api/models/<id>`: file list, `cardData.license`) and from HEAD requests on the checkpoint files. Speed figures are quoted from READMEs or model cards and labelled with their source. Papers were used only for numbers the repos do not give, and those are marked "paper".

## Short answer

- No open checkpoint does the Tesla job end to end: surround cameras in, an elevated chase view out, a learned amodal prior, 60 fps.
- The closest released pieces are feed-forward driving reconstructors. They predict 3D Gaussians in one forward pass, so there is no per-scene optimisation, but there is an explicit 3D layer in the middle. The best two are:
  - DGGT, which ships an Argoverse 2 checkpoint.
  - NVIDIA Instant NuRec: commercial-friendly weights, plus an official AV2-to-NCore converter.
- A one-step I2I cleaner goes on top: NVIDIA Fixer (commercial OK) or Difix (non-commercial).
- True LVSM-style models (no 3D at all) are only released for RealEstate10K/DL3DV at 256 to 512 px square, under non-commercial licences. They are the right architecture for stage 2 (a compact, trained chase-view decoder) but useless zero-shot on driving data.
- Camera-conditioned driving video diffusion (Cosmos, GEN3C, StreetCrafter, FreeVS) takes seconds to minutes per clip and often 8 GPUs. Use these only as offline teachers, if at all.

## Top 3 picks

1. **DGGT** (xiaomi-research/dggt) for the zero-training pilot. It has an AV2-trained checkpoint on HF (`model_latest_argoverse.pt`, 5.4 GB). Its AV2 preprocessing is the drivestudio format we already have. It has a dynamic head, a sky model with `forward_with_new_pose`, and a bundled one-step Difix-style refiner. Paper speed is 0.39 s per scene on an A100. Weights are CC BY-NC 4.0.
2. **Instant NuRec** (NVIDIA/instant-nurec) as the stronger reconstructor and teacher. It reports 28.26 dB on Waymo, against 26.25 for DGGT and 21.88 for STORM (paper). Weights are under the NVIDIA Open Model License. It writes a standard 3DGS PLY, which our Spark viewer can load directly, plus a sky cubemap and dynamic layers. The ncore repo ships an official AV2 converter. Risks: it was trained on NVIDIA F-theta cameras, so AV2 cameras need rectification, and NuRec asks for more than 24 GB VRAM.
3. **Fixer** (nv-tlabs/Fixer, nvidia/Fixer) as the I2I layer. It is a Cosmos-0.6B one-step model at 576x1024, 50.6 ms on an A100 and 26.5 ms on an H100 (model card). The licence is commercial OK, and training code is included. It is the drop-in successor to the Difix we are using now, which is non-commercial and takes 0.355 s on an A100 per its card. For a strict 60 fps target, train our own pix2pix-turbo/SD-Turbo LoRA instead (StreamDiffusion: SD-Turbo img2img at 93.9 fps on a 4090).

For stage 2, initialise the decoder from **Efficient-LVSM** (Ayakaee/Efficient-LVSM): RE10K weights, CC BY-NC-SA, 199M parameters, KV cache. Its README reports 24 ms to add a view at 4 inputs, against 123 ms for decoder-only LVSM.

## Comparison table

Legend for **Wts**: Y = downloadable now (file confirmed on HF or a direct link); G = Google Drive link in the README (not test-downloaded); N = not released. **Lic** is the weight licence. **NC** = non-commercial.

### A. Feed-forward driving reconstruction (multi-camera in, Gaussians out, render any pose)

| Model | Repo (stars, last push) | Wts | Lic (weights / code) | Train data | Inputs → outputs | Speed (source) | Arbitrary / elevated pose | VRAM | AV2 effort |
|---|---|---|---|---|---|---|---|---|---|
| **DGGT** (2512.03004) | [xiaomi-research/dggt](https://github.com/xiaomi-research/dggt) (603, 2026-01-15) | Y [HF](https://huggingface.co/xiaomi-research/dggt): waymo, nuscenes, **argoverse** ckpts + difix refiner | CC BY-NC 4.0 / Apache-2.0 | Waymo, nuScenes, AV2 (separate ckpts) | unposed images, 1 or 3 cams x 4+ frames → per-pixel Gaussians, cam poses, dynamics, lifespan, sky | 0.39 s/scene, A100 (paper) | Any pose via gsplat. Predicted frame is up to scale, so align to AV2 poses (`calculate_scale_factor` is in inference.py). Elevated views not evaluated | not stated; VGGT-1B backbone, likely fits 24 GB | **Low.** Same drivestudio preprocess and SegFormer sky masks we already ran. Add a chase-pose render (~50 lines) |
| **Instant NuRec** (2607.14203) | [NVIDIA/instant-nurec](https://github.com/NVIDIA/instant-nurec) (252, 2026-10-15) | Y [HF](https://huggingface.co/nvidia/instant-nurec): pa_front, pa_multiview (730 MB), pq_road | NVIDIA Open Model License (commercial OK) / Apache-2.0. The render extra uses nvdiffrast (NC) | ~40k internal NVIDIA AV clips; Waymo for eval only | 1/3/5 calibrated cams x 18 frames → static and dynamic 3DGS layers, sky cubemap, per-camera ISP affine | ~1.5 s per 10–20 s clip (paper) | PLY renders anywhere. Built-in render is source trajectory only. Bird's-eye and re-posed views shown qualitatively | NuRec minimum is ">24 GB" | **Medium.** [ncore AV2 converter](https://github.com/NVIDIA/ncore/tree/main/tools/data_converter/argoverse2), then F-theta rectification (docs/camera_rectification.md). AV2 front-center is portrait/narrow, so expect a domain gap |
| STORM (2501.00602) | [NVlabs/GaussianSTORM](https://github.com/NVlabs/GaussianSTORM) (374, 2025-05-22) | Unofficial only ([issue #1](https://github.com/NVlabs/GaussianSTORM/issues/1), Google Drive, color-tone bug reported) | NVIDIA License (NC) | Waymo | 3 cams x 4 frames → Gaussians + scene flow | ~200 ms (paper) | any pose | A100 tested | Medium. No AV2 loader ("might include") |
| EVolSplat4D (2601.15951) | [Miaosheng1/EVolSplat4D](https://github.com/Miaosheng1/EVolSplat4D) (21, 2026-10-08) | Y [HF](https://huggingface.co/cookiemiao/EVolSplat4D) pretrain_waymo.ckpt | Apache-2.0 (HF card) | Waymo | front images + DINO features + point-cloud prior + tracks → 3DGS | not stated | any pose | not stated | Medium-high (nerfstudio, needs point priors and tracks) |
| MCSplat (NeurIPS 2026) | [yfwang0824/MCSplat](https://github.com/yfwang0824/MCSplat) (13, 2026-10-26) | Y [HF](https://huggingface.co/Yifeifei/MCSplat) waymo + nuscenes, 4.58 GB each | Apache-2.0 | Waymo, nuScenes | 5–6 synced cams (Pi3 backbone) → Gaussians + bilateral grid | not stated | any pose | not stated | Medium (needs an AV2 loader) |
| ReconDrive (2603.07552) | [TuojingAI/ReconDrive](https://github.com/TuojingAI/ReconDrive) (172, 2026-03-30) | Y [HF](https://huggingface.co/TuojingAI/ReconDrive) stage1/2 | Apache-2.0 | nuScenes | surround cams → 4DGS (VGGT-based) | not stated | any pose | not stated | Medium (nuScenes only) |
| DrivingForward (2409.12753) | [fangzhou2000/DrivingForward](https://github.com/fangzhou2000/DrivingForward) (245, 2026-09-03) | G [Drive](https://drive.google.com/drive/folders/1IASOPK1RQeP-nLQvJUn7WQUtb_fwGlVS) | MIT (code) | nuScenes | 6 surround cams, SF/MF → Gaussians | not stated | any pose, small baseline | not stated | Medium |
| Omni-Scene (CVPR 2025) | [WU-CVGL/Omni-Scene](https://github.com/WU-CVGL/Omni-Scene) (254, 2025-03-24) | G [Drive](https://drive.google.com/drive/folders/1vgc8VjXhuo35KwFqbJiqdu5FEDg6AMRy) | MIT (code) | nuScenes | 6 cams → pixel + volume Gaussians | not stated | any pose | README: ">=1 A100" for the demo | High (mmdet3d, A100) |
| VGD (2510.19578) | [JHLin42in/VGD](https://github.com/JHLin42in/VGD) (15) | N ("Code will be released soon") | – | nuScenes | – | – | – | – | – |
| XYZCylinder | [yuyuyu223/XYZCylinder](https://github.com/yuyuyu223/XYZCylinder) (7) | N (HF "coming soon") | – | – | – | – | – | – | – |
| GeoForward | [LeapWM/GeoForward](https://github.com/LeapWM/GeoForward) (0, empty README) | N | – | – | – | – | – | – | – |
| WorldSplat (ICLR 2026) | [wm-research/worldsplat](https://github.com/wm-research/worldsplat) (144) | N (174-byte README) | – | nuScenes | – | – | – | – | – |

### B. LVSM family (no explicit 3D, transformer maps posed images + target rays to pixels)

| Model | Repo | Wts | Lic | Train data | Notes |
|---|---|---|---|---|---|
| LVSM (ICLR 2025) | [Haian-Jin/LVSM](https://github.com/Haian-Jin/LVSM) (571, 2025-08-04) | Y [HF coast01/LVSM](https://huggingface.co/coast01/LVSM) (256 and 512, enc-dec and dec-only) | CC BY-NC-SA 4.0 | RealEstate10K | Square 256/512; README says to fine-tune for other aspect ratios. Training: 64 GPUs, batch 512. Static scenes only |
| Efficient-LVSM | [Ayakaee/Efficient-LVSM](https://github.com/Ayakaee/Efficient-LVSM) (30, 2026-02-09) | Y [HF](https://huggingface.co/Ayakaee/efficient-lvsm) res256/res512 (2.96 GB) | CC BY-NC-SA 4.0 | RE10K | Separate encoder and decoder plus KV cache. README: 24 ms to add a view at 4 inputs vs 123 ms for LVSM dec-only. **Best init for our stage 2 decoder** |
| RayZer (ICCV 2025) | [hwjiang1510/RayZer](https://github.com/hwjiang1510/RayZer) (452) | Y [HF](https://huggingface.co/hwjiang/RayZer) | Apache-2.0 card (code is an Adobe re-implementation) | DL3DV | Self-supervised and pose-free. Not driving |
| WildRayZer (CVPR 2026) | [UVA-Computer-Vision-Lab/wild-rayzer](https://github.com/UVA-Computer-Vision-Lab/wild-rayzer) (13) | see repo | NOASSERTION | Dynamic RE10K | Dynamic scenes, self-supervised. Not driving |
| 4D-LVSM / Fast Spatial Memory (ECCV 2026) | [Mars-tin/fast-spatial-mem](https://github.com/Mars-tin/fast-spatial-mem) (112) | Y [HF marstin/fast-spatial-mem](https://huggingface.co/marstin/fast-spatial-mem) res128/256 | Apache-2.0 (card); code has mixed licences | non-driving 4D | Dynamic 4D-LVSM with test-time training. 256 px |
| FL4N | [cvlab-kaist/FL4N](https://github.com/cvlab-kaist/FL4N) (41, 2026-11-06) | Y [HF](https://huggingface.co/eunbeen-hong/FL4N) | FAIR noncommercial | Kubric4D, CamxTime | Latent 4D NVS on frozen LagerNVS (Meta, gated, NC) |

None of these has seen driving data or a 10 m elevated jump. All released driving-capable transformers predict Gaussians (group A).

### C. Camera-conditioned driving / video diffusion (offline teacher candidates)

| Model | Repo | Wts | Lic | Data | In → out | Cost | Fit |
|---|---|---|---|---|---|---|---|
| FreeVS (ICLR 2025) | [esdolo/FreeVS](https://github.com/esdolo/FreeVS) (164, 2025-02-22) | Y [HF Esdolo/FreeVS_WOD](https://huggingface.co/Esdolo/FreeVS_WOD) | none stated; README points to WOD terms | Waymo | lidar "pseudo-image" projected to the new pose + refs → SVD-based generation | multi-step SVD, seconds per frame | Same idea as our lidar-lift + I2I. Front-only path, no nuScenes/AV2 yet |
| StreetCrafter | [zju3dv/street_crafter](https://github.com/zju3dv/street_crafter) (338, 2026-09-15) | G (Drive, single and multi-cam Waymo) | Project Registration License | Waymo | lidar render + refs → video (Vista-based), then distilled into a per-scene GS | A100 80 GB | Quality mode only, not feed-forward |
| ReconDreamer | [GigaAI-research/ReconDreamer](https://github.com/GigaAI-research/ReconDreamer) (227) | G (per-scene weights) | none | Waymo | per-scene reconstruction + restorer | per-scene | No |
| GEN3C (CVPR 2025) | [nv-tlabs/GEN3C](https://github.com/nv-tlabs/GEN3C) (1455, 2026-06-15) | Y [HF GEN3C-Cosmos-7B](https://huggingface.co/nvidia/GEN3C-Cosmos-7B) | NVIDIA Open Model License (commercial OK) | general video | 3D cache from depth + any camera path → video. Multi-view input since 2025-08 | tested on H100/A100. ~43 GB peak even with full offload (README) | Offline teacher for elevated views. Too big for a 4090 |
| Lyra 2.0 | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) (2888) | Y [HF nvidia/Lyra-2.0](https://huggingface.co/nvidia/Lyra-2.0) | NVIDIA Internal Scientific R&D license (no production, no distribution) | general | image/video → 3D scene | H100-class | No (licence) |
| Cosmos-Transfer2.5 auto/multiview | [nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) (769, 2026-06-30) | Y [HF](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) (gated, auto/multiview) | NVIDIA Open Model License | NVIDIA AV (RDS-HQ) | world-scenario/lidar control videos → 7 camera views | "Multiview requires 8 GPUs" (GPUs ≥ active views) | Generates rig cameras, not a free virtual pose. The "plenoptic" mode does preset camera moves (elevation_up, distance_away) from one video. Offline only |
| Cosmos-Drive-Dreams | [nv-tlabs/Cosmos-Drive-Dreams](https://github.com/nv-tlabs/Cosmos-Drive-Dreams) (559) | Y (Transfer1-7B-AV, multiview) | NVIDIA Open Model License | RDS-HQ | HD map/lidar → multiview video | 7B, multi-GPU | Data generation, not NVS |
| Stable Virtual Camera | [Stability-AI/stable-virtual-camera](https://github.com/Stability-AI/stable-virtual-camera) (1684) | Y [HF](https://huggingface.co/stabilityai/stable-virtual-camera) (gated) | Stability non-commercial | general | N posed views → any camera path | multi-step diffusion | Generic. Could test as a teacher |

### D. One-step I2I (the "60 fps diffusion pass")

| Model | Repo | Wts | Lic | Res | Speed (source) | Notes |
|---|---|---|---|---|---|---|
| **Fixer** (Difix v2, Cosmos-0.6B) | [nv-tlabs/Fixer](https://github.com/nv-tlabs/Fixer) (166, 2026-07-30) | Y [HF nvidia/Fixer](https://huggingface.co/nvidia/Fixer) (4.05 GB) | NVIDIA Open Model License, **commercial OK** | 576x1024 | H100 26.5 ms, A100 50.6 ms, L40 87.8 ms (card) | Trained on 3DGUT renders. Docker (cosmos-predict2 container). Training code included |
| Difix / Difix-ref (SD-Turbo based) | [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) (1318, 2025-06-28) | Y [HF nvidia/difix](https://huggingface.co/nvidia/difix), [difix_ref](https://huggingface.co/nvidia/difix_ref) | NVIDIA License (NC) | ~512–576 | A100 0.355 s, H100 0.223 s (card) | What gen/ uses now. Trained on DL3DV |
| pix2pix-turbo | [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo) (2496, 2025-08-01) | Y (edge/sketch demos) | MIT code; SD-Turbo base is under the Stability licence | 512 | not stated in README | **Recipe for training our own paired one-step model** (Difix is built on it) |
| SD-Turbo + StreamDiffusion | [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) (10839, 2024-12-04) | Y [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | Stability community licence (check revenue limits) | 512 | img2img 93.9 fps on RTX 4090 (README) | Shows that 60 fps one-step I2I fits a 4090 |
| SDXS-512 | [IDKiro/sdxs](https://github.com/IDKiro/sdxs) (666, 2024-05-27) | Y [HF](https://huggingface.co/IDKiro/sdxs-512-0.9) | OpenRAIL++ | 512 | "~100 FPS" single GPU (README) | Smaller UNet; an image-conditioned variant is shown (sketch) |

### E. Driving world models

| Model | Status | Why not for this |
|---|---|---|
| Vista | [OpenDriveLab/Vista](https://github.com/OpenDriveLab/Vista) (912), Y [HF](https://huggingface.co/OpenDriveLab/Vista) Apache-2.0, nuScenes front camera | Predicts the future front view from actions. No camera-pose conditioning, slow SVD sampling |
| DriveDreamer | [JeffWang987/DriveDreamer](https://github.com/JeffWang987/DriveDreamer) (600, last push 2024-11) | Layout-to-video. No free-viewpoint decoder |
| OccWorld | [wzzheng/OccWorld](https://github.com/wzzheng/OccWorld) (598, 2024-04) | Occupancy only, no RGB |
| GAIA-1/2 | no weights | – |
| MagicDrive-V2, DiST-4D, UniScene | code and weights vary, AGPL or none | Generation from layout. Not reconstruction of a real drive |

## What "Tesla-like" means for us, given the above

Tesla's chain (by their description) is: driving net latent → small decoder → chase-view image → one-step I2I diffusion. The open equivalent today is:

`7 ring cams (+lidar) → feed-forward reconstructor (DGGT / Instant NuRec) → gsplat render at chase pose → Fixer one-step I2I`

This is feed-forward (seconds per clip, nothing optimised per scene), but not strictly pixels-in-pixels-out, and it is not real time per frame: reconstruction is about 0.4–1.5 s for a window. Rendering the Gaussians and the I2I pass are real time. To get a true learned decoder (and the amodal prior), stage 2 trains one, using the stage-1 outputs and real cameras as supervision.

## Stage 1: zero-training pilot (one AV2 log, ~2–4 GPU-hours, ~$1–3)

Log: val `3de5b5d6` (already preprocessed in drivestudio format at `data/processed/000`, with sky masks). This lets us compare directly against the StreetGS splat (PSNR 23.0) and the lidar-lift + Difix live track.

1. **DGGT, AV2 checkpoint** (4090 is fine):
   - Download `model_latest_argoverse.pt` (5.4 GB), `model_difix.pkl` (5.2 GB), the TAPIP3D tracker and the SegFormer ckpt from [HF](https://huggingface.co/xiaomi-research/dggt).
   - Run `inference.py --mode 2 --input_views 3 --sequence_length 4..8` over sliding windows of the 10.5 s segment. Get held-out PSNR on the side/rear cameras that were not used as inputs.
   - Add a chase render: Umeyama-align the predicted camera centres to the AV2 ego poses to get scale and rigid transform. Build the chase camera (10 m back, 5.6 m up, pitched down, same intrinsics as `gen/common.py CHASE`) and map it into the predicted frame. Call gsplat `rasterization` and `sky_model.forward_with_new_pose`, then run the bundled refiner or Fixer.
   - Check: whether the side and back of vehicles appear (they will only appear if they were seen in some frame of the window), floor holes behind the ego car, and temporal flicker between windows.
2. **Instant NuRec pa-multiview** (rent a 48 GB card, e.g. L40S or A6000; a 4090 is below the stated >24 GB):
   - Convert the log with the ncore AV2 converter and rectify front-center, front-left and front-right to the F-theta model.
   - Run `--model pa-multiview --merge --include-dynamic` and load the PLY plus `sky.npz` in our Spark viewer. Gaussians drop into `web/scenes/` with a small sky-cubemap shader.
   - Render the same chase path and run Fixer.
3. **I2I comparison** on the same raw chase renders: Difix (current), Fixer, and Fixer at a lower timestep. Record ms per frame on the 4090.
4. **Output:** a side-by-side video (StreetGS splat / lidar-lift+Difix / DGGT+Fixer / NuRec+Fixer), metrics on held-out real cameras, and timing.

Note: spending and renting a box need Waleed's OK (approval rule). Current Vast credit is about $8.67 (note of 2026-11-22).

## Stage 2: small training / distillation run

**Target model ("chase decoder"):** an Efficient-LVSM-style encoder–decoder.

- **Encoder:** per-view patch tokens from the 7 ring cams at 2 timesteps, downsampled to 224x320 (front-center cropped to landscape), patch 16. That gives about 7x2x280 ≈ 3.9k tokens. Plücker ray embeddings in the ego frame. Optional extra channel: projected lidar depth.
- **Decoder:** target Plücker rays for the chase camera at 288x512 → RGB, with cross-attention to the encoder KV cache. Encoder cost is paid once per timestep; the decoder runs per output frame.
- **Size and init:** about 150–200M parameters, initialised from Efficient-LVSM res256 (CC BY-NC-SA, fine for research). Change the patch embed and positional size for the new aspect ratio.
- **Head:** a one-step I2I head, either a LoRA on Fixer or pix2pix-turbo on SD-Turbo. Train it on (decoder output → teacher frame) for the photoreal and amodal look.

**Supervision (mixed):**
- (a) **Real held-out cameras** (always available, no teacher bias): from context at time t, predict all 7 cameras at t±0.5–1.5 s. The ego moves 5–15 m in that time, so this gives metre-scale viewpoint shifts with real ground truth. Also drop 1–2 cameras from the context and predict them.
- (b) **Pseudo-GT chase frames** (the only source of the elevated viewpoint): per log, render the chase path from a teacher and clean it with Fixer. Teacher options:
  - Per-scene StreetGS (best quality, ~55 min of 4090 per log, as already measured).
  - Instant NuRec / DGGT (seconds per log, but lower quality and its own artifacts).
  - Plan: StreetGS teachers for 20 logs, plus feed-forward teachers for up to 100 logs.
- (c) Optional: a small elevation curriculum. Render pseudo-GT at intermediate heights (1.5 → 3 → 5.6 m) so the decoder does not have to jump straight to the chase pose.

Losses: L1/MSE + LPIPS 0.2 on the decoder; adversarial/LPIPS for the I2I head, as in the pix2pix-turbo recipe.

**Data size (AV2 Sensor):**
- Raw: about 1.5–2 GB per log (7 ring cams at 20 Hz, ~15 s, plus lidar) → 30–40 GB for 20 logs, 150–200 GB for 100 logs. Download from the public S3 bucket with s5cmd; keep raw only on the box.
- Processed at 10 Hz, 320 px JPEG: ~150 frames x 7 cams x ~40 KB ≈ 45 MB per log, plus a chase pseudo-GT at 512 px of ~15 MB per log. About **1.2 GB for 20 logs, 6 GB for 100 logs**.

**GPU-hour estimate on one RTX 4090 (~$0.38/h on Vast):**

| Step | 20 logs | 100 logs |
|---|---|---|
| Preprocess (CPU-heavy) + sky masks | 1–2 h | 4–6 h |
| Teachers: StreetGS (55 min/log) | ~18 h | (not used for all; 20 logs as above) |
| Teachers: Instant NuRec / DGGT + render | ~1 h | ~4 h (48 GB card for NuRec) |
| Fixer pass on teacher frames (~0.1 s x 3k–15k frames) | <0.5 h | ~0.5 h |
| Chase decoder fine-tune (30–60k steps, batch 8, bf16, flash-attn) | 10–15 h | 20–30 h |
| One-step I2I head (LoRA, 5–10k pairs) | 3–5 h | 5–8 h |
| Eval + video | 1 h | 2 h |
| **Total** | **~35–40 h (~$14)** | **~55–70 h (~$22–27)** |

The decoder line is an estimate from tokens x parameters at a realistic 4090 throughput, not a measurement. Run a 1-hour throughput test first and rescale. A minimal proof (5 logs, feed-forward teachers only, 10k steps) fits in about 6–8 GPU-hours, roughly $3, which is within the current credit.

**Expected runtime of the trained path on a 4090:** encoder once per camera frame (10–20 Hz), decoder per display frame (the Efficient-LVSM README reports tens of ms at a few views; at our token count, expect 10–25 ms), plus an SD-Turbo-class one-step head (StreamDiffusion reports 93.9 fps img2img at 512 on a 4090). 30–60 fps at 288x512 looks reachable; full 576x1024 Fixer is about 20 fps.

**Evaluation:** PSNR/SSIM/LPIPS on held-out real cameras (time-shifted and dropped views), FID/KID of chase frames against Fixer'd StreetGS chase frames, and temporal warp error (gen/metrics.py already computes consistency).

## Licences in one line

AV2 itself is CC BY-NC-SA 4.0, so anything trained on it is non-commercial whatever the model licence. If we ever need a commercial path, it is Instant NuRec + Fixer + GEN3C/Cosmos (NVIDIA Open Model License) trained on non-AV2 data.

## Links checked

- HF model APIs: nvidia/instant-nurec, xiaomi-research/dggt, nvidia/Fixer, nvidia/difix, coast01/LVSM, Ayakaee/efficient-lvsm, hwjiang/RayZer, cookiemiao/EVolSplat4D, Yifeifei/MCSplat, TuojingAI/ReconDrive, Esdolo/FreeVS_WOD, nvidia/GEN3C-Cosmos-7B, nvidia/Cosmos-Transfer2.5-2B, nvidia/Lyra-2.0, stabilityai/stable-virtual-camera, OpenDriveLab/Vista, marstin/fast-spatial-mem, eunbeen-hong/FL4N
- Papers: DGGT [arXiv 2512.03004](https://arxiv.org/abs/2512.03004); Instant NuRec [arXiv 2607.14203](https://arxiv.org/abs/2607.14203); VGD [2510.19578](https://arxiv.org/abs/2510.19578); STORM [2501.00602](https://arxiv.org/abs/2501.00602); LVSM [2410.17242](https://arxiv.org/abs/2410.17242); Difix3D+ [2503.01774](https://arxiv.org/abs/2503.01774)
- NuRec hardware: https://docs.nvidia.com/nurec/basics/hardware.html (">24GB" GPU memory)
- Newer papers without code found: VGGD (2608.10682), StreetNVS (2606.01590), WV-GS (2611.07081), ViewMorpher3D (2601.07540), DrivingScene (2510.24734), SpectralSplat (2604.03462)
