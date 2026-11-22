# train/ — building a Reality View scene from a real drive

These scripts turn one Argoverse 2 surround-camera log into a dynamic Gaussian-splat scene and export it in the
format the web viewer reads (`web/scenes/<name>/`). They are the exact scripts used for `web/scenes/av2-miami-turn`.

## Data and licence

- **Argoverse 2 Sensor dataset**, val log `3de5b5d6-68c4-3c95-84ed-be7c83d829f8` (Miami). It is downloaded
  anonymously from the public S3 bucket over HTTPS, so no signup is needed.
- Licence: **CC BY-NC-SA 4.0** (Argo AI, see https://www.argoverse.org/about.html#terms-of-use). The exported scene
  is derived from it, so it is non-commercial and share-alike as well.
- Segment: 5.0 to 15.5 s after the start of the log, 105 lidar frames at 10 Hz. The car pulls away from a stop and
  makes a 90 degree right turn through a busy intersection, in sunny daylight, with about 90 moving vehicles.
- How the log was picked: I downloaded only `city_SE3_egovehicle.feather` for all 150 val logs and scored every
  10 to 12 s window by yaw change (needed at least 45 degrees) and distance travelled. For the top 12 I looked at a
  front-camera frame and the annotations, then chose the sunniest one with the most traffic.

## Method

[drivestudio](https://github.com/ziyc/drivestudio) (commit in `drivestudio_commit.txt` on the box), using the
**StreetGS** model: static background Gaussians, rigid per-vehicle Gaussians that follow the 3D box tracks, and a
cube-map sky. Config: `configs/av2_streetgs.yaml`. It differs from upstream `streetgs.yaml` in three ways: SH
degree 1 (so the exported ply matches the trained model), lidar depth weight 0.03 instead of 0.01 (firmer geometry
for the raised viewpoint), and 1.2M lidar init points. All 7 ring cameras are used at half resolution
(1024x775). Every 10th frame is held out for evaluation.

Where we differ from drivestudio's documented pipeline:

| drivestudio | here | why |
|---|---|---|
| `datasets/preprocess.py` with their av2-api fork (needs Rust) | `preprocess_av2.py` (pandas only) | same output layout, no Rust toolchain; original JPEGs are copied instead of re-encoded |
| no ego-car mask for AV2 | bottom of ring_front_center (288 px) and ring_rear_left/right (200 px) cropped | the hood/body stays fixed in those cameras and would get baked into the scene. Cropping the bottom keeps fx/fy/cx/cy valid; `patch_drivestudio.py` updates `original_size` to match |
| SegFormer via mmcv 1.2.7 / torch 1.8 | `sky_masks.py` with HF `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` (sky = class 10) | same mask files, runs in the main env |
| pytorch3d compiled from source | pure-torch shim of the 4 functions used (`patch_drivestudio.py`) | saves a long build |
| torch 2.0 / cu117 | torch 2.1.2 / cu118 (docker image), gsplat 1.3.0 built from source | RTX 4090 support |

## Commands (exactly what was run)

On the Mac (choosing the log, CPU only):
```bash
# list val logs, fetch only their pose files, score windows by yaw change + distance (see "How the log was picked")
bash train/download_log.sh val 3de5b5d6-68c4-3c95-84ed-be7c83d829f8 ~/av2raw      # optional local copy, 945 MB
```

On the GPU box (Vast.ai, 1x RTX 4090, image `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`, 80 GB disk):
```bash
mkdir -p /root/work && cd /root/work            # copy this train/ folder here
bash train/setup_box.sh                          # clone drivestudio, deps, gsplat 1.3.0, nvdiffrast, patches (~15 min)
bash train/run_train.sh 30000                    # download log, preprocess, sky masks, train
cd drivestudio && export PYTHONPATH=$(pwd)
python ../train/export_scene.py --ckpt output/av2/revis/checkpoint_final.pth \
    --raw_log data/argoverse/raw/val/3de5b5d6-68c4-3c95-84ed-be7c83d829f8 \
    --out /root/work/export/av2-miami-turn --name av2-miami-turn --gpu "RTX 4090" --train_minutes <m>
```
Then copy `/root/work/export/av2-miami-turn` to `web/scenes/av2-miami-turn` and destroy the box.

Spend is logged in `SPEND.md`.

## Output format

See the contract in the repo docs. Notes specific to this export:

- World frame = drivestudio's world = the AV2 ego frame at the first exported frame (origin at the rear axle,
  x forward, y left, z up). The AV2 ego origin sits at rear-axle height, so **the road is about 0.25-0.35 m below the ego pose**
  under the car, not at 0 (`meta.json`: `road_z_below_ego`). The road also climbs about 2 m over the segment, so do not assume a flat z = const ground.
- `sky.ply` is an extra file: the learned sky cube map baked into a 400 m dome of 17,250 large Gaussians. Load it
  first, or skip it and use a flat sky colour.
- `tracks.json` also carries `actor_classes` (track id to AV2 category). Actor ids are the first 8 characters of the
  AV2 track UUID. Actor poses come from the optimised drivestudio instance poses (box centre frame).
- Pedestrians and cyclists are not modelled separately (StreetGS has no non-rigid nodes), so they can show up as
  faint ghosts in the background.

## Result (av2-miami-turn)

- 30,000 iterations, 55 min on 1x RTX 4090. Held-out PSNR 23.0 / SSIM 0.66 / LPIPS 0.36 (70 images: every 10th frame,
  all 7 cameras). Training-batch PSNR about 25.9 at the end.
- background.ply: 1.5M Gaussians, SH0, 84 MB (trained with SH1; the DC term alone is written so the file fits under 95 MB).
  45 vehicle actors (0.1-1.8 MB each), sky.ply 1 MB.
- Previews: `web/scenes/av2-miami-turn/previews/` (Reality View camera 7 m behind and 4 m above the car, 12 degrees
  down, 70 degree hFOV, plus one held-out front-camera frame next to its render, 26.2 dB).
- drivestudio commit used: e59bda4fa681f829dbb1d65f0de582b0f633c450.
