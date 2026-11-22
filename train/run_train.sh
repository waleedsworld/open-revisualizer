#!/usr/bin/env bash
# Full GPU pipeline after setup_box.sh. Run from the directory that holds train/ and drivestudio/.
# usage: bash train/run_train.sh [iters]
set -euxo pipefail
ROOT=$(pwd); ITERS=${1:-30000}
LOG=3de5b5d6-68c4-3c95-84ed-be7c83d829f8
DATA=$ROOT/drivestudio/data/argoverse
# 1. raw log (anonymous S3 over HTTPS), 2. preprocess the 10.5 s segment, 3. sky masks
bash train/download_log.sh val $LOG $DATA/raw
python train/preprocess_av2.py --log_dir $DATA/raw/val/$LOG --out $DATA/processed/training --scene 0 --t0 5.0 --t1 15.5
python train/sky_masks.py --scene_dir $DATA/processed/training/000
# 4. train StreetGS-style model (background + rigid vehicles + sky env map)
cp train/configs/av2_streetgs.yaml drivestudio/configs/av2_streetgs.yaml
cd drivestudio
export PYTHONPATH=$(pwd)
python tools/train.py --config_file configs/av2_streetgs.yaml --output_root output --project av2 --run_name revis \
  dataset=argoverse/7cams trainer.optim.num_iters=$ITERS \
  data.scene_idx=0 data.start_timestep=0 data.end_timestep=-1 \
  "data.pixel_source.downscale_when_loading=[2,2,2,2,2,2,2]" \
  data.pixel_source.test_image_stride=10 data.pixel_source.load_smpl=False \
  render.render_novel=null
