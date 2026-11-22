#!/usr/bin/env bash
# Full GPU pipeline after setup_box.sh. Run from the directory that holds train/ and drivestudio/.
# usage: bash train/run_train.sh [iters]
# HD (native resolution) run: CONFIG=av2_streetgs_hd DOWNSCALE=1 SKY_WIDTH=2048 RUN=revis_hd bash train/run_train.sh 30000
set -euxo pipefail
ROOT=$(pwd); ITERS=${1:-30000}
CONFIG=${CONFIG:-av2_streetgs}; DOWNSCALE=${DOWNSCALE:-2}; SKY_WIDTH=${SKY_WIDTH:-1024}; RUN=${RUN:-revis}
DS="[$DOWNSCALE,$DOWNSCALE,$DOWNSCALE,$DOWNSCALE,$DOWNSCALE,$DOWNSCALE,$DOWNSCALE]"
# HD only skips rendering all 735 train images at the end (slow at full res); the held-out test set is still evaluated.
EXTRA=""; [ "$DOWNSCALE" = "1" ] && EXTRA="render.render_full=False"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=3de5b5d6-68c4-3c95-84ed-be7c83d829f8
DATA=$ROOT/drivestudio/data/argoverse
# 1. raw log (anonymous S3 over HTTPS), 2. preprocess the 10.5 s segment, 3. sky masks
bash train/download_log.sh val $LOG $DATA/raw
[ -f $DATA/processed/training/000/segment.json ] || python train/preprocess_av2.py --log_dir $DATA/raw/val/$LOG --out $DATA/processed/training --scene 0 --t0 5.0 --t1 15.5
python train/sky_masks.py --scene_dir $DATA/processed/training/000 --infer_width $SKY_WIDTH
# 4. train StreetGS-style model (background + rigid vehicles + sky env map)
cp train/configs/$CONFIG.yaml drivestudio/configs/$CONFIG.yaml
cd drivestudio
export PYTHONPATH=$(pwd)
python tools/train.py --config_file configs/$CONFIG.yaml --output_root output --project av2 --run_name $RUN \
  dataset=argoverse/7cams trainer.optim.num_iters=$ITERS \
  data.scene_idx=0 data.start_timestep=0 data.end_timestep=-1 \
  "data.pixel_source.downscale_when_loading=$DS" \
  data.pixel_source.test_image_stride=10 data.pixel_source.load_smpl=False \
  render.render_novel=null $EXTRA
