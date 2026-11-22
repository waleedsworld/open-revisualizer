#!/usr/bin/env bash
# Download one Argoverse 2 Sensor log (7 ring cameras, lidar, calibration, map, poses, annotations)
# anonymously over HTTPS. Stereo cameras are skipped.
# usage: download_log.sh <split> <log_id> <out_root>
set -euo pipefail
SPLIT=${1:-val}; LOG=${2:-3de5b5d6-68c4-3c95-84ed-be7c83d829f8}; OUT=${3:-data/argoverse/raw}
B=https://argoverse.s3.amazonaws.com; P=datasets/av2/sensor/$SPLIT/$LOG
mkdir -p "$OUT/$SPLIT/$LOG"; cd "$OUT/$SPLIT/$LOG"
: > keys.txt
for p in sensors/lidar calibration map sensors/cameras/ring_front_center sensors/cameras/ring_front_left \
         sensors/cameras/ring_front_right sensors/cameras/ring_side_left sensors/cameras/ring_side_right \
         sensors/cameras/ring_rear_left sensors/cameras/ring_rear_right; do
  tok=""
  while :; do
    r=$(curl -s "$B/?list-type=2&prefix=$P/$p/${tok:+&continuation-token=$tok}")
    echo "$r" | grep -oE "<Key>[^<]+</Key>" | sed -E 's/<\/?Key>//g' >> keys.txt
    tok=$( (echo "$r" | grep -oE "<NextContinuationToken>[^<]+" || true) | sed 's/<NextContinuationToken>//' \
          | python3 -c "import sys,urllib.parse;print(urllib.parse.quote(sys.stdin.read().strip(),safe=''))")
    [ -z "$tok" ] && break
  done
done
echo "$P/annotations.feather" >> keys.txt; echo "$P/city_SE3_egovehicle.feather" >> keys.txt
echo "$(wc -l < keys.txt) files"
sed "s#^$P/##" keys.txt | xargs -n1 dirname | sort -u | xargs mkdir -p
export B P
sed "s#^$P/##" keys.txt | xargs -P 32 -n 1 sh -c '[ -s "$0" ] || curl -sf --retry 3 -o "$0" "$B/$P/$0"'
echo done; du -sh .
