#!/usr/bin/env bash
# Usage: tools/make_gif.sh <frames_dir> <out.gif> [fps=12] [width=880]
set -euo pipefail
in="$1"; out="$2"; fps="${3:-12}"; w="${4:-880}"
pal="$(mktemp -t pal).png"
ffmpeg -loglevel error -y -framerate 30 -i "$in/f_%04d.png" -vf "fps=$fps,scale=$w:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=200" "$pal"
ffmpeg -loglevel error -y -framerate 30 -i "$in/f_%04d.png" -i "$pal" -lavfi "fps=$fps,scale=$w:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=sierra2_4a:diff_mode=rectangle" "$out"
rm -f "$pal"
ls -la "$out"
