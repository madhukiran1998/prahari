#!/usr/bin/env bash
# Starts the fake-camera rig: MediaMTX plus one looping ffmpeg per .mp4 in
# ./media (the ffmpeg loops are launched by MediaMTX itself, see mediamtx.yml).
#
#   ./sim/run_fake_cams.sh          # run in the foreground, Ctrl-C to stop
#
# Streams land on rtsp://localhost:8554/cam1 ... /cam5
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }

if [[ ! -x ./mediamtx ]]; then
  echo "mediamtx binary missing. Download it from" >&2
  echo "  https://github.com/bluenviron/mediamtx/releases (darwin_arm64 tarball)" >&2
  echo "and extract ./mediamtx into $HERE" >&2
  exit 1
fi

shopt -s nullglob
CLIPS=(media/*.mp4)
if (( ${#CLIPS[@]} == 0 )); then
  echo "No clips in $HERE/media - run ./sim/download_media.sh first." >&2
  exit 1
fi

# Normalise anything that has not been through our transcode yet (self-shot
# footage, files dropped in by hand). Canonical format = 640x360 h264, 15 fps,
# keyframe every 2 s, no audio - which is what lets mediamtx.yml stream with
# -c:v copy instead of burning CPU on five live encodes.
mkdir -p .normalized .down
for clip in "${CLIPS[@]}"; do
  base="$(basename "$clip" .mp4)"
  marker=".normalized/$base"
  if [[ -f "$marker" && "$marker" -nt "$clip" ]]; then
    continue
  fi
  echo "[normalise] $clip"
  tmp="$(mktemp -t prahari_norm).mp4"
  ffmpeg -y -hide_banner -loglevel error -i "$clip" -an \
    -vf "scale=640:360:force_original_aspect_ratio=increase,crop=640:360,fps=15" \
    -c:v libx264 -preset veryfast -crf 26 -g 30 -pix_fmt yuv420p "$tmp"
  mv "$tmp" "$clip"
  touch "$marker"
done

# Clear any camera left "down" by a previous demo.
rm -f .down/* 2>/dev/null || true

echo
echo "Starting MediaMTX with ${#CLIPS[@]} looping cameras:"
for clip in "${CLIPS[@]}"; do
  echo "   rtsp://localhost:8554/$(basename "$clip" .mp4)"
done
echo "   rtsp://localhost:8554/phone   (publish here from a phone app)"
echo
echo "Health check:  curl -s localhost:9997/v3/paths/list | python3 -m json.tool"
echo "Take one down: ./tools/cam_ctl.sh down cam4     (fires the camera-offline alert)"
echo

exec ./mediamtx mediamtx.yml
