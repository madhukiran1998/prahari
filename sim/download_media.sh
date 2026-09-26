#!/usr/bin/env bash
# Builds the fake-camera footage from CC0 Pexels clips.
#
# Pexels licence: free for commercial use, no attribution required.
# https://www.pexels.com/license/
#
# Two things worth knowing about why this script looks the way it does:
#
# 1. Almost all retail stock footage is handheld or a product close-up. Zone
#    rules need a FIXED camera and whole visible bodies, so only a handful of
#    clips are usable at all. The three sources below were picked by measuring
#    inter-frame motion and running the detector over candidates.
# 2. Each camera is a CROP of a 4K source, not the whole frame. Cropping gives
#    five genuinely different-looking viewpoints out of three clips, and - more
#    importantly - makes people 2-3x larger in the 640x360 frame we feed the
#    detector, which is the difference between reliable tracking and noise.
#
# Output: 640x360 h264, 15 fps, keyframe every 2 s, no audio.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA="$HERE/media"
RAW="$MEDIA/.raw"
mkdir -p "$MEDIA" "$RAW" "$HERE/.normalized"

# name : pexels id : crop(w:h:x:y in source pixels) : what it stands in for
CLIPS=(
  "cam1:20597684:1600:900:1150:1260:Entrance - main walkway, people entering"
  "cam2:20597684:1600:900:0:1260:Shop front - browsers at a display window"
  "cam3:3243946:1440:810:2400:1150:Display counter - lit case, staff behind it"
  "cam4:3243946:1600:900:0:900:Atrium - open floor, two levels"
  "cam5:15007111:1280:720:1180:560:Upper level - busy concourse"
)

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

for entry in "${CLIPS[@]}"; do
  IFS=':' read -r name id cw ch cx cy desc <<< "$entry"
  out="$MEDIA/$name.mp4"
  raw="$RAW/$id.mp4"

  if [[ -f "$out" ]]; then
    echo "[skip ] $name.mp4 already present"
    continue
  fi

  if [[ ! -s "$raw" ]]; then
    echo "[get  ] pexels/$id"
    if ! curl -sSL --fail -m 300 -A "$UA" -o "$raw" \
        "https://www.pexels.com/download/video/$id/"; then
      echo "[warn ] download failed for $id - skipping $name" >&2
      rm -f "$raw"
      continue
    fi
  fi

  echo "[build] $name.mp4  <- $id crop ${cw}x${ch}+${cx}+${cy}  ($desc)"
  ffmpeg -y -loglevel error -i "$raw" \
    -t 45 -an \
    -vf "crop=${cw}:${ch}:${cx}:${cy},scale=640:360,fps=15" \
    -c:v libx264 -preset veryfast -crf 24 -g 30 -pix_fmt yuv420p \
    "$out"
  touch "$HERE/.normalized/$name"   # already canonical; no re-transcode needed
done

echo
echo "Media ready in $MEDIA:"
ls -1sh "$MEDIA"/*.mp4 2>/dev/null || echo "  (none - check network, or drop your own .mp4 files here)"
cat <<'EOF'

Self-shot footage beats stock for a client demo: tape a phone high in a corner
and stage the scenarios (loiter near the door, lean over the counter with nobody
behind it, walk in "after hours"). Drop the clips into sim/media/ as cam1.mp4 ...
cam5.mp4 and re-run ./sim/run_fake_cams.sh - it normalises anything new for you.
Then redraw the zones in the dashboard's Zones mode.
EOF
