#!/usr/bin/env bash
# Take a simulated camera offline / bring it back, for the camera-health part of
# the demo.
#
#   ./tools/cam_ctl.sh down cam4
#   ./tools/cam_ctl.sh up   cam4
#   ./tools/cam_ctl.sh list
#
# "down" drops a flag file that the mediamtx.yml wrapper waits on, then kills
# that path's ffmpeg. MediaMTX respawns the wrapper, which parks until "up"
# removes the flag - so the stream stays dead until you say otherwise.
set -euo pipefail

SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")/../sim" && pwd)"
mkdir -p "$SIM/.down"

action="${1:-list}"
cam="${2:-}"

case "$action" in
  down)
    [[ -n "$cam" ]] || { echo "usage: cam_ctl.sh down <cam>" >&2; exit 1; }
    touch "$SIM/.down/$cam"
    # Match on "<port>/<cam>" so this keeps working whether mediamtx.yml
    # publishes to localhost, 127.0.0.1 or [::1].
    pkill -f "8554/$cam\$" || true
    echo "$cam is down - expect a camera-offline alert within ~15 s"
    ;;
  up)
    [[ -n "$cam" ]] || { echo "usage: cam_ctl.sh up <cam>" >&2; exit 1; }
    rm -f "$SIM/.down/$cam"
    echo "$cam coming back up (MediaMTX respawns the loop within a second)"
    ;;
  list)
    curl -s localhost:9997/v3/paths/list \
      | python3 -c '
import json, sys
for p in json.load(sys.stdin)["items"]:
    print("%-8s ready=%-5s readers=%d" % (p["name"], p["ready"], len(p["readers"])))' \
      2>/dev/null || echo "MediaMTX API not reachable - is the rig running?"
    ;;
  *)
    echo "usage: cam_ctl.sh {down|up} <cam> | list" >&2
    exit 1
    ;;
esac
