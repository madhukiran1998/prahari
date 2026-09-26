#!/usr/bin/env python3
"""Pre-meeting check: is the whole demo actually going to work right now?

Run this after starting the rig and the app, before you open the laptop in
front of anyone. It exercises the real system over HTTP - no mocks - and tells
you which part is broken rather than "something went wrong".

    ./sim/run_fake_cams.sh &
    python -m prahari.main &
    python tools/smoke_test.py

Exit code 0 means the demo in README's "Demo script" section will land.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prahari.config import load_cameras, load_settings  # noqa: E402

PASS, FAIL, WARN = "  ok  ", " FAIL ", " warn "
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> bool:
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    return status != FAIL


# A dropped connection is not a failed check. Retry transient errors so one
# blip does not read as "the demo is broken".
TRIES = 3


def get(url: str, timeout: float = 5.0):
    last: Exception | None = None
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            last = exc
            if attempt < TRIES - 1:
                time.sleep(1.5)
    raise last  # type: ignore[misc]


def head_ok(url: str, timeout: float = 10.0) -> tuple[bool, int]:
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.status == 200, len(resp.read())
        except urllib.error.HTTPError as exc:
            return False, exc.code
        except Exception:
            if attempt < TRIES - 1:
                time.sleep(1.5)
    return False, 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", type=int, default=150,
                    help="seconds to wait for the counter alert (default 150)")
    ap.add_argument("--base", default=None, help="dashboard base URL")
    args = ap.parse_args()

    settings = load_settings()
    cams = [c for c in load_cameras() if c.enabled]
    base = args.base or f"http://localhost:{settings.port}"

    print(f"checking {base}\n")

    # 1. the camera rig. Its admin API is optional - "are the cameras online?"
    # below is the check that actually matters, and it comes from the app.
    try:
        paths = get("http://127.0.0.1:9997/v3/paths/list")["items"]
        ready = [p["name"] for p in paths if p["ready"]]
        record(PASS if ready else WARN, "MediaMTX rig",
               f"{len(ready)} stream(s) ready: {', '.join(ready)}")
    except Exception as exc:
        record(WARN, "MediaMTX rig", f"admin API not reachable ({exc})")

    # 2. the app
    try:
        kpi = get(f"{base}/api/kpis")
        record(PASS, "Dashboard API", f"up, {kpi['uptime_s']}s uptime")
    except Exception as exc:
        record(FAIL, "Dashboard API", f"not reachable ({exc}). Run python -m prahari.main")
        return summarise()

    record(PASS if head_ok(f"{base}/")[0] else FAIL, "Dashboard page")

    # 3. every camera is delivering frames the detector can see.
    # Give the workers a moment first - RTSP negotiation and the model load take
    # a few seconds, and checking before then just measures the startup.
    warmup = time.time() + 45
    while time.time() < warmup:
        live = get(f"{base}/api/cams")
        if all(c["online"] and c["last_frame_age_s"] is not None for c in live):
            break
        time.sleep(3)
    live = get(f"{base}/api/cams")
    offline = [c["name"] for c in live if not c["online"]]
    record(PASS if not offline else FAIL, "Cameras online",
           f"{len(live) - len(offline)}/{len(live)}"
           + (f"; offline: {', '.join(offline)}" if offline else ""))

    stale = [c["name"] for c in live
             if c["online"] and (c["last_frame_age_s"] is None or c["last_frame_age_s"] > 5)]
    record(PASS if not stale else FAIL, "Frames arriving",
           "all fresh" if not stale else f"stale: {', '.join(stale)}")

    for cam in live:
        if head_ok(f"{base}/api/cams/{cam['id']}/frame.jpg")[0]:
            continue
        record(FAIL, f"Live frame for {cam['name']}")
        break
    else:
        record(PASS, "Live frames served", f"{len(live)} camera(s)")

    # 4. detector budget
    slow = float(settings.perf.get("slow_ms", 80))
    infer = kpi.get("infer_ms", 0)
    record(PASS if 0 < infer <= slow else (WARN if infer else FAIL), "Detector speed",
           f"{infer} ms/frame (budget {slow:.0f} ms)")

    # 5. zones the demo depends on
    counter = next((c for c in live if "zone_breach" in c["rules"]), None)
    if counter is None:
        record(FAIL, "Counter camera", "no camera has the zone_breach rule")
    else:
        have = set(counter["zones"])
        need = {"counter_customer", "counter_staff"}
        record(PASS if need <= have else FAIL, "Counter zones drawn",
               f"{counter['name']}: {', '.join(sorted(have)) or 'none'}")

    # 6. the headline alert, end to end
    print(f"\nwaiting up to {args.wait}s for the counter alert ...", flush=True)
    deadline = time.time() + args.wait
    alert = None
    while time.time() < deadline:
        try:
            rows = get(f"{base}/api/events?rule=zone_breach&severity=CRITICAL&limit=1")
        except Exception as exc:
            print(f"  (still waiting; {type(exc).__name__})", flush=True)
            time.sleep(3)
            continue
        if rows:
            alert = rows[0]
            break
        time.sleep(3)

    if alert is None:
        record(FAIL, "Counter alert",
               f"nothing in {args.wait}s. Check cam3 zones, or raise --wait "
               "(cooldown is "
               f"{settings.cooldown_for('zone_breach'):.0f}s between incidents)")
        return summarise()

    age = time.time() - alert["ts"]
    record(PASS, "Counter alert", f"\"{alert['title']}\" ({age:.0f}s ago)")

    ok, size = head_ok(f"{base}/snapshots/{alert['id']}.jpg")
    record(PASS if ok else FAIL, "Alert snapshot", f"{size // 1024} KB" if ok else "missing")

    post_s = float(settings.clip.get("post_s", 10))
    clip_deadline = time.time() + post_s + 20
    while time.time() < clip_deadline:
        ok, size = head_ok(f"{base}/clips/{alert['id']}.mp4")
        if ok:
            record(PASS, "Alert clip", f"{size // 1024} KB")
            break
        time.sleep(2)
    else:
        record(FAIL, "Alert clip", "never appeared - check the ingest worker log")

    # 7. things that are fine either way, but you should know before you present
    wa = settings.whatsapp
    record(PASS if wa.get("enabled") else WARN, "WhatsApp",
           "configured" if wa.get("enabled")
           else "off - alerts show on the dashboard and console only")
    record(WARN if settings.demo_force_closed else PASS, "Store hours",
           "demo_force_closed is ON - every camera counts as after-hours"
           if settings.demo_force_closed
           else f"open {settings.store_open}-{settings.store_close}")

    return summarise()


def summarise() -> int:
    failed = [name for status, name, _ in results if status == FAIL]
    warned = [name for status, name, _ in results if status == WARN]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("All checks passed - the demo is good to go."
          + (f" ({len(warned)} note(s): {', '.join(warned)})" if warned else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
