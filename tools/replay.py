#!/usr/bin/env python3
"""Run a clip through the real detector and the real rules, offline.

This is how you tune zones without waiting for the live pipeline:

    python tools/replay.py cam3                  # replay sim/media/cam3.mp4
    python tools/replay.py cam3 --preview out.jpg  # zones drawn on a frame
    python tools/replay.py cam1 --video ~/store.mp4 --loops 3

It reports every event the rules would have produced, plus per-zone occupancy,
so "why did nothing fire?" has an answer other than staring at the dashboard.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from prahari.config import load_cameras, load_settings  # noqa: E402
from prahari.geometry import bbox_in_polygon  # noqa: E402
from prahari.inference import _annotate, _extract, resolve_device  # noqa: E402
from prahari.rules.engine import RulesEngine  # noqa: E402
from prahari.types import Detection, FrameCtx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cam_id")
    ap.add_argument("--video", help="clip to replay (default sim/media/<cam_id>.mp4)")
    ap.add_argument("--loops", type=int, default=1, help="replay N times, for dwell rules")
    ap.add_argument("--preview", help="write a frame with the zones drawn on it")
    ap.add_argument("--force-closed", action="store_true", help="test the after-hours rule")
    args = ap.parse_args()

    settings = load_settings()
    if args.force_closed:
        settings.demo_force_closed = True
    cams = {c.id: c for c in load_cameras()}
    cam = cams.get(args.cam_id)
    if cam is None:
        print(f"no camera '{args.cam_id}' in config/cameras.yaml", file=sys.stderr)
        return 1

    video = Path(args.video) if args.video else ROOT / "sim" / "media" / f"{cam.id}.mp4"
    if not video.exists():
        print(f"no such clip: {video}", file=sys.stderr)
        return 1

    model_path = ROOT / settings.model
    from ultralytics import YOLO

    # Same device and tracker as the live pipeline, or tuning a zone offline
    # would not predict what the running system does with it.
    model = YOLO(str(model_path) if model_path.exists() else settings.model)
    model.to(resolve_device(settings.device))

    events: list = []
    engine = RulesEngine([cam], settings, on_event=events.append)

    step = 1.0 / settings.detect_fps
    clock = time.time()
    occupancy: dict[str, list[int]] = {name: [] for name in cam.zones}
    frames = 0
    preview_written = False

    print(f"replaying {video.name} as {cam.id} ({cam.name}), {settings.detect_fps} fps")
    print(f"zones: {', '.join(cam.zones) or '(none)'}")
    print(f"rules: {', '.join(cam.rules)}\n")

    for loop in range(args.loops):
        cap = cv2.VideoCapture(str(video))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        skip = max(1, round(src_fps / settings.detect_fps))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % skip:
                continue
            if frame.shape[1] != settings.img_width:
                scale = settings.img_width / frame.shape[1]
                frame = cv2.resize(
                    frame, (settings.img_width, int(frame.shape[0] * scale))
                )

            results = model.track(
                frame, persist=True, tracker=settings.tracker, classes=[0],
                conf=settings.conf_threshold, imgsz=settings.imgsz, verbose=False,
            )
            raw = _extract(results, cam.id, clock, frame)
            dets = [
                Detection(cam_id=cam.id, ts=clock, track_id=d["track_id"],
                          bbox=tuple(d["bbox"]), conf=d["conf"],
                          appearance=d["appearance"])
                for d in raw
            ]

            for name, poly in cam.zones.items():
                if len(poly) > 2:
                    occupancy[name].append(
                        sum(1 for d in dets if bbox_in_polygon(d.bbox, poly))
                    )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            before = len(events)
            engine.handle_frame(
                FrameCtx(cam_id=cam.id, ts=clock, detections=dets,
                         width=frame.shape[1], height=frame.shape[0],
                         luma=float(gray.mean()), phash=f"f{frames}")
            )
            engine.handle_tick(clock)
            for ev in events[before:]:
                offset = clock - (clock - frames * step)
                print(f"  t={offset:6.1f}s  {ev.severity:8} {ev.kind:8} "
                      f"{ev.rule:14} {ev.title}")

            if args.preview and not preview_written:
                cv2.imwrite(args.preview, _annotate(frame, raw, cam.zones, cam.name))
                preview_written = True

            frames += 1
            clock += step
        cap.release()

    print(f"\n{frames} frames ({frames * step:.0f}s of footage), "
          f"{len([e for e in events if e.kind == 'start'])} alert(s)")
    for name, counts in occupancy.items():
        if not counts:
            continue
        arr = np.array(counts)
        print(f"  zone {name:18} occupied {100 * (arr > 0).mean():5.1f}% of frames, "
              f"mean {arr.mean():.2f}, max {arr.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
