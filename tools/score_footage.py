#!/usr/bin/env python3
"""Scores candidate footage for whether the zone rules can actually work on it.

Stock video is chosen by humans for looking nice, which is uncorrelated with
being usable here. Four things decide it, and all four are measurable, so this
measures them rather than asking someone to watch clips:

  steadiness  every zone rule assumes the camera does not move. Measured by
              phase correlation between consecutive frames: a locked-off camera
              drifts under a pixel a second, a handheld one tens.
  presence    fraction of frames containing anybody at all. A beautiful empty
              showroom demonstrates nothing.
  body size   median person height as a fraction of the frame. Below about a
              fifth, boxes flicker and ByteTrack churns, which silently breaks
              every dwell timer.
  continuity  seconds a track survives, and how many track ids get burned per
              person actually present. Dwell rules need people to keep their id.

    python tools/score_footage.py sim/media/*.mp4
    python tools/score_footage.py --seconds 20 candidates/*.mp4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prahari.config import load_settings  # noqa: E402

PERSON_CLASS = 0

# Thresholds a clip must clear to be worth building a camera out of. Set from
# measuring the footage already in sim/media, which is known to work.
GOOD = {
    "drift_px_s": 1.5,
    "presence": 0.60,
    "body_frac": 0.20,
    "track_s": 3.0,
}


def sample(path: Path, fps: float, seconds: float):
    """Decode the clip at the detector's frame rate, resized as it will be seen."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    frames, idx = [], 0
    while len(frames) < int(seconds * fps):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            scale = 640 / frame.shape[1]
            frames.append(
                cv2.resize(frame, (640, max(1, int(frame.shape[0] * scale))))
            )
        idx += 1
    meta = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / src_fps,
    }
    cap.release()
    return frames, meta


def _background_mask(shape, boxes: np.ndarray) -> np.ndarray:
    """White everywhere except where people are, generously padded.

    Measuring camera movement on pixels that contain a walking person measures
    the person. On a close-up - where a shopper fills most of the frame - that
    is the difference between "hand-held" and "locked off", so the people have
    to be cut out before the background is compared at all.
    """
    mask = np.full(shape[:2], 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        pad = 0.15 * (x2 - x1)
        cv2.rectangle(
            mask,
            (int(max(0, x1 - pad)), int(max(0, y1 - pad))),
            (int(min(shape[1], x2 + pad)), int(min(shape[0], y2 + pad))),
            0,
            -1,
        )
    return mask


def analyse(frames, settings, fps, model_name=None, tracker="bytetrack.yaml",
            device=None) -> dict:
    """One pass: detect people, then measure the camera on what is left.

    Detection and steadiness are computed together because the second needs the
    first - the person boxes are what gets masked out of the background.
    """
    from ultralytics import YOLO

    model = YOLO(model_name or settings.model)
    counts, heights, shifts, times = [], [], [], []
    life: dict[int, int] = defaultdict(int)
    prev_gray = prev_pts = None

    for frame in frames:
        t0 = time.perf_counter()
        res = model.track(
            frame,
            persist=True,
            tracker=tracker,
            classes=[PERSON_CLASS],
            conf=settings.conf_threshold,
            imgsz=settings.imgsz,
            verbose=False,
            **({"device": device} if device else {}),
        )
        times.append((time.perf_counter() - t0) * 1000.0)
        boxes = getattr(res[0], "boxes", None) if res else None
        xyxy = (
            boxes.xyxy.cpu().numpy()
            if boxes is not None and boxes.xyxy is not None and len(boxes)
            else np.empty((0, 4))
        )
        counts.append(len(xyxy))
        heights += [float(b[3] - b[1]) / frame.shape[0] for b in xyxy]
        if boxes is not None and len(xyxy) and boxes.id is not None:
            for tid in boxes.id.cpu().numpy().astype(int):
                life[int(tid)] += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_pts is not None and len(prev_pts) >= 6:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            if nxt is not None:
                kept = status.ravel() == 1
                if kept.sum() >= 6:
                    moved = nxt[kept] - prev_pts[kept]
                    # Median of the background's displacement vectors: robust to
                    # the handful of features that latch onto a moving reflection.
                    dx, dy = np.median(moved.reshape(-1, 2), axis=0)
                    shifts.append(float((dx * dx + dy * dy) ** 0.5))
        prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=8,
            mask=_background_mask(frame.shape, xyxy),
        )
        prev_gray = gray

    present = [c for c in counts if c]
    lifetimes = [n / fps for n in life.values()]
    return {
        # The first few frames carry model warm-up, which is not what a
        # steady-state pipeline pays. Median, not mean.
        "ms": statistics.median(times[3:] or times),
        "drift": float(statistics.mean(shifts) * fps) if shifts else 0.0,
        "presence": len(present) / len(counts) if counts else 0.0,
        "people": statistics.median(present) if present else 0,
        "body_frac": statistics.median(heights) if heights else 0.0,
        "tracks": len(life),
        "track_s": statistics.median(lifetimes) if lifetimes else 0.0,
        # Track ids burned per person on screen. Near 1 is a camera where
        # people keep their identity; 5 means the dwell timers keep resetting.
        "churn": len(life) / statistics.median(present) if present else 0.0,
    }


def verdict(row: dict) -> str:
    if row["drift"] > GOOD["drift_px_s"]:
        return "REJECT camera moves"
    if row["presence"] < GOOD["presence"]:
        return "REJECT too empty"
    if row["body_frac"] < GOOD["body_frac"]:
        return "WEAK   people too small"
    if row["track_s"] < GOOD["track_s"]:
        return "WEAK   tracks break up"
    return "GOOD"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--model", default=None, help="weights to benchmark instead of settings.model")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--device", default=None, help="cpu, mps, cuda")
    args = ap.parse_args()
    settings = load_settings()
    fps = float(settings.detect_fps)

    label = f"{args.model or settings.model} / {args.tracker.replace('.yaml', '')}"
    label += f" / {args.device or 'default'}"
    print(f"{label:<26}{'source':>11} {'ms':>6} {'drift':>7} {'seen':>6} {'ppl':>4} "
          f"{'body':>6} {'trk':>5} {'churn':>6}  verdict")
    print("-" * 104)

    rows = []
    for path in args.clips:
        frames, meta = sample(path, fps, args.seconds)
        if not frames:
            print(f"{path.name:<26}  no frames")
            continue
        row = {"name": path.name, **meta}
        row.update(analyse(frames, settings, fps, args.model, args.tracker, args.device))
        row["verdict"] = verdict(row)
        rows.append(row)
        print(
            f"{row['name']:<26}{row['w']}x{row['h']:>5} {row['ms']:>6.0f} {row['drift']:>7.2f} "
            f"{row['presence'] * 100:>5.0f}% {row['people']:>4.0f} "
            f"{row['body_frac'] * 100:>5.0f}% {row['track_s']:>4.1f}s "
            f"{row['churn']:>6.1f}  {row['verdict']}"
        )

    print("-" * 96)
    print(f"drift = px/s of camera movement (want <{GOOD['drift_px_s']}), "
          f"seen = frames with anybody, body = person height as % of frame "
          f"(want >{int(GOOD['body_frac'] * 100)}%),")
    print(f"trk = median seconds a track survives (want >{GOOD['track_s']}), "
          f"churn = track ids burned per person on screen (want ~1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
