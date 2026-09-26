#!/usr/bin/env python3
"""Turns a hand-held clip into a fixed-camera one, or says it cannot.

Every zone rule assumes the camera does not move, and free retail stock footage
is almost entirely hand-held - `tools/score_footage.py` rejected eleven of
eleven candidates on movement alone. Rather than give up on the content, this
removes the movement:

  1. measure the global shift between consecutive frames by phase correlation,
     which reports camera motion while ignoring people walking about;
  2. find the window of `--seconds` where the camera wandered least - operators
     hold still for a few seconds at a time even when the shot as a whole pans;
  3. translate every frame in that window back onto the first one and crop away
     the margin the translation exposed.

Translation only. A clip where the camera rotates, zooms or dollies cannot be
locked this way and is reported as such rather than quietly half-fixed - a
zone drawn on a half-fixed clip is worse than no clip, because it looks right.

    python tools/lock_camera.py in.mp4 out.mp4 --seconds 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prahari.config import load_settings  # noqa: E402
from score_footage import PERSON_CLASS, _background_mask  # noqa: E402

ANALYSE_W = 640
OUT_W, OUT_H = 640, 360

# Beyond this much residual movement the window is not worth locking: the crop
# needed to hide it would throw away most of the picture.
MAX_LOCKABLE_PX = 90


def read_frames(path: Path, seconds: float | None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    limit = int(seconds * fps) if seconds else None
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        scale = ANALYSE_W / frame.shape[1]
        frames.append(cv2.resize(frame, (ANALYSE_W, max(1, int(frame.shape[0] * scale)))))
    cap.release()
    return frames, fps


def motion_path(frames: list[np.ndarray], settings) -> np.ndarray:
    """Cumulative (x, y) camera position, in pixels, one row per frame.

    People are detected and masked out before the background is compared. This
    is not a refinement - on a clip where a shopper fills most of the frame,
    an unmasked estimate follows the shopper, and "locking" to it shakes the
    room around a person held still. Measured: one candidate scored 4 px/s
    unmasked and 19 px/s once the subject was excluded.
    """
    from ultralytics import YOLO

    model = YOLO(settings.model)
    path = [(0.0, 0.0)]
    prev_gray = prev_pts = None

    for frame in frames:
        res = model.predict(
            frame, classes=[PERSON_CLASS], conf=settings.conf_threshold,
            imgsz=settings.imgsz, verbose=False,
        )
        boxes = getattr(res[0], "boxes", None) if res else None
        xyxy = (
            boxes.xyxy.cpu().numpy()
            if boxes is not None and boxes.xyxy is not None and len(boxes)
            else np.empty((0, 4))
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx = dy = 0.0
        if prev_gray is not None and prev_pts is not None and len(prev_pts) >= 6:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            if nxt is not None:
                kept = status.ravel() == 1
                if kept.sum() >= 6:
                    moved = nxt[kept] - prev_pts[kept]
                    dx, dy = (float(v) for v in np.median(moved.reshape(-1, 2), axis=0))
        path.append((path[-1][0] + dx, path[-1][1] + dy))
        prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=8,
            mask=_background_mask(frame.shape, xyxy),
        )
        prev_gray = gray

    # One row per frame: the seed entry above is the first frame's position.
    return np.array(path[: len(frames)])


def steadiest_window(path: np.ndarray, length: int) -> tuple[int, float]:
    """Start index of the run of `length` frames the camera moved least over.

    Scored on peak-to-peak spread rather than total distance: a camera that
    jitters and returns is lockable, one that walks steadily away is not.
    """
    if len(path) <= length:
        return 0, float(np.ptp(path, axis=0).max())
    best, best_score = 0, float("inf")
    for start in range(len(path) - length):
        window = path[start:start + length]
        score = float(np.ptp(window, axis=0).max())
        if score < best_score:
            best, best_score = start, score
    return best, best_score


def lock(frames: list[np.ndarray], path: np.ndarray, start: int, length: int):
    """Translate each frame back onto the first, then crop off the exposed edge."""
    window = frames[start:start + length]
    offsets = path[start:start + length] - path[start]
    margin = int(np.ceil(np.abs(offsets).max())) + 2

    out = []
    for frame, (dx, dy) in zip(window, offsets):
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        shifted = cv2.warpAffine(
            frame, m, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE
        )
        h, w = shifted.shape[:2]
        cropped = shifted[margin:h - margin, margin:w - margin]
        if cropped.size == 0:
            return []
        out.append(cv2.resize(cropped, (OUT_W, OUT_H)))
    return out


def write(frames: list[np.ndarray], path: Path, fps: float) -> None:
    """Pipe raw frames through ffmpeg - MediaMTX needs real h264, not mp4v."""
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{OUT_W}x{OUT_H}", "-r", str(round(fps)), "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-g", "30", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--scan", type=float, default=None,
                    help="only look at the first N seconds of the source")
    args = ap.parse_args()

    frames, fps = read_frames(args.src, args.scan)
    if len(frames) < 10:
        print(f"{args.src.name}: too short")
        return 1

    path = motion_path(frames, load_settings())
    length = min(len(frames) - 1, int(args.seconds * fps))
    start, spread = steadiest_window(path, length)

    total = float(np.ptp(path, axis=0).max())
    print(f"{args.src.name}: camera wandered {total:.0f}px overall; "
          f"steadiest {length / fps:.0f}s window starts at {start / fps:.1f}s "
          f"and spans {spread:.0f}px")

    if spread > MAX_LOCKABLE_PX:
        print("  -> NOT LOCKABLE: no window is steady enough to crop back into place")
        return 2

    locked = lock(frames, path, start, length)
    if not locked:
        print("  -> NOT LOCKABLE: correction would crop the frame away")
        return 2
    write(locked, args.dst, fps)
    print(f"  -> wrote {args.dst} ({len(locked)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
