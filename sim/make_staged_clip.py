#!/usr/bin/env python3
"""Builds sim/media/cam3.mp4 - a STAGED jewellery-counter scenario.

Why this clip is assembled rather than filmed
---------------------------------------------
The unattended-counter alert is the headline of the demo, so it needs footage
where a staff member is demonstrably present and then demonstrably absent.
That footage does not exist for free: retail stock video is handheld product
close-ups, and in the one usable fixed-camera counter scene the people behind
the counter are never cleanly detected (measured: zero detected feet above
y=120) and the clip is 8 seconds long, so no condition can hold for a 20-second
grace period.

So the scene is composited from real photographic material:

  * the room is a CC0 photograph of an empty jewellery showroom, which makes it
    a perfect clean plate and puts the demo in the client's own world;
  * the two people are real person cut-outs lifted from the stock crowd footage
    with a segmentation mask, not drawings.

Only the timing is fiction, and it is deliberately obvious so you can narrate it:

    0-14s   staff behind the counter, customer in front    quiet
    14-46s  staff steps away, customer stays               -> CRITICAL after the grace period
    46-60s  staff returns                                  -> resolved

Say "this camera is staged so the scenario repeats on demand" when you show it.
The other four cameras are untouched stock footage.

    python sim/make_staged_clip.py
    python sim/make_staged_clip.py --preview   # also writes frames to inspect
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "media" / ".raw"
OUT = HERE / "media" / "cam3.mp4"

# CC0 photo: "Luxurious Modern Jewelry Store Interior Design", pexels.com/photo/33257665
BACKDROP_ID = 33257665
BACKDROP = RAW / f"jewellery_store_{BACKDROP_ID}.jpg"
BACKDROP_URL = (
    f"https://images.pexels.com/photos/{BACKDROP_ID}/"
    f"pexels-photo-{BACKDROP_ID}.jpeg?auto=compress&cs=tinysrgb&w=1920"
)
# Where the people come from - a fixed-camera crowd clip we already downloaded.
PEOPLE_SOURCE = HERE / "media" / "cam2.mp4"

W, H = 640, 360
FPS = 15
DURATION_S = 60

# The showroom counter runs across this band. Re-pasting it over a composited
# figure is what makes them read as standing *behind* the counter - and it puts
# the bottom of their detection box on the counter line, which is where the
# counter_staff zone sits.
COUNTER_TOP, COUNTER_BOTTOM = 150, 205

STAFF_SPOT = (250, 196)      # centre x, foot y  (feet end up hidden by the counter)
STAFF_HEIGHT = 118
CUSTOMER_SPOT = (395, 262)
CUSTOMER_HEIGHT = 138

STAFF_PRESENT = [(0.0, 14.0), (46.0, 60.0)]
CUSTOMER_PRESENT = [(3.0, 60.0)]


def fetch_backdrop() -> np.ndarray:
    RAW.mkdir(parents=True, exist_ok=True)
    if not BACKDROP.exists():
        print("fetching showroom backdrop ...")
        subprocess.run(
            ["curl", "-sSL", "--fail", "-m", "120", "-A", "Mozilla/5.0",
             "-o", str(BACKDROP), BACKDROP_URL],
            check=True,
        )
    img = cv2.imread(str(BACKDROP))
    if img is None:
        raise SystemExit(f"could not read {BACKDROP}")
    # Crop to 16:9 around the counters, then down to the pipeline's frame size.
    h, w = img.shape[:2]
    want_h = int(w * 9 / 16)
    top = min(max(0, h - want_h), int(h * 0.22))
    return cv2.resize(img[top : top + want_h], (W, H))


def lift_people(count: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut real people out of the crowd footage using segmentation masks."""
    from ultralytics import YOLO

    weights = ROOT / "yolo11n-seg.pt"
    model = YOLO(str(weights) if weights.exists() else "yolo11n-seg.pt")

    cap = cv2.VideoCapture(str(PEOPLE_SOURCE))
    found: list[tuple[float, np.ndarray, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 5:
            continue
        res = model.predict(frame, classes=[0], conf=0.6, verbose=False, retina_masks=True)
        r = res[0]
        if r.masks is None:
            continue
        for box, conf, mask in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.masks.data.cpu().numpy(),
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            bh, bw = y2 - y1, x2 - x1
            # A whole standing body, clear of the frame edges.
            if bh < 80 or bw < 24 or not (1.7 <= bh / bw <= 4.0):
                continue
            if x1 < 6 or x2 > frame.shape[1] - 6 or y1 < 4 or y2 > frame.shape[0] - 4:
                continue
            m = (mask * 255).astype(np.uint8)
            if m.shape[:2] != frame.shape[:2]:
                m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
            patch, alpha = frame[y1:y2, x1:x2], m[y1:y2, x1:x2]
            if alpha.mean() < 70:  # mask barely covers the box - bad cut-out
                continue
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
            found.append((float(conf) * bh, patch.copy(), alpha))
    cap.release()

    if len(found) < count:
        raise SystemExit(
            f"only {len(found)} usable figures in {PEOPLE_SOURCE.name}; "
            "need a clip with clearly separated standing people"
        )
    found.sort(key=lambda f: f[0], reverse=True)
    return [(p, a) for _, p, a in found[:count]]


def paste(canvas, patch, alpha, spot, target_h):
    scale = target_h / patch.shape[0]
    patch = cv2.resize(patch, None, fx=scale, fy=scale)
    alpha = cv2.resize(alpha, None, fx=scale, fy=scale)
    ph, pw = patch.shape[:2]
    cx, foot_y = spot
    x0, y0 = int(cx - pw / 2), int(foot_y - ph)
    sx, sy = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + pw - sx), min(H, y0 + ph - sy)
    if x1 <= x0 or y1 <= y0:
        return
    patch = patch[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    a = (alpha[sy : sy + (y1 - y0), sx : sx + (x1 - x0)] / 255.0)[..., None]
    canvas[y0:y1, x0:x1] = (
        patch * a + canvas[y0:y1, x0:x1] * (1 - a)
    ).astype(np.uint8)


def active(t: float, windows) -> bool:
    return any(a <= t < b for a, b in windows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if not PEOPLE_SOURCE.exists():
        print(f"missing {PEOPLE_SOURCE} - run ./sim/download_media.sh first",
              file=sys.stderr)
        return 1

    plate = fetch_backdrop()
    print("lifting people out of the crowd footage ...")
    (staff_patch, staff_alpha), (cust_patch, cust_alpha) = lift_people(2)
    counter_strip = plate[COUNTER_TOP:COUNTER_BOTTOM].copy()
    rng = np.random.default_rng(7)

    tmpdir = Path(tempfile.mkdtemp(prefix="prahari_staged_"))
    total = DURATION_S * FPS
    print(f"compositing {total} frames ...")
    for i in range(total):
        t = i / FPS
        canvas = plate.copy()

        if active(t, STAFF_PRESENT):
            sway = int(round(2.5 * np.sin(t * 1.7)))
            paste(canvas, staff_patch, staff_alpha,
                  (STAFF_SPOT[0] + sway, STAFF_SPOT[1]), STAFF_HEIGHT)
            canvas[COUNTER_TOP:COUNTER_BOTTOM] = counter_strip  # they are behind it

        if active(t, CUSTOMER_PRESENT):
            sway = int(round(3.0 * np.sin(t * 1.1 + 2.0)))
            lean = int(round(2.0 * np.sin(t * 0.6)))
            paste(canvas, cust_patch, cust_alpha,
                  (CUSTOMER_SPOT[0] + sway, CUSTOMER_SPOT[1] + lean), CUSTOMER_HEIGHT)

        # Sensor grain: a real camera never sends two identical frames, and the
        # frozen-feed check relies on exactly that.
        noise = rng.normal(0, 1.8, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(tmpdir / f"{i:05d}.jpg"), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if args.preview and i in (30, 15 * 25, 15 * 50):
            cv2.imwrite(str(ROOT / f"staged_preview_{i // 15:02d}s.jpg"), canvas)

    print(f"encoding {OUT} ...")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", str(tmpdir / "%05d.jpg"), "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-g", "30", "-pix_fmt", "yuv420p", str(OUT)],
        check=True,
    )
    for f in tmpdir.glob("*.jpg"):
        f.unlink()
    tmpdir.rmdir()
    (HERE / ".normalized").mkdir(exist_ok=True)
    (HERE / ".normalized" / "cam3").touch()

    print(f"done: {OUT} ({DURATION_S}s)")
    print("staff away 14-46s, so the counter alert fires around 34s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
