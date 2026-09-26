"""Turns a list of buffered JPEG frames into an mp4.

Clips run at detect_fps (5), which is what the ring buffer holds. That looks
exactly like real CCTV export footage, so nobody asks why it is not 30 fps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

log = logging.getLogger("prahari.clips")


def write_clip(path: Path, frames: list[bytes], fps: float) -> Path | None:
    """Encode JPEG bytes to H.264. Returns the path, or None if there was nothing."""
    if not frames:
        log.warning("no frames buffered for %s", path.name)
        return None

    first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        log.warning("undecodable first frame for %s", path.name)
        return None
    height, width = first.shape[:2]

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        fps=max(1.0, fps),
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        codec="libx264",
        macro_block_size=8,  # 640x360 is divisible by 8, so no padding
        ffmpeg_log_level="error",
        output_params=["-preset", "veryfast", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for jpeg in frames:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            writer.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        writer.close()

    log.info("wrote %s (%d frames)", path.name, len(frames))
    return path
