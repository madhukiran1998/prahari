"""One process per camera: read RTSP, pace to detect_fps, keep a clip buffer.

Deliberately boring: a blocking read loop and a couple of queues. No asyncio
here - the only thing that must never happen is one camera's stall blocking
another's, and separate processes give that for free.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

# OpenCV's default RTSP transport is UDP, which drops packets and produces
# smeared frames on a busy machine. This must be set before any VideoCapture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .clips import write_clip  # noqa: E402
from .config import CameraCfg, Settings  # noqa: E402
from .types import CamStatus  # noqa: E402

log = logging.getLogger("prahari.ingest")

RECONNECT_BACKOFF_S = (1, 2, 4, 8, 16, 30)
JPEG_QUALITY = 80


def _fingerprint(frame: np.ndarray) -> str:
    """Exact hash of the decoded pixels.

    Deliberately not a perceptual hash: a real camera always has sensor noise,
    so consecutive frames are never byte-identical. A perceptual hash would call
    an empty shop at 3am "frozen" every night; an exact hash only matches when
    the decoder is genuinely repeating a frame.
    """
    return hashlib.blake2b(frame.tobytes(), digest_size=8).hexdigest()


class _ClipRecorder:
    """Collects pre-roll + post-roll frames for one event, then encodes off-thread."""

    def __init__(self, event_id: str, pre_frames: list[bytes], until: float):
        self.event_id = event_id
        self.frames = list(pre_frames)
        self.until = until

    def add(self, jpeg: bytes) -> None:
        self.frames.append(jpeg)

    def done(self, now: float) -> bool:
        return now >= self.until


def camera_worker(
    cam: CameraCfg,
    settings: Settings,
    frame_q: mp.Queue,
    bus_q: mp.Queue,
    cmd_q: mp.Queue,
    stop: "mp.synchronize.Event",
) -> None:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
    )

    detect_fps = float(settings.detect_fps)
    interval = 1.0 / detect_fps
    pre_s = float(settings.clip.get("pre_s", 5))
    post_s = float(settings.clip.get("post_s", 10))
    ring: deque[bytes] = deque(maxlen=max(1, int(pre_s * detect_fps)))
    recorders: list[_ClipRecorder] = []

    online = None  # None = nothing reported yet
    attempt = 0
    last_emit = 0.0

    def publish_status(is_online: bool, detail: str = "") -> None:
        nonlocal online
        if online == is_online:
            return
        online = is_online
        bus_q.put(
            {
                "type": "cam_status",
                "status": asdict(
                    CamStatus(
                        cam_id=cam.id, online=is_online, ts=time.time(), detail=detail
                    )
                ),
            }
        )
        log.info("%s | %s%s", cam.id, "online" if is_online else "OFFLINE",
                 f" ({detail})" if detail else "")

    def drain_commands() -> None:
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                return
            if cmd.get("type") == "clip":
                recorders.append(
                    _ClipRecorder(cmd["event_id"], list(ring), time.time() + post_s)
                )
                log.info("%s | clip requested for %s", cam.id, cmd["event_id"])
            elif cmd.get("type") == "set_fps":
                _set_fps(float(cmd["fps"]))

    def _set_fps(fps: float) -> None:
        nonlocal detect_fps, interval, ring
        if fps <= 0 or abs(fps - detect_fps) < 0.01:
            return
        detect_fps = fps
        interval = 1.0 / detect_fps
        ring = deque(ring, maxlen=max(1, int(pre_s * detect_fps)))
        log.warning("%s | detect fps -> %.1f", cam.id, detect_fps)

    def flush_recorders(now: float, force: bool = False) -> None:
        for rec in [r for r in recorders if force or r.done(now)]:
            recorders.remove(rec)
            path = settings.clips_dir / f"{rec.event_id}.mp4"
            threading.Thread(
                target=_encode_and_report,
                args=(path, rec.frames, detect_fps, rec.event_id, cam.id, bus_q),
                daemon=True,
            ).start()

    while not stop.is_set():
        cap = cv2.VideoCapture(cam.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass  # not every backend honours it; harmless

        if not cap.isOpened():
            cap.release()
            publish_status(False, "cannot open stream")
            delay = RECONNECT_BACKOFF_S[min(attempt, len(RECONNECT_BACKOFF_S) - 1)]
            attempt += 1
            log.warning("%s | open failed, retrying in %ds", cam.id, delay)
            stop.wait(delay)
            continue

        attempt = 0
        publish_status(True)

        while not stop.is_set():
            if not cap.grab():
                publish_status(False, "stream ended")
                break

            now = time.time()
            drain_commands()
            flush_recorders(now)

            if now - last_emit < interval:
                continue  # grabbed and discarded: keeps the RTSP buffer drained
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                publish_status(False, "decode failed")
                break
            last_emit = now

            if frame.shape[1] != settings.img_width:
                scale = settings.img_width / frame.shape[1]
                frame = cv2.resize(
                    frame, (settings.img_width, max(1, int(frame.shape[0] * scale)))
                )

            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                continue
            jpeg = buf.tobytes()

            ring.append(jpeg)
            for rec in recorders:
                rec.add(jpeg)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _put_latest(
                frame_q,
                {
                    "type": "frame",
                    "cam_id": cam.id,
                    "ts": now,
                    "jpeg": jpeg,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "luma": float(gray.mean()),
                    "phash": _fingerprint(frame),
                },
            )

        cap.release()
        if not stop.is_set():
            delay = RECONNECT_BACKOFF_S[min(attempt, len(RECONNECT_BACKOFF_S) - 1)]
            attempt += 1
            log.warning("%s | reconnecting in %ds", cam.id, delay)
            stop.wait(delay)

    flush_recorders(time.time(), force=True)
    log.info("%s | ingest stopped", cam.id)


def _encode_and_report(
    path: Path, frames: list[bytes], fps: float, event_id: str, cam_id: str, bus_q
) -> None:
    try:
        written = write_clip(path, frames, fps)
    except Exception:
        log.exception("%s | clip encode failed for %s", cam_id, event_id)
        return
    if written:
        bus_q.put({"type": "clip_ready", "event_id": event_id, "path": str(written)})


def _put_latest(q: mp.Queue, item: dict) -> None:
    """Enqueue, discarding the oldest frame if the detector has fallen behind.

    Latest-wins: an alert on a frame from ten seconds ago is worse than no alert.
    """
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass
