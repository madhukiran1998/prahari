"""Single process, all cameras: YOLO person detection + ByteTrack IDs.

One YOLO object per camera. That is not wasteful paranoia - `model.track(
persist=True)` keeps tracker state on the model instance, so sharing one model
across streams would let cam1's track IDs leak into cam3 and silently corrupt
every dwell timer in the system.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import statistics
import time
from collections import deque

import cv2
import numpy as np

from .config import CameraCfg, Settings

log = logging.getLogger("prahari.inference")

PERSON_CLASS = 0
BATCH = 8  # frames drained per loop before we go back for more

ZONE_COLOURS = {
    "counter_customer": (0, 165, 255),  # amber - the watched side
    "counter_staff": (255, 160, 0),  # blue - where staff should be
    "entrance": (0, 200, 0),
    "entry_line": (255, 255, 0),
}
DEFAULT_ZONE_COLOUR = (200, 200, 200)
BOX_COLOUR = (0, 220, 60)


def inference_worker(
    cams: list[CameraCfg],
    settings: Settings,
    frame_q: mp.Queue,
    bus_q: mp.Queue,
    stop: "mp.synchronize.Event",
    cmd_q: mp.Queue | None = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
    )
    from ultralytics import YOLO

    device = resolve_device(settings.device)
    log.info(
        "loading %s on %s for %d cameras, tracking with %s",
        settings.model, device, len(cams), settings.tracker,
    )
    models = {cam.id: YOLO(settings.model).to(device) for cam in cams}
    display_ids = {cam.id: _DisplayIds() for cam in cams}
    zones = {cam.id: cam.zones for cam in cams}
    names = {cam.id: cam.name for cam in cams}
    log.info("models ready")

    timings: deque[float] = deque(maxlen=60)
    degraded = False
    slow_ms = float(settings.perf.get("slow_ms", 80))
    slow_fps = float(settings.perf.get("slow_detect_fps", 3))

    while not stop.is_set():
        # Zones redrawn in the dashboard, so the overlays match the rules.
        while cmd_q is not None:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                break
            if cmd.get("type") == "zones":
                zones[cmd["cam_id"]] = cmd["zones"]
                log.info("%s | zone overlay updated", cmd["cam_id"])

        batch = _drain(frame_q, BATCH, timeout=0.25)
        for msg in batch:
            cam_id = msg["cam_id"]
            model = models.get(cam_id)
            if model is None:
                continue

            frame = cv2.imdecode(np.frombuffer(msg["jpeg"], np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=settings.tracker,
                    classes=[PERSON_CLASS],
                    conf=settings.conf_threshold,
                    imgsz=settings.imgsz,
                    verbose=False,
                )
            except Exception:
                log.exception("%s | inference failed", cam_id)
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            detections = _extract(results, cam_id, msg["ts"], frame)
            labels = display_ids[cam_id]
            labels.retire({d["track_id"] for d in detections})
            for det in detections:
                det["label"] = labels.label(det["track_id"])
            annotated = _annotate(
                frame, detections, zones.get(cam_id, {}), names.get(cam_id, cam_id)
            )
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            bus_q.put(
                {
                    "type": "det",
                    "cam_id": cam_id,
                    "ts": msg["ts"],
                    "detections": detections,
                    "jpeg": buf.tobytes() if ok else msg["jpeg"],
                    "width": msg["width"],
                    "height": msg["height"],
                    "luma": msg["luma"],
                    "phash": msg["phash"],
                    "infer_ms": elapsed_ms,
                }
            )

        if not degraded and len(timings) == timings.maxlen:
            median = statistics.median(timings)
            if median > slow_ms:
                degraded = True
                log.warning(
                    "median inference %.0f ms > %.0f ms budget - dropping to %.0f fps",
                    median,
                    slow_ms,
                    slow_fps,
                )
                bus_q.put({"type": "perf_degrade", "fps": slow_fps, "median_ms": median})

    log.info("inference stopped")


class _DisplayIds:
    """Small, reusable labels for tracks, per camera.

    ByteTrack and OC-SORT ids climb forever - on a looping demo clip they reach
    four digits within minutes, and "#2317" on screen reads as noise rather
    than as a person the system is following. This hands out the lowest free
    number instead and takes it back when the track goes away, so a camera with
    five people on it shows #1 to #5 all day.
    """

    def __init__(self) -> None:
        self.assigned: dict[int, int] = {}
        self.free: set[int] = set()
        self.next = 1

    def label(self, track_id: int) -> int:
        if track_id in self.assigned:
            return self.assigned[track_id]
        if self.free:
            n = min(self.free)
            self.free.discard(n)
        else:
            n = self.next
            self.next += 1
        self.assigned[track_id] = n
        return n

    def retire(self, live: set[int]) -> None:
        for tid in [t for t in self.assigned if t not in live]:
            self.free.add(self.assigned.pop(tid))


def resolve_device(preference: str) -> str:
    """Turn `device: auto` into whatever this machine actually has.

    Falls back rather than raising: a demo laptop that has lost its GPU should
    run slowly, not refuse to start. Torch is imported here rather than at
    module scope because this runs in the inference process only.
    """
    import torch

    wanted = (preference or "auto").strip().lower()
    if wanted != "auto":
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _drain(q: mp.Queue, limit: int, timeout: float) -> list[dict]:
    out: list[dict] = []
    try:
        out.append(q.get(timeout=timeout))
    except queue.Empty:
        return out
    while len(out) < limit:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def _torso_signature(frame: np.ndarray, box) -> list[float] | None:
    """Mean hue/saturation/value of a person's torso.

    The crop is the middle of the upper half of the box: below the head, above
    the legs, inset from the sides. That is the part of a person that is
    reliably shirt rather than background, which is what makes it usable as a
    uniform signature.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None  # too small to sample without picking up the background
    cx1 = int(max(0, x1 + w * 0.25))
    cx2 = int(min(frame.shape[1], x2 - w * 0.25))
    cy1 = int(max(0, y1 + h * 0.15))
    cy2 = int(min(frame.shape[0], y1 + h * 0.50))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    hsv = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2HSV)
    return [float(v) for v in hsv.reshape(-1, 3).mean(axis=0)]


def _extract(results, cam_id: str, ts: float, frame: np.ndarray) -> list[dict]:
    """Pull boxes + track IDs out of an ultralytics result into plain dicts."""
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
    else:
        # Detected but not yet tracked (first frames after a reconnect).
        # Negative placeholders keep them distinct without colliding with real IDs.
        ids = np.array([-(i + 1) for i in range(len(xyxy))])

    return [
        {
            "cam_id": cam_id,
            "ts": ts,
            "track_id": int(tid),
            "bbox": [float(v) for v in box],
            "conf": float(conf),
            "appearance": _torso_signature(frame, box),
        }
        for box, conf, tid in zip(xyxy, confs, ids)
    ]


def _annotate(
    frame: np.ndarray, detections: list[dict], zones: dict, cam_name: str
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()

    for name, points in zones.items():
        if not points or len(points) < 2:
            continue
        colour = ZONE_COLOURS.get(name, DEFAULT_ZONE_COLOUR)
        pts = np.array(points, dtype=np.int32)
        if len(points) == 2:  # a counting line, not an area
            cv2.line(out, tuple(pts[0]), tuple(pts[1]), colour, 2)
            cv2.putText(
                out, name, tuple(pts[0] + np.array([4, -6])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA,
            )
            continue
        cv2.fillPoly(overlay, [pts], colour)
        cv2.polylines(out, [pts], True, colour, 2)
        cv2.putText(
            out, name, tuple(pts[0] + np.array([4, 16])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOUR, 2)
        label = f"#{det.get('label', det['track_id'])}"
        cv2.putText(
            out, label, (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOUR, 1, cv2.LINE_AA,
        )
        ax, ay = int((x1 + x2) / 2), y2
        cv2.circle(out, (ax, ay), 3, (0, 0, 255), -1)  # the anchor rules test

    banner = f"{cam_name}  {time.strftime('%H:%M:%S')}  {len(detections)} person(s)"
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(
        out, banner, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        cv2.LINE_AA,
    )
    return out
