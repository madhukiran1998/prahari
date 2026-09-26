"""Data that moves between processes and into the rules.

Everything here is a plain dataclass so rules can be unit-tested by handing them
synthetic values - no camera, no model, no I/O.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"

# Event lifecycle
START = "start"
UPDATE = "update"
RESOLVED = "resolved"


@dataclass
class Detection:
    """One tracked person in one frame."""

    cam_id: str
    ts: float
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    conf: float
    # Mean hue/saturation/value of the torso crop, from the detector. Used to
    # recognise a uniform; None when the crop was too small to sample.
    appearance: tuple[float, float, float] | None = None
    # Filled in by StaffTracker before the rules run, never by the detector.
    is_staff: bool = False

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box - where the person's feet are."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class FrameCtx:
    """Everything the rules know about one processed frame."""

    cam_id: str
    ts: float
    detections: list[Detection]
    width: int = 640
    height: int = 360
    luma: float = 128.0  # mean brightness, for blackout detection
    phash: str = ""  # cheap frame fingerprint, for frozen-feed detection
    # Has StaffTracker learned at least one uniform on this camera yet? Rules
    # that punish being a customer must stay quiet until it has, or they accuse
    # the shop's own staff for the first few seconds after every restart.
    staff_known: bool = False


@dataclass
class Event:
    """Something a rule wants to say. Persisted, broadcast, maybe notified."""

    cam_id: str
    rule: str
    severity: str
    title: str
    kind: str = START
    ts: float = field(default_factory=time.time)
    track_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Identifies the ongoing condition, so a single incident is one alert.
    # Defaults to (rule, cam) - rules that can have several concurrent
    # incidents (loitering, one per person) set it explicitly.
    dedup_key: str | None = None
    # A moment, not a condition: it will never be resolved, so the engine must
    # not hold it open in the dedup table. Line crossings are the example.
    transient: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def key(self) -> str:
        return self.dedup_key or f"{self.rule}:{self.cam_id}"


@dataclass
class CamStatus:
    """Ingest worker reporting on its stream."""

    cam_id: str
    online: bool
    ts: float
    detail: str = ""
