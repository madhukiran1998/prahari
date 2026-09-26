"""Rule interface.

A rule is a small state machine over one camera. It takes plain dataclasses in
and returns Events out - no database, no network, no frames. That is what makes
tests/test_rules.py able to drive a whole incident in a few lines.

Rules emit edges, not levels: one `start` when a condition begins, one
`resolved` when it ends. The engine handles cooldowns and persistence.
"""

from __future__ import annotations

from ..config import CameraCfg, Settings
from ..types import CamStatus, Event, FrameCtx


class Rule:
    name = "rule"

    def __init__(self, cam: CameraCfg, settings: Settings):
        self.cam = cam
        self.settings = settings

    # --- hooks; override what you need ------------------------------------
    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        """A frame was processed. `ctx.detections` are the people in it."""
        return []

    def on_tick(self, ts: float) -> list[Event]:
        """Called roughly every second, whether or not frames are arriving."""
        return []

    def on_status(self, status: CamStatus) -> list[Event]:
        """The ingest worker reported the stream going up or down."""
        return []

    def reset(self, ts: float) -> list[Event]:
        """The stream was lost. Clear state and resolve anything still firing,
        so a dead camera cannot leave a CRITICAL stuck on the dashboard."""
        return []

    # --- helpers ----------------------------------------------------------
    def zone(self, name: str) -> list[list[float]]:
        return self.cam.zones.get(name, [])

    def event(self, **kw) -> Event:
        kw.setdefault("cam_id", self.cam.id)
        kw.setdefault("rule", self.name)
        return Event(**kw)
