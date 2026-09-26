"""Anyone on camera while the store is shut.

Deliberately blunt: no zones, no dwell timer. Outside opening hours any person
at all is worth waking the owner for. `demo_force_closed: true` in
settings.yaml makes it fire during a daytime meeting.
"""

from __future__ import annotations

from ..config import CameraCfg, Settings
from ..types import CRITICAL, RESOLVED, START, Event, FrameCtx
from .base import Rule


class AfterHoursRule(Rule):
    name = "after_hours"

    def __init__(self, cam: CameraCfg, settings: Settings):
        super().__init__(cam, settings)
        self.clear_s = settings.threshold("after_hours_clear_s")
        self.firing = False
        self.last_seen = 0.0

    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        if not self.settings.is_closed(ctx.ts):
            return self._clear(ctx.ts)
        if not ctx.detections:
            return []

        self.last_seen = ctx.ts
        if self.firing:
            return []
        self.firing = True
        return [
            self.event(
                severity=CRITICAL,
                kind=START,
                ts=ctx.ts,
                track_id=ctx.detections[0].track_id,
                title=f"Movement detected after hours ({len(ctx.detections)} person(s))",
                meta={"people": len(ctx.detections)},
            )
        ]

    def on_tick(self, ts: float) -> list[Event]:
        if self.firing and ts - self.last_seen > self.clear_s:
            return self._clear(ts)
        return []

    def _clear(self, ts: float) -> list[Event]:
        if not self.firing:
            return []
        self.firing = False
        return [
            self.event(
                severity=CRITICAL,
                kind=RESOLVED,
                ts=ts,
                title="After-hours movement stopped",
                meta={},
            )
        ]

    def reset(self, ts: float) -> list[Event]:
        return self._clear(ts)
