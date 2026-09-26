"""Counts people crossing an `entry_line`, for the footfall KPI.

Emits INFO events rather than keeping a private counter, so the count survives a
restart and "how many people came in on Tuesday?" is a SQL query. The dashboard
filters INFO out of the alert feed.
"""

from __future__ import annotations

from ..config import CameraCfg, Settings
from ..geometry import bbox_anchor, line_crossing
from ..types import INFO, START, Event, FrameCtx
from .base import Rule

ENTRY_LINE = "entry_line"


class FootfallRule(Rule):
    name = "footfall"

    def __init__(self, cam: CameraCfg, settings: Settings):
        super().__init__(cam, settings)
        self.last_anchor: dict[int, tuple[float, float]] = {}

    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        line = self.zone(ENTRY_LINE)
        if not line:
            return []

        events: list[Event] = []
        seen: set[int] = set()
        for det in ctx.detections:
            seen.add(det.track_id)
            anchor = bbox_anchor(det.bbox)
            prev = self.last_anchor.get(det.track_id)
            self.last_anchor[det.track_id] = anchor
            if prev is None:
                continue
            direction = line_crossing(prev, anchor, line)
            if direction == 0:
                continue
            events.append(
                self.event(
                    severity=INFO,
                    kind=START,
                    ts=ctx.ts,
                    track_id=det.track_id,
                    title="Entry" if direction > 0 else "Exit",
                    transient=True,
                    meta={"direction": "in" if direction > 0 else "out"},
                )
            )

        for tid in [t for t in self.last_anchor if t not in seen]:
            del self.last_anchor[tid]
        return events

    def reset(self, ts: float) -> list[Event]:
        self.last_anchor.clear()
        return []
