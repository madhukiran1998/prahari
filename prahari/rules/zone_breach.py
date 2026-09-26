"""Unattended-counter rule - the demo's headline alert.

Fires when somebody is standing at a display counter and no staff member is
behind it for `zone_breach_grace_s` seconds.

Honest framing for the client: this is NOT case-open detection. Detecting an
opened display case needs either a reed/vibration sensor on the case or a model
trained on that specific gesture. Staff-absence at an occupied counter is the
achievable proxy, and it catches the situation that actually matters - a
customer alone with the stock.
"""

from __future__ import annotations

from ..config import CameraCfg, Settings
from ..geometry import bbox_in_polygon
from ..types import CRITICAL, RESOLVED, START, Event, FrameCtx
from .base import Rule

CUSTOMER_ZONE = "counter_customer"
STAFF_ZONE = "counter_staff"


class ZoneBreachRule(Rule):
    name = "zone_breach"

    def __init__(self, cam: CameraCfg, settings: Settings):
        super().__init__(cam, settings)
        self.grace_s = settings.threshold("zone_breach_grace_s")
        self.since: float | None = None  # when the condition started holding
        self.firing = False

    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        customer_zone = self.zone(CUSTOMER_ZONE)
        staff_zone = self.zone(STAFF_ZONE)
        if not customer_zone or not staff_zone:
            return []  # camera has no counter drawn on it

        customers = [d for d in ctx.detections if bbox_in_polygon(d.bbox, customer_zone)]
        staff = [d for d in ctx.detections if bbox_in_polygon(d.bbox, staff_zone)]

        if not customers or staff:
            return self._clear(ctx.ts)

        if self.since is None:
            self.since = ctx.ts
        held = ctx.ts - self.since
        if held < self.grace_s or self.firing:
            return []

        self.firing = True
        return [
            self.event(
                severity=CRITICAL,
                kind=START,
                ts=ctx.ts,
                title=f"Customer at open counter, no staff present ({held:.0f}s)",
                track_id=customers[0].track_id,
                meta={
                    "customers": len(customers),
                    "held_s": round(held, 1),
                    "zone": CUSTOMER_ZONE,
                },
            )
        ]

    def _clear(self, ts: float) -> list[Event]:
        self.since = None
        if not self.firing:
            return []
        self.firing = False
        return [
            self.event(
                severity=CRITICAL,
                kind=RESOLVED,
                ts=ts,
                title="Counter attended again",
                meta={},
            )
        ]

    def reset(self, ts: float) -> list[Event]:
        return self._clear(ts)
