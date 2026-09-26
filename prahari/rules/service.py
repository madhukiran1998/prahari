"""Was the customer actually served?

The rules next door protect stock. This one protects revenue, and it is the
only thing here the owner has never been able to measure: a jewellery shop
takes thirty to eighty walk-ins a day, each worth a large ticket, and nobody
knows how many of them stood at a case and left without anyone speaking to
them. Every one of those is a lost sale with a clip attached.

Two things come out of it:

  time-to-greet   how long a customer stood at the counter before a staff
                  member was there with them. Recorded per customer, INFO, so
                  the daily average is a SQL query.
  walk-away       a customer who dwelt at the counter past `unserved_min_s`
                  and left with nobody having joined them. WARNING.

It needs `counter_customer` and `counter_staff` drawn, and it needs
StaffTracker to have decided who is who - a camera without both stays quiet.

Deliberately *not* an alarm. A walk-away is not an incident to respond to; it
is a number that should be smaller tomorrow. It stays WARNING, and the demo
narration should be "here is what this cost you", never "somebody did wrong".
"""

from __future__ import annotations

from ..config import CameraCfg, Settings
from ..geometry import point_in_polygon
from ..types import INFO, START, WARNING, Event, FrameCtx
from .base import Rule

CUSTOMER_ZONE = "counter_customer"
STAFF_ZONE = "counter_staff"

# A customer whose track vanishes for longer than this has left, rather than
# been briefly occluded by someone walking in front of them.
GONE_S = 3.0


class _Visit:
    """One customer's stay at the counter."""

    __slots__ = ("started", "last_seen", "greeted_at")

    def __init__(self, ts: float):
        self.started = ts
        self.last_seen = ts
        self.greeted_at: float | None = None

    @property
    def greeted(self) -> bool:
        return self.greeted_at is not None

    def wait_s(self, until: float) -> float:
        return (self.greeted_at or until) - self.started


class ServiceRule(Rule):
    name = "service"

    def __init__(self, cam: CameraCfg, settings: Settings):
        super().__init__(cam, settings)
        self.visits: dict[int, _Visit] = {}

    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        customer_zone = self.zone(CUSTOMER_ZONE)
        staff_zone = self.zone(STAFF_ZONE)
        if not customer_zone or not staff_zone:
            return []

        staff_here = any(
            det.is_staff and point_in_polygon(det.anchor, staff_zone)
            for det in ctx.detections
        )

        events: list[Event] = []
        for det in ctx.detections:
            if det.is_staff or not point_in_polygon(det.anchor, customer_zone):
                continue
            visit = self.visits.get(det.track_id)
            if visit is None:
                visit = self.visits[det.track_id] = _Visit(ctx.ts)
            visit.last_seen = ctx.ts
            if staff_here and not visit.greeted:
                visit.greeted_at = ctx.ts
                events.append(
                    self.event(
                        severity=INFO,
                        kind=START,
                        ts=ctx.ts,
                        track_id=det.track_id,
                        title="Customer served",
                        transient=True,
                        meta={"wait_s": round(visit.wait_s(ctx.ts), 1), "served": True},
                    )
                )
        return events

    def on_tick(self, ts: float) -> list[Event]:
        """Close out visits whose customer has left the frame."""
        events: list[Event] = []
        for track_id in [t for t, v in self.visits.items() if ts - v.last_seen > GONE_S]:
            visit = self.visits.pop(track_id)
            waited = visit.wait_s(visit.last_seen)
            if visit.greeted or waited < self.settings.threshold("unserved_min_s"):
                continue
            events.append(
                self.event(
                    severity=WARNING,
                    kind=START,
                    ts=ts,
                    track_id=track_id,
                    title=f"Customer left unattended after {int(waited)}s",
                    # A moment, not a condition - the customer has already gone,
                    # so there is nothing to resolve later. Being transient also
                    # exempts it from the cooldown, which matters: two customers
                    # leaving in the same minute must be two lines, not one.
                    transient=True,
                    meta={"wait_s": round(waited, 1), "served": False},
                )
            )
        return events

    def reset(self, ts: float) -> list[Event]:
        # A lost stream is not a walk-away; drop the visits without reporting.
        self.visits.clear()
        return []
