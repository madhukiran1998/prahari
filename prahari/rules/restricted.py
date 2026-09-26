"""Somebody is where nobody should be.

Two zones, both optional:

  restricted   nobody at all - the strongroom, the safe, the back office after
               the last staff member leaves. Any person is CRITICAL.
  staff_only   behind the counter, the till, the stockroom. A *customer* there
               is CRITICAL; staff are expected and stay silent.

`staff_only` needs StaffTracker to have learned a uniform first, and stays
silent until it has - otherwise the shop's own staff are accused for the first
few seconds after every restart, which is exactly the alert you do not want as
the opening line of a demo. Learning takes one staff member standing behind
their own counter for `staff_learn_s`, so it settles itself.

Of everything here this is the most reliable rule - a point in a polygon and
nothing else - which is why it is the one to demonstrate first.
"""

from __future__ import annotations

from ..config import CameraCfg, Settings
from ..geometry import point_in_polygon
from ..types import CRITICAL, RESOLVED, START, Event, FrameCtx
from .base import Rule

NOBODY_ZONE = "restricted"
STAFF_ONLY_ZONE = "staff_only"


class RestrictedRule(Rule):
    name = "restricted"

    def __init__(self, cam: CameraCfg, settings: Settings):
        super().__init__(cam, settings)
        self.active: set[str] = set()

    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        events: list[Event] = []
        for zone_name, title in (
            (NOBODY_ZONE, "Person in a restricted area"),
            (STAFF_ONLY_ZONE, "Customer in a staff-only area"),
        ):
            zone = self.zone(zone_name)
            if not zone:
                continue
            if zone_name == STAFF_ONLY_ZONE and not ctx.staff_known:
                # We do not yet know what a uniform looks like here, so every
                # staff member would read as a trespasser. Wait until someone
                # has stood behind the counter long enough to teach us.
                continue
            intruders = [
                det
                for det in ctx.detections
                if point_in_polygon(det.anchor, zone)
                # In a nobody-allowed zone, being staff is no excuse.
                and (zone_name == NOBODY_ZONE or not det.is_staff)
            ]
            events += self._transition(zone_name, title, bool(intruders), ctx, intruders)
        return events

    def _transition(
        self, zone_name: str, title: str, occupied: bool, ctx: FrameCtx, intruders: list
    ) -> list[Event]:
        was = zone_name in self.active
        if occupied and not was:
            self.active.add(zone_name)
            return [
                self.event(
                    severity=CRITICAL,
                    kind=START,
                    ts=ctx.ts,
                    track_id=intruders[0].track_id,
                    title=title,
                    dedup_key=f"restricted:{self.cam.id}:{zone_name}",
                    meta={"zone": zone_name, "people": len(intruders)},
                )
            ]
        if was and not occupied:
            self.active.discard(zone_name)
            return [
                self.event(
                    severity=CRITICAL,
                    kind=RESOLVED,
                    ts=ctx.ts,
                    title=f"{title} - clear",
                    dedup_key=f"restricted:{self.cam.id}:{zone_name}",
                    meta={"zone": zone_name},
                )
            ]
        return []

    def reset(self, ts: float) -> list[Event]:
        events = [
            self.event(
                severity=CRITICAL,
                kind=RESOLVED,
                ts=ts,
                title="Restricted area - stream lost",
                dedup_key=f"restricted:{self.cam.id}:{zone_name}",
                meta={"zone": zone_name},
            )
            for zone_name in sorted(self.active)
        ]
        self.active.clear()
        return events
