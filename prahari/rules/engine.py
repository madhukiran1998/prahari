"""Dispatches frames to each camera's rules and decides which events survive.

Rules report edges. The engine adds the two things that keep a demo from turning
into an alert storm:

  dedup    - while a condition is still true, one alert. `resolved` closes it.
  cooldown - after a condition clears, the same rule on the same camera stays
             quiet for `cooldown_s` so one shopper does not produce ten alerts.

Suppressed events are dropped entirely rather than stored, except `resolved`
events for conditions that were never announced (they close nothing).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from ..config import CameraCfg, Settings
from ..staff import StaffTracker
from ..types import CamStatus, Event, FrameCtx, RESOLVED, START
from .after_hours import AfterHoursRule
from .base import Rule
from .camera_health import CameraHealthRule
from .footfall import FootfallRule
from .loitering import LoiteringRule
from .restricted import RestrictedRule
from .service import ServiceRule
from .zone_breach import ZoneBreachRule

log = logging.getLogger("prahari.rules")

RULES: dict[str, type[Rule]] = {
    ZoneBreachRule.name: ZoneBreachRule,
    LoiteringRule.name: LoiteringRule,
    AfterHoursRule.name: AfterHoursRule,
    CameraHealthRule.name: CameraHealthRule,
    FootfallRule.name: FootfallRule,
    RestrictedRule.name: RestrictedRule,
    ServiceRule.name: ServiceRule,
}


# Which rule reads which zone. Used to arm a rule when its zone is drawn.
ZONE_RULES: dict[str, tuple[str, ...]] = {
    "entrance": ("loitering",),
    "entry_line": ("footfall",),
    "counter_customer": ("zone_breach", "service"),
    "counter_staff": ("zone_breach", "service"),
    "restricted": ("restricted",),
    "staff_only": ("restricted",),
}


def build_rules(cam: CameraCfg, settings: Settings) -> list[Rule]:
    rules = []
    for name in cam.rules:
        cls = RULES.get(name)
        if cls is None:
            log.warning("%s | unknown rule '%s' in cameras.yaml - ignored", cam.id, name)
            continue
        rules.append(cls(cam, settings))
    return rules


class RulesEngine:
    def __init__(
        self,
        cams: Iterable[CameraCfg],
        settings: Settings,
        on_event: Callable[[Event], None] | None = None,
    ):
        self.settings = settings
        self.cams = {c.id: c for c in cams}
        self.rules = {c.id: build_rules(c, settings) for c in self.cams.values()}
        self.on_event = on_event
        self.active: dict[str, str] = {}  # dedup key -> event id
        self.last_cleared: dict[str, float] = {}  # (rule, cam) -> ts condition ended
        self.last_frame_ts: dict[str, float] = {}
        # Runs before the rules, so every rule sees detections already labelled
        # staff or customer rather than each working it out for itself.
        self.staff = {c.id: StaffTracker(c, settings) for c in self.cams.values()}

    # --- inputs ------------------------------------------------------------
    def handle_frame(self, ctx: FrameCtx) -> list[Event]:
        previous = self.last_frame_ts.get(ctx.cam_id)
        self.last_frame_ts[ctx.cam_id] = ctx.ts
        tracker = self.staff.get(ctx.cam_id)
        if tracker is not None:
            # Elapsed time is measured from the last frame rather than assumed
            # from detect_fps, so a camera running behind still accrues dwell
            # in real seconds.
            dt = ctx.ts - previous if previous is not None else 0.0
            tracker.update(ctx, max(0.0, min(dt, 5.0)))
            ctx.staff_known = bool(tracker.known)
        return self._collect(ctx.cam_id, lambda r: r.on_frame(ctx))

    def handle_status(self, status: CamStatus) -> list[Event]:
        events = self._collect(status.cam_id, lambda r: r.on_status(status))
        if not status.online:
            tracker = self.staff.get(status.cam_id)
            if tracker is not None:
                tracker.reset()
            # A dead stream freezes every other rule mid-condition; clear them so
            # nothing stays stuck on the dashboard until the camera returns.
            events += self._collect(
                status.cam_id,
                lambda r: r.reset(status.ts) if r.name != CameraHealthRule.name else [],
            )
        return events

    def handle_tick(self, ts: float) -> list[Event]:
        events: list[Event] = []
        for cam_id in self.rules:
            events += self._collect(cam_id, lambda r: r.on_tick(ts))
        return events

    # --- plumbing ----------------------------------------------------------
    def _collect(self, cam_id: str, call) -> list[Event]:
        out: list[Event] = []
        for rule in self.rules.get(cam_id, []):
            try:
                produced = call(rule) or []
            except Exception:
                log.exception("%s | rule %s raised", cam_id, rule.name)
                continue
            for ev in produced:
                if self._admit(ev):
                    out.append(ev)
                    if self.on_event:
                        self.on_event(ev)
        return out

    def _admit(self, ev: Event) -> bool:
        key = ev.key()
        gate = f"{ev.rule}:{ev.cam_id}"

        if ev.kind == RESOLVED:
            if key not in self.active:
                return False  # closing something we never announced
            del self.active[key]
            self.last_cleared[gate] = ev.ts
            return True

        if ev.kind == START:
            if ev.transient:
                return True  # a moment, not a condition - nothing to hold open
            if key in self.active:
                return False  # already announced, still true
            cooldown = self.settings.cooldown_for(ev.rule)
            since_cleared = ev.ts - self.last_cleared.get(gate, float("-inf"))
            if since_cleared < cooldown:
                log.debug(
                    "%s | %s suppressed, %.0fs into %.0fs cooldown",
                    ev.cam_id,
                    ev.rule,
                    since_cleared,
                    cooldown,
                )
                return False
            self.active[key] = ev.id
            return True

        return True  # UPDATE events pass straight through

    def arm(self, cam_id: str, rule_name: str) -> bool:
        """Turn a rule on for one camera at runtime. Returns True if it was off.

        Drawing a zone in the dashboard is an unambiguous request for the rule
        that reads it - nobody outlines a strongroom for fun. Without this the
        zone saves, the overlay appears, and nothing ever fires, which is
        indistinguishable from the system being broken.
        """
        cam = self.cams.get(cam_id)
        cls = RULES.get(rule_name)
        if cam is None or cls is None:
            return False
        if any(r.name == rule_name for r in self.rules.get(cam_id, [])):
            return False
        if rule_name not in cam.rules:
            cam.rules.append(rule_name)
        self.rules.setdefault(cam_id, []).append(cls(cam, self.settings))
        log.info("%s | armed rule '%s' (its zone was drawn)", cam_id, rule_name)
        return True

    # --- introspection for the API ----------------------------------------
    def active_keys(self) -> list[str]:
        return sorted(self.active)
