"""Rules are fed synthetic detection sequences and must produce exact events."""

import dataclasses

import pytest

from conftest import frame, person
from prahari.rules.after_hours import AfterHoursRule
from prahari.rules.camera_health import CameraHealthRule
from prahari.rules.engine import RulesEngine
from prahari.rules.footfall import FootfallRule
from prahari.rules.loitering import LoiteringRule
from prahari.rules.zone_breach import ZoneBreachRule
from prahari.types import CRITICAL, INFO, RESOLVED, START, WARNING, CamStatus

CUSTOMER = (320, 300)  # inside counter_customer
STAFF = (320, 120)  # inside counter_staff
OUTSIDE = (600, 350)  # outside both


def feed(rule, cam_id, points_by_ts):
    """Run a rule over {ts: [(track_id, (x, y)), ...]} and collect its events."""
    out = []
    for ts in sorted(points_by_ts):
        dets = [person(tid, x, y, cam_id=cam_id, ts=ts) for tid, (x, y) in points_by_ts[ts]]
        out += rule.on_frame(frame(cam_id, ts, dets))
    return out


# --- zone_breach ---------------------------------------------------------
def test_zone_breach_fires_after_grace(settings, counter_cam):
    rule = ZoneBreachRule(counter_cam, settings)
    grace = settings.threshold("zone_breach_grace_s")

    before = feed(rule, "cam3", {t: [(1, CUSTOMER)] for t in range(0, int(grace))})
    assert before == []

    late = rule.on_frame(frame("cam3", grace + 1, [person(1, *CUSTOMER)]))
    assert len(late) == 1
    assert late[0].severity == CRITICAL and late[0].kind == START


def test_zone_breach_silent_while_staff_present(settings, counter_cam):
    rule = ZoneBreachRule(counter_cam, settings)
    grace = settings.threshold("zone_breach_grace_s")
    events = feed(
        rule,
        "cam3",
        {t: [(1, CUSTOMER), (2, STAFF)] for t in range(0, int(grace) + 10)},
    )
    assert events == []


def test_zone_breach_grace_restarts_when_staff_returns(settings, counter_cam):
    rule = ZoneBreachRule(counter_cam, settings)
    grace = settings.threshold("zone_breach_grace_s")
    # Customer alone for most of the grace window, then staff step in.
    feed(rule, "cam3", {t: [(1, CUSTOMER)] for t in range(0, int(grace) - 2)})
    rule.on_frame(frame("cam3", grace, [person(1, *CUSTOMER), person(2, *STAFF)]))
    # Staff leave again; the clock must start from scratch.
    assert rule.on_frame(frame("cam3", grace + 1, [person(1, *CUSTOMER)])) == []
    fired = rule.on_frame(frame("cam3", grace + 1 + grace, [person(1, *CUSTOMER)]))
    assert len(fired) == 1


def test_zone_breach_resolves_when_staff_return(settings, counter_cam):
    rule = ZoneBreachRule(counter_cam, settings)
    grace = settings.threshold("zone_breach_grace_s")
    feed(rule, "cam3", {t: [(1, CUSTOMER)] for t in range(0, int(grace) + 2)})
    assert rule.firing
    resolved = rule.on_frame(
        frame("cam3", grace + 3, [person(1, *CUSTOMER), person(2, *STAFF)])
    )
    assert [e.kind for e in resolved] == [RESOLVED]


def test_zone_breach_needs_both_zones(settings, counter_cam):
    bare = dataclasses.replace(counter_cam, zones={})
    rule = ZoneBreachRule(bare, settings)
    assert feed(rule, "cam3", {t: [(1, CUSTOMER)] for t in range(0, 60)}) == []


# --- loitering -----------------------------------------------------------
def test_loitering_warns_then_escalates(settings, entrance_cam):
    rule = LoiteringRule(entrance_cam, settings)
    warn = settings.threshold("loiter_warn_s")
    crit = settings.threshold("loiter_crit_s")
    spot = (300, 300)

    events = feed(
        rule, "cam1", {t / 2: [(1, spot)] for t in range(0, int(crit * 2) + 4)}
    )
    sev = [e.severity for e in events]
    assert sev == [WARNING, CRITICAL], f"expected one of each, got {sev}"
    assert events[0].meta["dwell_s"] >= warn


def test_loitering_ignores_passers_by(settings, entrance_cam):
    rule = LoiteringRule(entrance_cam, settings)
    # In the zone for only a few seconds, then gone.
    events = feed(rule, "cam1", {t: [(1, (300, 300))] for t in range(0, 5)})
    assert events == []


def test_loitering_survives_track_id_churn(settings, entrance_cam):
    """The same person re-identified as a new track must keep their dwell."""
    rule = LoiteringRule(entrance_cam, settings)
    warn = settings.threshold("loiter_warn_s")
    spot = (300, 300)
    half = int(warn) // 2

    feed(rule, "cam1", {t: [(1, spot)] for t in range(0, half)})
    rule.on_frame(frame("cam1", half, []))  # occlusion: track lost
    events = feed(
        rule, "cam1", {float(t): [(2, spot)] for t in range(half + 1, int(warn) + 3)}
    )
    assert [e.severity for e in events] == [WARNING]


def test_loitering_does_not_merge_a_different_person(settings, entrance_cam):
    rule = LoiteringRule(entrance_cam, settings)
    warn = settings.threshold("loiter_warn_s")
    half = int(warn) // 2

    feed(rule, "cam1", {t: [(1, (200, 300))] for t in range(0, half)})
    rule.on_frame(frame("cam1", half, []))
    # New track, far away - must start from zero and stay quiet.
    events = feed(
        rule, "cam1", {float(t): [(2, (500, 250))] for t in range(half + 1, int(warn) + 2)}
    )
    assert events == []


def test_loitering_gap_does_not_inflate_dwell(settings, entrance_cam):
    """A long stream stall must not be counted as time spent standing there."""
    rule = LoiteringRule(entrance_cam, settings)
    crit = settings.threshold("loiter_crit_s")
    spot = (300, 300)
    rule.on_frame(frame("cam1", 0, [person(1, *spot)]))
    events = rule.on_frame(frame("cam1", crit * 3, [person(1, *spot)]))
    assert events == []


def test_loitering_resolves_on_departure(settings, entrance_cam):
    rule = LoiteringRule(entrance_cam, settings)
    warn = settings.threshold("loiter_warn_s")
    feed(rule, "cam1", {t / 2: [(1, (300, 300))] for t in range(0, int(warn * 2) + 4)})
    left = rule.on_frame(frame("cam1", warn + 10, []))
    assert any(e.kind == RESOLVED for e in left)


# --- after_hours ---------------------------------------------------------
def test_after_hours_fires_only_when_closed(settings, entrance_cam):
    open_settings = dataclasses.replace(
        settings, demo_force_closed=False, store_open="00:00", store_close="23:59"
    )
    rule = AfterHoursRule(entrance_cam, open_settings)
    assert rule.on_frame(frame("cam1", 1000, [person(1, 300, 300)])) == []

    closed = dataclasses.replace(settings, demo_force_closed=True)
    rule = AfterHoursRule(entrance_cam, closed)
    events = rule.on_frame(frame("cam1", 1000, [person(1, 300, 300)]))
    assert len(events) == 1 and events[0].severity == CRITICAL


def test_after_hours_one_alert_per_incident(settings, entrance_cam):
    closed = dataclasses.replace(settings, demo_force_closed=True)
    rule = AfterHoursRule(entrance_cam, closed)
    events = feed(rule, "cam1", {t: [(1, (300, 300))] for t in range(0, 30)})
    assert len(events) == 1


def test_after_hours_resolves_after_quiet_period(settings, entrance_cam):
    closed = dataclasses.replace(settings, demo_force_closed=True)
    rule = AfterHoursRule(entrance_cam, closed)
    rule.on_frame(frame("cam1", 0, [person(1, 300, 300)]))
    clear = settings.threshold("after_hours_clear_s")
    assert rule.on_tick(clear / 2) == []
    assert [e.kind for e in rule.on_tick(clear + 1)] == [RESOLVED]


# --- camera_health -------------------------------------------------------
def test_camera_offline_after_threshold(settings, entrance_cam):
    rule = CameraHealthRule(entrance_cam, settings)
    down = settings.threshold("camera_down_s")
    rule.on_status(CamStatus(cam_id="cam1", online=False, ts=0.0))
    assert rule.on_tick(down / 2) == []
    events = rule.on_tick(down + 1)
    assert len(events) == 1 and events[0].severity == WARNING
    assert events[0].meta["fault"] == "offline"


def test_camera_offline_resolves_on_recovery(settings, entrance_cam):
    rule = CameraHealthRule(entrance_cam, settings)
    down = settings.threshold("camera_down_s")
    rule.on_status(CamStatus(cam_id="cam1", online=False, ts=0.0))
    rule.on_tick(down + 1)
    back = rule.on_status(CamStatus(cam_id="cam1", online=True, ts=down + 2))
    assert [e.kind for e in back] == [RESOLVED]


def test_frozen_feed_detected(settings, entrance_cam):
    rule = CameraHealthRule(entrance_cam, settings)
    frozen = settings.threshold("camera_frozen_s")
    events = []
    for t in range(0, int(frozen) + 3):
        events += rule.on_frame(frame("cam1", float(t), [], phash="same"))
    faults = [e.meta["fault"] for e in events if e.kind == START]
    assert faults == ["frozen"]


def test_moving_feed_is_not_frozen(settings, entrance_cam):
    rule = CameraHealthRule(entrance_cam, settings)
    frozen = settings.threshold("camera_frozen_s")
    events = []
    for t in range(0, int(frozen) + 3):
        events += rule.on_frame(frame("cam1", float(t), [], phash=f"h{t}"))
    assert events == []


def test_blackout_only_while_open(settings, entrance_cam):
    dark = settings.threshold("camera_dark_s")
    closed = dataclasses.replace(settings, demo_force_closed=True)
    rule = CameraHealthRule(entrance_cam, closed)
    events = []
    for t in range(0, int(dark) + 3):
        events += rule.on_frame(frame("cam1", float(t), [], luma=1.0, phash=f"h{t}"))
    assert events == []  # dark at night is just night

    open_settings = dataclasses.replace(
        settings, demo_force_closed=False, store_open="00:00", store_close="23:59"
    )
    rule = CameraHealthRule(entrance_cam, open_settings)
    events = []
    for t in range(0, int(dark) + 3):
        events += rule.on_frame(frame("cam1", float(t), [], luma=1.0, phash=f"h{t}"))
    assert [e.meta["fault"] for e in events if e.kind == START] == ["blackout"]


# --- footfall ------------------------------------------------------------
def test_footfall_counts_crossings_with_direction(settings, entrance_cam):
    rule = FootfallRule(entrance_cam, settings)
    # entry_line runs across y=220; walking from y=300 up to y=150 crosses it.
    events = feed(rule, "cam1", {0: [(1, (300, 300))], 1: [(1, (300, 150))]})
    assert len(events) == 1
    assert events[0].severity == INFO
    assert events[0].meta["direction"] in {"in", "out"}
    assert events[0].transient


def test_footfall_ignores_movement_along_the_line(settings, entrance_cam):
    rule = FootfallRule(entrance_cam, settings)
    events = feed(rule, "cam1", {0: [(1, (150, 300))], 1: [(1, (500, 300))]})
    assert events == []


# --- engine: dedup + cooldown -------------------------------------------
def test_engine_dedups_ongoing_condition(settings, counter_cam):
    engine = RulesEngine([counter_cam], settings)
    grace = settings.threshold("zone_breach_grace_s")
    fired = []
    for t in range(0, int(grace) + 20):
        fired += engine.handle_frame(frame("cam3", float(t), [person(1, *CUSTOMER)]))
    starts = [e for e in fired if e.kind == START]
    assert len(starts) == 1, "one continuous incident must be one alert"


def test_engine_cooldown_blocks_immediate_repeat(settings, counter_cam):
    engine = RulesEngine([counter_cam], settings)
    grace = settings.threshold("zone_breach_grace_s")
    cooldown = settings.cooldown_for("zone_breach")

    def run_incident(t0):
        out = []
        for t in range(int(t0), int(t0 + grace) + 2):
            out += engine.handle_frame(frame("cam3", float(t), [person(1, *CUSTOMER)]))
        # staff return -> condition clears
        out += engine.handle_frame(
            frame("cam3", t0 + grace + 3, [person(1, *CUSTOMER), person(2, *STAFF)])
        )
        return out

    first = run_incident(0)
    assert len([e for e in first if e.kind == START]) == 1

    second = run_incident(grace + 5)  # well inside the cooldown
    assert [e for e in second if e.kind == START] == []

    third = run_incident(grace + 5 + cooldown + 10)
    assert len([e for e in third if e.kind == START]) == 1


def test_engine_never_stores_orphan_resolves(settings, counter_cam):
    engine = RulesEngine([counter_cam], settings)
    # Staff present the whole time: nothing ever fires, so nothing may resolve.
    out = []
    for t in range(0, 40):
        out += engine.handle_frame(
            frame("cam3", float(t), [person(1, *CUSTOMER), person(2, *STAFF)])
        )
    assert out == []


def test_engine_clears_rules_when_stream_dies(settings, counter_cam):
    engine = RulesEngine([counter_cam], settings)
    grace = settings.threshold("zone_breach_grace_s")
    for t in range(0, int(grace) + 2):
        engine.handle_frame(frame("cam3", float(t), [person(1, *CUSTOMER)]))
    assert engine.active_keys()

    engine.handle_status(CamStatus(cam_id="cam3", online=False, ts=grace + 5))
    assert engine.active_keys() == [], "a dead camera must not leave a stuck CRITICAL"


def test_engine_survives_a_broken_rule(settings, counter_cam):
    engine = RulesEngine([counter_cam], settings)

    class Exploding:
        name = "boom"

        def on_frame(self, ctx):
            raise RuntimeError("bad rule")

    engine.rules["cam3"].insert(0, Exploding())
    grace = settings.threshold("zone_breach_grace_s")
    out = []
    for t in range(0, int(grace) + 2):
        out += engine.handle_frame(frame("cam3", float(t), [person(1, *CUSTOMER)]))
    assert len([e for e in out if e.kind == START]) == 1


def test_engine_skips_unknown_rule_names(settings, counter_cam):
    cam = dataclasses.replace(counter_cam, rules=["zone_breach", "nope"])
    engine = RulesEngine([cam], settings)
    assert [r.name for r in engine.rules["cam3"]] == ["zone_breach"]


def test_drawing_a_zone_arms_the_rule_that_reads_it(settings, entrance_cam):
    """A zone drawn in the dashboard must do something. Saving one and finding
    nothing ever fires is indistinguishable from the system being broken."""
    from prahari.rules.engine import RulesEngine

    cam = dataclasses.replace(entrance_cam, rules=["loitering"], zones=dict(entrance_cam.zones))
    engine = RulesEngine([cam], settings)
    assert [r.name for r in engine.rules[cam.id]] == ["loitering"]

    assert engine.arm(cam.id, "footfall") is True
    assert "footfall" in [r.name for r in engine.rules[cam.id]]
    assert "footfall" in cam.rules

    # Idempotent: arming twice must not stack two copies of the rule, or every
    # line crossing would be counted twice.
    assert engine.arm(cam.id, "footfall") is False
    assert [r.name for r in engine.rules[cam.id]].count("footfall") == 1


def test_arming_an_unknown_rule_is_ignored(settings, entrance_cam):
    from prahari.rules.engine import RulesEngine

    engine = RulesEngine([entrance_cam], settings)
    assert engine.arm(entrance_cam.id, "not_a_rule") is False
    assert engine.arm("no_such_cam", "footfall") is False
