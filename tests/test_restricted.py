"""Restricted areas: the most reliable rule here, and the one to demo first."""

from conftest import frame, person
from prahari.rules.restricted import RestrictedRule
from prahari.types import CRITICAL, RESOLVED, START

INSIDE_RESTRICTED = (585, 150)
INSIDE_STAFF_ONLY = (300, 120)
OUTSIDE = (100, 340)


def _feed(rule, ts, dets, staff_known=True):
    """staff_known defaults on: the interesting cases are after the system has
    learned a uniform. test_before_a_uniform_is_learned covers the other side."""
    return rule.on_frame(frame("cam9", ts, dets, staff_known=staff_known))


def test_anyone_in_a_nobody_zone_is_critical(settings, restricted_cam):
    rule = RestrictedRule(restricted_cam, settings)
    events = _feed(rule, 1.0, [person(1, *INSIDE_RESTRICTED, cam_id="cam9")])
    assert len(events) == 1
    assert events[0].severity == CRITICAL
    assert events[0].kind == START
    assert events[0].meta["zone"] == "restricted"


def test_being_staff_is_no_excuse_in_a_nobody_zone(settings, restricted_cam):
    """The strongroom does not care who you are."""
    rule = RestrictedRule(restricted_cam, settings)
    events = _feed(
        rule, 1.0, [person(1, *INSIDE_RESTRICTED, cam_id="cam9", is_staff=True)]
    )
    assert len(events) == 1
    assert events[0].severity == CRITICAL


def test_staff_behind_their_own_counter_are_silent(settings, restricted_cam):
    rule = RestrictedRule(restricted_cam, settings)
    events = _feed(
        rule, 1.0, [person(1, *INSIDE_STAFF_ONLY, cam_id="cam9", is_staff=True)]
    )
    assert events == []


def test_a_customer_behind_the_counter_is_critical(settings, restricted_cam):
    rule = RestrictedRule(restricted_cam, settings)
    events = _feed(rule, 1.0, [person(1, *INSIDE_STAFF_ONLY, cam_id="cam9")])
    assert len(events) == 1
    assert events[0].meta["zone"] == "staff_only"


def test_one_alert_per_incident_then_a_resolve(settings, restricted_cam):
    rule = RestrictedRule(restricted_cam, settings)
    intruder = [person(1, *INSIDE_RESTRICTED, cam_id="cam9")]

    assert len(_feed(rule, 1.0, intruder)) == 1
    assert _feed(rule, 2.0, intruder) == [], "still there, already said so"
    assert _feed(rule, 3.0, intruder) == []

    events = _feed(rule, 4.0, [])
    assert len(events) == 1
    assert events[0].kind == RESOLVED


def test_before_a_uniform_is_learned_the_staff_area_stays_quiet(settings, restricted_cam):
    """The opening seconds of a demo must not accuse the shop's own staff."""
    rule = RestrictedRule(restricted_cam, settings)
    events = _feed(
        rule, 1.0, [person(1, *INSIDE_STAFF_ONLY, cam_id="cam9")], staff_known=False
    )
    assert events == []


def test_a_nobody_zone_does_not_wait_to_learn_anything(settings, restricted_cam):
    """`restricted` is pure geometry, so it works from the first frame."""
    rule = RestrictedRule(restricted_cam, settings)
    events = _feed(
        rule, 1.0, [person(1, *INSIDE_RESTRICTED, cam_id="cam9")], staff_known=False
    )
    assert len(events) == 1


def test_walking_past_outside_the_zone_says_nothing(settings, restricted_cam):
    rule = RestrictedRule(restricted_cam, settings)
    assert _feed(rule, 1.0, [person(1, *OUTSIDE, cam_id="cam9")]) == []


def test_both_zones_fire_independently(settings, restricted_cam):
    rule = RestrictedRule(restricted_cam, settings)
    events = _feed(
        rule,
        1.0,
        [
            person(1, *INSIDE_RESTRICTED, cam_id="cam9"),
            person(2, *INSIDE_STAFF_ONLY, cam_id="cam9"),
        ],
    )
    assert {e.meta["zone"] for e in events} == {"restricted", "staff_only"}


def test_a_dead_camera_does_not_leave_a_critical_stuck_on_screen(
    settings, restricted_cam
):
    rule = RestrictedRule(restricted_cam, settings)
    _feed(rule, 1.0, [person(1, *INSIDE_RESTRICTED, cam_id="cam9")])
    events = rule.reset(5.0)
    assert len(events) == 1
    assert events[0].kind == RESOLVED
    assert rule.active == set()


def test_a_camera_without_the_zones_stays_quiet(settings, counter_cam):
    rule = RestrictedRule(counter_cam, settings)
    assert _feed(rule, 1.0, [person(1, 300, 300)]) == []
