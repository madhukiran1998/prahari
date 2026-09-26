"""Time-to-greet and the walk-away: the two numbers the owner has never had."""

from conftest import frame, person
from prahari.rules.service import ServiceRule
from prahari.types import INFO, WARNING

CUSTOMER_SPOT = (300, 300)  # inside counter_customer
STAFF_SPOT = (300, 120)  # inside counter_staff
AWAY = (600, 350)  # outside both


def _feed(rule, ts, dets):
    return rule.on_frame(frame("cam3", ts, dets))


def test_greeting_records_how_long_the_customer_waited(settings, counter_cam):
    rule = ServiceRule(counter_cam, settings)

    events = _feed(rule, 0.0, [person(1, *CUSTOMER_SPOT)])
    assert events == [], "nothing to say while they are still waiting"

    for ts in (1.0, 2.0, 3.0):
        assert _feed(rule, ts, [person(1, *CUSTOMER_SPOT)]) == []

    events = _feed(
        rule,
        12.0,
        [person(1, *CUSTOMER_SPOT), person(2, *STAFF_SPOT, is_staff=True)],
    )
    assert len(events) == 1
    assert events[0].severity == INFO
    assert events[0].meta["served"] is True
    assert events[0].meta["wait_s"] == 12.0


def test_a_customer_is_greeted_once_not_every_frame(settings, counter_cam):
    rule = ServiceRule(counter_cam, settings)
    both = [person(1, *CUSTOMER_SPOT), person(2, *STAFF_SPOT, is_staff=True)]
    assert len(_feed(rule, 1.0, both)) == 1
    assert _feed(rule, 2.0, both) == []
    assert _feed(rule, 3.0, both) == []


def test_staff_behind_the_counter_are_not_counted_as_customers(settings, counter_cam):
    """A staff member standing on the customer side must not open a visit -
    otherwise every quiet hour produces walk-aways for your own employees."""
    rule = ServiceRule(counter_cam, settings)
    _feed(rule, 0.0, [person(1, *CUSTOMER_SPOT, is_staff=True)])
    assert rule.visits == {}


def test_leaving_unserved_is_reported_with_the_wait(settings, counter_cam):
    rule = ServiceRule(counter_cam, settings)
    wait = settings.threshold("unserved_min_s")

    for ts in range(0, int(wait) + 5):
        _feed(rule, float(ts), [person(1, *CUSTOMER_SPOT)])

    last_seen = float(int(wait) + 4)
    assert rule.on_tick(last_seen + 1.0) == [], "still only briefly out of sight"

    events = rule.on_tick(last_seen + 5.0)
    assert len(events) == 1
    assert events[0].severity == WARNING
    assert events[0].meta["served"] is False
    assert events[0].meta["wait_s"] == last_seen
    assert "unattended" in events[0].title


def test_a_brief_visit_is_not_a_lost_sale(settings, counter_cam):
    """Someone glancing at a case on their way past is not a walk-away."""
    rule = ServiceRule(counter_cam, settings)
    _feed(rule, 0.0, [person(1, *CUSTOMER_SPOT)])
    _feed(rule, 2.0, [person(1, *CUSTOMER_SPOT)])
    assert rule.on_tick(20.0) == []


def test_a_served_customer_never_becomes_a_walk_away(settings, counter_cam):
    rule = ServiceRule(counter_cam, settings)
    wait = settings.threshold("unserved_min_s")
    _feed(rule, 0.0, [person(1, *CUSTOMER_SPOT)])
    _feed(rule, 2.0, [person(1, *CUSTOMER_SPOT), person(2, *STAFF_SPOT, is_staff=True)])
    for ts in range(3, int(wait) + 5):
        _feed(rule, float(ts), [person(1, *CUSTOMER_SPOT)])
    assert rule.on_tick(1000.0) == []


def test_two_customers_leaving_together_are_two_lost_sales(settings, counter_cam):
    rule = ServiceRule(counter_cam, settings)
    wait = int(settings.threshold("unserved_min_s"))
    for ts in range(0, wait + 5):
        _feed(rule, float(ts), [person(1, 280, 300), person(2, 340, 300)])
    events = rule.on_tick(float(wait + 20))
    assert len(events) == 2
    assert {e.track_id for e in events} == {1, 2}


def test_a_lost_stream_is_not_a_walk_away(settings, counter_cam):
    """The customer did not leave - the camera did."""
    rule = ServiceRule(counter_cam, settings)
    wait = int(settings.threshold("unserved_min_s"))
    for ts in range(0, wait + 5):
        _feed(rule, float(ts), [person(1, *CUSTOMER_SPOT)])
    assert rule.reset(float(wait + 6)) == []
    assert rule.on_tick(float(wait + 100)) == []


def test_a_camera_without_both_zones_stays_quiet(settings, entrance_cam):
    rule = ServiceRule(entrance_cam, settings)
    assert rule.on_frame(frame("cam1", 0.0, [person(1, 300, 300, cam_id="cam1")])) == []
