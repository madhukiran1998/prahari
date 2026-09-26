"""Staff classification: learned from position, applied by colour."""

from prahari.staff import StaffTracker, colour_distance
from conftest import frame, person

NAVY = (110.0, 180.0, 90.0)  # a uniform
BEIGE = (20.0, 40.0, 200.0)  # a customer's coat

IN_STAFF_ZONE = (300, 120)
IN_CUSTOMER_ZONE = (300, 300)


def _run(tracker, positions, appearance, track_id=1, start=0.0, step=1.0):
    """Walk one person through a series of positions, one second apart."""
    ts = start
    for x, y in positions:
        det = person(track_id, x, y, ts=ts, appearance=appearance)
        tracker.update(frame("cam3", ts, [det]), step)
        ts += step
    return det


def test_hue_distance_wraps_around_red(settings):
    # OpenCV hue is 0-180, so 179 and 1 are two apart, not 178.
    assert colour_distance((179, 100, 50), (1, 100, 50)) == 4.0
    assert colour_distance((10, 100, 50), (10, 140, 50)) == 40.0


def test_standing_behind_the_counter_makes_you_staff(settings, counter_cam):
    tracker = StaffTracker(counter_cam, settings)
    learn_s = settings.threshold("staff_learn_s")

    det = _run(tracker, [IN_STAFF_ZONE] * int(learn_s - 1), NAVY)
    assert det.is_staff is False, "promoted before serving the full dwell"

    det = _run(tracker, [IN_STAFF_ZONE] * 3, NAVY, start=learn_s)
    assert det.is_staff is True


def test_learned_uniform_is_recognised_in_front_of_the_counter(settings, counter_cam):
    """The whole point: a staff member who steps out from behind the counter
    must not become a customer, or every service timer restarts on them."""
    tracker = StaffTracker(counter_cam, settings)
    _run(tracker, [IN_STAFF_ZONE] * 8, NAVY, track_id=1)

    # A *new* track id, same uniform, standing on the customer side.
    det = _run(tracker, [IN_CUSTOMER_ZONE], NAVY, track_id=2, start=100.0)
    assert det.is_staff is True


def test_a_different_colour_stays_a_customer(settings, counter_cam):
    tracker = StaffTracker(counter_cam, settings)
    _run(tracker, [IN_STAFF_ZONE] * 8, NAVY, track_id=1)

    det = _run(tracker, [IN_CUSTOMER_ZONE], BEIGE, track_id=2, start=100.0)
    assert det.is_staff is False


def test_no_staff_zone_means_nobody_is_ever_staff(settings, entrance_cam):
    """A camera without counter_staff drawn must stay silent rather than guess."""
    tracker = StaffTracker(entrance_cam, settings)
    det = _run(tracker, [(300, 300)] * 20, NAVY)
    assert det.is_staff is False
    assert tracker.known == []


def test_uniforms_survive_a_stream_drop_but_track_ids_do_not(settings, counter_cam):
    tracker = StaffTracker(counter_cam, settings)
    _run(tracker, [IN_STAFF_ZONE] * 8, NAVY)
    assert tracker.known

    tracker.reset()
    assert tracker.tracks == {}
    assert tracker.known, "the shop did not change uniform because a cable fell out"


def test_known_uniforms_do_not_accumulate_duplicates(settings, counter_cam):
    tracker = StaffTracker(counter_cam, settings)
    for tid in range(1, 6):  # same person, re-acquired five times
        _run(tracker, [IN_STAFF_ZONE] * 8, NAVY, track_id=tid, start=tid * 100.0)
    assert len(tracker.known) == 1


def test_device_preference_is_honoured_and_auto_resolves():
    """`device: auto` must land on something torch will accept, and an explicit
    choice must survive - a laptop that has lost its GPU should run slowly
    rather than refuse to start."""
    from prahari.inference import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}
    assert resolve_device("") in {"cuda", "mps", "cpu"}
