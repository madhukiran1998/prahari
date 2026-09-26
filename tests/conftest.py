import pytest

from prahari.config import CameraCfg, load_settings

# Rooms used by the rule tests. Frame is 640x360.
COUNTER_CUSTOMER = [[120, 180], [520, 180], [520, 340], [120, 340]]
COUNTER_STAFF = [[120, 60], [520, 60], [520, 170], [120, 170]]
ENTRANCE = [[80, 200], [560, 200], [560, 350], [80, 350]]
ENTRY_LINE = [[100, 220], [540, 220]]


@pytest.fixture
def settings():
    """The real config/settings.yaml - so a missing threshold fails a test here
    rather than at 3pm in front of the client."""
    return load_settings()


@pytest.fixture
def counter_cam():
    return CameraCfg(
        id="cam3",
        name="Diamond counter",
        url="rtsp://test/cam3",
        rules=["zone_breach"],
        zones={"counter_customer": COUNTER_CUSTOMER, "counter_staff": COUNTER_STAFF},
    )


@pytest.fixture
def entrance_cam():
    return CameraCfg(
        id="cam1",
        name="Entrance",
        url="rtsp://test/cam1",
        rules=["loitering", "footfall"],
        zones={"entrance": ENTRANCE, "entry_line": ENTRY_LINE},
    )


def person(
    track_id: int,
    x: float,
    y: float,
    cam_id: str = "cam3",
    ts: float = 0.0,
    appearance=None,
    is_staff: bool = False,
):
    """A detection whose feet land exactly on (x, y)."""
    from prahari.types import Detection

    return Detection(
        cam_id=cam_id,
        ts=ts,
        track_id=track_id,
        bbox=(x - 20, y - 80, x + 20, y),
        conf=0.9,
        appearance=appearance,
        is_staff=is_staff,
    )


def frame(cam_id: str, ts: float, detections, **kw):
    from prahari.types import FrameCtx

    return FrameCtx(cam_id=cam_id, ts=ts, detections=detections, **kw)


# Kept clear of COUNTER_STAFF (x 120-520) so a single person cannot land
# in both zones and make a one-zone assertion look like a rule bug.
RESTRICTED = [[540, 60], [630, 60], [630, 200], [540, 200]]


@pytest.fixture
def restricted_cam():
    return CameraCfg(
        id="cam9",
        name="Strongroom",
        url="rtsp://test/cam9",
        rules=["restricted"],
        zones={"restricted": RESTRICTED, "staff_only": COUNTER_STAFF},
    )
