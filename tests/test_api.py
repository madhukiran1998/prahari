"""The HTTP surface the dashboard depends on, exercised in-process."""

import asyncio
import queue

import pytest
from fastapi.testclient import TestClient

from prahari import config
from prahari.api import Hub, create_app, mjpeg_frames
from prahari.config import CameraCfg, load_settings
from prahari.notify import Notifier
from prahari.store import Store
from prahari.types import CRITICAL, INFO, WARNING, Event


@pytest.fixture(autouse=True)
def isolate_zone_overrides(tmp_path, monkeypatch):
    """Saving zones writes to a real file next to cameras.yaml. Without this the
    zone tests would overwrite the shipped zones and quietly break the demo."""
    monkeypatch.setattr(config, "ZONES_OVERRIDE", tmp_path / "zones.local.yaml")


@pytest.fixture
def hub(tmp_path):
    settings = load_settings()
    settings.data_dir = tmp_path
    (tmp_path / "clips").mkdir(exist_ok=True)
    (tmp_path / "snapshots").mkdir(exist_ok=True)
    cams = [
        CameraCfg(id="cam3", name="Display counter", url="rtsp://x/cam3",
                  rules=["zone_breach"],
                  zones={"counter_customer": [[1, 1], [2, 1], [2, 2]]}),
        CameraCfg(id="cam1", name="Entrance", url="rtsp://x/cam1", rules=["footfall"]),
    ]
    return Hub(
        settings, cams, Store(tmp_path / "t.db"), Notifier({"enabled": False}),
        {c.id: queue.Queue() for c in cams}, queue.Queue(),
    )


@pytest.fixture
def client(hub):
    with TestClient(create_app(hub)) as c:
        yield c


def add(hub, **kw):
    kw.setdefault("cam_id", "cam3")
    kw.setdefault("rule", "zone_breach")
    kw.setdefault("severity", CRITICAL)
    kw.setdefault("title", "Counter unattended")
    ev = Event(**kw)
    hub.store.add_event(ev, cam_name="Display counter")
    return ev


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Prahari" in r.text


def test_cams_listing(client):
    rows = client.get("/api/cams").json()
    assert [c["id"] for c in rows] == ["cam3", "cam1"]
    assert rows[0]["zones"]["counter_customer"]


def test_events_hide_footfall_noise(client, hub):
    add(hub)
    add(hub, rule="footfall", severity=INFO, cam_id="cam1", title="Entry")

    assert [e["severity"] for e in client.get("/api/events").json()] == [CRITICAL]
    assert len(client.get("/api/events?include_info=true").json()) == 2


def test_events_filters(client, hub):
    add(hub)
    add(hub, rule="loitering", severity=WARNING, cam_id="cam1", title="Loitering")

    assert len(client.get("/api/events?rule=loitering").json()) == 1
    assert len(client.get("/api/events?cam_id=cam3").json()) == 1
    assert len(client.get("/api/events?q=unattended").json()) == 1


def test_acknowledge(client, hub):
    ev = add(hub)
    assert client.post(f"/api/events/{ev.id}/ack").json()["ok"] is True
    assert hub.store.get(ev.id)["status"] == "ack"
    assert client.post("/api/events/nope/ack").status_code == 404


def test_frame_endpoint(client, hub):
    assert client.get("/api/cams/cam3/frame.jpg").status_code == 404
    hub.set_frame("cam3", b"\xff\xd8jpegbytes", 123.0, 1, 12.0)
    r = client.get("/api/cams/cam3/frame.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == b"\xff\xd8jpegbytes"


def test_mjpeg_emits_one_part_per_new_frame(hub):
    async def run():
        gen = mjpeg_frames(hub, "cam3", boundary="B")
        hub.set_frame("cam3", b"\xff\xd8first", 1.0, 1, 10.0)
        first = await gen.__anext__()
        hub.set_frame("cam3", b"\xff\xd8second", 2.0, 1, 10.0)
        second = await gen.__anext__()
        # A frame that has not changed must not be sent again.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), timeout=0.4)
        await gen.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert first.startswith(b"--B\r\nContent-Type: image/jpeg\r\n")
    assert b"Content-Length: 7\r\n\r\n\xff\xd8first\r\n" in first
    assert b"\xff\xd8second" in second


def test_mjpeg_route_is_wired_up(client):
    """Deliberately not fetched over HTTP: the response never ends, so a test
    client blocks forever trying to close it. The generator above carries the
    behaviour; this just proves the route exists and reaches its guard."""
    app = client.app
    assert any(getattr(r, "path", "") == "/api/cams/{cam_id}/stream.mjpg"
               for r in app.routes)
    assert client.get("/api/cams/nope/stream.mjpg").status_code == 404


def test_mjpeg_unknown_camera(client):
    assert client.get("/api/cams/nope/stream.mjpg").status_code == 404


def test_saving_zones_takes_effect_immediately(client, hub):
    new = {"counter_customer": [[10, 10], [90, 10], [90, 90], [10, 90]],
           "counter_staff": [[10, 0], [90, 0], [90, 9]]}
    r = client.post("/api/cams/cam3/zones", json={"zones": new})
    assert r.status_code == 200

    # The rules read cam.zones every frame, so the live object must have changed.
    assert hub.cams["cam3"].zones["counter_staff"] == [[10.0, 0.0], [90.0, 0.0], [90.0, 9.0]]
    assert "counter_customer" in hub.cams["cam3"].zones
    # ... and the detector must be told, or its overlay would lie.
    assert hub.infer_q.get_nowait()["cam_id"] == "cam3"


def test_saving_zones_rejects_rubbish(client):
    assert client.post("/api/cams/cam3/zones", json={}).status_code == 400
    assert client.post("/api/cams/cam3/zones",
                       json={"zones": {"z": [[1, 1]]}}).status_code == 400
    assert client.post("/api/cams/cam3/zones",
                       json={"zones": {"z": [["a", "b"], [1, 2]]}}).status_code == 400
    assert client.post("/api/cams/nope/zones",
                       json={"zones": {}}).status_code == 404


def test_evidence_paths_cannot_escape_the_data_directory(client, hub):
    (hub.settings.snapshots_dir / "abc123.jpg").write_bytes(b"\xff\xd8shot")
    assert client.get("/snapshots/abc123.jpg").status_code == 200
    # Event ids are hex; anything else is someone probing.
    assert client.get("/snapshots/..%2f..%2fetc%2fpasswd.jpg").status_code in (307, 404)
    assert client.get("/clips/missing.mp4").status_code == 404


def test_kpis(client, hub):
    add(hub)
    hub.set_frame("cam3", b"x", 1.0, 3, 20.0)
    hub.online["cam3"] = True
    k = client.get("/api/kpis").json()
    assert k["cameras_total"] == 2
    assert k["cameras_online"] == 1
    assert k["people_now"] == 3
    assert k["critical_today"] == 1


def test_websocket_greets_with_current_state(client, hub):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        second = ws.receive_json()
    kinds = {first["type"], second["type"]}
    assert kinds == {"kpi", "cams"}


def test_critical_event_captures_evidence_and_asks_for_a_clip(hub, client):
    hub.set_frame("cam3", b"\xff\xd8frame", 5.0, 1, 10.0)
    ev = Event(cam_id="cam3", rule="zone_breach", severity=CRITICAL,
               title="Customer at open counter")
    hub.handle_event(ev)

    row = hub.store.get(ev.id)
    assert row["snapshot"], "a CRITICAL must save the frame that caused it"
    assert (hub.settings.snapshots_dir / f"{ev.id}.jpg").read_bytes() == b"\xff\xd8frame"
    assert hub.cmd_qs["cam3"].get_nowait() == {"type": "clip", "event_id": ev.id}


def test_warning_event_does_not_burn_a_clip(hub):
    hub.set_frame("cam1", b"\xff\xd8frame", 5.0, 1, 10.0)
    hub.handle_event(Event(cam_id="cam1", rule="loitering", severity=WARNING,
                           title="Loitering"))
    assert hub.cmd_qs["cam1"].empty()
