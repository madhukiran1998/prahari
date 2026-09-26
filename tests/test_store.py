"""The event log: it has to survive a restart and answer questions about the past."""

import time

import pytest

from prahari.store import Store
from prahari.types import CRITICAL, INFO, RESOLVED, START, WARNING, Event


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def make(rule="zone_breach", severity=CRITICAL, cam="cam3", title="Counter unattended",
         ts=None, kind=START, meta=None):
    return Event(
        cam_id=cam, rule=rule, severity=severity, title=title, kind=kind,
        ts=ts if ts is not None else time.time(), meta=meta or {},
    )


def test_round_trip(store):
    ev = make(meta={"held_s": 21.5})
    row = store.add_event(ev, cam_name="Display counter")

    assert row["id"] == ev.id
    assert row["cam_name"] == "Display counter"
    assert row["meta"]["held_s"] == 21.5
    assert row["status"] == "new"


def test_attach_evidence(store):
    ev = make()
    store.add_event(ev)
    store.attach(ev.id, snapshot="/snaps/x.jpg")
    store.attach(ev.id, clip="/clips/x.mp4")

    row = store.get(ev.id)
    assert row["snapshot"] == "/snaps/x.jpg"
    assert row["clip"] == "/clips/x.mp4", "attaching a clip must not wipe the snapshot"


def test_acknowledge(store):
    ev = make()
    store.add_event(ev)
    assert store.acknowledge(ev.id) is True
    assert store.get(ev.id)["status"] == "ack"
    assert store.acknowledge("nope") is False


def test_query_filters(store):
    now = time.time()
    store.add_event(make(rule="loitering", severity=WARNING, cam="cam1", ts=now - 100))
    store.add_event(make(rule="zone_breach", severity=CRITICAL, cam="cam3", ts=now - 50))
    store.add_event(make(rule="zone_breach", severity=CRITICAL, cam="cam3",
                         ts=now - 10, kind=RESOLVED))

    assert len(store.query(kind=START)) == 2
    assert len(store.query(rule="loitering")) == 1
    assert len(store.query(cam_id="cam3", kind=START)) == 1
    assert len(store.query(severity=WARNING)) == 1
    assert len(store.query(since=now - 60, kind=START)) == 1
    assert len(store.query(kind=None)) == 3


def test_exclude_severity_keeps_footfall_out_of_the_alert_log(store):
    store.add_event(make(severity=CRITICAL))
    store.add_event(make(rule="footfall", severity=INFO, cam="cam1", title="Entry"))

    assert len(store.query()) == 2
    rows = store.query(exclude_severity=INFO)
    assert [r["severity"] for r in rows] == [CRITICAL]


def test_query_is_newest_first(store):
    now = time.time()
    for i in range(5):
        store.add_event(make(title=f"event {i}", ts=now - (10 - i)))
    rows = store.query()
    assert [r["title"] for r in rows] == [f"event {i}" for i in (4, 3, 2, 1, 0)]


def test_search_finds_by_title_and_meta(store):
    store.add_event(make(title="Customer at open counter, no staff present"))
    store.add_event(make(rule="loitering", severity=WARNING, cam="cam1",
                         title="Person loitering near entrance"))

    assert len(store.query(text="counter")) == 1
    assert len(store.query(text="loitering")) == 1
    assert store.query(text="counter")[0]["title"].startswith("Customer")
    assert store.query(text="submarine") == []


def test_search_survives_nonsense_input(store):
    """FTS5 rejects some punctuation outright; the box must not 500 on it."""
    store.add_event(make(title="Counter unattended"))
    assert isinstance(store.query(text='"unbalanced'), list)
    assert isinstance(store.query(text="AND OR"), list)


def test_kpis_count_the_right_things(store):
    now = time.time()
    day = now - (now % 86400)
    store.add_event(make(severity=CRITICAL, ts=now))
    store.add_event(make(rule="loitering", severity=WARNING, cam="cam1", ts=now))
    store.add_event(make(rule="footfall", severity=INFO, cam="cam1", ts=now,
                         meta={"direction": "in"}))
    store.add_event(make(rule="footfall", severity=INFO, cam="cam1", ts=now,
                         meta={"direction": "out"}))
    resolved = make(severity=CRITICAL, ts=now, kind=RESOLVED)
    store.add_event(resolved)

    k = store.kpis(day_start=day)
    assert k["alerts_today"] == 2, "INFO and resolved rows are not alerts"
    assert k["critical_today"] == 1
    assert k["footfall_today"] == 1, "only entries count as footfall"
    assert k["pending"] == 2


def test_kpis_ignore_acknowledged_for_pending(store):
    ev = make()
    store.add_event(ev)
    assert store.kpis()["pending"] == 1
    store.acknowledge(ev.id)
    assert store.kpis()["pending"] == 0


def test_reopening_the_database_keeps_events(tmp_path):
    path = tmp_path / "persist.db"
    ev = make()
    Store(path).add_event(ev)
    assert Store(path).get(ev.id) is not None
