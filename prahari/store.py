"""SQLite event log. WAL so the API can read while the rules thread writes.

Also carries an FTS index over title+meta, which is what makes "show me
yesterday's counter alerts" a query rather than a scroll.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .types import INFO, START, Event

log = logging.getLogger("prahari.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        TEXT PRIMARY KEY,
    ts        REAL NOT NULL,
    cam_id    TEXT NOT NULL,
    cam_name  TEXT,
    rule      TEXT NOT NULL,
    severity  TEXT NOT NULL,
    kind      TEXT NOT NULL,
    title     TEXT NOT NULL,
    track_id  INTEGER,
    meta      TEXT,
    snapshot  TEXT,
    clip      TEXT,
    status    TEXT NOT NULL DEFAULT 'new'
);
CREATE INDEX IF NOT EXISTS events_ts    ON events(ts DESC);
CREATE INDEX IF NOT EXISTS events_cam   ON events(cam_id, ts DESC);
CREATE INDEX IF NOT EXISTS events_rule  ON events(rule, ts DESC);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.fts = self._init_fts()
        self.db.commit()

    def _init_fts(self) -> bool:
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS events_fts "
                "USING fts5(event_id UNINDEXED, title, meta)"
            )
            return True
        except sqlite3.OperationalError:
            log.warning("sqlite built without FTS5 - search falls back to LIKE")
            return False

    # --- writes -----------------------------------------------------------
    def add_event(self, ev: Event, cam_name: str = "") -> dict[str, Any]:
        meta = json.dumps(ev.meta, default=str)
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO events "
                "(id, ts, cam_id, cam_name, rule, severity, kind, title, track_id, meta) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ev.id, ev.ts, ev.cam_id, cam_name, ev.rule, ev.severity, ev.kind,
                 ev.title, ev.track_id, meta),
            )
            if self.fts:
                self.db.execute(
                    "INSERT INTO events_fts (event_id, title, meta) VALUES (?,?,?)",
                    (ev.id, ev.title, meta),
                )
            self.db.commit()
        return self.get(ev.id) or {}

    def attach(self, event_id: str, *, snapshot: str | None = None,
               clip: str | None = None) -> None:
        sets, args = [], []
        if snapshot is not None:
            sets.append("snapshot=?")
            args.append(snapshot)
        if clip is not None:
            sets.append("clip=?")
            args.append(clip)
        if not sets:
            return
        args.append(event_id)
        with self._lock:
            self.db.execute(f"UPDATE events SET {','.join(sets)} WHERE id=?", args)
            self.db.commit()

    def acknowledge(self, event_id: str) -> bool:
        with self._lock:
            cur = self.db.execute(
                "UPDATE events SET status='ack' WHERE id=?", (event_id,)
            )
            self.db.commit()
            return cur.rowcount > 0

    # --- reads ------------------------------------------------------------
    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM events WHERE id=?", (event_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def query(
        self,
        since: float | None = None,
        rule: str | None = None,
        cam_id: str | None = None,
        severity: str | None = None,
        kind: str | None = START,
        text: str | None = None,
        limit: int = 100,
        exclude_severity: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE 1=1"
        args: list[Any] = []
        if exclude_severity:
            sql += " AND severity != ?"
            args.append(exclude_severity)
        if since is not None:
            sql += " AND ts >= ?"
            args.append(since)
        if rule:
            sql += " AND rule = ?"
            args.append(rule)
        if cam_id:
            sql += " AND cam_id = ?"
            args.append(cam_id)
        if severity:
            sql += " AND severity = ?"
            args.append(severity)
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if text:
            ids = self._search_ids(text)
            if not ids:
                return []
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args += ids
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self.db.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]

    def _search_ids(self, text: str) -> list[str]:
        with self._lock:
            if self.fts:
                try:
                    rows = self.db.execute(
                        "SELECT event_id FROM events_fts WHERE events_fts MATCH ? "
                        "LIMIT 500",
                        (text,),
                    ).fetchall()
                    return [r[0] for r in rows]
                except sqlite3.OperationalError:
                    pass  # user typed something FTS cannot parse; fall through
            like = f"%{text}%"
            rows = self.db.execute(
                "SELECT id FROM events WHERE title LIKE ? OR meta LIKE ? LIMIT 500",
                (like, like),
            ).fetchall()
            return [r[0] for r in rows]

    def kpis(self, day_start: float | None = None) -> dict[str, Any]:
        if day_start is None:
            day_start = _midnight()
        with self._lock:
            alerts = self.db.execute(
                "SELECT COUNT(*) FROM events WHERE ts>=? AND kind=? AND severity!=?",
                (day_start, START, INFO),
            ).fetchone()[0]
            critical = self.db.execute(
                "SELECT COUNT(*) FROM events WHERE ts>=? AND kind=? AND severity='CRITICAL'",
                (day_start, START),
            ).fetchone()[0]
            pending = self.db.execute(
                "SELECT COUNT(*) FROM events WHERE kind=? AND severity!=? AND status='new'",
                (START, INFO),
            ).fetchone()[0]
            footfall = self.db.execute(
                "SELECT COUNT(*) FROM events WHERE ts>=? AND rule='footfall' "
                "AND meta LIKE '%\"in\"%'",
                (day_start,),
            ).fetchone()[0]
            exits = self.db.execute(
                "SELECT COUNT(*) FROM events WHERE ts>=? AND rule='footfall' "
                "AND meta LIKE '%\"out\"%'",
                (day_start,),
            ).fetchone()[0]
            service = self.db.execute(
                "SELECT meta FROM events WHERE ts>=? AND rule='service' AND kind=?",
                (day_start, START),
            ).fetchall()

        # Service metrics are derived in Python rather than SQL: the numbers
        # live inside the meta JSON, and a handful of rows a day is not worth a
        # generated column or a JSON1 dependency.
        waits: list[float] = []
        walkaways = 0
        for (meta,) in service:
            try:
                data = json.loads(meta) if meta else {}
            except (TypeError, ValueError):
                continue
            if data.get("served"):
                waits.append(float(data.get("wait_s", 0.0)))
            else:
                walkaways += 1

        return {
            "alerts_today": alerts,
            "critical_today": critical,
            "pending": pending,
            "footfall_today": footfall,
            "exits_today": exits,
            # Never negative: a miscount on the exit line must not report the
            # shop as holding fewer than nobody.
            "occupancy": max(0, footfall - exits),
            "served_today": len(waits),
            "walkaways_today": walkaways,
            "avg_greet_s": round(sum(waits) / len(waits), 1) if waits else None,
        }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["meta"] = json.loads(d.get("meta") or "{}")
    except json.JSONDecodeError:
        d["meta"] = {}
    return d


def _midnight() -> float:
    now = time.localtime()
    return time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
