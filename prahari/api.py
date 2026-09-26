"""FastAPI surface: REST for history, WebSocket for live alerts, JPEG for feeds.

The Hub is the join between the worker processes (which speak through queues on
a plain thread) and the asyncio world (websockets, WhatsApp). Everything that
crosses that boundary goes through Hub.push, which is thread-safe.

Camera tiles are polled JPEGs rather than WebRTC. At 1 fps on a LAN that is
indistinguishable from streaming for a wall-display, and it removes an entire
category of demo-day failure.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .config import CameraCfg, Settings, save_zone_override
from .notify import Notifier
from .inference import resolve_device
from .rules.engine import ZONE_RULES
from .store import Store
from .types import CRITICAL, INFO, START, Event

log = logging.getLogger("prahari.api")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
KPI_INTERVAL_S = 3.0
# Live counts are pushed at most this often per camera. Fast enough that the
# panel matches the picture, slow enough not to flood five websocket clients.
LIVE_PUSH_S = 0.25


class Hub:
    """Shared state between the rules thread and the HTTP/WS layer."""

    def __init__(
        self,
        settings: Settings,
        cams: list[CameraCfg],
        store: Store,
        notifier: Notifier,
        cmd_qs: dict[str, Any],
        infer_q: Any = None,
    ):
        self.settings = settings
        self.cams = {c.id: c for c in cams}
        self.store = store
        self.notifier = notifier
        self.cmd_qs = cmd_qs
        self.infer_q = infer_q  # tells the detector to redraw zone overlays
        self.engine = None  # set by main once the rules engine exists

        self.loop: asyncio.AbstractEventLoop | None = None
        self.out: asyncio.Queue | None = None
        self.clients: set[WebSocket] = set()

        self.frames: dict[str, bytes] = {}
        self.frame_ts: dict[str, float] = {}
        self.online: dict[str, bool] = {c.id: False for c in cams}
        self.infer_ms: dict[str, float] = {}
        self.people: dict[str, int] = {c.id: 0 for c in cams}
        self.zone_counts: dict[str, dict[str, int]] = {}
        self._last_live: dict[str, float] = {}
        # Arrival times of recent frames, per camera. The dashboard reports the
        # fps actually delivered rather than the fps requested in settings -
        # the configured number is an intention, and an intention is not
        # something to put on screen next to measurements.
        self._arrivals: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
        self.started = time.time()

    # --- called from the rules thread -------------------------------------
    def set_frame(self, cam_id: str, jpeg: bytes, ts: float, people: int,
                  infer_ms: float, zones: dict[str, int] | None = None) -> None:
        self.frames[cam_id] = jpeg
        self.frame_ts[cam_id] = ts
        self.people[cam_id] = people
        self.infer_ms[cam_id] = infer_ms
        self._arrivals[cam_id].append(ts)
        if zones is not None:
            self.zone_counts[cam_id] = zones
        # Push the counts that belong to THIS frame. Polling every few seconds
        # meant the panel and the picture on screen came from different
        # moments, so the banner could read 5 while the panel read 4 - which
        # makes the whole thing look guessed rather than measured.
        if ts - self._last_live.get(cam_id, 0.0) >= LIVE_PUSH_S:
            self._last_live[cam_id] = ts
            self.push({
                "type": "live",
                "cam_id": cam_id,
                "people": people,
                "zone_counts": self.zone_counts.get(cam_id, {}),
                "fps": self.measured_fps(cam_id),
            })

    def set_online(self, cam_id: str, online: bool, detail: str = "") -> None:
        if self.online.get(cam_id) == online:
            return
        self.online[cam_id] = online
        self.push({
            "type": "cam_status",
            "cam_id": cam_id,
            "online": online,
            "detail": detail,
        })

    def handle_event(self, ev: Event) -> None:
        """Persist, capture evidence, notify, broadcast. Never raises upward."""
        try:
            cam_name = self.cams[ev.cam_id].name if ev.cam_id in self.cams else ev.cam_id
            row = self.store.add_event(ev, cam_name=cam_name)

            if ev.severity == CRITICAL and ev.kind == START:
                snapshot = self._save_snapshot(ev)
                if snapshot:
                    row["snapshot"] = str(snapshot)
                    self.store.attach(ev.id, snapshot=str(snapshot))
                self._request_clip(ev)
                self._notify(ev, cam_name, snapshot)

            self.push({"type": "event", "event": row})
        except Exception:
            log.exception("failed to handle event %s", ev.id)

    def _save_snapshot(self, ev: Event) -> Path | None:
        jpeg = self.frames.get(ev.cam_id)
        if not jpeg:
            return None
        path = self.settings.snapshots_dir / f"{ev.id}.jpg"
        path.write_bytes(jpeg)
        return path

    def _request_clip(self, ev: Event) -> None:
        cmd_q = self.cmd_qs.get(ev.cam_id)
        if cmd_q is None:
            return
        try:
            cmd_q.put_nowait({"type": "clip", "event_id": ev.id})
        except Exception:
            log.warning("could not request clip for %s", ev.id)

    def _notify(self, ev: Event, cam_name: str, snapshot: Path | None) -> None:
        when = time.strftime("%d %b %H:%M:%S", time.localtime(ev.ts))
        body = f"{cam_name} - {when}"
        if self.loop is None:
            log.info("ALERT | %s | %s", ev.title, body)
            return
        asyncio.run_coroutine_threadsafe(
            self.notifier.send_alert(ev.title, body, snapshot), self.loop
        )

    def push(self, msg: dict) -> None:
        """Hand a message to the asyncio side from any thread."""
        if self.loop is None or self.out is None:
            return
        try:
            self.loop.call_soon_threadsafe(self.out.put_nowait, msg)
        except RuntimeError:
            pass  # loop is shutting down

    # --- asyncio side -----------------------------------------------------
    async def broadcast(self, msg: dict) -> None:
        if not self.clients:
            return
        payload = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                self.clients.discard(ws)

    def kpi_payload(self) -> dict:
        kpis = self.store.kpis()
        kpis["cameras_online"] = sum(1 for v in self.online.values() if v)
        kpis["cameras_total"] = len(self.cams)
        kpis["people_now"] = sum(self.people.values())
        median = [v for v in self.infer_ms.values() if v]
        kpis["infer_ms"] = round(sum(median) / len(median), 1) if median else 0.0
        kpis["uptime_s"] = round(time.time() - self.started)
        # The owner view colours the greeting time against the shop's own
        # target, so the threshold has to travel with the numbers.
        kpis["greet_target_s"] = self.settings.thresholds.get("greet_target_s", 60)
        # Named on the dashboard rather than buried in a log: an engineer
        # evaluating this will ask what is running and on what.
        kpis["model"] = self.settings.model
        kpis["device"] = resolve_device(self.settings.device)
        return {"type": "kpi", "kpi": kpis}

    def measured_fps(self, cam_id: str) -> float:
        """Frames per second this camera actually delivered, over its last few.

        Two arrivals are not a rate, so anything shorter than three frames or
        a collapsed time span reports 0.0 rather than a fabricated number.
        """
        stamps = self._arrivals.get(cam_id)
        if not stamps or len(stamps) < 3:
            return 0.0
        span = stamps[-1] - stamps[0]
        return round((len(stamps) - 1) / span, 1) if span > 0 else 0.0

    def cams_payload(self) -> list[dict]:
        now = time.time()
        out = []
        for cam_id, cam in self.cams.items():
            ts = self.frame_ts.get(cam_id, 0.0)
            out.append({
                "id": cam_id,
                "name": cam.name,
                "online": self.online.get(cam_id, False),
                "rules": cam.rules,
                "zones": cam.zones,
                "people": self.people.get(cam_id, 0),
                "last_frame_age_s": round(now - ts, 1) if ts else None,
                "infer_ms": round(self.infer_ms.get(cam_id, 0.0), 1),
                "zone_counts": self.zone_counts.get(cam_id, {}),
                "fps": self.measured_fps(cam_id),
            })
        return out


MJPEG_BOUNDARY = "prahariframe"


async def mjpeg_frames(hub: Hub, cam_id: str, boundary: str = MJPEG_BOUNDARY):
    """Yield each new annotated frame as one multipart part, forever.

    Split out from the endpoint so it can be driven directly in a test - an
    endless streaming response is close to untestable through a test client.
    """
    last_ts = None
    # Poll at twice the detect rate so a new frame goes out promptly.
    interval = 1.0 / max(2.0, hub.settings.detect_fps * 2)
    while True:
        jpeg = hub.frames.get(cam_id)
        ts = hub.frame_ts.get(cam_id)
        if jpeg and ts != last_ts:
            last_ts = ts
            yield (
                f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                + jpeg
                + b"\r\n"
            )
            continue
        await asyncio.sleep(interval)


def create_app(hub: Hub) -> FastAPI:
    async def lifespan(_: FastAPI):
        hub.loop = asyncio.get_running_loop()
        hub.out = asyncio.Queue(maxsize=1000)
        tasks = [asyncio.create_task(_pump()), asyncio.create_task(_kpi_ticker())]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()

    app = FastAPI(title="Prahari", docs_url="/api/docs", lifespan=lifespan)

    async def _pump() -> None:
        assert hub.out is not None
        while True:
            msg = await hub.out.get()
            await hub.broadcast(msg)

    async def _kpi_ticker() -> None:
        while True:
            await hub.broadcast(hub.kpi_payload())
            await asyncio.sleep(KPI_INTERVAL_S)

    # --- pages ------------------------------------------------------------
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    # --- cameras ----------------------------------------------------------
    @app.get("/api/cams")
    async def cams() -> JSONResponse:
        return JSONResponse(hub.cams_payload())

    @app.post("/api/cams/{cam_id}/zones")
    async def set_zones(cam_id: str, body: dict) -> JSONResponse:
        """Save zones drawn in the dashboard. Takes effect immediately.

        The rules read `cam.zones` on every frame, so mutating it in place is
        the whole hot-reload mechanism - no restart, no reload endpoint.
        """
        cam = hub.cams.get(cam_id)
        if cam is None:
            raise HTTPException(status_code=404, detail="unknown camera")

        zones = body.get("zones")
        if not isinstance(zones, dict):
            raise HTTPException(status_code=400, detail="expected {'zones': {...}}")
        cleaned: dict[str, list[list[float]]] = {}
        for name, points in zones.items():
            if not isinstance(points, list) or len(points) < 2:
                raise HTTPException(
                    status_code=400, detail=f"zone '{name}' needs at least 2 points"
                )
            try:
                cleaned[str(name)] = [[float(p[0]), float(p[1])] for p in points]
            except (TypeError, IndexError, ValueError):
                raise HTTPException(
                    status_code=400, detail=f"zone '{name}' has malformed points"
                ) from None

        cam.zones.clear()
        cam.zones.update(cleaned)
        save_zone_override(cam_id, cleaned)

        # Arm whatever rules those zones feed, so a zone drawn in a meeting
        # actually does something rather than silently sitting there.
        armed: list[str] = []
        if hub.engine is not None:
            for name, points in cleaned.items():
                if len(points) < 2:
                    continue
                for rule in ZONE_RULES.get(name, ()):
                    if hub.engine.arm(cam_id, rule):
                        armed.append(rule)
        if hub.infer_q is not None:
            try:
                hub.infer_q.put_nowait({"type": "zones", "cam_id": cam_id, "zones": cleaned})
            except Exception:
                log.warning("could not push new zones to the detector")
        await hub.broadcast({"type": "cams", "cams": hub.cams_payload()})
        log.info(
            "%s | zones updated: %s%s", cam_id, ", ".join(cleaned) or "(none)",
            f" | armed {', '.join(armed)}" if armed else "",
        )
        return JSONResponse({"ok": True, "zones": cleaned, "armed": armed,
                             "rules": cam.rules})

    @app.get("/api/cams/{cam_id}/frame.jpg")
    async def frame(cam_id: str) -> Response:
        """The latest annotated frame, once. Used for snapshots and checks."""
        jpeg = hub.frames.get(cam_id)
        if not jpeg:
            raise HTTPException(status_code=404, detail="no frame yet")
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/cams/{cam_id}/stream.mjpg")
    async def stream(cam_id: str) -> StreamingResponse:
        """The camera tiles, as one long-lived connection per camera.

        The dashboard used to re-fetch frame.jpg once a second per tile. This
        replaces N requests per second with one streamed response per camera,
        which is both smoother and far less to go wrong on a display left open
        all day.
        """
        if cam_id not in hub.cams:
            raise HTTPException(status_code=404, detail="unknown camera")
        return StreamingResponse(
            mjpeg_frames(hub, cam_id),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # --- events -----------------------------------------------------------
    @app.get("/api/events")
    async def events(
        since: float | None = None,
        rule: str | None = None,
        cam_id: str | None = None,
        severity: str | None = None,
        kind: str | None = START,
        q: str | None = None,
        limit: int = Query(100, le=500),
        include_info: bool = False,
    ) -> JSONResponse:
        """Alerts, newest first.

        INFO rows (footfall line crossings) are data for the KPIs, not things
        anyone needs to read, so they are left out unless asked for.
        """
        rows = hub.store.query(
            since=since, rule=rule, cam_id=cam_id, severity=severity,
            kind=kind, text=q, limit=limit,
            exclude_severity=None if include_info else INFO,
        )
        return JSONResponse(rows)

    @app.post("/api/events/{event_id}/ack")
    async def ack(event_id: str) -> JSONResponse:
        if not hub.store.acknowledge(event_id):
            raise HTTPException(status_code=404, detail="unknown event")
        await hub.broadcast({"type": "ack", "event_id": event_id})
        return JSONResponse({"ok": True, "id": event_id})

    @app.get("/api/kpis")
    async def kpis() -> JSONResponse:
        return JSONResponse(hub.kpi_payload()["kpi"])

    # --- evidence ---------------------------------------------------------
    @app.get("/clips/{event_id}.mp4")
    async def clip(event_id: str) -> FileResponse:
        path = hub.settings.clips_dir / f"{_safe(event_id)}.mp4"
        if not path.exists():
            raise HTTPException(status_code=404, detail="clip not ready")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/snapshots/{event_id}.jpg")
    async def snapshot(event_id: str) -> FileResponse:
        path = hub.settings.snapshots_dir / f"{_safe(event_id)}.jpg"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no snapshot")
        return FileResponse(path, media_type="image/jpeg")

    # --- live -------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        try:
            await ws.send_text(json.dumps(hub.kpi_payload(), default=str))
            await ws.send_text(
                json.dumps({"type": "cams", "cams": hub.cams_payload()}, default=str)
            )
            while True:
                await ws.receive_text()  # client keepalives; nothing to act on
        except WebSocketDisconnect:
            pass
        except Exception:
            log.debug("websocket closed", exc_info=True)
        finally:
            hub.clients.discard(ws)

    return app


def _safe(name: str) -> str:
    """Event ids are hex, so anything else is someone probing for ../../etc."""
    return "".join(c for c in name if c.isalnum())
