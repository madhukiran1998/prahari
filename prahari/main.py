"""Orchestrator: spawn the workers, run the rules, serve the dashboard.

Process layout (one machine, no broker, no container):

    ingest x N  --frame_q-->  inference  --bus_q-->  main
        ^                                              |
        +---------------- cmd_q (clips, fps) ----------+

The main process runs uvicorn on the asyncio loop and the rules engine on a
plain thread that drains bus_q. A camera worker crashing takes down nothing but
itself, and the supervisor restarts it.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from pathlib import Path

# Must be set before OpenCV opens any capture, in this process and its children.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

from .api import Hub, create_app  # noqa: E402
from .config import ROOT, CameraCfg, Settings, load_cameras, load_settings  # noqa: E402
from .ingest import camera_worker  # noqa: E402
from .inference import inference_worker  # noqa: E402
from .notify import Notifier  # noqa: E402
from .rules.engine import RulesEngine  # noqa: E402
from .store import Store  # noqa: E402
from .geometry import bbox_in_polygon
from .types import CamStatus, Detection, FrameCtx  # noqa: E402

log = logging.getLogger("prahari")

TICK_S = 1.0
SUPERVISE_S = 5.0
STAGGER_S = 0.7  # gap between worker starts, so cameras don't all dial at once


def bus_loop(
    bus_q: mp.Queue,
    hub: Hub,
    engine: RulesEngine,
    store: Store,
    cmd_qs: dict[str, mp.Queue],
    stop: threading.Event,
) -> None:
    """Drain the worker bus, feed the rules, tick the clocks."""
    last_tick = 0.0
    while not stop.is_set():
        try:
            msg = bus_q.get(timeout=0.25)
        except queue.Empty:
            msg = None
        except (EOFError, OSError):
            break

        if msg is not None:
            try:
                _handle(msg, hub, engine, store, cmd_qs)
            except Exception:
                log.exception("bus message failed: %s", msg.get("type"))

        now = time.time()
        if now - last_tick >= TICK_S:
            last_tick = now
            try:
                engine.handle_tick(now)
            except Exception:
                log.exception("tick failed")


def _handle(
    msg: dict, hub: Hub, engine: RulesEngine, store: Store, cmd_qs: dict[str, mp.Queue]
) -> None:
    kind = msg.get("type")

    if kind == "det":
        dets = [
            Detection(
                cam_id=d["cam_id"],
                ts=d["ts"],
                track_id=d["track_id"],
                bbox=tuple(d["bbox"]),
                conf=d["conf"],
                appearance=(
                    tuple(d["appearance"]) if d.get("appearance") else None
                ),
            )
            for d in msg["detections"]
        ]
        # Live occupancy per zone. Computed here rather than in the API because
        # this is the only place the detections and the zones meet, and it is
        # what lets the dashboard show the system understanding space rather
        # than just counting people.
        cam = hub.cams.get(msg["cam_id"])
        zones = {}
        if cam is not None:
            for name, poly in cam.zones.items():
                if len(poly) > 2:
                    zones[name] = sum(
                        1 for d in dets if bbox_in_polygon(d.bbox, poly)
                    )
        hub.set_frame(
            msg["cam_id"], msg["jpeg"], msg["ts"], len(dets),
            msg.get("infer_ms", 0.0), zones,
        )
        hub.set_online(msg["cam_id"], True)
        engine.handle_frame(
            FrameCtx(
                cam_id=msg["cam_id"],
                ts=msg["ts"],
                detections=dets,
                width=msg["width"],
                height=msg["height"],
                luma=msg["luma"],
                phash=msg["phash"],
            )
        )

    elif kind == "cam_status":
        status = CamStatus(**msg["status"])
        hub.set_online(status.cam_id, status.online, status.detail)
        engine.handle_status(status)

    elif kind == "clip_ready":
        store.attach(msg["event_id"], clip=msg["path"])
        hub.push({"type": "clip", "event_id": msg["event_id"]})
        log.info("clip ready for %s", msg["event_id"])

    elif kind == "perf_degrade":
        fps = msg["fps"]
        log.warning(
            "inference is slow (%.0f ms median) - asking every camera for %.0f fps",
            msg.get("median_ms", 0.0),
            fps,
        )
        for cmd_q in cmd_qs.values():
            try:
                cmd_q.put_nowait({"type": "set_fps", "fps": fps})
            except Exception:
                pass


class Supervisor(threading.Thread):
    """Restarts a worker that has died. One flaky camera must not end the demo."""

    def __init__(self, specs: dict[str, tuple], stop: threading.Event, ctx):
        super().__init__(daemon=True, name="supervisor")
        self.specs = specs  # name -> (target, args)
        self.stop = stop
        self.ctx = ctx
        self.procs: dict[str, mp.Process] = {}

    def start_all(self) -> None:
        for i, name in enumerate(self.specs):
            if i:
                # Stagger the starts. Every camera negotiating RTSP in the same
                # instant makes some of them time out, and a worker that loses
                # that race then sits out a 30 s connect timeout per retry. An
                # NVR with a dozen channels makes this much worse.
                time.sleep(STAGGER_S)
            self._spawn(name)

    def _spawn(self, name: str) -> None:
        target, args = self.specs[name]
        proc = self.ctx.Process(target=target, args=args, name=name, daemon=True)
        proc.start()
        self.procs[name] = proc
        log.info("started worker %s (pid %s)", name, proc.pid)

    def run(self) -> None:
        while not self.stop.wait(SUPERVISE_S):
            for name, proc in list(self.procs.items()):
                if proc.is_alive():
                    continue
                log.error("worker %s died (exit %s) - restarting", name, proc.exitcode)
                self._spawn(name)

    def shutdown(self) -> None:
        for proc in self.procs.values():
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


def resolve_model(settings: Settings) -> None:
    """Prefer weights sitting in the repo root over a fresh download."""
    local = ROOT / settings.model
    if local.exists():
        settings.model = str(local)


def run(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.settings) if args.settings else None)
    cams = [c for c in load_cameras(Path(args.cameras) if args.cameras else None)
            if c.enabled]
    resolve_model(settings)

    if args.force_closed:
        settings.demo_force_closed = True
        log.warning("demo_force_closed is ON - every camera is 'after hours'")

    log.info("%d camera(s): %s", len(cams), ", ".join(f"{c.id}={c.name}" for c in cams))

    ctx = mp.get_context("spawn")
    frame_q = ctx.Queue(maxsize=max(4, 2 * len(cams)))
    bus_q = ctx.Queue(maxsize=400)
    cmd_qs = {c.id: ctx.Queue(maxsize=32) for c in cams}
    infer_cmd_q = ctx.Queue(maxsize=32)
    worker_stop = ctx.Event()

    specs: dict[str, tuple] = {
        f"ingest:{c.id}": (
            camera_worker,
            (c, settings, frame_q, bus_q, cmd_qs[c.id], worker_stop),
        )
        for c in cams
    }
    if not args.ingest_only:
        specs["inference"] = (
            inference_worker,
            (cams, settings, frame_q, bus_q, worker_stop, infer_cmd_q),
        )

    thread_stop = threading.Event()
    supervisor = Supervisor(specs, thread_stop, ctx)
    supervisor.start_all()
    supervisor.start()

    if args.ingest_only:
        _ingest_only_loop(frame_q, bus_q, thread_stop)
        worker_stop.set()
        thread_stop.set()
        supervisor.shutdown()
        return

    store = Store(settings.db_path)
    notifier = Notifier(settings.whatsapp)
    hub = Hub(settings, cams, store, notifier, cmd_qs, infer_cmd_q)
    engine = RulesEngine(cams, settings, on_event=hub.handle_event)
    # So the zones endpoint can arm a rule the moment its zone is drawn.
    hub.engine = engine

    bus = threading.Thread(
        target=bus_loop,
        args=(bus_q, hub, engine, store, cmd_qs, thread_stop),
        daemon=True,
        name="bus",
    )
    bus.start()

    import uvicorn

    app = create_app(hub)
    log.info("dashboard on http://localhost:%d", settings.port)
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level="warning",
            access_log=False,
        )
    finally:
        log.info("shutting down")
        thread_stop.set()
        worker_stop.set()
        supervisor.shutdown()


def _ingest_only_loop(frame_q: mp.Queue, bus_q: mp.Queue, stop: threading.Event) -> None:
    """Phase-2 checkpoint mode: prove the streams read and reconnect, no model."""
    counts: dict[str, int] = {}
    last_report = time.time()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("ingest-only mode: Ctrl-C to stop")
    while not stop.is_set():
        try:
            msg = frame_q.get(timeout=0.25)
            counts[msg["cam_id"]] = counts.get(msg["cam_id"], 0) + 1
        except queue.Empty:
            pass
        while True:
            try:
                event = bus_q.get_nowait()
            except queue.Empty:
                break
            if event.get("type") == "cam_status":
                st = event["status"]
                log.info(
                    "%s | %s %s", st["cam_id"],
                    "ONLINE" if st["online"] else "OFFLINE", st.get("detail", ""),
                )
        now = time.time()
        if now - last_report >= 5.0:
            span = now - last_report
            log.info(
                "frames/s | %s",
                "  ".join(f"{k}={v / span:.1f}" for k, v in sorted(counts.items()))
                or "(none)",
            )
            counts.clear()
            last_report = now


def main() -> None:
    parser = argparse.ArgumentParser(prog="prahari", description=__doc__)
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--cameras", help="path to cameras.yaml")
    parser.add_argument("--ingest-only", action="store_true",
                        help="read streams and report frame rates; no model, no API")
    parser.add_argument("--force-closed", action="store_true",
                        help="pretend the store is shut, to demo the after-hours rule")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)


if __name__ == "__main__":
    main()
