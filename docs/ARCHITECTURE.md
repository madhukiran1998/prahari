# Prahari — developer guide

How the code is laid out, how data moves through it, and how to change it
safely. Read the [README](../README.md) first for what the system does and how
to run the demo.

---

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest                                   # no cameras, model or network needed
```

To run the full system you also need `ffmpeg`, the `mediamtx` binary in `sim/`,
the demo footage (`./sim/download_media.sh`, `python sim/make_staged_clip.py`)
and the YOLO weights — see *First-time setup* in the README. None of these are
in git: weights, footage, `data/` and `config/zones.local.yaml` are all
gitignored.

---

## Processes

```
ingest:cam1 ─┐
ingest:cam2 ─┤  frame_q          bus_q
   ...       ├──────────> inference ──────────> main process
ingest:camN ─┘  (JPEG,           (detections,    ├─ bus thread: rules engine
      ^          latest wins)     annotated JPEG) ├─ uvicorn: API + WebSocket
      |                                           └─ supervisor thread
      └──────────── cmd_q per camera (clip requests, fps changes) ─────┘
```

| Process / thread | File | Job |
|---|---|---|
| `ingest:<cam>` (one per camera) | `prahari/ingest.py` | Reads RTSP over TCP, reconnects with backoff, paces to `detect_fps`, resizes to 640 wide, keeps a pre-roll ring buffer, writes event clips, reports online/offline. |
| `inference` (one) | `prahari/inference.py` | One YOLO model **per camera** (tracker state lives on the model — sharing one would mix track IDs across cameras). Runs `model.track`, samples torso colour for staff detection, draws the overlay, auto-drops fps if median inference exceeds `perf.slow_ms`. |
| bus thread (main) | `prahari/main.py:bus_loop` | Drains `bus_q`, builds `FrameCtx`, feeds the rules engine, ticks it once a second. |
| uvicorn (main) | `prahari/api.py` | Dashboard, REST, MJPEG streams, WebSocket push. |
| supervisor (main) | `prahari/main.py:Supervisor` | Restarts any worker process that dies. Staggers start-up so cameras don't all negotiate RTSP at once. |

Processes use the `spawn` context. `frame_q` is small on purpose: when the
detector falls behind, the oldest frame is dropped (`ingest._put_latest`), so
alerts are never raised on stale frames.

### Messages on `bus_q`

| `type` | From | Meaning |
|---|---|---|
| `det` | inference | Detections + annotated JPEG + luma + frame hash for one frame |
| `cam_status` | ingest | Stream went online/offline (`CamStatus`) |
| `clip_ready` | ingest | An event clip finished encoding |
| `perf_degrade` | inference | Inference is slow; main broadcasts `set_fps` to every camera |

---

## The rules

Everything in `prahari/rules/` is pure: dataclasses in (`prahari/types.py`),
`Event`s out, no I/O. That is what lets the tests drive whole incidents without
a camera.

**Contract** (`rules/base.py`): a rule is a small state machine per camera with
four hooks — `on_frame(ctx)`, `on_tick(ts)`, `on_status(status)`,
`reset(ts)`. Rules emit **edges**: one `start` when a condition begins, one
`resolved` when it ends. Events marked `transient=True` (e.g. a footfall line
crossing) are moments, never resolved.

**Engine** (`rules/engine.py`), per frame:

1. `StaffTracker` (`prahari/staff.py`) labels each detection `is_staff` by
   learned shirt colour, and sets `ctx.staff_known`.
2. Each rule enabled for that camera runs; an exception in one rule is logged
   and does not affect the others.
3. `_admit` applies **dedup** (one open alert per `dedup_key` while the
   condition holds) and **cooldown** (after it clears, the same rule on the same
   camera stays quiet for `cooldown_s` / `rule_cooldown_s`).
4. Admitted events go to `Hub.handle_event` → SQLite, snapshot, clip request,
   WebSocket, and WhatsApp for CRITICAL.

When a camera goes offline, every rule except `camera_health` is `reset`, so
nothing stays stuck on the dashboard.

| Rule | File | Zones it reads | Uses staff labels? |
|---|---|---|---|
| `zone_breach` | `zone_breach.py` | `counter_customer`, `counter_staff` | No — position only |
| `service` | `service.py` | `counter_customer`, `counter_staff` | Yes |
| `restricted` | `restricted.py` | `restricted`, `staff_only` | Yes (`staff_only` silent until a uniform is learned) |
| `loitering` | `loitering.py` | `entrance` | No |
| `footfall` | `footfall.py` | `entry_line` (2 points) | No |
| `after_hours` | `after_hours.py` | — | No |
| `camera_health` | `camera_health.py` | — | No |

A person is "in" a zone when the **bottom-centre of their box** (their feet) is
inside the polygon — `geometry.bbox_anchor`.

### Adding a rule

1. Subclass `Rule` in `prahari/rules/<name>.py`, set `name`, implement the hooks
   you need. Read thresholds with `settings.threshold("...")` and add them to
   `config/settings.yaml` — **no magic numbers in Python**.
2. Register it in `RULES` in `rules/engine.py`, and in `ZONE_RULES` if drawing a
   zone should switch it on.
3. Enable it per camera in `config/cameras.yaml` (`rules: [...]`).
4. Test it like `tests/test_rules.py` does: build `FrameCtx`s with synthetic
   `Detection`s and assert on the events. Fixtures are in `tests/conftest.py`.

---

## Configuration

| File | In git | What |
|---|---|---|
| `config/settings.yaml` | yes | Every threshold, cooldown, model/tracker choice, store hours, WhatsApp, port. |
| `config/cameras.yaml` | yes | Cameras, RTSP URLs, rules per camera, default zones. |
| `config/zones.local.yaml` | no | Zones drawn in the dashboard. Overrides `cameras.yaml` per camera. |

Zones are pixel coordinates in the 640×360 frame the detector sees. Saving zones
in the dashboard (`POST /api/cams/{id}/zones`) mutates `CameraCfg.zones` in
place, so rules and overlays pick them up with no restart.

**Secrets:** `whatsapp.token` lives in `settings.yaml`, which is committed. Keep
it empty in git; put the real token in only on the demo machine, and don't
commit that change.

---

## Storage

Everything goes under `data/` (gitignored):

- `data/prahari.db` — SQLite in WAL mode, one `events` table plus an FTS5 index
  for text search (`prahari/store.py`). Footfall and time-to-greet are INFO rows,
  so KPIs are SQL queries and survive restarts.
- `data/snapshots/<event_id>.jpg` — annotated frame at the moment of the alert.
- `data/clips/<event_id>.mp4` — `clip.pre_s` before + `clip.post_s` after, at
  the detection frame rate (5 fps).

There is no retention policy yet; `data/` grows without bound.

---

## HTTP API

Served by `prahari/api.py` on `server.port` (8008). No authentication.

| Method | Path | |
|---|---|---|
| GET | `/` | Dashboard (`web/index.html`, single file, no build step) |
| GET | `/api/cams` | Cameras with status, zones, live people/zone counts, fps |
| POST | `/api/cams/{id}/zones` | Replace a camera's zones: `{"zones": {"name": [[x,y],...]}}` |
| GET | `/api/cams/{id}/frame.jpg` | Latest annotated frame |
| GET | `/api/cams/{id}/stream.mjpg` | Live annotated MJPEG stream |
| GET | `/api/events` | Alerts, newest first. Filters: `since`, `rule`, `cam_id`, `severity`, `kind`, `q` (text), `limit` ≤ 500, `include_info` |
| POST | `/api/events/{id}/ack` | Acknowledge an alert |
| GET | `/api/kpis` | Owner-view numbers |
| GET | `/clips/{id}.mp4`, `/snapshots/{id}.jpg` | Evidence for an event |
| WS | `/ws` | Pushes `kpi`, `cams`, `live`, `cam_status`, `event`, `clip`, `ack` messages |

---

## Tests and tools

`pytest` runs the rules, geometry, staff tracker, store, clip writer and API
(through FastAPI's test client). No camera or model needed.

For anything involving real footage, use the tools in `tools/` (described in the
README): `replay.py` runs a clip through the real rules offline,
`score_footage.py` says whether footage is usable at all, and `smoke_test.py`
checks the running system end to end over HTTP.
