# Prahari

Camera intelligence for a jewellery shop. It takes RTSP video in and sends
alerts out, for four situations:

- a customer at a display counter with no staff behind it;
- someone loitering by the door;
- anyone moving about after closing;
- a camera that has quietly stopped working.

Prahari is a **proof of concept** (POC). It runs on one laptop and needs no
cloud. The only outbound connection is the optional WhatsApp alert.

| Doc | For |
|---|---|
| **This README** | Developer guide: setup, architecture, code map, how to extend it |
| [docs/STATUS.md](docs/STATUS.md) | What's built, known issues, next steps. **Read this second.** |
| [docs/DEMO.md](docs/DEMO.md) | Step-by-step script for showing it to a client |

**Contents:**
[Quick start](#quick-start) ·
[How it works](#how-it-works) ·
[Code map](#code-map) ·
[The rules](#the-rules) ·
[Configuration](#configuration) ·
[Storage](#storage) ·
[HTTP API](#http-api) ·
[Common tasks](#common-tasks) ·
[Tests and tools](#tests-and-tools) ·
[Known limits](#known-limits) ·
[Troubleshooting](#troubleshooting)

---

## Quick start

### 1. One-time setup

You need Python 3.11+ and `ffmpeg` (`brew install ffmpeg`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'                                         # app + pytest

python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"    # ~6 MB model weights
./sim/download_media.sh                                         # demo footage
python sim/make_staged_clip.py                                  # the counter scene (cam3)
```

Also download the `mediamtx` binary (a small RTSP server) for your OS from
[github.com/bluenviron/mediamtx/releases](https://github.com/bluenviron/mediamtx/releases).
Put the extracted `mediamtx` file in `sim/`, next to `sim/mediamtx.yml`.

None of the weights, footage, the binary or runtime data are in git. They are
large and can be regenerated with the commands above.

### 2. Run it

Use three terminals:

```bash
./sim/run_fake_cams.sh      # 5 simulated cameras on rtsp://localhost:8554/cam1..cam5
python -m prahari.main      # detector, rules and dashboard
python tools/smoke_test.py  # optional: end-to-end check, prints pass/fail per part
```

Open **http://localhost:8008**. The dashboard opens on the **owner view**
(KPIs); **Security view** in the header shows the live cameras and the alert
log.

### 3. Check the code without cameras

```bash
pytest                      # 102 tests, a few seconds; no camera, model or network
```

---

## How it works

### The big picture

```
 CAMERAS                PROCESSES                                        OUTPUTS
 ───────                ─────────                                        ───────
 cam1 ─RTSP─> [ingest:cam1] ─┐
 cam2 ─RTSP─> [ingest:cam2] ─┤  frame_q               bus_q
  ...                        ├─────────> [inference] ─────────> [main process]
 camN ─RTSP─> [ingest:camN] ─┘  JPEG frames      people boxes     │
                   ^            (5 fps each)     + track IDs      ├─ rules engine ──> alerts
                   │                             + overlay image  ├─ SQLite (data/prahari.db)
                   │                                              ├─ dashboard + WebSocket
                   └──────── cmd_q: "record a clip", "slow down" ─┤─ WhatsApp
                                                                  └─ supervisor (restarts dead workers)
```

There are three kinds of process:

1. **Ingest, one per camera** (`prahari/ingest.py`). Opens the RTSP stream
   (over TCP), reconnects with backoff if it drops, and keeps 5 frames per
   second, shrunk to 640 px wide. It holds the last 5 seconds in memory so an
   alert clip can include what happened *before* the alert.
2. **Inference, one in total** (`prahari/inference.py`). Runs YOLO11n
   person detection plus OC-SORT tracking, which gives each person a stable
   track ID across frames. It uses one model instance per camera, because the
   tracker keeps its state on the model, and sharing one would mix IDs between
   cameras. It also samples each person's shirt colour and draws the boxes and
   zones onto the picture.
3. **Main process** (`prahari/main.py`). A background thread feeds every
   frame's detections to the **rules engine**. Uvicorn serves the **dashboard
   and API**. A supervisor thread restarts any worker that dies.

**Why separate processes?** So one hung camera can never stall the others, and
a crash in one worker takes down nothing else.

**Why do frames get dropped?** If the detector falls behind, the oldest frame
in the queue is thrown away (`ingest._put_latest`). An alert based on a frame
from ten seconds ago is worse than no alert. If inference is slow for long
enough, every camera is also told to drop from 5 to 3 fps.

### Life of one alert, step by step

Here is the headline "unattended counter" alert, followed through the code:

1. `ingest:cam3` reads a frame, resizes it, and puts a JPEG on `frame_q`. It
   also adds the frame to its 5-second ring buffer.
2. `inference` decodes it, and `model.track()` finds two people with track IDs
   7 and 9. It samples each person's torso colour, draws the overlay, and puts
   a `det` message on `bus_q`.
3. The main process's `bus_loop` turns the message into a `FrameCtx` holding a
   list of `Detection`s (`prahari/types.py`) and calls
   `RulesEngine.handle_frame`.
4. The engine first runs `StaffTracker` (`prahari/staff.py`), which marks each
   detection as staff or not, based on shirt colours it has learned.
5. Each rule enabled for cam3 runs. `ZoneBreachRule` sees someone in
   `counter_customer` and nobody in `counter_staff`, and starts a timer. On a
   later frame, 20 s on, it returns an `Event(kind="start", severity="CRITICAL")`.
6. `RulesEngine._admit` checks two things. **Dedup:** is this alert already
   open? **Cooldown:** did the same rule's last alert on this camera clear
   less than 120 s ago? If neither, the event is let through.
7. `Hub.handle_event` (`prahari/api.py`) then:
   - saves the event to SQLite;
   - writes a snapshot JPEG;
   - pushes the event to every open dashboard over the WebSocket;
   - sends it to WhatsApp, because it is CRITICAL;
   - puts a `clip` command on cam3's `cmd_q`.
8. `ingest:cam3` records 10 more seconds after the ring buffer's 5, encodes an
   MP4, and sends `clip_ready`. The event row is updated and the dashboard shows
   the clip.
9. When staff return, the rule emits `kind="resolved"` and the alert closes.

### Messages between processes

| On queue | `type` | From → to | Meaning |
|---|---|---|---|
| `frame_q` | `frame` | ingest → inference | One JPEG frame plus brightness and a hash of the frame |
| `bus_q` | `det` | inference → main | Detections plus annotated JPEG for one frame |
| `bus_q` | `cam_status` | ingest → main | Stream went online or offline |
| `bus_q` | `clip_ready` | ingest → main | An event clip finished encoding |
| `bus_q` | `perf_degrade` | inference → main | Inference is too slow; lower the fps |
| `cmd_q` (per camera) | `clip`, `set_fps` | main → ingest | Record a clip; change the frame rate |
| `infer_cmd_q` | `zones` | main → inference | Zones were redrawn; update the overlay |

---

## Code map

```
prahari/
  main.py            Entry point. Starts processes, bus loop, supervisor, web server.
  ingest.py          Per-camera RTSP reader, frame pacing, ring buffer, clip recording.
  inference.py       YOLO + tracker, torso-colour sampling, overlay drawing.
  api.py             FastAPI app + Hub (in-memory live state, event fan-out).
  types.py           The data model: Detection, FrameCtx, Event, CamStatus.
  config.py          Loads settings.yaml, cameras.yaml, zones.local.yaml.
  geometry.py        Point-in-polygon, line crossing, IoU. Pure functions.
  staff.py           Learns staff uniforms by shirt colour.
  store.py           SQLite event log (WAL mode + FTS5 text search), KPIs.
  clips.py           Encodes buffered JPEGs to H.264 MP4.
  notify.py          WhatsApp via Meta Cloud API; logs to console when disabled.
  rules/
    base.py          The Rule interface every rule implements.
    engine.py        Runs rules per camera; dedup and cooldown; rule registry.
    zone_breach.py   Unattended counter.
    service.py       Walk-aways and time-to-greet.
    restricted.py    No-go and staff-only zones.
    loitering.py     Dwell time at the entrance.
    footfall.py      Line-crossing counter.
    after_hours.py   Anyone present while the store is closed.
    camera_health.py Offline, frozen or blacked-out camera.
web/index.html       The whole dashboard: one file, plain JS, no build step.
config/              settings.yaml, cameras.yaml (+ zones.local.yaml, not in git).
sim/                 Fake camera rig: MediaMTX + looping ffmpeg, footage scripts.
tools/               Smoke test, offline replay, footage scoring, camera control.
tests/               pytest suite. Shared fixtures are in conftest.py.
```

---

## The rules

### Four ideas to know first

1. **Rules are pure.** A rule takes dataclasses in and returns `Event`s. It does
   no database, network or file work. That is why the tests can drive whole
   incidents in a few lines, with no camera.
2. **Rules report edges, not levels.** A rule emits one `start` when a
   condition begins and one `resolved` when it ends, not an alert on every
   frame. Events marked `transient=True` (a footfall crossing, say) are one-off
   moments and are never resolved.
3. **A person's position is their feet.** Someone is "in" a zone when the
   bottom-centre of their box is inside the polygon (`geometry.bbox_anchor`).
   Using the centre of the box would put tall people in the wrong zone.
4. **The engine, not the rule, stops alert storms.** *Dedup* allows one open
   alert per condition. *Cooldown* keeps a rule quiet on a camera for
   `cooldown_s` after the condition clears. When a camera goes offline, every
   rule except `camera_health` is reset, so no alert stays stuck.

Every rule implements up to four hooks from `rules/base.py`:

| Hook | Called when |
|---|---|
| `on_frame(ctx)` | A frame has been processed |
| `on_tick(ts)` | About once a second, whether frames arrive or not |
| `on_status(status)` | A camera stream went up or down |
| `reset(ts)` | The stream was lost |

### The rules and what they need

| Rule | Fires when | Severity | Zones it reads | Uses staff labels |
|---|---|---|---|---|
| `zone_breach` | A customer is at the counter and nobody has been behind it for `zone_breach_grace_s` (20 s) | CRITICAL | `counter_customer`, `counter_staff` | No, position only |
| `service` | A customer waited at the counter and left unserved; also records time-to-greet | WARNING / INFO | `counter_customer`, `counter_staff` | Yes |
| `restricted` | Anyone in `restricted`, or a customer in `staff_only` | CRITICAL | `restricted`, `staff_only` | Yes |
| `loitering` | One person's total time in the zone passes `loiter_warn_s`, then `loiter_crit_s` | WARNING → CRITICAL | `entrance` | No |
| `footfall` | A tracked person crosses the counting line | INFO | `entry_line` (2 points) | No |
| `after_hours` | Anyone at all while the store is closed | CRITICAL | none | No |
| `camera_health` | Stream offline, feed frozen, or view blacked out while the shop is open | WARNING | none | No |

A camera without the zone a rule needs simply stays quiet, so it's safe to
enable a rule before its zone is drawn. Drawing a zone in the dashboard turns on
the rules that read it (`ZONE_RULES` in `rules/engine.py`).

### Staff vs customers

No off-the-shelf detector knows a staff uniform, and training one would need
the client's own staff. So `prahari/staff.py` learns them instead. Anyone who
stands in `counter_staff` for `staff_learn_s` (6 s) is taken to be staff, and
their shirt colour is remembered. From then on, anyone with that colour
anywhere in the frame is labelled `is_staff`. Learning is per camera and needs
no enrolment. Learned colours survive a stream drop.

Until a camera has learned at least one uniform, `staff_only` stays silent.
Otherwise it would flag the shop's own staff.

> The counter alert is **not** detection of a display case being opened. That
> needs a sensor on the case. It detects a customer at the counter with no staff
> present, which is the situation that comes before a loss.

---

## Configuration

Everything tunable lives in config files. **There are no thresholds in the
Python.** If you find a magic number in code, treat it as a bug and move it to
`settings.yaml`.

| File | In git | Contains |
|---|---|---|
| `config/settings.yaml` | yes | Model, tracker, confidence, fps, every threshold and cooldown, store hours, clip lengths, WhatsApp, port. Each setting is commented with *why* it has its value. |
| `config/cameras.yaml` | yes | One entry per camera: RTSP URL, enabled rules, default zones. |
| `config/zones.local.yaml` | no | Zones drawn in the dashboard. Overrides `cameras.yaml` per camera; delete it to reset. |

Zones are lists of `[x, y]` pixel coordinates in the **640×360** frame the
detector sees. A 2-point zone is a line (`entry_line`); anything else is a
polygon. Saving zones in the dashboard updates the running rules straight
away, with no restart.

**Secrets:** `whatsapp.token` sits in `settings.yaml`, which is committed. Keep
it empty in git; set it only on the machine running the demo.

---

## Storage

Everything is written under `data/`, which is not in git:

| Path | What |
|---|---|
| `data/prahari.db` | SQLite (WAL mode). One `events` table plus an FTS5 index for text search. Footfall and time-to-greet are stored as INFO events, so KPIs are SQL queries and survive restarts. |
| `data/snapshots/<event_id>.jpg` | Annotated frame at the moment of the alert (CRITICAL alerts only) |
| `data/clips/<event_id>.mp4` | 5 s before + 10 s after the alert, at 5 fps (CRITICAL alerts only) |

There is no retention policy yet, so `data/` keeps growing.

---

## HTTP API

Served by `prahari/api.py` on port 8008. There is no authentication.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The dashboard |
| GET | `/api/cams` | Every camera: online, zones, people now, per-zone counts, fps |
| POST | `/api/cams/{id}/zones` | Replace a camera's zones. Body: `{"zones": {"name": [[x,y], ...]}}` |
| GET | `/api/cams/{id}/frame.jpg` | Latest annotated frame |
| GET | `/api/cams/{id}/stream.mjpg` | Live annotated video (MJPEG) |
| GET | `/api/events` | Alerts, newest first. Filters: `since`, `rule`, `cam_id`, `severity`, `kind`, `q` (text search), `limit` (≤ 500), `include_info` |
| POST | `/api/events/{id}/ack` | Acknowledge an alert |
| GET | `/api/kpis` | Owner-view numbers |
| GET | `/clips/{id}.mp4`, `/snapshots/{id}.jpg` | Evidence for an event |
| WS | `/ws` | Live push. Message types: `kpi`, `cams`, `live`, `cam_status`, `event`, `clip`, `ack` |

---

## Common tasks

### Add a new rule

1. Create `prahari/rules/<name>.py` with a subclass of `Rule`. Set `name` and
   implement the hooks you need. Read thresholds with
   `self.settings.threshold("my_threshold")` and add them to `settings.yaml`.
   `zone_breach.py` is the simplest example to copy.
2. Register the class in `RULES` in `prahari/rules/engine.py`. If it reads a
   zone, add it to `ZONE_RULES` as well.
3. Enable it on a camera in `config/cameras.yaml`: `rules: [..., <name>]`.
4. Write tests in the style of `tests/test_rules.py`: build `FrameCtx`
   objects with fake `Detection`s, step time forward, and assert on the events.

### Tune zones or rules on new footage

```bash
python tools/score_footage.py clip.mp4          # is this footage usable at all?
python tools/replay.py cam3 --preview z.jpg     # see the zones drawn on a frame
python tools/replay.py cam3 --loops 3           # every event the clip would produce
```

`replay.py` runs the real detector and rules offline. It also reports how often
each zone was occupied, so "why did nothing fire?" has an answer.

### Connect real cameras

Point `url` at the NVR channel and disable or delete the simulated entries:

```yaml
- id: counter
  name: Diamond counter
  url: rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102   # Hikvision sub-stream
  rules: [zone_breach, after_hours, camera_health]
  zones: {}          # draw them in the dashboard
```

Use the **sub-stream** (channel `x02`). It's already about 640×360, which
matches what the detector wants, and it leaves the main stream free for the
NVR's own recording.

### Change the model or tracker

Set `model`, `tracker`, `conf_threshold` and `imgsz` in `settings.yaml`. The
comments there record what was measured: which models were compared, and why
confidence is 0.3. Re-check with `replay.py` after changing any of them.

---

## Tests and tools

| Command | What it does |
|---|---|
| `pytest` | Tests rules, geometry, staff tracker, store, clips and API. No camera needed. |
| `python tools/smoke_test.py` | Checks the **running** system end to end over HTTP |
| `python tools/replay.py <cam>` | Runs a clip through the real rules offline |
| `python tools/score_footage.py clip.mp4` | Measures camera movement, person size and track survival; rejects footage that won't work |
| `python tools/lock_camera.py in.mp4 out.mp4` | Stabilises a hand-held clip (translation only) |
| `./tools/cam_ctl.sh down cam4` / `up cam4` / `list` | Take a simulated camera offline or back online |

About the demo footage: free stock retail video is almost never usable. Of
eleven clips from Pexels, all eleven were rejected for camera movement alone.
So `cam3` (the counter) is **staged**: a CC0 photo of an empty showroom with
real person cut-outs composited in (see `sim/make_staged_clip.py`). The other
four cameras are unedited stock footage, with `cam4` stabilised by
`lock_camera.py`.

---

## Known limits

- **People only.** No faces, bags or objects, and no detection of a display
  case being opened.
- **Fixed cameras only.** Every zone assumes the camera doesn't move.
- **The tracker loses people** during long occlusions. `loitering`
  re-attaches a new track ID to a recent one by box overlap, but someone who
  leaves and comes back a minute later counts as a new person.
- **No authentication**, and the server listens on all network interfaces.
  Fine on a demo laptop; not fine in a shop.
- **No retention.** Clips and snapshots pile up in `data/`.
- **Clips are 5 fps**, the detection rate.

Open bugs and the plan for fixing them are in [docs/STATUS.md](docs/STATUS.md).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Address already in use` on start | Another program has the port. Change `server.port` in `settings.yaml`. |
| Cameras show "offline" | The camera rig isn't running, or `ffmpeg`/`mediamtx` is missing. Run `./tools/cam_ctl.sh list`. |
| Detector speed warning | It drops to 3 fps by itself. Check that the startup log says `on mps` or `on cuda`, not `on cpu`. For more speed, set `imgsz: 480`. |
| No counter alert | Check cam3 has both counter zones: `python tools/replay.py cam3`. There is a 120 s cooldown between incidents. |
| `Can't assign requested address` for local connections | Something else on the machine has used up the ephemeral ports. Check with `netstat -an \| grep -c TIME_WAIT`. |
