# Prahari

Camera intelligence for a jewellery shop. RTSP in, alerts out: a person at an
unattended display counter, someone loitering by the door, movement after
closing, or a camera that has quietly stopped working.

This is a POC. It runs on one laptop, needs no cloud, and the only outbound
connection is the WhatsApp alert.

**Docs:** this README covers running and demoing it.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the developer guide: the
processes, how the rules work, the API, and how to add a rule.
[docs/STATUS.md](docs/STATUS.md) covers what's built, known issues and next
steps.

---

## Run it

Three commands, three terminals.

```bash
pip install -e '.[dev]'     # first time only; Python 3.11+ (dev adds pytest)
./sim/run_fake_cams.sh      # 5 simulated cameras on rtsp://localhost:8554/camN
python -m prahari.main      # detector, rules and dashboard
```

Then open **http://localhost:8008**.

Before showing anyone, check the whole thing actually works:

```bash
python tools/smoke_test.py
```

It walks the real system over HTTP and prints a pass/fail line per piece —
streams, frames, detector speed, zones, and a live counter alert with its
snapshot and clip.

### First-time setup

`pip install -e .` pulls in ultralytics, opencv, fastapi and friends. Two things
are fetched on demand and worth doing before you travel:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"   # ~6 MB of weights
./sim/download_media.sh                                        # demo footage
python sim/make_staged_clip.py                                 # the counter scenario
```

You also need `ffmpeg` (`brew install ffmpeg`) and the `mediamtx` binary in
`sim/` — download the darwin/linux build from
[github.com/bluenviron/mediamtx/releases](https://github.com/bluenviron/mediamtx/releases)
and drop the extracted `mediamtx` next to `sim/mediamtx.yml`.

---

## Demo script

0. **Open on the owner view.** The dashboard starts on the numbers an owner
   cares about - customers today, how many are in the shop, how long people
   wait before they are served, and how many walked out without being served
   at all - plus a strip reporting the system's own uptime, speed and counting
   drift. **Security view** in the header switches to the live console below.
   The choice is remembered per browser.
1. **Everything is watching.** Press Security view. Five cameras, live boxes and
   track numbers, zone overlays burned into the picture. The sentence across
   the top says what the system thinks is happening right now.
2. **The counter alert.** On *Display counter*, a customer is at the case with a
   staff member behind it. The staff member steps away; twenty seconds later the
   tile turns red and a CRITICAL lands in the log with a snapshot and a 15-second
   clip. (This camera is staged so the scenario repeats — say so.)
3. **Loitering.** *Entrance* raises a WARNING when someone stays in the doorway
   zone past the threshold, and escalates if they stay twice as long.
4. **After hours.** Set `demo_force_closed: true` in `config/settings.yaml`, or
   start with `python -m prahari.main --force-closed`. Every person on any camera
   is now a CRITICAL.
5. **A camera dies.** `./tools/cam_ctl.sh down cam4` — within 15 seconds the tile
   greys out and a camera-offline warning appears. `./tools/cam_ctl.sh up cam4`
   brings it back. This is the alert nobody else demos, and it is the one that
   matters: a dead camera looks exactly like a quiet shop.
6. **Draw a zone live.** Press **Zones** on any tile, click a polygon on the
   picture, pick what it is, save. It takes effect immediately — no restart. Do
   this on the client's own footage in the meeting.
7. **WhatsApp.** With `whatsapp.enabled: true`, CRITICAL alerts also arrive on
   the owner's phone with the snapshot attached.

### WhatsApp on the day

Free-form image messages only reach a phone inside a 24-hour window that the
recipient opens by messaging your number first. Have the owner send "hi" to the
test number as the meeting starts and every alert that follows is free. Cold
alerts outside that window need an approved message template — that is a pilot
task, not a POC one.

---

## Configuration

Everything tunable lives in two files. There are no thresholds in the Python.

**`config/settings.yaml`** — detector settings, alert thresholds, cooldowns,
store hours, clip lengths, WhatsApp credentials, the port.

**`config/cameras.yaml`** — one entry per camera: its RTSP URL, which rules run
on it, and the zones drawn on it. Zones are pixel coordinates in the 640×360
frame the detector sees.

Zones saved from the dashboard are written to `config/zones.local.yaml` rather
than back over `cameras.yaml`, so the documented defaults stay readable. Delete
that file to go back to them.

### The rules

| Rule | Fires when | Needs |
|---|---|---|
| `zone_breach` | Somebody is at the counter and nobody is behind it for `zone_breach_grace_s` | `counter_customer`, `counter_staff` |
| `service` | A customer waited at the counter and left with nobody having served them | `counter_customer`, `counter_staff` |
| `restricted` | Anybody in `restricted`, or a customer in `staff_only` | `restricted` and/or `staff_only` |
| `loitering` | One person's cumulative dwell in a zone passes `loiter_warn_s`, then `loiter_crit_s` | `entrance` |
| `after_hours` | Any person while the store is closed | — |
| `camera_health` | Stream offline, frozen, or blacked out | — |
| `footfall` | A tracked person crosses the counting line | `entry_line` |


A camera without the zone a rule needs simply stays quiet, so it is safe to
enable a rule before you have drawn its zone.

### Staff and customers

Nothing detects a uniform out of the box, and training one needs the client's
own staff. So `prahari/staff.py` learns instead: stand inside `counter_staff`
for `staff_learn_s` and your shirt colour is remembered, after which the same
colour reads as staff anywhere in the frame - including in front of the
counter, where position alone would call you a customer.

It is per camera, it needs no enrolment step, and it survives a stream drop
(track ids restart; the uniform does not). Until a camera has learned at least
one uniform, `staff_only` stays silent rather than accusing the shop's own
staff - which is the alert you least want as the opening line of a demo.

**On `zone_breach`, be straight with the client.** This is not open-case
detection. Knowing a display case has been opened needs a sensor on the case or
a model trained on that gesture. What this detects is a customer at the counter
with no staff present, which is the situation that actually precedes a loss.

---

## How it fits together

```
[5 ffmpeg loops] ─> MediaMTX ─> rtsp://localhost:8554/camN
                                     │
                    ingest worker per camera (own process)
                    reads RTSP, paces to 5 fps, keeps a 15 s clip buffer
                                     │ frames
                    inference (one process, one YOLO per camera)
                    person detection + OC-SORT track IDs, draws the overlay
                                     │ detections
                    rules engine (thread in the main process)
                    zone membership, dwell timers, cooldowns, dedup
                                     │ events
              SQLite  ·  clip writer  ·  WhatsApp  ·  WebSocket ─> dashboard
```

Each camera is its own process, so one dead stream cannot stall another, and the
supervisor restarts any worker that dies. The rules are pure — dataclasses in,
events out, no I/O — which is why `tests/test_rules.py` can drive a whole
incident in a few lines.

---

## Tools

```bash
python tools/score_footage.py clip.mp4      # will the rules work on this footage?
python tools/lock_camera.py in.mp4 out.mp4  # turn a hand-held clip into a fixed one
python tools/replay.py cam3 --loops 3       # run a clip through the real rules, offline
python tools/replay.py cam1 --preview z.jpg # dump a frame with the zones drawn on it
python tools/smoke_test.py                  # pre-meeting end-to-end check
./tools/cam_ctl.sh down cam4                # take a camera offline on purpose
./tools/cam_ctl.sh list                     # what the rig is publishing
pytest                                      # 102 tests, a few seconds
```

`score_footage.py` answers "will this camera work" before you spend an evening
drawing zones on footage that cannot support them. It reports camera movement
measured on the background with the people masked out, how much of the frame a
person fills, and how long a track survives - and rejects the clip if any of
those is out of range. Point it at a client's exported footage and you can tell
them in a minute whether their cameras are usable.

Free stock retail footage almost never is: of eleven candidates pulled from
Pexels, **eleven were rejected on camera movement alone**. That is why `cam3` is
staged rather than filmed. `lock_camera.py` rescues some of them - it finds the
window where the operator held stillest and translates every frame back onto the
first, which took the current `cam4` clip from 4.7 px/s to 0.8. It corrects
translation only, and says so rather than half-fixing a clip that pans or zooms.

`replay.py` is how you tune zones on new footage: it reports every event a clip
would have produced plus how often each zone was occupied, so "why did nothing
fire?" has an answer.

---

## Using real cameras

Point `url` at the NVR channel and delete the simulated entries:

```yaml
- id: counter
  name: Diamond counter
  url: rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102   # Hikvision sub-stream
  rules: [zone_breach, after_hours, camera_health]
  zones: {}          # draw them in the dashboard
```

Use the **sub-stream** (channel `x02`), not the main stream: it is already about
640×360, which is what the detector wants, and it leaves the main stream free for
the NVR's own recording.

A phone works too — run IP Webcam or Larix, publish to
`rtsp://<laptop-ip>:8554/phone`, and set `enabled: true` on the `phone` camera.
That is the most convincing demo available: draw a counter zone on the room you
are standing in and walk into it.

---

## Known limits

- **People only.** No face recognition, no bag or object tracking, no
  distinguishing staff from customers except by which zone they stand in.
- **Fixed cameras only.** Every zone rule assumes the camera does not move.
- **Staged counter camera.** `cam3` is composited: a CC0 photo of an empty
  showroom with real person cut-outs, because no free fixed-camera counter
  footage exists where staff are reliably detected. The other four cameras are
  untouched stock footage. See the comment at the top of `sim/make_staged_clip.py`.
- **The tracker loses people.** OC-SORT (like ByteTrack before it) drops IDs
  under occlusion. The loitering rule re-attaches a new track ID to a
  recent one by box overlap, which handles brief occlusion but not somebody
  leaving and returning a minute later.
- **One machine, no auth.** Anyone on the network can open the dashboard. Fine
  for a POC on a laptop; not fine in a shop.
- **SQLite, no retention policy.** Clips accumulate in `data/clips/` forever.

## If something breaks

| Symptom | Cause |
|---|---|
| `Address already in use` on start | Something else has the port. Change `server.port` in `settings.yaml`. |
| Cameras show "offline" | The rig is not running, or `ffmpeg`/`mediamtx` is missing. `./tools/cam_ctl.sh list`. |
| Detector speed warning | It drops to 3 fps by itself. Check the startup log says `on mps` (or `cuda`) rather than `on cpu` - `device: auto` should find the GPU. To go further, set `imgsz: 480` in `settings.yaml`. |
| No counter alert | Check `cam3` has both counter zones: `python tools/replay.py cam3`. Remember the 120 s cooldown between incidents. |
| Nothing local can connect, `Can't assign requested address` | Ephemeral ports exhausted by something else on the machine; `netstat -an \| grep -c TIME_WAIT`. |
