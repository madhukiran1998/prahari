# Project status

Where Prahari stands as of **26 September 2026**, for anyone picking it up.

## What it is

A proof of concept of camera intelligence for a jewellery shop, built to be
demoed to a shop owner from one laptop. RTSP cameras in; alerts, a live
dashboard, evidence clips and WhatsApp messages out. No cloud.

## What's built and working

**Pipeline**
- One ingest process per camera, with RTSP-over-TCP, reconnect backoff,
  pacing to 5 fps and a pre-roll buffer for clips.
- A shared inference process: YOLO11n person detection plus OC-SORT tracking,
  running on Apple GPU, CUDA or CPU. It drops to 3 fps by itself when
  inference is slow.
- A supervisor that restarts dead workers.

**Rules** (all configurable, no thresholds in code)
- Unattended counter (`zone_breach`): a customer at the counter with no staff
  behind it for 20 s is CRITICAL.
- Walk-aways and time-to-greet (`service`): revenue metrics for the owner.
- Restricted and staff-only zones (`restricted`).
- Loitering at the entrance, WARNING then CRITICAL (`loitering`), with
  re-attachment of lost track IDs.
- Footfall line counting (`footfall`).
- After-hours presence (`after_hours`).
- Camera health (`camera_health`): offline, frozen feed, or blacked-out lens.

**Staff recognition**
- Learns uniforms by shirt colour, with no enrolment step
  (`prahari/staff.py`).

**Dashboard** (`web/index.html`)
- An owner view with KPIs (customers today, in the shop now, time to greet,
  walk-aways) and system health.
- A security view with five live annotated tiles, an alert log with snapshots
  and clips, and acknowledge.
- Zones drawn live on any camera, applied with no restart.

**Alerts**
- Stored in SQLite with text search, plus a snapshot and a 15-second clip per
  event.
- WhatsApp through the Meta Cloud API for CRITICAL alerts. Off by default.

**Demo rig**
- Five simulated cameras (MediaMTX with looping ffmpeg).
- Scripts to download and normalise stock footage.
- A composited counter scene (`cam3`), because no usable free footage of a
  fixed counter camera exists.

**Tooling**
- `smoke_test.py`: a pre-meeting check of the running system.
- `replay.py`: offline tuning of rules on a clip.
- `score_footage.py`: tells you whether footage is usable.
- `lock_camera.py`: stabilises hand-held footage.
- `cam_ctl.sh`: takes a simulated camera down or up.

**Tests**
- 102 tests passing. They run in seconds and need no camera or model.

## Known issues (from code review)

Worth fixing before the counter alert is shown on real footage:

1. **`zone_breach` ignores the learned staff labels.** It decides staff and
   customers purely by which zone someone is standing in
   (`prahari/rules/zone_breach.py`). If a staff member walks round to the
   customer side to show a piece, they count as a customer, nobody is behind
   the counter, and a CRITICAL fires after 20 s. `service` and `restricted`
   already use `is_staff`; `zone_breach` should too.
2. **The counter timer has no tolerance.** A single frame where the customer is
   missed, or where a false detection appears in the staff zone, resets the
   20 s countdown. Several comments in `config/cameras.yaml` exist to work
   around this. The condition should have to be false for a second or two
   before the timer clears.

Accepted limits for a POC (details in the README, *Known limits*):
- No login on the dashboard, which listens on every network interface.
- No retention: `data/` clips and snapshots grow forever.
- Clips are 5 fps, the detection rate.
- People only: no faces, objects or case-open detection.
- Fixed cameras only.
- The tracker loses IDs over long occlusions.
- The WhatsApp token lives in `settings.yaml`. Keep it empty in git.
- WhatsApp alerts sent without first getting a message from the owner need an
  approved template.

## Suggested next steps

1. Fix the two `zone_breach` issues above, with tests.
2. Try it on the client's own exported footage: run `score_footage.py`, draw
   zones, then tune with `replay.py`.
3. Before any pilot in a real shop:
   - add dashboard authentication;
   - add retention for `data/`;
   - move secrets to environment variables;
   - get a WhatsApp message template approved.
