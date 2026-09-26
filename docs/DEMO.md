# Demo script

How to show Prahari to a shop owner. Start the system as described in the
[README](../README.md#quick-start), and run `python tools/smoke_test.py`
before the meeting to confirm every part works.

0. **Open on the owner view.** The dashboard opens on the numbers an owner
   cares about:
   - customers today;
   - how many people are in the shop now;
   - how long people wait before someone serves them;
   - how many walked out without being served.

   Below these, a strip shows the system's own health: uptime, speed and how
   far the counts have drifted. **Security view** in the header switches to
   the live console. Each browser remembers which view you last used.
1. **Everything is watching.** Press **Security view**. Five cameras show live
   boxes, track numbers and zone outlines drawn onto the picture. The sentence
   across the top says what the system thinks is happening right now.
2. **The counter alert.** On *Display counter*, a customer stands at the case
   with a staff member behind it. The staff member steps away. Twenty seconds
   later the tile turns red, and a CRITICAL alert appears in the log with a
   snapshot and a 15-second clip. This camera's footage is staged so the
   scene repeats; tell the client so.
3. **Loitering.** *Entrance* raises a WARNING when someone stays in the
   doorway zone past the threshold, and a CRITICAL if they stay twice as long.
4. **After hours.** Set `demo_force_closed: true` in `config/settings.yaml`,
   or start with `python -m prahari.main --force-closed`. Every person on any
   camera now raises a CRITICAL.
5. **A camera dies.** Run `./tools/cam_ctl.sh down cam4`. Within 15 seconds
   the tile greys out and a camera-offline warning appears.
   `./tools/cam_ctl.sh up cam4` brings it back. Few other systems demo this
   alert, and it matters most: a dead camera looks exactly like a quiet shop.
6. **Draw a zone live.** Press **Zones** on any tile, click out a polygon on
   the picture, pick what the zone is for, and save. It takes effect at once,
   with no restart. Do this on the client's own footage during the meeting.
7. **WhatsApp.** With `whatsapp.enabled: true`, CRITICAL alerts also arrive on
   the owner's phone with the snapshot attached.

## Being straight about the counter alert

This is not detection of a display case being opened. That needs a sensor on
the case, or a model trained on that gesture. What Prahari detects is a
customer at the counter with no staff present, which is the situation that
comes before a loss.

## WhatsApp on the day

Free-form image messages only reach a phone within 24 hours of the recipient
messaging your number. Have the owner send "hi" to the test number as the
meeting starts, and every alert after that is free. Sending outside that window
needs a message template approved by Meta. That's a pilot task, not a POC one.

## Most convincing demo: a phone as the camera

Run IP Webcam or Larix on a phone and publish to
`rtsp://<laptop-ip>:8554/phone`. Then set `enabled: true` on the `phone` camera
in `config/cameras.yaml`. Draw a counter zone on the room you're standing in
and walk into it.
