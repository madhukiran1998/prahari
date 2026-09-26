"""Is the camera actually working?

Three failure modes, all of which look like "no alerts" to a naive system and
are therefore the most dangerous:

  offline  - the stream dropped (ingest tells us, and we also time out on frames)
  frozen   - frames still arrive but never change (encoder wedged, common on
             cheap NVR channels; a still image will never trigger any rule)
  blackout - the lens was covered or the lights went out during opening hours
"""

from __future__ import annotations

from ..config import CameraCfg, Settings
from ..types import RESOLVED, START, WARNING, CamStatus, Event, FrameCtx
from .base import Rule


class CameraHealthRule(Rule):
    name = "camera_health"

    def __init__(self, cam: CameraCfg, settings: Settings):
        super().__init__(cam, settings)
        self.down_s = settings.threshold("camera_down_s")
        self.frozen_s = settings.threshold("camera_frozen_s")
        self.dark_s = settings.threshold("camera_dark_s")
        self.dark_luma = settings.threshold("camera_dark_luma")

        self.last_frame_ts: float | None = None
        self.offline_since: float | None = None
        self.last_phash = ""
        self.phash_since: float | None = None
        self.dark_since: float | None = None
        self.firing: set[str] = set()  # which of offline/frozen/blackout are live

    # ------------------------------------------------------------------
    def on_status(self, status: CamStatus) -> list[Event]:
        if status.online:
            self.offline_since = None
            return self._clear("offline", status.ts, "Camera back online")
        if self.offline_since is None:
            self.offline_since = status.ts
        return []

    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        events: list[Event] = []
        self.last_frame_ts = ctx.ts
        if self.offline_since is not None:
            self.offline_since = None
            events += self._clear("offline", ctx.ts, "Camera back online")

        # Frozen feed: identical fingerprint for too long.
        if ctx.phash and ctx.phash == self.last_phash:
            if self.phash_since is None:
                self.phash_since = ctx.ts
            elif ctx.ts - self.phash_since >= self.frozen_s:
                events += self._raise(
                    "frozen",
                    ctx.ts,
                    f"Camera feed frozen for {ctx.ts - self.phash_since:.0f}s",
                )
        else:
            self.last_phash = ctx.phash
            self.phash_since = None
            events += self._clear("frozen", ctx.ts, "Camera feed moving again")

        # Blackout, but only while the store is open - dark frames overnight
        # are just night.
        if ctx.luma < self.dark_luma and not self.settings.is_closed(ctx.ts):
            if self.dark_since is None:
                self.dark_since = ctx.ts
            elif ctx.ts - self.dark_since >= self.dark_s:
                events += self._raise(
                    "blackout", ctx.ts, "Camera view blacked out or obstructed"
                )
        else:
            self.dark_since = None
            events += self._clear("blackout", ctx.ts, "Camera view restored")

        return events

    def on_tick(self, ts: float) -> list[Event]:
        """Frames can stop without the reader noticing an error, so time out here
        as well as trusting the ingest worker's status messages."""
        stale = self.last_frame_ts is not None and ts - self.last_frame_ts > self.down_s
        if stale and self.offline_since is None:
            self.offline_since = self.last_frame_ts
        if self.offline_since is None:
            return []
        if ts - self.offline_since < self.down_s:
            return []
        return self._raise(
            "offline", ts, f"Camera offline for {ts - self.offline_since:.0f}s"
        )

    # ------------------------------------------------------------------
    def _raise(self, kind: str, ts: float, title: str) -> list[Event]:
        if kind in self.firing:
            return []
        self.firing.add(kind)
        return [
            self.event(
                severity=WARNING,
                kind=START,
                ts=ts,
                title=f"{self.cam.name}: {title}",
                dedup_key=f"{self.name}:{self.cam.id}:{kind}",
                meta={"fault": kind},
            )
        ]

    def _clear(self, kind: str, ts: float, title: str) -> list[Event]:
        if kind not in self.firing:
            return []
        self.firing.discard(kind)
        return [
            self.event(
                severity=WARNING,
                kind=RESOLVED,
                ts=ts,
                title=f"{self.cam.name}: {title}",
                dedup_key=f"{self.name}:{self.cam.id}:{kind}",
                meta={"fault": kind},
            )
        ]

    def reset(self, ts: float) -> list[Event]:
        return []  # a lost stream is exactly what this rule exists to report
