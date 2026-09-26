"""Loitering rule: cumulative dwell inside a zone, per person.

The hard part is not the timer, it is ByteTrack losing an ID when someone is
briefly occluded and handing the same person a fresh one. Without the re-attach
below, a person who lingers for five minutes reads as twenty people who each
stayed fifteen seconds, and the rule never fires. When a new track appears in
the zone close to where a track vanished moments ago, we inherit the old dwell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import CameraCfg, Settings
from ..geometry import bbox_in_polygon, iou
from ..types import CRITICAL, RESOLVED, START, WARNING, Event, FrameCtx
from .base import Rule

LOITER_ZONE = "entrance"

# One frame's gap should not add more than this to a dwell timer; anything
# larger is a stream hiccup, not someone standing still for that long.
MAX_STEP_S = 1.0


@dataclass
class _Track:
    dwell: float = 0.0
    last_ts: float = 0.0
    last_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    level: int = 0  # 0 = quiet, 1 = warned, 2 = escalated


@dataclass
class _Ghost:
    """A track that just left, kept briefly in case it comes back with a new id."""

    dwell: float
    level: int
    last_ts: float
    last_bbox: tuple[float, float, float, float]
    fired: bool = False
    key: str = ""


class LoiteringRule(Rule):
    name = "loitering"

    def __init__(self, cam: CameraCfg, settings: Settings):
        super().__init__(cam, settings)
        self.warn_s = settings.threshold("loiter_warn_s")
        self.crit_s = settings.threshold("loiter_crit_s")
        self.merge_s = settings.threshold("loiter_merge_s")
        self.merge_iou = settings.threshold("loiter_merge_iou")
        self.tracks: dict[int, _Track] = {}
        self.ghosts: list[_Ghost] = []
        self.firing: set[int] = set()

    def on_frame(self, ctx: FrameCtx) -> list[Event]:
        zone = self.zone(LOITER_ZONE)
        if not zone:
            return []

        events: list[Event] = []
        inside = {
            d.track_id: d for d in ctx.detections if bbox_in_polygon(d.bbox, zone)
        }

        for tid, det in inside.items():
            track = self.tracks.get(tid)
            if track is None:
                track = self._adopt(det.bbox, ctx.ts)
                self.tracks[tid] = track
                if track.level > 0:
                    self.firing.add(tid)
            else:
                track.dwell += min(ctx.ts - track.last_ts, MAX_STEP_S)
            track.last_ts = ctx.ts
            track.last_bbox = det.bbox
            events += self._check_thresholds(tid, track, ctx.ts)

        for tid in [t for t in self.tracks if t not in inside]:
            events += self._retire(tid, ctx.ts)

        self._prune_ghosts(ctx.ts)
        return events

    def on_tick(self, ts: float) -> list[Event]:
        self._prune_ghosts(ts)
        return []

    # ------------------------------------------------------------------
    def _check_thresholds(self, tid: int, track: _Track, ts: float) -> list[Event]:
        if track.dwell >= self.crit_s and track.level < 2:
            track.level = 2
            self.firing.add(tid)
            return [self._alert(tid, track, ts, CRITICAL)]
        if track.dwell >= self.warn_s and track.level < 1:
            track.level = 1
            self.firing.add(tid)
            return [self._alert(tid, track, ts, WARNING)]
        return []

    def _alert(self, tid: int, track: _Track, ts: float, severity: str) -> Event:
        return self.event(
            severity=severity,
            kind=START,
            ts=ts,
            track_id=tid,
            title=f"Person loitering near entrance for {track.dwell:.0f}s",
            dedup_key=f"{self.name}:{self.cam.id}:{tid}:{severity}",
            meta={"dwell_s": round(track.dwell, 1), "zone": LOITER_ZONE},
        )

    def _adopt(self, bbox, ts: float) -> _Track:
        """Reuse a recently-lost track's dwell if this looks like the same person."""
        best, best_iou = None, self.merge_iou
        for ghost in self.ghosts:
            if ts - ghost.last_ts > self.merge_s:
                continue
            overlap = iou(bbox, ghost.last_bbox)
            if overlap >= best_iou:
                best, best_iou = ghost, overlap
        if best is None:
            return _Track(last_ts=ts, last_bbox=bbox)
        self.ghosts.remove(best)
        return _Track(dwell=best.dwell, last_ts=ts, last_bbox=bbox, level=best.level)

    def _retire(self, tid: int, ts: float) -> list[Event]:
        track = self.tracks.pop(tid)
        self.ghosts.append(
            _Ghost(
                dwell=track.dwell,
                level=track.level,
                last_ts=track.last_ts,
                last_bbox=track.last_bbox,
                fired=tid in self.firing,
            )
        )
        if tid not in self.firing:
            return []
        self.firing.discard(tid)
        # Resolve both severities; the engine drops the one that was never active.
        return [
            self.event(
                severity=sev,
                kind=RESOLVED,
                ts=ts,
                track_id=tid,
                title="Loiterer left the entrance",
                dedup_key=f"{self.name}:{self.cam.id}:{tid}:{sev}",
                meta={"dwell_s": round(track.dwell, 1)},
            )
            for sev in (WARNING, CRITICAL)
        ]

    def _prune_ghosts(self, ts: float) -> None:
        self.ghosts = [g for g in self.ghosts if ts - g.last_ts <= self.merge_s]

    def reset(self, ts: float) -> list[Event]:
        events: list[Event] = []
        for tid in list(self.tracks):
            events += self._retire(tid, ts)
        self.ghosts.clear()
        return events
