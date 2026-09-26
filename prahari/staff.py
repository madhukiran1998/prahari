"""Tells staff from customers, by watching who stands behind the counter.

There is no staff/customer class in any detector, and training one needs the
client's own uniforms. So this learns instead: anybody who spends a few seconds
inside the `counter_staff` zone is, by definition, staff. Their shirt colour is
remembered, and from then on the same colour is recognised as staff anywhere
else in the frame - including in front of the counter, where the location prior
alone would call them a customer.

Two properties worth keeping:

  * it needs no enrolment step, so it works on the first minute of a client's
    own footage;
  * it is sticky per track. A staff member who turns away and loses their
    colour match for a frame does not flicker into being a customer, because
    that flicker would restart every service timer watching them.

Colour is compared in hue/saturation only. Brightness is dropped deliberately:
the same uniform under a display spotlight and in the aisle differs mostly in
V, and matching on it would split one staff member into two people.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import CameraCfg, Settings
from .geometry import point_in_polygon
from .types import Detection, FrameCtx

STAFF_ZONE = "counter_staff"

# Tracks not seen for this long are forgotten, so track-id churn cannot grow
# the table without bound over a day.
FORGET_S = 60.0


def colour_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Distance between two torso signatures in hue/saturation.

    Hue is circular over OpenCV's 0-180 range, so red at 179 and red at 1 are
    two apart, not 178. Hue is weighted double: it carries the uniform's
    identity, while saturation mostly reports how well-lit the person is.
    """
    dh = abs(a[0] - b[0])
    dh = min(dh, 180.0 - dh)
    ds = abs(a[1] - b[1])
    return 2.0 * dh + ds


@dataclass
class _Track:
    """What we have learned about one track id on one camera."""

    staff_s: float = 0.0  # seconds spent inside the staff zone
    last_ts: float = 0.0
    signature: tuple[float, float, float] | None = None
    samples: int = 0
    is_staff: bool = False

    def observe(self, appearance: tuple[float, float, float] | None) -> None:
        """Fold one frame's colour into a running mean for this track."""
        if appearance is None:
            return
        if self.signature is None:
            self.signature = appearance
            self.samples = 1
            return
        # Running mean. Hue is averaged naively rather than circularly: a torso
        # crop of one person does not straddle the red wrap-around in practice,
        # and the cost of getting it wrong is one mis-sorted track.
        n = self.samples
        self.signature = tuple(
            (old * n + new) / (n + 1) for old, new in zip(self.signature, appearance)
        )
        self.samples = n + 1


@dataclass
class StaffTracker:
    """Per-camera staff classifier. Mutates `Detection.is_staff` in place."""

    cam: CameraCfg
    settings: Settings
    tracks: dict[int, _Track] = field(default_factory=dict)
    known: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def learn_s(self) -> float:
        return self.settings.threshold("staff_learn_s")

    @property
    def match_dist(self) -> float:
        return self.settings.threshold("staff_match_dist")

    def update(self, ctx: FrameCtx, dt: float) -> None:
        """Label every detection in this frame. `dt` is seconds since the last."""
        zone = self.cam.zones.get(STAFF_ZONE, [])
        for det in ctx.detections:
            track = self.tracks.setdefault(det.track_id, _Track())
            track.last_ts = ctx.ts
            track.observe(det.appearance)

            if zone and point_in_polygon(det.anchor, zone):
                track.staff_s += dt
                if track.staff_s >= self.learn_s and not track.is_staff:
                    self._promote(track)

            if not track.is_staff and self._matches_known(track.signature):
                track.is_staff = True

            det.is_staff = track.is_staff

        self._forget(ctx.ts)

    def _promote(self, track: _Track) -> None:
        """This track has earned staff status; remember how they look."""
        track.is_staff = True
        if track.signature is None:
            return
        # Do not store a signature we would already have matched - otherwise a
        # staff member re-acquired under a new track id adds a near-duplicate
        # every time they are occluded, and the table drifts.
        if self._matches_known(track.signature):
            return
        self.known.append(track.signature)

    def _matches_known(self, signature: tuple[float, float, float] | None) -> bool:
        if signature is None or not self.known:
            return False
        return any(
            colour_distance(signature, k) <= self.match_dist for k in self.known
        )

    def _forget(self, now: float) -> None:
        stale = [t for t, tr in self.tracks.items() if now - tr.last_ts > FORGET_S]
        for tid in stale:
            del self.tracks[tid]

    def reset(self) -> None:
        """Stream died. Track ids restart from scratch, so the per-track table
        is meaningless - but the learned uniforms are still true, so they stay."""
        self.tracks.clear()


def staff_present(detections: list[Detection]) -> bool:
    return any(d.is_staff for d in detections)
