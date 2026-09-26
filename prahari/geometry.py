"""Zone maths. Pure functions on plain tuples - no OpenCV, no model objects.

Polygons are lists of (x, y) in frame pixel coordinates. Lines are two points.
"""

from __future__ import annotations

Point = tuple[float, float]
Polygon = list[Point] | list[list[float]]
BBox = tuple[float, float, float, float]


def bbox_anchor(bbox: BBox) -> Point:
    """Bottom-centre of a box: where a standing person meets the floor.

    Using the centroid instead puts tall people in the wrong zone whenever a
    zone boundary runs across the floor near their feet.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    """Ray-casting point-in-polygon. Points exactly on an edge may go either way."""
    if not poly or len(poly) < 3:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        # Does the horizontal ray at y cross edge j->i?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def bbox_in_polygon(bbox: BBox, poly: Polygon) -> bool:
    return point_in_polygon(bbox_anchor(bbox), poly)


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cross(o: Point, a: Point, b: Point) -> float:
    """Z of (a-o) x (b-o). Sign tells which side of line o->a the point b is on."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def line_side(pt: Point, p1: Point, p2: Point) -> int:
    """-1, 0 or +1 for which side of the directed line p1->p2 the point lies on."""
    c = _cross(p1, p2, pt)
    if c > 0:
        return 1
    if c < 0:
        return -1
    return 0


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """True if segment a1-a2 properly crosses segment b1-b2."""
    d1 = _cross(b1, b2, a1)
    d2 = _cross(b1, b2, a2)
    d3 = _cross(a1, a2, b1)
    d4 = _cross(a1, a2, b2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def line_crossing(prev: Point, cur: Point, line: Polygon) -> int:
    """Direction a track crossed a counting line between two positions.

    Returns +1 or -1 for a crossing (sign = which way), 0 for no crossing.
    Requires an actual segment intersection, so someone walking parallel to the
    line - on either side - never registers.
    """
    if not line or len(line) < 2:
        return 0
    p1 = (line[0][0], line[0][1])
    p2 = (line[1][0], line[1][1])
    if not segments_intersect(prev, cur, p1, p2):
        return 0
    before = line_side(prev, p1, p2)
    after = line_side(cur, p1, p2)
    if before == after:
        return 0
    return 1 if after > 0 else -1
