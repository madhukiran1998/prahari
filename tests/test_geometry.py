from prahari.geometry import (
    bbox_anchor,
    bbox_in_polygon,
    iou,
    line_crossing,
    line_side,
    point_in_polygon,
    segments_intersect,
)

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_anchor_is_bottom_centre():
    assert bbox_anchor((10, 20, 30, 60)) == (20.0, 60.0)


def test_point_in_polygon_basic():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((5, -1), SQUARE)


def test_point_in_polygon_concave():
    # A "U" shape: the notch in the middle is outside.
    u = [[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10]]
    assert point_in_polygon((1, 5), u)
    assert point_in_polygon((5, 1), u)
    assert not point_in_polygon((5, 8), u)


def test_point_in_polygon_degenerate():
    assert not point_in_polygon((1, 1), [])
    assert not point_in_polygon((1, 1), [[0, 0], [1, 1]])


def test_bbox_in_polygon_uses_feet_not_centre():
    # Tall person whose centre is above the zone but whose feet are inside it.
    bbox = (4, -20, 6, 5)
    assert bbox_in_polygon(bbox, SQUARE)
    # ... and one standing below the zone, centre inside, feet outside.
    assert not bbox_in_polygon((4, 5, 6, 15), SQUARE)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 1 / 3


def test_line_side():
    assert line_side((0, 5), (0, 0), (10, 0)) == 1
    assert line_side((0, -5), (0, 0), (10, 0)) == -1
    assert line_side((5, 0), (0, 0), (10, 0)) == 0


def test_segments_intersect():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_intersect((0, 0), (1, 1), (5, 5), (6, 6))


def test_line_crossing_direction():
    line = [[0, 0], [10, 0]]
    assert line_crossing((5, -5), (5, 5), line) == 1
    assert line_crossing((5, 5), (5, -5), line) == -1


def test_line_crossing_ignores_parallel_movement():
    line = [[0, 0], [10, 0]]
    assert line_crossing((1, 5), (9, 5), line) == 0
    # Movement past the end of the line segment does not count either.
    assert line_crossing((20, -5), (20, 5), line) == 0


def test_line_crossing_needs_a_line():
    assert line_crossing((0, 0), (1, 1), []) == 0
