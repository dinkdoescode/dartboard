import pytest

from dartscore.geometry import SEGMENT_ORDER, STANDARD as D, polar_to_xy, segment_center_angle, xy_to_polar
from dartscore.scoring import parse_throw, score_polar, score_xy


@pytest.mark.parametrize("x,y,label,points", [
    (0, 0, "BULL", 50),
    (3, 4, "BULL", 50),             # r = 5
    (0, 10, "25", 25),
    (0, 103, "T20", 60),
    (0, 166, "D20", 40),
    (0, 50, "S20", 20),
    (0, 140, "S20", 20),
    (0, 175, "MISS", 0),
    (103, 0, "T6", 18),             # 3 o'clock
    (0, -103, "T3", 9),             # 6 o'clock
    (-166, 0, "D11", 22),           # 9 o'clock
])
def test_known_points(x, y, label, points):
    t = score_xy(x, y)
    assert t.label == label
    assert t.points == points


@pytest.mark.parametrize("number", SEGMENT_ORDER)
def test_every_segment_every_ring(number):
    a = segment_center_angle(number)
    assert score_polar(60, a).label == f"S{number}"
    assert score_polar(103, a).label == f"T{number}"
    assert score_polar(130, a).label == f"S{number}"
    assert score_polar(166, a).label == f"D{number}"
    assert score_polar(166, a).points == 2 * number
    assert score_polar(103, a).points == 3 * number


def test_segment_order_clockwise():
    labels = [score_polar(80, k * 18).segment for k in range(20)]
    assert labels == list(SEGMENT_ORDER)


def test_segment_edges():
    # Just inside either edge of the 20 segment (spans -9..+9 degrees).
    assert score_polar(80, 8.9).segment == 20
    assert score_polar(80, -8.9).segment == 20
    assert score_polar(80, 9.1).segment == 1
    assert score_polar(80, -9.1).segment == 5


def test_ring_boundaries():
    a = 0
    assert score_polar(D.inner_bull, a).label == "BULL"
    assert score_polar(D.inner_bull + 0.01, a).label == "25"
    assert score_polar(D.outer_bull + 0.01, a).label == "S20"
    assert score_polar(D.triple_inner + 0.01, a).label == "T20"
    assert score_polar(D.triple_outer + 0.01, a).label == "S20"
    assert score_polar(D.double_inner + 0.01, a).label == "D20"
    assert score_polar(D.double_outer, a).label == "D20"
    assert score_polar(D.double_outer + 0.01, a).label == "MISS"


def test_polar_roundtrip():
    for r, a in [(10, 0), (50, 45), (120, 200), (169, 359)]:
        r2, a2 = xy_to_polar(*polar_to_xy(r, a))
        assert r2 == pytest.approx(r)
        assert a2 == pytest.approx(a)


@pytest.mark.parametrize("text,label,points", [
    ("T20", "T20", 60), ("d16", "D16", 32), ("5", "S5", 5), ("s19", "S19", 19),
    ("bull", "BULL", 50), ("50", "BULL", 50), ("25", "25", 25), ("miss", "MISS", 0),
])
def test_parse_throw(text, label, points):
    t = parse_throw(text)
    assert (t.label, t.points) == (label, points)


def test_parse_throw_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_throw("T21")
