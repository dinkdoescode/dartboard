from dartscore.game import Cricket, X01, checkout
from dartscore.scoring import parse_throw as P


def visit(game, *labels):
    for lbl in labels:
        game.throw(P(lbl))
    game.next_turn()


def test_501_basic_turns():
    g = X01(["A", "B"])
    visit(g, "T20", "T20", "T20")
    assert g.scores["A"] == 321
    assert g.current == 1
    visit(g, "S1", "S5", "S20")
    assert g.scores["B"] == 475
    assert g.current == 0


def test_turn_over_after_three_darts_and_auto_advance():
    g = X01(["A", "B"])
    for lbl in ("S1", "S1", "S1"):
        g.throw(P(lbl))
    assert g.turn_over
    g.throw(P("S20"))          # manual play: 4th dart starts B's visit
    assert g.current == 1
    assert g.scores["B"] == 481


def test_bust_reverts_visit():
    g = X01(["A"], start=50)
    g.throw(P("S10"))
    g.throw(P("T20"))          # would go below zero
    assert g.scores["A"] == 50
    assert g.turn_over
    assert "BUST" in g.message


def test_double_out_required():
    g = X01(["A"], start=40)
    g.throw(P("T10"))
    g.throw(P("S10"))          # reaches 0 on a single: bust
    assert g.scores["A"] == 40
    assert g.winner is None


def test_leaving_one_is_bust_with_double_out():
    g = X01(["A"], start=21)
    g.throw(P("S20"))
    assert g.scores["A"] == 21


def test_checkout_win():
    g = X01(["A", "B"], start=40)
    g.throw(P("D20"))
    assert g.winner == "A"


def test_bull_checks_out():
    g = X01(["A"], start=50)
    g.throw(P("bull"))
    assert g.winner == "A"


def test_single_out():
    g = X01(["A"], start=20, double_out=False)
    g.throw(P("S20"))
    assert g.winner == "A"


def test_double_in():
    g = X01(["A"], start=501, double_in=True)
    g.throw(P("T20"))
    assert g.scores["A"] == 501
    g.throw(P("D20"))
    g.throw(P("T20"))
    assert g.scores["A"] == 501 - 40 - 60


def test_undo():
    g = X01(["A", "B"])
    g.throw(P("T20"))
    g.throw(P("T19"))
    assert g.undo()
    assert g.scores["A"] == 441
    assert len(g.turn) == 1
    g.next_turn()
    assert g.undo()
    assert g.current == 0


def test_checkout_suggestions():
    assert checkout(170) == ["T20", "T20", "BULL"]
    assert checkout(40, 1) == ["D20"]
    assert checkout(2, 1) == ["D1"]
    assert checkout(169) is None          # famous bogey number
    assert checkout(171) is None
    assert checkout(1) is None
    route = checkout(121)
    assert route and len(route) <= 3
    pts = sum(P(x).points for x in route)
    assert pts == 121 and P(route[-1]).is_double


def test_cricket_marks_and_points():
    g = Cricket(["A", "B"])
    visit(g, "T20", "S20", "S20")          # closes 20, scores 40 (B hasn't closed)
    assert g.marks["A"][20] == 3
    assert g.points["A"] == 40
    visit(g, "T20", "S1", "S1")            # B closes 20; no points for B
    assert g.points["B"] == 0
    visit(g, "S20", "S1", "S1")            # 20 closed by everyone: no more points
    assert g.points["A"] == 40


def test_cricket_bull_and_win():
    g = Cricket(["A", "B"])
    for n in (20, 19, 18, 17, 16, 15):
        visit(g, f"T{n}")
        visit(g, "miss")
    g.throw(P("bull"))
    assert g.winner is None
    g.throw(P("25"))
    assert g.winner == "A"
