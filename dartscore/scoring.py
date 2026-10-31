"""Scoring engine: board position -> Throw.

Pure functions with no OpenCV / hardware dependency, so the maths can be
tested on its own and reused by any input source.
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import (
    SEGMENT_ORDER, STANDARD, BoardDimensions, polar_to_xy, segment_index,
    xy_to_polar,
)

BULL = 25  # "segment" number used for both bull rings


@dataclass(frozen=True)
class Throw:
    segment: int      # 1-20, 25 for bull, 0 for a miss
    multiplier: int   # 0 (miss), 1, 2 or 3
    x: float | None = None  # board position in mm, if known
    y: float | None = None

    @property
    def points(self) -> int:
        return self.segment * self.multiplier

    @property
    def is_double(self) -> bool:
        return self.multiplier == 2  # includes the bullseye (D25 = 50)

    @property
    def label(self) -> str:
        if self.multiplier == 0:
            return "MISS"
        if self.segment == BULL:
            return "BULL" if self.multiplier == 2 else "25"
        return {1: "S", 2: "D", 3: "T"}[self.multiplier] + str(self.segment)

    def to_dict(self) -> dict:
        return {
            "label": self.label, "segment": self.segment,
            "multiplier": self.multiplier, "points": self.points,
            "x": self.x, "y": self.y,
        }


MISS = Throw(0, 0)


def score_xy(x: float, y: float, dims: BoardDimensions = STANDARD) -> Throw:
    """Score a point given in board coordinates (mm, y up)."""
    r, theta = xy_to_polar(x, y)
    seg, mult = _classify(r, theta, dims)
    return Throw(seg, mult, x, y)


def score_polar(r: float, theta_deg: float, dims: BoardDimensions = STANDARD) -> Throw:
    """Score a point given as radius (mm) and angle (deg clockwise from up)."""
    x, y = polar_to_xy(r, theta_deg)
    seg, mult = _classify(r, theta_deg % 360.0, dims)
    return Throw(seg, mult, x, y)


def _classify(r: float, theta: float, d: BoardDimensions) -> tuple[int, int]:
    if r <= d.inner_bull:
        return BULL, 2
    if r <= d.outer_bull:
        return BULL, 1
    if r > d.double_outer:
        return 0, 0
    number = SEGMENT_ORDER[segment_index(theta)]
    if d.triple_inner < r <= d.triple_outer:
        return number, 3
    if d.double_inner < r <= d.double_outer:
        return number, 2
    return number, 1


def parse_throw(text: str) -> Throw:
    """Parse manual input like 'T20', 'd16', '5', 'S5', 'bull', '25', 'miss'."""
    t = text.strip().upper()
    if t in ("MISS", "M", "0", "X"):
        return MISS
    if t in ("BULL", "DB", "D25", "50", "BE"):
        return Throw(BULL, 2)
    if t in ("25", "SB", "S25", "OB"):
        return Throw(BULL, 1)
    mult = 1
    if t[0] in "SDT":
        mult = "SDT".index(t[0]) + 1
        t = t[1:]
    n = int(t)
    if n not in SEGMENT_ORDER:
        raise ValueError(f"not a dartboard number: {text!r}")
    return Throw(n, mult)
