"""Standard dartboard geometry (WDF/BDO regulation dimensions).

Coordinate convention used throughout the project ("board coordinates"):
    * units are millimetres
    * origin is the centre of the bullseye
    * +x points right, +y points UP (towards the 20 segment)
    * angles (theta) are measured in degrees CLOCKWISE from +y (12 o'clock),
      so the centre of the 20 segment is theta=0 and the centre of 6 is theta=90.

Every input source (camera, multiple cameras, contact sensors, mouse clicks)
ultimately produces a point in this frame, which is what the scoring engine
consumes. That is the seam that keeps hardware and game logic independent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Segment numbers clockwise starting at the top.
SEGMENT_ORDER: tuple[int, ...] = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
)
SEGMENT_ANGLE = 360.0 / len(SEGMENT_ORDER)  # 18 degrees


@dataclass(frozen=True)
class BoardDimensions:
    """Radii in millimetres, measured from the board centre to the wire."""

    inner_bull: float = 6.35       # double bull / "bullseye" (50)
    outer_bull: float = 15.9       # single bull (25)
    triple_inner: float = 99.0
    triple_outer: float = 107.0
    double_inner: float = 162.0
    double_outer: float = 170.0    # edge of the scoring area
    board_radius: float = 225.5    # physical board edge (451 mm diameter)


STANDARD = BoardDimensions()


def polar_to_xy(r: float, theta_deg: float) -> tuple[float, float]:
    """Board polar (r, theta clockwise-from-up) -> board (x, y)."""
    t = math.radians(theta_deg)
    return r * math.sin(t), r * math.cos(t)


def xy_to_polar(x: float, y: float) -> tuple[float, float]:
    """Board (x, y) -> (r, theta) with theta in [0, 360) clockwise from up."""
    r = math.hypot(x, y)
    theta = math.degrees(math.atan2(x, y)) % 360.0
    return r, theta


def segment_index(theta_deg: float) -> int:
    """Index into SEGMENT_ORDER for an angle (clockwise from up)."""
    return int(((theta_deg + SEGMENT_ANGLE / 2) % 360.0) // SEGMENT_ANGLE)


def segment_center_angle(number: int) -> float:
    """Angle of the centre line of a numbered segment."""
    return SEGMENT_ORDER.index(number) * SEGMENT_ANGLE


# Calibration reference points: where the outer double wire meets the segment
# wires at four well-spread positions. They are easy to click on a real board.
CALIBRATION_POINTS: dict[str, tuple[float, float]] = {
    "20/1":  polar_to_xy(STANDARD.double_outer, 9.0),
    "6/10":  polar_to_xy(STANDARD.double_outer, 99.0),
    "3/19":  polar_to_xy(STANDARD.double_outer, 189.0),
    "11/14": polar_to_xy(STANDARD.double_outer, 279.0),
}
