"""Synthetic dartboard images: a stand-in for real camera footage.

`VirtualCamera` is a pinhole camera looking at the board from any angle, so we
can test calibration and detection under realistic perspective with exact
ground truth. `render_board` draws a regulation board; `draw_dart` draws a dart
stuck in it as a 3-D object (tip on the board, body sticking out).

Also has overlay helpers used by the UI / CLI to annotate real frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import Calibration, canvas_transform
from .geometry import SEGMENT_ANGLE, SEGMENT_ORDER, STANDARD, polar_to_xy

# BGR colours
BLACK = (30, 30, 30)
CREAM = (175, 215, 230)
RED = (40, 35, 200)
GREEN = (50, 140, 30)
WIRE = (185, 185, 185)
WALL = (120, 125, 130)
SURROUND = (20, 20, 20)


@dataclass
class VirtualCamera:
    """Pinhole camera. Board plane is z=0 (x right, y up, z towards player)."""

    width: int = 1280
    height: int = 720
    distance: float = 900.0     # mm from board centre
    azimuth: float = 0.0        # deg, + = camera to the right of the board
    elevation: float = 0.0      # deg, + = camera above the board centre
    roll: float = 0.0           # deg, camera rotation about its view axis
    focal: float | None = None  # px; default frames the whole board
    target: tuple[float, float] = (0.0, 0.0)
    K: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    t: np.ndarray = field(init=False)

    def __post_init__(self):
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        C = self.distance * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
        tgt = np.array([*self.target, 0.0])
        fwd = tgt - C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])
        rr = math.radians(self.roll)
        Rz = np.array([[math.cos(rr), -math.sin(rr), 0], [math.sin(rr), math.cos(rr), 0], [0, 0, 1]])
        self.R = Rz @ R
        self.t = -self.R @ C
        f = self.focal or 0.85 * min(self.width, self.height) * self.distance / (2 * STANDARD.board_radius)
        self.K = np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1]])

    @property
    def board_to_image(self) -> np.ndarray:
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return H / H[2, 2]

    def ground_truth_calibration(self) -> Calibration:
        return Calibration(H=np.linalg.inv(self.board_to_image), image_size=(self.width, self.height),
                           meta={"method": "ground-truth"})

    def project(self, pts3d) -> np.ndarray:
        p = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        cam = (self.R @ p.T).T + self.t
        uv = (self.K @ cam.T).T
        return uv[:, :2] / uv[:, 2:3]

    def depth(self, pt3d) -> float:
        return float((self.R @ np.asarray(pt3d, dtype=np.float64) + self.t)[2])


def render_board_topdown(px_per_mm: float = 2.0, radius_mm: float = 230.0) -> np.ndarray:
    """Top-down board image; board centre at the image centre, 20 at the top."""
    size = int(2 * radius_mm * px_per_mm)
    c = size / 2
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    bx = (xs + 0.5 - c) / px_per_mm
    by = (c - ys - 0.5) / px_per_mm
    r = np.hypot(bx, by)
    theta = np.degrees(np.arctan2(bx, by)) % 360
    idx = (((theta + SEGMENT_ANGLE / 2) % 360) // SEGMENT_ANGLE).astype(int)
    even = idx % 2 == 0
    d = STANDARD

    img = np.empty((size, size, 3), np.uint8)
    img[:] = WALL
    img[r <= d.board_radius] = SURROUND
    single = r <= d.double_outer
    img[single & even] = BLACK
    img[single & ~even] = CREAM
    ring = ((r > d.triple_inner) & (r <= d.triple_outer)) | ((r > d.double_inner) & (r <= d.double_outer))
    img[ring & even] = RED
    img[ring & ~even] = GREEN
    img[r <= d.outer_bull] = GREEN
    img[r <= d.inner_bull] = RED

    def to_px(x, y):
        return int(round(c + x * px_per_mm)), int(round(c - y * px_per_mm))

    lw = max(1, int(round(px_per_mm * 0.6)))
    for rad in (d.inner_bull, d.outer_bull, d.triple_inner, d.triple_outer, d.double_inner, d.double_outer):
        cv2.circle(img, to_px(0, 0), int(round(rad * px_per_mm)), WIRE, lw, cv2.LINE_AA)
    for k in range(20):
        a = SEGMENT_ANGLE / 2 + k * SEGMENT_ANGLE
        cv2.line(img, to_px(*polar_to_xy(d.outer_bull, a)), to_px(*polar_to_xy(d.double_outer, a)), WIRE, lw, cv2.LINE_AA)
    scale = 0.55 * px_per_mm
    for k, n in enumerate(SEGMENT_ORDER):
        x, y = polar_to_xy(196, k * SEGMENT_ANGLE)
        txt = str(n)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        u, v = to_px(x, y)
        cv2.putText(img, txt, (u - tw // 2, v + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 2, cv2.LINE_AA)
    return img


_TOPDOWN_CACHE: dict = {}


def render_board(cam: VirtualCamera, px_per_mm: float = 3.0, noise: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Board as seen by `cam`."""
    key = px_per_mm
    if key not in _TOPDOWN_CACHE:
        _TOPDOWN_CACHE[key] = render_board_topdown(px_per_mm, 230.0)
    top = _TOPDOWN_CACHE[key]
    M = cam.board_to_image @ np.linalg.inv(canvas_transform(px_per_mm, 230.0))
    img = cv2.warpPerspective(top, M, (cam.width, cam.height), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=WALL)
    if noise > 0:
        rng = rng or np.random.default_rng()
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


@dataclass
class Dart:
    x: float                 # impact point, board mm
    y: float
    tilt_x: float = 0.0      # deg the tail leans right
    tilt_y: float = 10.0     # deg the tail leans up (darts usually sit tail-high)
    flight_color: tuple = (200, 120, 20)

    def direction(self) -> np.ndarray:
        d = np.array([math.tan(math.radians(self.tilt_x)), math.tan(math.radians(self.tilt_y)), 1.0])
        return d / np.linalg.norm(d)


def draw_dart(img: np.ndarray, cam: VirtualCamera, dart: Dart) -> np.ndarray:
    """Draw a dart stuck in the board (painter's order: tip first, flight last)."""
    tip = np.array([dart.x, dart.y, 0.0])
    d = dart.direction()

    def at(s):
        return tip + d * s

    def width_px(s, mm):
        f = cam.K[0, 0]
        return max(1, int(round(mm * f / cam.depth(at(s)))))

    def seg(s0, s1, mm, color):
        a, b = cam.project([at(s0), at(s1)])
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                 color, width_px((s0 + s1) / 2, mm), cv2.LINE_AA)

    seg(0, 28, 2.4, (95, 95, 100))        # steel point
    seg(28, 75, 7.0, (55, 55, 60))        # barrel
    seg(75, 112, 4.0, (30, 30, 30))       # shaft
    n1 = np.cross(d, [0.0, 1.0, 0.0])
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(d, n1)
    for n in (n1, n2):
        quad = [at(108), at(108) + n * 16, at(150) + n * 16, at(150) - n * 16, at(108) - n * 16]
        pts = np.round(cam.project(quad)).astype(np.int32)
        cv2.fillConvexPoly(img, pts, dart.flight_color, cv2.LINE_AA)
    return img


# -- overlays -----------------------------------------------------------------
def draw_board_overlay(img: np.ndarray, cal: Calibration, color=(0, 255, 255), thickness=1) -> np.ndarray:
    """Draw the calibrated ring/segment geometry on top of an image."""
    out = img.copy()
    d = STANDARD
    angles = np.arange(0, 361, 2.0)
    for rad in (d.inner_bull, d.outer_bull, d.triple_inner, d.triple_outer, d.double_inner, d.double_outer):
        pts = cal.board_to_image([polar_to_xy(rad, a) for a in angles])
        cv2.polylines(out, [np.round(pts).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    for k in range(20):
        a = SEGMENT_ANGLE / 2 + k * SEGMENT_ANGLE
        pts = cal.board_to_image([polar_to_xy(d.outer_bull, a), polar_to_xy(d.double_outer, a)])
        cv2.line(out, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)), color, thickness, cv2.LINE_AA)
    for k, n in enumerate(SEGMENT_ORDER):
        u, v = cal.board_to_image([polar_to_xy(190, k * SEGMENT_ANGLE)])[0]
        cv2.putText(out, str(n), (int(u) - 8, int(v) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def draw_hit(img: np.ndarray, uv, label: str, color=(255, 0, 255)) -> None:
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(img, label, (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
