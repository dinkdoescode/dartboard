"""A fake camera: renders a board with darts being thrown and pulled out.

Lets you run the full camera pipeline (calibration -> detection -> scoring ->
game) with no hardware at all, and records ground truth for testing.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import polar_to_xy, segment_center_angle
from ..render import Dart, VirtualCamera, draw_dart, render_board
from ..scoring import Throw, score_xy
from .base import FrameSource

FLIGHT_COLORS = [(200, 120, 20), (30, 180, 230), (180, 40, 160)]
SKIN = (120, 160, 210)


@dataclass
class ThrownDart:
    dart: Dart
    truth: Throw


class SyntheticSource(FrameSource):
    def __init__(self, seed: int = 0, azimuth: float = 15.0, elevation: float = 25.0,
                 roll: float = 0.0, noise: float = 2.0, auto: bool = True,
                 frames_per_dart: int = 15, hand_frames: int = 12,
                 width: int = 1280, height: int = 720, spread_mm: float = 25.0):
        self.name = "synthetic"
        self.rng = np.random.default_rng(seed)
        self.cam = VirtualCamera(width=width, height=height, azimuth=azimuth, elevation=elevation, roll=roll)
        self.noise = noise
        self.auto = auto
        self.frames_per_dart = frames_per_dart
        self.hand_frames = hand_frames
        self.spread = spread_mm
        self.board = render_board(self.cam)
        self.darts: list[ThrownDart] = []
        self.history: list[ThrownDart] = []   # every dart ever thrown (ground truth)
        self._static = self.board.copy()
        self._frame = 0
        self._hand_t: int | None = None

    # -- manual control (tests) ------------------------------------------------
    def throw_at(self, x: float, y: float, tilt_x: float | None = None, tilt_y: float | None = None) -> Throw:
        tx = self.rng.uniform(-15, 15) if tilt_x is None else tilt_x
        ty = self.rng.uniform(-5, 20) if tilt_y is None else tilt_y
        dart = Dart(x, y, tx, ty, FLIGHT_COLORS[len(self.darts) % 3])
        td = ThrownDart(dart, score_xy(x, y))
        self.darts.append(td)
        self.history.append(td)
        self._redraw()
        return td.truth

    def remove_darts(self) -> None:
        self.darts.clear()
        self._redraw()

    def _redraw(self) -> None:
        img = self.board.copy()
        for td in self.darts:
            draw_dart(img, self.cam, td.dart)
        self._static = img

    # -- scripted player -------------------------------------------------------
    def _aim(self) -> tuple[float, float]:
        target = self.rng.choice(["T20", "T20", "T19", "BULL", "D16"])
        if target == "BULL":
            ax, ay = 0.0, 0.0
        else:
            n = int(target[1:])
            r = 103.0 if target[0] == "T" else 166.0
            ax, ay = polar_to_xy(r, segment_center_angle(n))
        return ax + self.rng.normal(0, self.spread), ay + self.rng.normal(0, self.spread)

    def _step_script(self) -> None:
        f = self._frame
        if self._hand_t is not None:
            self._hand_t += 1
            if self._hand_t == self.hand_frames // 2:
                self.remove_darts()
            if self._hand_t >= self.hand_frames:
                self._hand_t = None
                self._frame = 0
            return
        if f > 0 and f % self.frames_per_dart == 0:
            if len(self.darts) < 3:
                self.throw_at(*self._aim())
            else:
                self._hand_t = 0

    def read(self) -> np.ndarray:
        if self.auto:
            self._step_script()
        self._frame += 1
        img = self._static.copy()
        if self._hand_t is not None:
            t = self._hand_t / max(1, self.hand_frames)
            cx = int(self.cam.width * (0.2 + 0.6 * t))
            cv2.ellipse(img, (cx, self.cam.height // 2), (110, 70), 20, 0, 360, SKIN, -1)
        if self.noise > 0:
            n = self.rng.normal(0, self.noise, img.shape[:2])[..., None]
            img = np.clip(img + n, 0, 255).astype(np.uint8)
        return img
