"""Impact sources: the seam between hardware and scoring.

An ImpactSource emits events in BOARD coordinates (mm), never pixels:
    Impact   - a dart landed at (x, y)
    Takeout  - the darts were pulled out of the board (end of turn)

Camera pipeline = FrameSource + Calibration + DartDetector -> CameraImpactSource
Multiple cameras = several CameraImpactSources -> FusedImpactSource
Contact sensors / electronic board = write a new ImpactSource that maps a
sensor id to a segment centre; scoring and games don't change.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import Calibration
from .capture.base import FrameSource
from .detection import DartDetector, Detection, DiffDetector
from .geometry import STANDARD, polar_to_xy


@dataclass
class Impact:
    x: float
    y: float
    confidence: float = 1.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    pixel: tuple[float, float] | None = None


@dataclass
class Takeout:
    source: str = ""
    timestamp: float = field(default_factory=time.time)


Event = Impact | Takeout


class ImpactSource(ABC):
    name: str = "impacts"

    @abstractmethod
    def poll(self) -> list[Event]:
        """Non-blocking: return any events since the last call."""

    def close(self) -> None:
        pass


class ManualImpactSource(ImpactSource):
    """Thread-safe queue for clicks / keyboard input / tests."""

    def __init__(self, name: str = "manual"):
        self.name = name
        self._q: deque[Event] = deque()
        self._lock = threading.Lock()

    def push(self, event: Event) -> None:
        with self._lock:
            self._q.append(event)

    def poll(self):
        with self._lock:
            out = list(self._q)
            self._q.clear()
        return out


class CameraImpactSource(ImpactSource):
    """Watches a camera; emits an Impact when a new dart appears and settles,
    and a Takeout when the board returns to its empty state.

    State machine per frame:
      * moving (big frame-to-frame change: dart in flight, hand)  -> wait
      * settled for `settle_frames`:
          - looks like the empty board again and darts were present -> Takeout
          - differs from the last settled frame                     -> detect dart
    """

    def __init__(self, frames: FrameSource, calibration: Calibration,
                 detector: DartDetector | None = None, settle_frames: int = 3,
                 motion_frac: float = 0.0015, change_frac: float = 0.0004,
                 min_confidence: float = 0.0):
        self.frames = frames
        self.cal = calibration
        self.detector = detector or DiffDetector()
        self.name = f"camera[{frames.name}]"
        self.settle_frames = settle_frames
        self.motion_frac = motion_frac
        self.change_frac = change_frac
        self.min_confidence = min_confidence
        self._roi: np.ndarray | None = None
        self._prev = None
        self._reference = None      # last settled frame (with current darts)
        self._empty = None          # last settled frame with no darts
        self._stable = 0
        self.darts_on_board = 0
        self.last_frame: np.ndarray | None = None
        self.last_detection: Detection | None = None
        self.hits: list[tuple[float, float]] = []   # pixel positions this turn

    def roi(self, shape) -> np.ndarray:
        if self._roi is None or self._roi.shape != shape[:2]:
            pts = self.cal.board_to_image([polar_to_xy(STANDARD.board_radius, a) for a in range(0, 360, 3)])
            self._roi = np.zeros(shape[:2], np.uint8)
            cv2.fillPoly(self._roi, [np.round(pts).astype(np.int32)], 255)
        return self._roi

    def _changed(self, a, b) -> float:
        d = cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0)).max(axis=2)
        m = (d > 30) & (self.roi(a.shape) > 0)
        return float(m.sum()) / max(1, int((self.roi(a.shape) > 0).sum()))

    def poll(self):
        frame = self.frames.read()
        if frame is None:
            return []
        self.last_frame = frame
        if self._reference is None:
            self._reference = self._empty = self._prev = frame
            return []
        motion = self._changed(frame, self._prev)
        self._prev = frame
        if motion > self.motion_frac:
            self._stable = 0
            return []
        self._stable += 1
        if self._stable != self.settle_frames:
            return []
        # Just settled: what changed since the last settled state?
        if self._changed(frame, self._reference) < self.change_frac:
            return []
        if self.darts_on_board and self._changed(frame, self._empty) < self.change_frac:
            self._reference = self._empty = frame
            self.darts_on_board = 0
            self.hits.clear()
            return [Takeout(self.name)]
        det = self.detector.detect(self._reference, frame, self.roi(frame.shape))
        self._reference = frame
        self.last_detection = det
        if det is None or det.confidence < self.min_confidence:
            return []
        x, y = self.cal.image_to_board([det.tip])[0]
        self.darts_on_board += 1
        self.hits.append(det.tip)
        return [Impact(float(x), float(y), det.confidence, self.name, pixel=det.tip)]

    def close(self):
        self.frames.close()


class FusedImpactSource(ImpactSource):
    """Combines several cameras looking at the same board.

    Impacts reported by different sources within `window_s` seconds are treated
    as the same dart and averaged (confidence-weighted). Takeouts are forwarded
    once. A smarter fusion (e.g. pick the camera with the best view of that
    board region) can replace `_fuse` without touching anything else.
    """

    def __init__(self, sources: list[ImpactSource], window_s: float = 0.4):
        self.sources = sources
        self.window = window_s
        self.name = "fused"
        self._pending: list[Impact] = []
        self._last_takeout = 0.0

    def poll(self):
        out: list[Event] = []
        now = time.time()
        for src in self.sources:
            for ev in src.poll():
                if isinstance(ev, Takeout):
                    if now - self._last_takeout > 2.0:
                        self._last_takeout = now
                        out.append(Takeout(self.name))
                else:
                    self._pending.append(ev)
        if self._pending and now - self._pending[0].timestamp >= self.window:
            group = [i for i in self._pending if i.timestamp - self._pending[0].timestamp < self.window]
            self._pending = self._pending[len(group):]
            out.append(self._fuse(group))
        return out

    @staticmethod
    def _fuse(group: list[Impact]) -> Impact:
        w = np.array([max(i.confidence, 1e-3) for i in group])
        x = float(np.average([i.x for i in group], weights=w))
        y = float(np.average([i.y for i in group], weights=w))
        return Impact(x, y, float(w.max()), "fused")

    def close(self):
        for s in self.sources:
            s.close()
