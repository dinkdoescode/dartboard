"""Dart-detector abstraction: find the tip of a newly arrived dart in an image.

Swap in a different strategy (ML keypoint model, IR blob tracking, ...) by
implementing `DartDetector`; the rest of the pipeline stays unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    tip: tuple[float, float]        # image pixel (u, v) of the impact point
    confidence: float               # 0..1, heuristic
    area: int = 0                   # blob size in pixels
    contour: np.ndarray | None = None


class DartDetector(ABC):
    @abstractmethod
    def detect(self, before: np.ndarray, after: np.ndarray, roi: np.ndarray | None = None) -> Detection | None:
        """Compare a frame before and after a throw; return the new dart's tip."""
