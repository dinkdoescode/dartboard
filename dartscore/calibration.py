"""Calibration: mapping between image pixels and board coordinates (mm).

A dartboard is a flat plane, so from any single camera viewpoint (including
oblique angles) the pixel <-> board mapping is exactly a 3x3 homography,
ignoring lens distortion. Four known reference points are enough; more points
make it robust (least squares / RANSAC).

Each camera owns one Calibration. Swapping hardware means producing a new
calibration file, not touching the scoring or game code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import CALIBRATION_POINTS


@dataclass
class Calibration:
    # Homography that maps image pixels (u, v) -> board millimetres (x, y up).
    H: np.ndarray
    image_size: tuple[int, int] | None = None  # (width, height)
    meta: dict = field(default_factory=dict)

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_points(cls, image_pts, board_pts, ransac: bool = False, **meta) -> "Calibration":
        src = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(board_pts, dtype=np.float64).reshape(-1, 2)
        if len(src) < 4:
            raise ValueError("need at least 4 point correspondences")
        method = cv2.RANSAC if ransac else 0
        H, _ = cv2.findHomography(src, dst, method, 2.0)
        if H is None:
            raise ValueError("could not compute homography (degenerate points?)")
        return cls(H=H, meta=meta)

    @classmethod
    def from_reference_clicks(cls, clicks: dict[str, tuple[float, float]], **meta) -> "Calibration":
        """Build from user clicks on the named CALIBRATION_POINTS."""
        names = [n for n in CALIBRATION_POINTS if n in clicks]
        return cls.from_points(
            [clicks[n] for n in names], [CALIBRATION_POINTS[n] for n in names],
            method="manual", **meta,
        )

    # -- mapping ----------------------------------------------------------
    def image_to_board(self, pts) -> np.ndarray:
        return _apply(self.H, pts)

    def board_to_image(self, pts) -> np.ndarray:
        return _apply(self.H_inv, pts)

    def board_scale_at(self, u: float, v: float) -> float:
        """Approximate mm-per-pixel near an image location."""
        p = self.image_to_board([[u, v], [u + 1, v], [u, v + 1]])
        return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2)

    def rectify(self, image: np.ndarray, px_per_mm: float = 2.0, radius_mm: float = 230.0) -> np.ndarray:
        """Warp the image into a top-down board view (centre in the middle)."""
        size = int(2 * radius_mm * px_per_mm)
        board_to_canvas = canvas_transform(px_per_mm, radius_mm)
        M = board_to_canvas @ self.H
        return cv2.warpPerspective(image, M, (size, size))

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {"H": self.H.tolist(), "image_size": self.image_size, "meta": self.meta}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        size = tuple(data["image_size"]) if data.get("image_size") else None
        return cls(H=np.array(data["H"], dtype=np.float64), image_size=size, meta=data.get("meta", {}))


def canvas_transform(px_per_mm: float, radius_mm: float) -> np.ndarray:
    """Board mm (y up, centred) -> top-down canvas pixels (y down)."""
    c = radius_mm * px_per_mm
    return np.array([[px_per_mm, 0, c], [0, -px_per_mm, c], [0, 0, 1]], dtype=np.float64)


def _apply(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)
