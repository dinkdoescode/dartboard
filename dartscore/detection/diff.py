"""Frame-difference dart detector.

1. Difference the frame before and after the throw -> blob of the new dart.
2. Principal axis of the blob = the dart's direction in the image.
3. The tip is the NARROW end (the flight end is wide); the impact point is the
   extreme pixel at that end.

This is the classic single-camera approach. Its weakness is a camera looking
along the dart (the body hides the tip); that's what side-mounted cameras and
multi-camera fusion fix later.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import DartDetector, Detection


class DiffDetector(DartDetector):
    def __init__(self, threshold: int = 25, min_area: int = 80, max_area_frac: float = 0.08,
                 blur: int = 5, group_px: int = 25):
        self.threshold = threshold
        self.group_px = group_px
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.blur = blur

    def diff_mask(self, before: np.ndarray, after: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        a = cv2.GaussianBlur(_gray(before), (self.blur, self.blur), 0)
        b = cv2.GaussianBlur(_gray(after), (self.blur, self.blur), 0)
        # Colour difference catches darts that match the board's brightness.
        d = cv2.absdiff(a, b)
        if before.ndim == 3:
            dc = cv2.absdiff(cv2.GaussianBlur(before, (self.blur, self.blur), 0),
                             cv2.GaussianBlur(after, (self.blur, self.blur), 0)).max(axis=2)
            d = np.maximum(d, dc)
        mask = (d > self.threshold).astype(np.uint8) * 255
        if roi is not None:
            mask &= roi
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        return mask

    def detect(self, before, after, roi=None):
        mask = self.diff_mask(before, after, roi)
        # A dart often shows up as several fragments (dark shaft over a dark
        # segment differs little), so group fragments that lie close together.
        grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.group_px, self.group_px)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped)
        if n <= 1:
            return None
        sizes = [int(np.count_nonzero(mask[labels == j])) for j in range(1, n)]
        i = 1 + int(np.argmax(sizes))
        blob = (labels == i) & (mask > 0)
        area = int(blob.sum())
        if area < self.min_area or area > self.max_area_frac * mask.size:
            return None
        vs, us = np.nonzero(blob)
        pts = np.column_stack([us, vs]).astype(np.float64)
        tip, elongation = _narrow_end(pts)
        contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Heuristic only (used to weight multi-camera fusion): long thin blobs
        # are unambiguous darts, round blobs (flight facing the camera) less so.
        conf = float(np.clip((elongation - 1.0) / 3.0, 0.05, 1.0))
        return Detection(tip=tip, confidence=conf, area=area, contour=contours[0] if contours else None)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _narrow_end(pts: np.ndarray) -> tuple[tuple[float, float], float]:
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    axis, perp = evecs[:, 1], evecs[:, 0]
    t = centered @ axis
    s = centered @ perp
    lo, hi = t.min(), t.max()
    length = hi - lo
    elongation = float(np.sqrt(evals[1] / max(evals[0], 1e-6)))

    def width(sel):
        if sel.sum() < 3:
            return 0.0
        lo_s, hi_s = np.percentile(s[sel], [5, 95])
        return hi_s - lo_s

    w_lo = width(t < lo + 0.25 * length)
    w_hi = width(t > hi - 0.25 * length)
    if w_lo <= w_hi:
        end_sel, extreme = t < lo + 0.03 * length + 1, lo
    else:
        end_sel, extreme = t > hi - 0.03 * length - 1, hi
    s_tip = float(np.median(s[end_sel]))
    tip = mean + axis * extreme + perp * s_tip
    return (float(tip[0]), float(tip[1])), elongation
