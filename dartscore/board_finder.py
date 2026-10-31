"""Automatic board detection / calibration from a single image of the board.

Pipeline:
  1. Segment the red/green ring colours (HSV thresholds).
  2. Fit an ellipse to the outer hull -> affine first guess (circle -> ellipse).
  3. Rectify, then find the rotation from the red/green alternation of the
     double ring (red on even segments: 20, 18, 13, ...). The pattern repeats
     every 36 deg, so we pick the candidate that puts 20 nearest the top of the
     image -- i.e. we assume the board hangs roughly upright (+/-18 deg).
  4. Refine to a full perspective homography from ~150 ring-edge and
     segment-boundary correspondences (RANSAC), iterating a few times.

Works well on clean images (the synthetic renderer, good photos). For awkward
lighting or boards whose 20 is not near the top, use manual 4-click
calibration instead (see `python -m dartscore calibrate`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .calibration import Calibration, canvas_transform
from .geometry import SEGMENT_ANGLE, STANDARD, polar_to_xy

PX_PER_MM = 2.0
CANVAS_R = 200.0
RED, GREEN = 1, 2


class BoardNotFound(RuntimeError):
    pass


@dataclass
class FinderResult:
    calibration: Calibration
    ellipse: tuple
    correspondences: int
    residual_mm: float


def color_labels(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel label: 0 = other, 1 = red, 2 = green."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 90) & (v > 60)
    red = sat & ((h <= 10) | (h >= 165))
    green = sat & (h >= 40) & (h <= 95)
    out = np.zeros(h.shape, np.uint8)
    out[red] = RED
    out[green] = GREEN
    return out


def find_board(bgr: np.ndarray, iterations: int = 4) -> FinderResult:
    labels = color_labels(bgr)
    ellipse = _fit_outer_ellipse(labels)
    H = _ellipse_homography(ellipse)                 # image -> pseudo-board
    H = _fix_rotation(labels, H)
    n, resid = 0, float("nan")
    # Double ring + bullseye alone pin down the homography (a circle, its
    # centre and the segment angles). Bring in the triple ring once the
    # estimate is close enough for its features to be matched reliably.
    for i in range(iterations):
        H, n, resid = _refine(labels, H, use_triple=i >= iterations // 2)
    cal = Calibration(H=H, image_size=(bgr.shape[1], bgr.shape[0]),
                      meta={"method": "auto", "correspondences": n, "residual_mm": resid})
    return FinderResult(cal, ellipse, n, resid)


# --------------------------------------------------------------------------
def _fit_outer_ellipse(labels: np.ndarray):
    mask = (labels > 0).astype(np.uint8) * 255
    k = max(3, int(min(mask.shape) / 100)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise BoardNotFound("no red/green regions found")
    # The double ring is the outermost coloured structure; merge the big blobs
    # (wires can split it) and take the convex hull.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    big = [c for c in contours if cv2.contourArea(c) > 0.02 * cv2.contourArea(contours[0])]
    hull = cv2.convexHull(np.vstack(big))
    if len(hull) < 5 or cv2.contourArea(hull) < 0.005 * mask.size:
        raise BoardNotFound("coloured region too small to be a dartboard")
    return cv2.fitEllipse(hull)


def _ellipse_homography(ellipse) -> np.ndarray:
    (cx, cy), (w, h), ang = ellipse
    a = math.radians(ang)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    S = np.diag([w / 2 / STANDARD.double_outer, h / 2 / STANDARD.double_outer, 1.0])
    F = np.diag([1.0, -1.0, 1.0])                       # board y-up -> image y-down
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    board_to_image = T @ R @ S @ F
    return np.linalg.inv(board_to_image)


def _rot_cw(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _rectified_labels(labels: np.ndarray, H: np.ndarray) -> np.ndarray:
    size = int(2 * CANVAS_R * PX_PER_MM)
    M = canvas_transform(PX_PER_MM, CANVAS_R) @ H
    return cv2.warpPerspective(labels, M, (size, size), flags=cv2.INTER_NEAREST)


def _sample(rect: np.ndarray, r, theta) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    t = np.radians(theta)
    c = CANVAS_R * PX_PER_MM
    u = np.round(c + r * np.sin(t) * PX_PER_MM).astype(int)
    v = np.round(c - r * np.cos(t) * PX_PER_MM).astype(int)
    ok = (u >= 0) & (v >= 0) & (u < rect.shape[1]) & (v < rect.shape[0])
    out = np.zeros(np.broadcast(u, v).shape, np.uint8)
    out[ok] = rect[v[ok], u[ok]]
    return out


def _expected_color(theta_true: np.ndarray) -> np.ndarray:
    idx = (((theta_true + SEGMENT_ANGLE / 2) % 360) // SEGMENT_ANGLE).astype(int)
    return np.where(idx % 2 == 0, RED, GREEN)


def _fix_rotation(labels: np.ndarray, H: np.ndarray) -> np.ndarray:
    rect = _rectified_labels(labels, H)
    thetas = np.arange(0, 360, 0.5)
    obs = np.concatenate([_sample(rect, 166.0, thetas), _sample(rect, 103.0, thetas)])
    th2 = np.concatenate([thetas, thetas])
    valid = obs > 0
    if valid.sum() < 100:
        raise BoardNotFound("could not see the double/triple rings")
    best_phi, best_score = 0.0, -1
    for phi in np.arange(0, 36, 0.5):
        score = np.sum(obs[valid] == _expected_color(th2[valid] - phi))
        if score > best_score:
            best_phi, best_score = phi, score
    # Ten candidates fit the colour pattern; choose the one with 20 at the top.
    best_H, best_up = H, -2.0
    for k in range(10):
        Hk = _rot_cw(-(best_phi + 36 * k)) @ H
        inv = np.linalg.inv(Hk)
        c, top = _project(inv, [(0, 0), (0, 150)])
        d = top - c
        up = -d[1] / (np.linalg.norm(d) + 1e-9)
        if up > best_up:
            best_H, best_up = Hk, up
    return best_H


def _project(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def _refine(labels: np.ndarray, H: np.ndarray, use_triple: bool = True):
    """One refinement step. Returns (H, n_features, median_residual_mm).

    Features are measured in the current rectified estimate, then we solve for
    a correcting homography G (H_new = G @ H) with robust least squares:
      * ring-edge points must land on their circle (radial residual only --
        we do NOT know their true angle, and pinning it biases the fit),
      * red/green boundary points must land on their boundary angle,
      * the bullseye must land on the origin.
    """
    rect = _rectified_labels(labels, H)
    d = STANDARD

    # The bullseye pins the true centre: under perspective the centre of the
    # outer ellipse is NOT the board centre.
    bull = _find_bull(rect)

    edge_pts, edge_r = [], []
    rs = np.arange(80.0, 190.0, 0.25)
    rings = [(d.double_inner, d.double_outer)]
    if use_triple:
        rings.append((d.triple_inner, d.triple_outer))
    for theta in np.arange(0, 360, 3.0):
        col = _sample(rect, rs, np.full_like(rs, theta)) > 0
        for lo, hi in rings:
            run = _best_run(rs, col, lo, hi)
            if run is None:
                continue
            for r_meas, r_true in zip(run, (lo, hi)):
                edge_pts.append(polar_to_xy(r_meas, theta))
                edge_r.append(r_true)

    bnd_pts, bnd_ang = [], []
    for r in ((103.0, 166.0) if use_triple else (166.0,)):
        for k in range(20):
            b = SEGMENT_ANGLE / 2 + k * SEGMENT_ANGLE
            ts = np.arange(b - 7, b + 7, 0.1)
            lab = _sample(rect, r, ts)
            left, right = (RED, GREEN) if k % 2 == 0 else (GREEN, RED)
            est = _split_point(ts, lab, left, right)
            if est is not None and abs(est - b) < 6:
                bnd_pts.append(polar_to_xy(r, est))
                bnd_ang.append(b)

    if len(edge_pts) < 8 or len(bnd_pts) < 4:
        raise BoardNotFound("not enough ring features for refinement")
    E = np.asarray(edge_pts)
    Er = np.asarray(edge_r)
    B = np.asarray(bnd_pts)
    Ba = np.radians(bnd_ang)

    def apply(G, P):
        q = np.column_stack([P, np.ones(len(P))]) @ G.T
        return q[:, :2] / q[:, 2:3]

    def G_of(x):
        return np.array([[1 + x[0], x[1], x[2]], [x[3], 1 + x[4], x[5]], [x[6] * 1e-3, x[7] * 1e-3, 1.0]])

    def residuals(x):
        G = G_of(x)
        e = apply(G, E)
        res = [np.hypot(e[:, 0], e[:, 1]) - Er]
        b = apply(G, B)
        ang = np.arctan2(b[:, 0], b[:, 1])
        dang = (ang - Ba + np.pi) % (2 * np.pi) - np.pi
        res.append(dang * np.hypot(b[:, 0], b[:, 1]))
        if bull is not None:
            res.append(5.0 * apply(G, np.array([bull]))[0])
        return np.concatenate(res)

    sol = least_squares(residuals, np.zeros(8), loss="soft_l1", f_scale=1.0)
    G = G_of(sol.x)
    n = len(E) + len(B) + (bull is not None)
    return G @ H, n, float(np.median(np.abs(residuals(sol.x))))


def _find_bull(rect: np.ndarray, search_mm: float = 45.0):
    """Centroid (board mm, current estimate) of the red inner bull, if visible."""
    c = CANVAS_R * PX_PER_MM
    rpx = int(search_mm * PX_PER_MM)
    win = rect[int(c) - rpx:int(c) + rpx, int(c) - rpx:int(c) + rpx]
    red = (win == RED).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(red)
    best, best_d = None, 1e9
    for i in range(1, n):
        area_mm2 = stats[i, cv2.CC_STAT_AREA] / PX_PER_MM ** 2
        if not 40 < area_mm2 < 400:   # inner bull is ~127 mm^2
            continue
        u, v = cents[i]
        x = (u + int(c) - rpx - c) / PX_PER_MM
        y = (c - (v + int(c) - rpx)) / PX_PER_MM
        dist = math.hypot(x, y)
        if dist < best_d:
            best, best_d = (x, y), dist
    return best


def _best_run(rs, col, lo, hi, tol=8.0):
    """Coloured run along a ray best matching [lo, hi]; returns (start, end)."""
    best, best_overlap = None, 0.0
    i, n = 0, len(col)
    while i < n:
        if not col[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and col[j + 1]:
            j += 1
        a, b = rs[i], rs[j]
        overlap = min(b, hi) - max(a, lo)
        if overlap > best_overlap and abs(a - lo) < tol and abs(b - hi) < tol:
            best, best_overlap = (a, b + (rs[1] - rs[0])), overlap
        i = j + 1
    return best


def _split_point(ts, lab, left, right):
    """Angle that best separates `left`-coloured from `right`-coloured samples."""
    is_l, is_r = lab == left, lab == right
    if is_l.sum() < 5 or is_r.sum() < 5:
        return None
    # errors(t_i) = #right-coloured before i + #left-coloured from i onward
    err = np.concatenate([[0], np.cumsum(is_r)])[:-1] + (is_l.sum() - np.concatenate([[0], np.cumsum(is_l)])[:-1])
    i = int(np.argmin(err))
    return float(ts[i])
