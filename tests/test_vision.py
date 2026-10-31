"""Calibration, board finding and dart detection against synthetic images
with exact ground truth."""
import numpy as np
import pytest

from dartscore.board_finder import find_board
from dartscore.calibration import Calibration
from dartscore.capture.synthetic import SyntheticSource
from dartscore.detection import DiffDetector
from dartscore.geometry import CALIBRATION_POINTS, STANDARD, polar_to_xy
from dartscore.render import VirtualCamera, render_board
from dartscore.scoring import score_xy

VIEWS = [(0, 0, 0), (25, 10, 0), (-30, 20, 5), (10, -35, -8), (40, 30, 12)]
PROBE = [polar_to_xy(r, a) for r in (10, 60, 103, 140, 166) for a in range(0, 360, 15)]


def test_calibration_roundtrip(tmp_path):
    cam = VirtualCamera(azimuth=20, elevation=15)
    cal = cam.ground_truth_calibration()
    img = cal.board_to_image(PROBE)
    assert np.allclose(cal.image_to_board(img), PROBE, atol=1e-6)
    cal.save(tmp_path / "c.json")
    cal2 = Calibration.load(tmp_path / "c.json")
    assert np.allclose(cal2.H, cal.H)


def test_four_click_calibration_is_exact():
    cam = VirtualCamera(azimuth=-25, elevation=30, roll=10)
    gt = cam.ground_truth_calibration()
    clicks = {n: tuple(gt.board_to_image([p])[0]) for n, p in CALIBRATION_POINTS.items()}
    cal = Calibration.from_reference_clicks(clicks)
    err = np.linalg.norm(cal.image_to_board(gt.board_to_image(PROBE)) - PROBE, axis=1)
    assert err.max() < 0.01


def test_click_noise_tolerance():
    """Clicks off by ~1.5 px still give roughly 1-2 mm accuracy."""
    rng = np.random.default_rng(0)
    cam = VirtualCamera(azimuth=15, elevation=20)
    gt = cam.ground_truth_calibration()
    clicks = {n: tuple(gt.board_to_image([p])[0] + rng.normal(0, 1.5, 2)) for n, p in CALIBRATION_POINTS.items()}
    cal = Calibration.from_reference_clicks(clicks)
    err = np.linalg.norm(cal.image_to_board(gt.board_to_image(PROBE)) - PROBE, axis=1)
    assert np.median(err) < 3.0


@pytest.mark.parametrize("az,el,roll", VIEWS)
def test_auto_calibration_accuracy(az, el, roll):
    cam = VirtualCamera(azimuth=az, elevation=el, roll=roll)
    img = render_board(cam, noise=3, rng=np.random.default_rng(1))
    res = find_board(img)
    img_pts = cam.ground_truth_calibration().board_to_image(PROBE)
    err = np.linalg.norm(res.calibration.image_to_board(img_pts) - PROBE, axis=1)
    assert err.max() < 1.5, f"max error {err.max():.2f} mm"


@pytest.mark.parametrize("az,el", [(15, 25), (-30, 35), (35, -20)])
def test_dart_detection_accuracy(az, el):
    src = SyntheticSource(seed=2, azimuth=az, elevation=el, auto=False)
    cal = src.cam.ground_truth_calibration()
    det = DiffDetector()
    rng = np.random.default_rng(3)
    errors = []
    for _ in range(24):
        if len(src.darts) == 3:
            src.remove_darts()
        before = src.read()
        r, a = rng.uniform(0, 170), rng.uniform(0, 360)
        x, y = polar_to_xy(r, a)
        src.throw_at(x, y)
        d = det.detect(before, src.read())
        assert d is not None
        bx, by = cal.image_to_board([d.tip])[0]
        errors.append(np.hypot(bx - x, by - y))
    errors = np.array(errors)
    assert np.median(errors) < 3.0
    assert np.mean(errors < 6.0) >= 0.9   # occasional occlusion by earlier darts


def test_detection_scores_clear_targets():
    """Darts in the middle of a bed are scored correctly end-to-end."""
    src = SyntheticSource(seed=4, azimuth=20, elevation=25, auto=False)
    cal = find_board(src.read()).calibration
    det = DiffDetector()
    for target, (r, a) in {"T20": (103, 0), "D16": (166, 234), "S5": (60, 342), "25": (11, 120)}.items():
        src.remove_darts()
        before = src.read()
        src.throw_at(*polar_to_xy(r, a), tilt_x=5, tilt_y=10)
        tip = det.detect(before, src.read()).tip
        assert score_xy(*cal.image_to_board([tip])[0]).label == target
