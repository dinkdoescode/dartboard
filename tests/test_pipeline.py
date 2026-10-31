"""Full pipeline with a simulated camera: frames -> impacts -> game."""
import time

from dartscore.capture.synthetic import SyntheticSource
from dartscore.game import X01
from dartscore.geometry import polar_to_xy
from dartscore.impacts import CameraImpactSource, FusedImpactSource, Impact, ManualImpactSource, Takeout
from dartscore.session import Session


def pump(source, session, frames):
    for _ in range(frames):
        for ev in source.poll():
            session.handle(ev)


def test_camera_pipeline_scores_a_visit_and_takeout():
    sim = SyntheticSource(seed=5, azimuth=15, elevation=25, auto=False)
    cam = CameraImpactSource(sim, sim.cam.ground_truth_calibration())
    session = Session(X01(["A", "B"]))
    pump(cam, session, 3)                    # learn the empty board
    for r, a in [(103, 0), (103, 0.5 * 18 * 0.4), (166, 18 * 5)]:   # T20, T20, D6
        sim.throw_at(*polar_to_xy(r, a), tilt_x=0, tilt_y=8)
        pump(cam, session, 6)
    assert [t["label"] for t in session.state()["turn"]] == ["T20", "T20", "D6"]
    assert session.game.scores["A"] == 501 - 132
    sim.remove_darts()
    pump(cam, session, 6)
    assert session.game.current == 1         # takeout ended the visit


def test_scripted_simulator_runs_a_game():
    sim = SyntheticSource(seed=6, auto=True, frames_per_dart=8, hand_frames=8)
    cam = CameraImpactSource(sim, sim.cam.ground_truth_calibration())
    session = Session(X01(["A", "B"]))
    pump(cam, session, 8 * 4 * 3 + 40)       # ~3 visits
    thrown = len(sim.history)
    scored = sum(len(v["darts"]) for v in session.game.log) + len(session.game.turn)
    assert thrown >= 6
    assert scored >= thrown - 1              # allow one unsettled at the end
    assert len(session.game.log) >= 2


def test_manual_source_and_session():
    src = ManualImpactSource()
    session = Session(X01(["A"]))
    src.push(Impact(0, 103))
    src.push(Impact(0, 0))
    src.push(Takeout())
    pump(src, session, 1)
    assert session.game.scores["A"] == 501 - 110


def test_fusion_averages_cameras():
    a, b = ManualImpactSource("a"), ManualImpactSource("b")
    fused = FusedImpactSource([a, b], window_s=0.05)
    a.push(Impact(0, 100, confidence=1.0))
    b.push(Impact(0, 106, confidence=1.0))
    out = fused.poll()
    time.sleep(0.06)
    out += fused.poll()
    assert len(out) == 1
    assert abs(out[0].y - 103) < 1e-6
