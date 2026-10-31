"""dartscore: automatic dartboard scoring.

Layers (each swappable):
    capture     -> frames from a camera / file / simulator
    detection   -> dart tip in pixels
    calibration -> pixels <-> board millimetres
    impacts     -> board-coordinate events (the hardware seam)
    scoring     -> board position -> Throw (S/D/T, bull, miss)
    game        -> X01, Cricket
    session/ui  -> glue + web dashboard
"""
from .scoring import Throw, parse_throw, score_polar, score_xy

__all__ = ["Throw", "parse_throw", "score_polar", "score_xy"]
