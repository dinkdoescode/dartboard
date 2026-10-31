"""Concrete frame sources: webcam, image file, video file."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import FrameSource


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0, width: int | None = 1280, height: int | None = 720):
        self.name = f"webcam:{index}"
        # CAP_DSHOW opens much faster than the default backend on Windows.
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open webcam {index}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class ImageSource(FrameSource):
    """A still image, returned on every read. Good for calibration practice."""

    def __init__(self, path: str | Path):
        self.name = str(path)
        self.frame = cv2.imread(str(path))
        if self.frame is None:
            raise FileNotFoundError(path)

    def read(self):
        return self.frame.copy()


class VideoSource(FrameSource):
    def __init__(self, path: str | Path, loop: bool = False):
        self.name = str(path)
        self.loop = loop
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class ListSource(FrameSource):
    """Plays back a fixed list of frames (used by tests)."""

    def __init__(self, frames: list[np.ndarray]):
        self.name = "list"
        self.frames = list(frames)
        self.i = 0

    def read(self):
        if self.i >= len(self.frames):
            return None
        self.i += 1
        return self.frames[self.i - 1]


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def open_source(spec: str) -> FrameSource:
    """Create a source from a string: 'webcam', 'webcam:1', 'synthetic',
    'synthetic:seed=3', or a path to an image or video file."""
    if spec.startswith("webcam"):
        idx = int(spec.split(":", 1)[1]) if ":" in spec else 0
        return WebcamSource(idx)
    if spec.startswith("synthetic"):
        from .synthetic import SyntheticSource
        opts = {}
        if ":" in spec:
            for kv in spec.split(":", 1)[1].split(","):
                k, v = kv.split("=")
                opts[k] = float(v) if "." in v else int(v)
        return SyntheticSource(**opts)
    p = Path(spec)
    if p.suffix.lower() in VIDEO_EXT:
        return VideoSource(p)
    return ImageSource(p)
