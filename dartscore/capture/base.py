"""Frame-source abstraction: anything that produces images of the board.

Implement `FrameSource` for new camera hardware (a dedicated USB board camera,
an IP camera, a Raspberry Pi camera, an IR camera...). Everything downstream
only calls `read()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    name: str = "frames"

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if no frame is available."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
