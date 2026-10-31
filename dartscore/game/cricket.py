"""Cricket: close 15-20 and bull (3 marks each); score on numbers you've
closed and an opponent hasn't. Win = everything closed and not behind."""
from __future__ import annotations

from ..scoring import BULL, Throw
from .base import Game

TARGETS = (20, 19, 18, 17, 16, 15, BULL)


class Cricket(Game):
    name = "cricket"

    def __init__(self, players: list[str]):
        super().__init__(players)
        self.marks = {p: {t: 0 for t in TARGETS} for p in self.players}
        self.points = {p: 0 for p in self.players}

    def _apply(self, t: Throw) -> None:
        if t.segment not in TARGETS:
            return
        p = self.players[self.current]
        hits = t.multiplier
        have = self.marks[p][t.segment]
        closing = min(hits, 3 - have)
        self.marks[p][t.segment] = have + closing
        extra = hits - closing
        if extra and any(self.marks[o][t.segment] < 3 for o in self.players if o != p):
            self.points[p] += extra * t.segment
        if all(m >= 3 for m in self.marks[p].values()) and \
                all(self.points[p] >= self.points[o] for o in self.players):
            self.winner = p
            self.message = f"{p} wins!"

    def _state(self) -> dict:
        return {
            "targets": [str(t) if t != BULL else "BULL" for t in TARGETS],
            "marks": {p: [self.marks[p][t] for t in TARGETS] for p in self.players},
            "scores": self.points,
        }
