"""X01 games (501, 301, ...): count down to exactly zero."""
from __future__ import annotations

from functools import lru_cache

from ..geometry import SEGMENT_ORDER
from ..scoring import BULL, Throw
from .base import Game


class X01(Game):
    def __init__(self, players: list[str], start: int = 501, double_out: bool = True, double_in: bool = False):
        super().__init__(players)
        self._init.update(start=start, double_out=double_out, double_in=double_in)
        self.start = start
        self.name = str(start)
        self.double_out = double_out
        self.double_in = double_in
        self.scores = {p: start for p in self.players}
        self.opened = {p: not double_in for p in self.players}
        self.turn_start = start
        self.darts_thrown = {p: 0 for p in self.players}
        self.points_scored = {p: 0 for p in self.players}

    @property
    def player(self) -> str:
        return self.players[self.current]

    def _apply(self, t: Throw) -> None:
        p = self.player
        if len(self.turn) == 1:
            self.turn_start = self.scores[p]
        self.darts_thrown[p] += 1
        if not self.opened[p]:
            if not t.is_double:
                return
            self.opened[p] = True
        remaining = self.scores[p] - t.points
        if remaining < 0 or (self.double_out and remaining == 1) or \
                (remaining == 0 and self.double_out and not t.is_double):
            self.scores[p] = self.turn_start
            self.points_scored[p] -= sum(x.points for x in self.turn[:-1])
            self.message = f"BUST! {p} stays on {self.turn_start}"
            self.busted = True
            return
        self.scores[p] = remaining
        self.points_scored[p] += t.points
        if remaining == 0:
            self.winner = p
            self.message = f"{p} wins with {t.label}!"

    def _state(self) -> dict:
        p = self.player
        darts_left = 3 - len(self.turn) if not self.turn_over else 3
        rem = self.scores[p]
        return {
            "scores": self.scores,
            "averages": {q: round(3 * self.points_scored[q] / self.darts_thrown[q], 1) if self.darts_thrown[q] else 0.0
                         for q in self.players},
            "checkout": checkout(rem, darts_left, self.double_out) if self.opened[p] else None,
        }


def _all_throws() -> list[Throw]:
    out = [Throw(n, m) for m in (3, 2, 1) for n in sorted(SEGMENT_ORDER, reverse=True)]
    out += [Throw(BULL, 2), Throw(BULL, 1)]
    return sorted(out, key=lambda t: -t.points)


_THROWS = _all_throws()
_PREFERRED_DOUBLES = [16, 20, 8, 18, 12, 10, 4, 6, 14, 2, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, BULL]


@lru_cache(maxsize=None)
def _finishers(double_out: bool) -> list[Throw]:
    if not double_out:
        return _THROWS
    return [Throw(n, 2) for n in _PREFERRED_DOUBLES]


@lru_cache(maxsize=4096)
def _checkout(score: int, darts: int, double_out: bool) -> tuple[str, ...] | None:
    if darts <= 0 or score <= 0:
        return None
    for f in _finishers(double_out):
        if f.points == score:
            return (f.label,)
    if darts == 1:
        return None
    for t in _THROWS:
        rest = score - t.points
        if rest <= 1 and double_out:
            continue
        sub = _checkout(rest, darts - 1, double_out)
        if sub:
            return (t.label, *sub)
    return None


def checkout(score: int, darts: int = 3, double_out: bool = True) -> list[str] | None:
    """A route to finish `score` in at most `darts` darts, e.g. ['T20', 'T20', 'BULL']."""
    if score > 180:
        return None
    r = _checkout(score, darts, double_out)
    return list(r) if r else None
