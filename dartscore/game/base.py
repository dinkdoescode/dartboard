"""Game base class: turn tracking, 3 darts per visit, undo."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from ..scoring import Throw

DARTS_PER_TURN = 3


class Game(ABC):
    name = "game"

    def __init__(self, players: list[str]):
        if not players:
            raise ValueError("need at least one player")
        self.players = list(players)
        self.current = 0
        self.turn: list[Throw] = []      # darts thrown in the current visit
        self.turn_over = False           # visit finished; waiting for takeout
        self.winner: str | None = None
        self.message = ""
        self.log: list[dict] = []        # finished visits
        self._undo: list[dict] = []

    # -- public API -------------------------------------------------------------
    def throw(self, t: Throw) -> None:
        if self.winner:
            return
        if self.turn_over:
            # Manual play has no takeout event; a new dart starts the next visit.
            self.next_turn()
        self._push_undo()
        self.message = ""
        self.turn.append(t)
        self._apply(t)
        if not self.winner and len(self.turn) >= DARTS_PER_TURN:
            self.turn_over = True

    def next_turn(self) -> None:
        if self.winner or not self.turn:
            return
        self._push_undo()
        self.log.append({"player": self.players[self.current], "darts": [t.label for t in self.turn]})
        self._end_turn()
        self.turn = []
        self.turn_over = False
        self.current = (self.current + 1) % len(self.players)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.__dict__.update(self._undo.pop())
        return True

    def state(self) -> dict:
        return {
            "game": self.name,
            "players": self.players,
            "current": self.current,
            "turn": [t.to_dict() for t in self.turn],
            "turn_over": self.turn_over,
            "winner": self.winner,
            "message": self.message,
            "log": self.log[-12:],
            **self._state(),
        }

    # -- hooks for concrete games ----------------------------------------------
    @abstractmethod
    def _apply(self, t: Throw) -> None: ...

    def _end_turn(self) -> None:
        pass

    @abstractmethod
    def _state(self) -> dict: ...

    def _push_undo(self) -> None:
        snap = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "_undo"}
        self._undo.append(snap)
        del self._undo[:-200]
