"""Game base class: turn tracking, 3 darts per visit, undo and corrections.

The game is stored as an event list (darts and "next player" markers). The
scores are always the result of replaying that list, so correcting any dart --
even one from several visits ago -- is just: change the event, replay.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from ..scoring import Throw

DARTS_PER_TURN = 3
NEXT = "next"   # event marker: the visit ended, next player's turn


class Game(ABC):
    name = "game"

    def __init__(self, players: list[str]):
        if not players:
            raise ValueError("need at least one player")
        self.players = list(players)
        self.current = 0
        self.turn: list[Throw] = []      # darts thrown in the current visit
        self.busted = False              # rest of this visit doesn't count
        self.winner: str | None = None
        self.message = ""
        self.log: list[dict] = []        # finished visits
        self._events: list = []          # Throw | NEXT, the source of truth
        self._init: dict = {"players": self.players}   # constructor args, for replay
        self._undo: list[dict] = []

    @property
    def turn_over(self) -> bool:
        """Visit finished; waiting for the darts to be pulled out."""
        return self.busted or len(self.turn) >= DARTS_PER_TURN

    # -- live play ----------------------------------------------------------------
    def throw(self, t: Throw) -> None:
        if self.winner:
            return
        self._push_undo()
        if len(self.turn) >= DARTS_PER_TURN:
            # No takeout seen (manual play, or a missed takeout): a 4th dart
            # starts the next visit.
            self._events.append(NEXT)
            self._do_next()
        self._events.append(t)
        self._do_throw(t)

    def next_turn(self) -> None:
        if self.winner or not self.turn:
            return
        self._push_undo()
        self._events.append(NEXT)
        self._do_next()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.__dict__.update(self._undo.pop())
        return True

    # -- corrections --------------------------------------------------------------
    def visits(self) -> list[dict]:
        """Every visit so far (the last one is the current visit)."""
        out = [{"player": self.players[0], "darts": [], "events": []}]
        p = 0
        for i, e in enumerate(self._events):
            if e == NEXT:
                p = (p + 1) % len(self.players)
                out.append({"player": self.players[p], "darts": [], "events": []})
            else:
                out[-1]["darts"].append(e)
                out[-1]["events"].append(i)
        return out

    def edit(self, visit: int, dart: int, new: Throw | None) -> None:
        """Replace dart `dart` of visit `visit` with `new`, delete it (new=None),
        or add a missed dart (dart == number of darts in that visit)."""
        visits = self.visits()
        if visit < 0:
            visit += len(visits)
        if not 0 <= visit < len(visits):
            raise IndexError("no such visit")
        v = visits[visit]
        n = len(v["darts"])
        events = list(self._events)
        if dart < n:
            pos = v["events"][dart]
            if new is None:
                del events[pos]
            else:
                events[pos] = new
        elif dart == n and new is not None:
            if n >= DARTS_PER_TURN:
                raise ValueError("a visit has at most 3 darts")
            # Insert after the visit's last dart, or right after the marker
            # that opened it (or at the very end for the current visit).
            if n:
                pos = v["events"][-1] + 1
            else:
                nexts = [i for i, e in enumerate(events) if e == NEXT]
                pos = nexts[visit - 1] + 1 if visit > 0 else 0
            events.insert(pos, new)
        else:
            raise IndexError("no such dart")
        self._push_undo()
        self._replay(events)
        self.message = "Corrected" + (f": {self.message}" if self.message else "")

    def _replay(self, events: list) -> None:
        fresh = type(self)(**self._init)
        used = []
        for e in events:
            if fresh.winner:
                break   # an earlier dart now wins the game; later ones never happened
            used.append(e)
            if e == NEXT:
                fresh._do_next()
            else:
                fresh._do_throw(e)
        fresh._events = used
        undo = self._undo
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        self._undo = undo

    # -- state ----------------------------------------------------------------------
    def state(self) -> dict:
        visits = self.visits()
        return {
            "game": self.name,
            "players": self.players,
            "current": self.current,
            "turn": [t.to_dict() for t in self.turn],
            "turn_over": self.turn_over,
            "busted": self.busted,
            "winner": self.winner,
            "message": self.message,
            "log": self.log[-12:],
            "visits": [{"index": i, "player": v["player"], "darts": [t.label for t in v["darts"]],
                        "points": sum(t.points for t in v["darts"])}
                       for i, v in enumerate(visits)][-15:],
            **self._state(),
        }

    # -- internals ------------------------------------------------------------------
    def _do_throw(self, t: Throw) -> None:
        self.message = ""
        self.turn.append(t)
        if self.busted:
            return   # thrown after a bust: recorded, but doesn't count
        self._apply(t)

    def _do_next(self) -> None:
        self.log.append({"player": self.players[self.current], "darts": [t.label for t in self.turn]})
        self._end_turn()
        self.turn = []
        self.busted = False
        self.current = (self.current + 1) % len(self.players)

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
