"""Session: wires impact sources -> scoring engine -> game, thread-safely."""
from __future__ import annotations

import threading
import time

from .game import Game, make_game
from .impacts import Event, ImpactSource, Takeout
from .scoring import Throw, score_xy


class Session:
    def __init__(self, game: Game):
        self.game = game
        self.lock = threading.RLock()
        self.markers: list[dict] = []    # board positions of darts in this visit
        self.feed: list[str] = []        # human-readable event log
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    # -- events ---------------------------------------------------------------
    def handle(self, ev: Event) -> Throw | None:
        with self.lock:
            if isinstance(ev, Takeout):
                self.game.next_turn()
                self.markers.clear()
                self._log(f"darts removed ({ev.source})")
                return None
            t = score_xy(ev.x, ev.y)
            return self._throw(t, ev.source)

    def throw(self, t: Throw, source: str = "manual") -> Throw:
        with self.lock:
            return self._throw(t, source)

    def _throw(self, t: Throw, source: str) -> Throw:
        if self.game.turn_over:
            self.markers.clear()
        self.game.throw(t)
        if t.x is not None:
            self.markers.append({"x": t.x, "y": t.y, "label": t.label})
        self._log(f"{t.label} ({t.points}) via {source}")
        return t

    def undo(self) -> None:
        with self.lock:
            if self.game.undo():
                if self.markers:
                    self.markers.pop()
                self._log("undo")

    def next_turn(self) -> None:
        self.handle(Takeout("manual"))

    def new_game(self, kind: str, players: list[str], **opts) -> None:
        with self.lock:
            self.game = make_game(kind, players, **opts)
            self.markers.clear()
            self._log(f"new game: {kind} for {', '.join(players)}")

    def state(self) -> dict:
        with self.lock:
            return {**self.game.state(), "markers": list(self.markers), "feed": self.feed[-15:]}

    def _log(self, msg: str) -> None:
        self.feed.append(time.strftime("%H:%M:%S ") + msg)
        del self.feed[:-200]

    # -- background sources ---------------------------------------------------
    def attach(self, source: ImpactSource, interval: float = 0.03) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    for ev in source.poll():
                        self.handle(ev)
                except Exception as e:  # keep the app alive; report the problem
                    self._log(f"error in {source.name}: {e}")
                    time.sleep(1.0)
                time.sleep(interval)
            source.close()

        th = threading.Thread(target=loop, name=source.name, daemon=True)
        th.start()
        self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
