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
        self.feed: list[str] = []        # human-readable event log
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    # -- events ---------------------------------------------------------------
    def handle(self, ev: Event) -> Throw | None:
        with self.lock:
            if isinstance(ev, Takeout):
                self.game.next_turn()
                self._log(f"darts removed ({ev.source})")
                return None
            t = score_xy(ev.x, ev.y)
            return self._throw(t, ev.source)

    def throw(self, t: Throw, source: str = "manual") -> Throw:
        with self.lock:
            return self._throw(t, source)

    def _throw(self, t: Throw, source: str) -> Throw:
        self.game.throw(t)
        self._log(f"{t.label} ({t.points}) via {source}")
        return t

    def undo(self) -> None:
        with self.lock:
            if self.game.undo():
                self._log("undo")

    def edit(self, visit: int, dart: int, new: Throw | None) -> None:
        """Correct a dart: replace it, delete it (None), or add a missed one."""
        with self.lock:
            visits = self.game.visits()
            v = visits[visit]
            old = v["darts"][dart].label if dart < len(v["darts"]) else "(missing)"
            self.game.edit(visit, dart, new)
            what = new.label if new else "deleted"
            self._log(f"corrected {v['player']} dart {dart + 1}: {old} -> {what}")

    def next_turn(self) -> None:
        self.handle(Takeout("manual"))

    def new_game(self, kind: str, players: list[str], **opts) -> None:
        with self.lock:
            self.game = make_game(kind, players, **opts)
            self._log(f"new game: {kind} for {', '.join(players)}")

    def state(self) -> dict:
        with self.lock:
            markers = [{"x": t.x, "y": t.y, "label": t.label} for t in self.game.turn if t.x is not None]
            return {**self.game.state(), "markers": markers, "feed": self.feed[-15:]}

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
