from .base import Game
from .cricket import Cricket
from .x01 import X01, checkout


def make_game(kind: str, players: list[str], **opts) -> Game:
    kind = kind.lower()
    if kind == "cricket":
        return Cricket(players)
    if kind.isdigit():
        return X01(players, start=int(kind), **opts)
    raise ValueError(f"unknown game: {kind}")


__all__ = ["Game", "X01", "Cricket", "checkout", "make_game"]
