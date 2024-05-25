from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class GameResult(Enum):
    WHITE_WON = 1
    DRAW = 0
    BLACK_WON = -1

    @classmethod
    def from_string(cls, result: str) -> GameResult:
        if result == "1-0":
            return GameResult.WHITE_WON

        if result == "1/2-1/2":
            return GameResult.DRAW

        if result == "0-1":
            return GameResult.BLACK_WON

        raise AssertionError(f"invalid result: {result}")


@dataclass
class Game:
    __slots__ = ["moves", "result"]
    moves: list[str]
    result: GameResult
