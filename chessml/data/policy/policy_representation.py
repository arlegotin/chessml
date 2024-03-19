from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Iterable
from functools import cached_property


class PolicyRepresentation(ABC):
    board_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    board_numbers = ["1", "2", "3", "4", "5", "6", "7", "8"]
    piece_promotions = ["", "n", "b", "r", "q"]

    @cached_property
    def board_squares(self) -> List[str]:
        """
        Reshaped BOARD_SQUARES should match shape made by board_to_position_array:

        [['r' 'n' 'b' 'q' 'k' 'b' 'n' 'r']
        ['p' 'p' 'p' 'p' 'p' 'p' 'p' 'p']
        [None None None None None None None None]
        [None None None None None None None None]
        [None None None None None None None None]
        [None None None None None None None None]
        ['P' 'P' 'P' 'P' 'P' 'P' 'P' 'P']
        ['R' 'N' 'B' 'Q' 'K' 'B' 'N' 'R']]

        [['a8' 'b8' 'c8' 'd8' 'e8' 'f8' 'g8' 'h8']
        ['a7' 'b7' 'c7' 'd7' 'e7' 'f7' 'g7' 'h7']
        ['a6' 'b6' 'c6' 'd6' 'e6' 'f6' 'g6' 'h6']
        ['a5' 'b5' 'c5' 'd5' 'e5' 'f5' 'g5' 'h5']
        ['a4' 'b4' 'c4' 'd4' 'e4' 'f4' 'g4' 'h4']
        ['a3' 'b3' 'c3' 'd3' 'e3' 'f3' 'g3' 'h3']
        ['a2' 'b2' 'c2' 'd2' 'e2' 'f2' 'g2' 'h2']
        ['a1' 'b1' 'c1' 'd1' 'e1' 'f1' 'g1' 'h1']]
        """
        return [f"{l}{n}" for n in self.board_numbers for l in self.board_letters]

    @abstractmethod
    def __call__(self, moves: Iterable[str]) -> np.ndarray:
        """
        Should return moves as tensor
        """

    @abstractmethod
    def to_string(self, tensor: np.ndarray) -> List[str]:
        """
        Should convert moves tensor to list of strings
        """

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self(["e2e4"]).shape


class PolicyAsVector(PolicyRepresentation):
    """
    Represents moves as a vector.
    Components of this vector correspond to all_moves.
    Value is 1 if this is the move, otherwise, it's 0.
    """

    @cached_property
    def all_moves(self) -> List[str]:
        return [
            f"{f}{t}{p}"
            for f in self.board_squares
            for t in self.board_squares
            for p in self.piece_promotions
            if f != t
            and (
                p == ""
                or (f[1] == "2" and t[1] == "1")
                or (f[1] == "7" and t[1] == "8")
            )
        ]

    def __call__(self, moves: Iterable[str]) -> np.ndarray:
        for move in moves:
            if move not in self.all_moves:
                print(self.all_moves)
                print(type(move))
                raise RuntimeError(f"unknown move {move}. Please check all_moves")

        return np.array([1.0 if x in moves else 0.0 for x in self.all_moves])

    def to_string(self, tensor: np.ndarray) -> List[str]:
        indexes = (-tensor).argsort()
        return [self.all_moves[i] for i in indexes]
