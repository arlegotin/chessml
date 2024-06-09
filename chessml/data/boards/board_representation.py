from abc import ABC, abstractmethod
from chess import Board, SQUARES_180, WHITE, BLACK
import numpy as np
from chessml.data.assets import PIECE_CLASSES, PIECE_CLASSES_NUMBER, BOARD_SIZE
from functools import cached_property
from typing import Tuple


class BoardRepresentation(ABC):
    """
    Helps to represent a board as a tensor
    """

    @abstractmethod
    def __call__(self, board: Board) -> np.ndarray:
        """
        Should return board as tensor
        """

    @cached_property
    def starting_position(self) -> np.ndarray:
        return self(Board())

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.starting_position.shape


class OnlyPieces(BoardRepresentation):
    """
    Converts Board into hot-encoded numpy array with shape CxHxW=13x8x8,
    where 13 for piece classes (counting empty square as a class)
    and 8 is the board size
    """

    def __call__(self, board: Board) -> np.ndarray:
        pieces = []

        for square in SQUARES_180:
            piece = board.piece_at(square)
            piece_symbol = None if piece is None else piece.symbol()

            assert piece_symbol in PIECE_CLASSES, f"invalid piece {piece_symbol}"

            piece_class = PIECE_CLASSES[piece_symbol]
            pieces.append(piece_class)

        reshaped = np.reshape(pieces, (BOARD_SIZE, BOARD_SIZE))

        pieces = np.eye(PIECE_CLASSES_NUMBER)[reshaped]
        pieces = np.swapaxes(pieces, 2, 0)
        pieces = pieces.astype(np.float32)

        return pieces

    # def to_string(self, tensor: np.ndarray) -> str:
    #     indices_tensor = np.argmax(tensor, axis=0)

    #     class_symbols = {v: k for k, v in PIECE_CLASSES.items()}
    #     class_symbols[0] = "."

    #     lines = []
    #     for row in indices_tensor.T:
    #         lines.append(" ".join(map(lambda cls: class_symbols[cls], row)))

    #     return "\n".join(lines)


class FullPosition(OnlyPieces):
    """
    Converts Board into hot-encoded numpy array with shape CxHxW=13x8x8:
    where 19 comes from:

    - 13 for piece classes (counting empty square as a class)
    - 1 for en passant square
    - 4 for castling rights (all squares are filled with ones if castling is possible)
    - 1 for who's turn it is (all squares are filled with ones if white to move)

    and 8 is the board size
    """

    def __call__(self, board: Board) -> np.ndarray:
        # Pieces
        pieces = super().__call__(board)

        # En passant
        en_passant = np.reshape(
            [[1 if board.ep_square == square else 0 for square in SQUARES_180]],
            (1, BOARD_SIZE, BOARD_SIZE),
        )

        # Castling & turn
        ones = np.ones((1, BOARD_SIZE, BOARD_SIZE))

        white_kingside_castling = ones * board.has_kingside_castling_rights(WHITE)
        white_queenside_castling = ones * board.has_queenside_castling_rights(WHITE)
        black_kingside_castling = ones * board.has_kingside_castling_rights(BLACK)
        black_queenside_castling = ones * board.has_queenside_castling_rights(BLACK)

        white_turn = ones * board.turn

        return np.concatenate(
            (
                pieces,
                en_passant,
                white_kingside_castling,
                white_queenside_castling,
                black_kingside_castling,
                black_queenside_castling,
                white_turn,
            )
        ).astype(np.float32)
