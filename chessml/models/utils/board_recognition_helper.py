from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, TypeAlias

import cv2
from chess import BLACK, WHITE, Board, Color

from chessml.data.assets import BOARD_SIZE, INVERTED_PIECE_CLASSES
from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import (
    BoardDetector,
    InvalidBoardGeometryError,
)
from chessml.models.lightning.piece_classifier_model import (
    PieceClassifier,
    PieceDecodingError,
)
from chessml.models.lightning.square_classifier_model import SquareClassifier


class BoardOrientation(Enum):
    WHITE_AT_BOTTOM = auto()
    BLACK_AT_BOTTOM = auto()


class RecognitionFailureReason(Enum):
    NO_BOARD = auto()
    INVALID_GEOMETRY = auto()
    DECODING_FAILED = auto()
    INVALID_PLACEMENT = auto()


@dataclass(frozen=True)
class RecognitionSuccess:
    source_placement: str
    board_image: Picture


@dataclass(frozen=True)
class RecognitionFailure:
    reason: RecognitionFailureReason


RecognitionResult: TypeAlias = RecognitionSuccess | RecognitionFailure


def _rotate_placement(placement: str) -> str:
    return "/".join(row[::-1] for row in reversed(placement.split("/")))


def _is_valid_source_placement(placement: str) -> bool:
    candidates = (placement, _rotate_placement(placement))
    for candidate in candidates:
        for turn in ("w", "b"):
            try:
                board = Board(f"{candidate} {turn} - - 0 1")
            except ValueError:
                continue
            if board.is_valid():
                return True
    return False


def _iterate_squares(
    board_image: Picture,
    square_size: int,
) -> Iterator[Picture]:
    resized = cv2.resize(
        board_image.cv2,
        (BOARD_SIZE * square_size, BOARD_SIZE * square_size),
        interpolation=cv2.INTER_CUBIC,
    )
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            yield Picture(
                resized[
                    row * square_size : (row + 1) * square_size,
                    column * square_size : (column + 1) * square_size,
                ]
            )


def _source_placement(class_indexes: list[int]) -> str:
    rows = []
    for start in range(0, len(class_indexes), BOARD_SIZE):
        row = []
        empty_count = 0
        for class_index in class_indexes[start : start + BOARD_SIZE]:
            if class_index == -1:
                empty_count += 1
                continue
            if empty_count:
                row.append(str(empty_count))
                empty_count = 0
            row.append(INVERTED_PIECE_CLASSES[class_index])
        if empty_count:
            row.append(str(empty_count))
        rows.append("".join(row))
    return "/".join(rows)


def build_fen(
    result: RecognitionSuccess,
    *,
    orientation: BoardOrientation,
    turn: Color,
    castling: str,
    en_passant: str,
    halfmove_clock: int,
    fullmove_number: int,
) -> str:
    if not isinstance(result, RecognitionSuccess):
        raise TypeError("build_fen requires RecognitionSuccess")
    if not isinstance(orientation, BoardOrientation):
        raise ValueError("Invalid board orientation")
    if turn is WHITE:
        active_color = "w"
    elif turn is BLACK:
        active_color = "b"
    else:
        raise ValueError("Turn must be chess.WHITE or chess.BLACK")
    if (
        not isinstance(castling, str)
        or (
            castling != "-"
            and (
                not castling
                or castling
                != "".join(symbol for symbol in "KQkq" if symbol in castling)
            )
        )
    ):
        raise ValueError("Castling must be '-' or a canonical KQkq subset")
    if not isinstance(en_passant, str):
        raise ValueError("En-passant must be '-' or a target square")
    if type(halfmove_clock) is not int or halfmove_clock < 0:
        raise ValueError("Halfmove clock must be a non-negative integer")
    if type(fullmove_number) is not int or fullmove_number <= 0:
        raise ValueError("Fullmove number must be a positive integer")

    placement = result.source_placement
    if orientation is BoardOrientation.BLACK_AT_BOTTOM:
        placement = _rotate_placement(placement)
    fen = (
        f"{placement} {active_color} {castling} {en_passant} "
        f"{halfmove_clock} {fullmove_number}"
    )

    try:
        board = Board(fen)
    except ValueError as error:
        raise ValueError("Invalid FEN context") from error
    if not board.is_valid():
        raise ValueError("FEN context is inconsistent with the placement")
    return fen


class BoardRecognitionHelper:
    def __init__(
        self,
        board_detector: BoardDetector,
        square_classifier: SquareClassifier,
        piece_classifier: PieceClassifier,
    ):
        self.board_detector = board_detector
        self.square_classifier = square_classifier
        self.piece_classifier = piece_classifier

    def recognize(self, original_image: Picture) -> RecognitionResult:
        try:
            board_image = self.board_detector.extract_board_image(original_image)
        except InvalidBoardGeometryError:
            return RecognitionFailure(
                RecognitionFailureReason.INVALID_GEOMETRY
            )

        squares = list(_iterate_squares(board_image, square_size=64))
        square_classes = self.square_classifier.classify_squares(
            [square.bw for square in squares]
        )
        if len(square_classes) != len(squares):
            raise ValueError("SquareClassifier must return one label per square")

        occupied_indexes = [
            index
            for index, square_class in enumerate(square_classes)
            if square_class == 1
        ]
        pieces = [squares[index].bw for index in occupied_indexes]

        if pieces:
            try:
                piece_classes = self.piece_classifier.classify_pieces(pieces)
            except PieceDecodingError:
                return RecognitionFailure(
                    RecognitionFailureReason.DECODING_FAILED
                )
        else:
            piece_classes = []

        if len(piece_classes) != len(occupied_indexes):
            raise ValueError(
                "PieceClassifier must return one label per occupied square"
            )

        class_indexes = [-1] * len(squares)
        for index, piece_class in zip(occupied_indexes, piece_classes):
            class_indexes[index] = piece_class

        source_placement = _source_placement(class_indexes)
        if not _is_valid_source_placement(source_placement):
            return RecognitionFailure(
                RecognitionFailureReason.INVALID_PLACEMENT
            )
        return RecognitionSuccess(
            source_placement=source_placement,
            board_image=board_image,
        )
