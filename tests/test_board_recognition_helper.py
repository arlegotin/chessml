from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from chess import BLACK, WHITE, Board, square

from chessml.data.constants import PIECE_CLASSES
from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import (
    InvalidBoardGeometryError,
)
from chessml.models.lightning.piece_classifier_model import PieceDecodingError
from chessml.models.utils.board_recognition_helper import (
    BoardOrientation,
    BoardRecognitionHelper,
    RecognitionFailure,
    RecognitionFailureReason,
    RecognitionSuccess,
    build_fen,
)


class BoardDetectorStub:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)

    def extract_board_image(self, original_image):
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return original_image


class SquareClassifierStub:
    def __init__(self, occupied):
        self.occupied = occupied

    def classify_squares(self, images):
        assert len(images) == 64
        return self.occupied


class PieceClassifierStub:
    def __init__(self, classes, error=None):
        self.classes = classes
        self.error = error

    def classify_pieces(self, images):
        if self.error is not None:
            raise self.error
        assert len(images) == len(self.classes)
        return self.classes


def classifier_outputs(position):
    board = Board(f"{position} w - - 0 1")
    occupied = []
    classes = []

    for row in range(8):
        for column in range(8):
            piece = board.piece_at(square(column, 7 - row))
            occupied.append(int(piece is not None))
            if piece is not None:
                classes.append(PIECE_CLASSES[piece.symbol()])

    return occupied, classes


def helper_for(position, board_detector=None, piece_error=None):
    occupied, classes = classifier_outputs(position)
    return BoardRecognitionHelper(
        board_detector=board_detector or BoardDetectorStub(),
        square_classifier=SquareClassifierStub(occupied),
        piece_classifier=PieceClassifierStub(classes, error=piece_error),
    )


def recognize(position, board_detector=None, piece_error=None):
    return helper_for(
        position,
        board_detector=board_detector,
        piece_error=piece_error,
    ).recognize(IMAGE)


def rotate_position(position):
    return "/".join(row[::-1] for row in reversed(position.split("/")))


IMAGE = Picture(np.zeros((8, 8, 3), dtype=np.uint8))
ASYMMETRIC_POSITION = "r3k2r/8/8/3p4/8/2N5/8/R3K2R"
CONTEXT_POSITION = "r3k2r/8/8/3pP3/8/8/8/R3K2R"
INVALID_POSITION = "8/8/8/8/8/8/4k3/4K3"
EMPTY_POSITION = "8/8/8/8/8/8/8/8"


def test_recognize_returns_a_frozen_source_oriented_success():
    result = recognize(ASYMMETRIC_POSITION)

    assert isinstance(result, RecognitionSuccess)
    assert result.source_placement == ASYMMETRIC_POSITION
    assert result.board_image is IMAGE
    assert not hasattr(result, "board")
    assert not hasattr(result, "get_fen")
    assert not hasattr(result, "flipped")
    assert set(RecognitionFailureReason) == {
        RecognitionFailureReason.NO_BOARD,
        RecognitionFailureReason.INVALID_GEOMETRY,
        RecognitionFailureReason.DECODING_FAILED,
        RecognitionFailureReason.INVALID_PLACEMENT,
    }

    with pytest.raises(FrozenInstanceError):
        result.source_placement = "8/8/8/8/8/8/8/8"

    failure = RecognitionFailure(RecognitionFailureReason.NO_BOARD)
    with pytest.raises(FrozenInstanceError):
        failure.reason = RecognitionFailureReason.INVALID_GEOMETRY


def test_history_changes_only_when_the_caller_supplies_different_context():
    result = RecognitionSuccess(ASYMMETRIC_POSITION, IMAGE)

    first = build_fen(
        result,
        orientation=BoardOrientation.WHITE_AT_BOTTOM,
        turn=WHITE,
        castling="KQkq",
        en_passant="-",
        halfmove_clock=4,
        fullmove_number=12,
    )
    second = build_fen(
        result,
        orientation=BoardOrientation.WHITE_AT_BOTTOM,
        turn=BLACK,
        castling="-",
        en_passant="-",
        halfmove_clock=9,
        fullmove_number=33,
    )

    assert first == f"{ASYMMETRIC_POSITION} w KQkq - 4 12"
    assert second == f"{ASYMMETRIC_POSITION} b - - 9 33"
    assert result.source_placement == ASYMMETRIC_POSITION


@pytest.mark.parametrize(
    ("source_placement", "orientation"),
    [
        (CONTEXT_POSITION, BoardOrientation.WHITE_AT_BOTTOM),
        (
            rotate_position(CONTEXT_POSITION),
            BoardOrientation.BLACK_AT_BOTTOM,
        ),
    ],
)
def test_build_fen_uses_explicit_orientation_and_preserves_every_field(
    source_placement,
    orientation,
):
    result = RecognitionSuccess(source_placement, IMAGE)

    fen = build_fen(
        result,
        orientation=orientation,
        turn=WHITE,
        castling="KQkq",
        en_passant="d6",
        halfmove_clock=7,
        fullmove_number=23,
    )

    assert fen == f"{CONTEXT_POSITION} w KQkq d6 7 23"


def test_build_fen_requires_every_context_argument():
    result = RecognitionSuccess(CONTEXT_POSITION, IMAGE)

    with pytest.raises(TypeError):
        build_fen(
            result,
            orientation=BoardOrientation.WHITE_AT_BOTTOM,
            turn=WHITE,
            castling="KQkq",
            en_passant="d6",
            halfmove_clock=7,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("orientation", "white"),
        ("turn", "w"),
        ("castling", "QK"),
        ("castling", "KK"),
        ("en_passant", "a3"),
        ("halfmove_clock", -1),
        ("fullmove_number", 0),
    ],
)
def test_build_fen_rejects_invalid_or_normalized_context(field, value):
    result = RecognitionSuccess(CONTEXT_POSITION, IMAGE)
    context = {
        "orientation": BoardOrientation.WHITE_AT_BOTTOM,
        "turn": WHITE,
        "castling": "KQkq",
        "en_passant": "d6",
        "halfmove_clock": 7,
        "fullmove_number": 23,
    }
    context[field] = value

    with pytest.raises(ValueError):
        build_fen(result, **context)


def test_recognize_maps_invalid_geometry_to_a_failure():
    result = recognize(
        ASYMMETRIC_POSITION,
        board_detector=BoardDetectorStub(
            [InvalidBoardGeometryError("bad geometry")]
        ),
    )

    assert result == RecognitionFailure(
        RecognitionFailureReason.INVALID_GEOMETRY
    )


def test_recognize_maps_constrained_decoding_to_a_failure():
    result = recognize(
        ASYMMETRIC_POSITION,
        piece_error=PieceDecodingError("decoder failed"),
    )

    assert result == RecognitionFailure(
        RecognitionFailureReason.DECODING_FAILED
    )


@pytest.mark.parametrize("position", [INVALID_POSITION, EMPTY_POSITION])
def test_recognize_rejects_a_structurally_invalid_placement(position):
    result = recognize(position)

    assert result == RecognitionFailure(
        RecognitionFailureReason.INVALID_PLACEMENT
    )


def test_a_failed_frame_does_not_prevent_the_next_recognition():
    detector = BoardDetectorStub(
        [InvalidBoardGeometryError("bad geometry"), None]
    )
    helper = helper_for(ASYMMETRIC_POSITION, board_detector=detector)

    first = helper.recognize(IMAGE)
    second = helper.recognize(IMAGE)

    assert first == RecognitionFailure(
        RecognitionFailureReason.INVALID_GEOMETRY
    )
    assert isinstance(second, RecognitionSuccess)
    assert second.source_placement == ASYMMETRIC_POSITION


def test_recognize_does_not_hide_model_execution_errors():
    with pytest.raises(RuntimeError, match="device execution failed") as raised:
        recognize(
            ASYMMETRIC_POSITION,
            piece_error=RuntimeError("device execution failed"),
        )

    assert type(raised.value) is RuntimeError
