import numpy as np
from chess import Board, square

from chessml.data.assets import PIECE_CLASSES
from chessml.data.images.picture import Picture
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper


class BoardDetectorStub:
    def extract_board_image(self, original_image):
        return original_image


class SquareClassifierStub:
    def __init__(self, occupied):
        self.occupied = occupied

    def classify_squares(self, images):
        assert len(images) == 64
        return self.occupied


class PieceClassifierStub:
    def __init__(self, classes):
        self.classes = classes

    def classify_pieces(self, images):
        assert len(images) == len(self.classes)
        return self.classes


class MetaPredictorStub:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, board):
        assert board.shape == (13, 8, 8)
        assert board.dtype == np.float32
        return self.prediction


def classifier_outputs(board):
    occupied = []
    classes = []

    for row in range(8):
        for column in range(8):
            piece = board.piece_at(square(column, 7 - row))
            occupied.append(int(piece is not None))
            if piece is not None:
                classes.append(PIECE_CLASSES[piece.symbol()])

    return occupied, classes


def recognize(position, prediction):
    source_board = Board(f"{position} w - - 0 1")
    occupied, classes = classifier_outputs(source_board)
    helper = BoardRecognitionHelper(
        board_detector=BoardDetectorStub(),
        square_classifier=SquareClassifierStub(occupied),
        piece_classifier=PieceClassifierStub(classes),
        meta_predictor=MetaPredictorStub(prediction),
    )

    return helper.recognize(Picture(np.zeros((8, 8, 3), dtype=np.uint8)))


def rotate_position(position):
    return "/".join(row[::-1] for row in reversed(position.split("/")))


CANONICAL_POSITION = "r3k2r/8/8/3p4/8/2N5/8/R3K2R"


def test_recognize_uses_meta_prediction_for_unflipped_position():
    result = recognize(
        CANONICAL_POSITION,
        (True, True, True, True, True, False),
    )

    assert result.get_fen() == f"{CANONICAL_POSITION} w KQkq - 0 1"
    assert result.flipped is False


def test_recognize_canonicalizes_a_flipped_position_exactly_once():
    result = recognize(
        rotate_position(CANONICAL_POSITION),
        (True, True, True, True, False, True),
    )

    assert result.get_fen() == f"{CANONICAL_POSITION} b KQkq - 0 1"
    assert result.flipped is True
