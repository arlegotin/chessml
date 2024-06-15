import numpy as np
from chessml.data.assets import PIECE_CLASSES, BOARD_SIZE, INVERTED_PIECE_CLASSES
import cv2
from typing import Iterator, Optional
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.meta_predictor_model import MetaPredictor
from chessml.data.images.picture import Picture
from chessml.data.boards.board_representation import OnlyPieces
from chess import Board
from chessml.utils import reset_dir
from pathlib import Path


class RecognitionResult:
    def __init__(self, board_image: Picture):
        self.board_image = board_image
        self.board = Board()
        self.flipped = False

    def iterate_squares(
        self, square_size: int
    ) -> Iterator[tuple[np.ndarray, int, int]]:

        if self.flipped:
            row_range = range(BOARD_SIZE - 1, -1, -1)
            col_range = range(BOARD_SIZE - 1, -1, -1)
        else:
            row_range = range(BOARD_SIZE)
            col_range = range(BOARD_SIZE)

        resized_image = cv2.resize(
            self.board_image.cv2,
            (BOARD_SIZE * square_size, BOARD_SIZE * square_size),
            interpolation=cv2.INTER_CUBIC,
        )

        if self.flipped:
            row_range = range(BOARD_SIZE - 1, -1, -1)
            col_range = range(BOARD_SIZE - 1, -1, -1)
        else:
            row_range = range(BOARD_SIZE)
            col_range = range(BOARD_SIZE)

        for row in row_range:
            y_start = row * square_size
            y_end = y_start + square_size
            rank_index = BOARD_SIZE - 1 - row
            for col in col_range:
                x_start = col * square_size
                x_end = x_start + square_size
                file_index = col

                yield Picture(
                    resized_image[y_start:y_end, x_start:x_end]
                ), rank_index, file_index

    def get_fen(self) -> str:
        return self.board.fen()


class BoardRecognitionHelper:
    def __init__(
        self,
        board_detector: BoardDetector,
        piece_classifier: PieceClassifier,
        meta_predictor: MetaPredictor,
    ):
        self.board_detector = board_detector
        self.piece_classifier = piece_classifier
        self.meta_predictor = meta_predictor

    def recognize(self, original_image: Picture) -> RecognitionResult:
        result = RecognitionResult(
            board_image=self.board_detector.extract_board_image(original_image)
        )

        squares, ranks, files = zip(*result.iterate_squares(square_size=128))
        # result.board_image.pil.save("output/tmp/board.png")
        # for i, s in enumerate(squares):
        #     s.pil.save(f"output/tmp/{i}.png")
        # quit()

        class_indexes = self.piece_classifier.classify_pieces([s.bw for s in squares])

        classified_squares = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for class_index, rank, file in zip(class_indexes, ranks, files):
            classified_squares[rank][file] = class_index

        fen_rows = []
        for row in reversed(classified_squares):
            fen_row = []
            empty_count = 0

            for class_index in row:
                if class_index == PIECE_CLASSES[None]:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row.append(str(empty_count))
                        empty_count = 0
                    piece = INVERTED_PIECE_CLASSES[class_index]
                    fen_row.append(piece)

            if empty_count > 0:
                fen_row.append(str(empty_count))

            fen_rows.append("".join(fen_row))

        fen_position = "/".join(fen_rows)
        result.board.set_fen(f"{fen_position} w - - 0 1")

        (
            white_kingside_castling,
            white_queenside_castling,
            black_kingside_castling,
            black_queenside_castling,
            white_turn,
            flipped,
        ) = self.meta_predictor.predict(OnlyPieces()(result.board))

        castling = (
            "".join(
                [
                    "K" if white_kingside_castling else "",
                    "Q" if white_queenside_castling else "",
                    "k" if black_kingside_castling else "",
                    "q" if black_queenside_castling else "",
                ]
            )
            or "-"
        )

        # if flipped:
        #     fen_position = "/".join(row[::-1] for row in fen_rows[::-1])

        result.board.set_fen(
            f"{fen_position} {'w' if white_turn else 'b'} {castling} - 0 1"
        )

        result.flipped = flipped

        return result
