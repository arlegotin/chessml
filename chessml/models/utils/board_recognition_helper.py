import numpy as np
from chessml.data.assets import PIECE_CLASSES, BOARD_SIZE, INVERTED_PIECE_CLASSES
import cv2
from typing import Iterator, Optional
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.data.images.picture import Picture

class RecognitionResult:
    def __init__(self, board_image: Optional[Picture] = None):
        self.classified_squares = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.flipped = False
        self.white_castle_short = False
        self.white_castle_long = False
        self.black_castle_short = False
        self.black_castle_long = False
        self.white_to_move = True

        self.board_image = board_image

    def iterate_squares(
        self, square_size: int
    ) -> Iterator[tuple[np.ndarray, int, int]]:

        if self.board_image is None:
            raise RuntimeError("board image is not set")

        if self.flipped:
            row_range = range(BOARD_SIZE - 1, -1, -1)
            col_range = range(BOARD_SIZE - 1, -1, -1)
        else:
            row_range = range(BOARD_SIZE)
            col_range = range(BOARD_SIZE)

        resized_image = cv2.resize(
            self.board_image.cv2,
            (BOARD_SIZE * square_size, BOARD_SIZE * square_size),
            interpolation=cv2.INTER_LINEAR,
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

                yield Picture(resized_image[
                    y_start:y_end, x_start:x_end
                ]), rank_index, file_index

    def set_square_class(self, rank_index: int, file_index: int, class_index: int):
        self.classified_squares[rank_index][file_index] = class_index

    def get_fen_placement(self) -> str:
        fen_rows = []

        for row in reversed(self.classified_squares):
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

        return "/".join(fen_rows)
    
    def get_fen(self) -> str:
        fen_placement = self.get_fen_placement()

        active_color = 'w' if self.white_to_move else 'b'

        castling_availability = []
        if self.white_castle_short:
            castling_availability.append('K')
        if self.white_castle_long:
            castling_availability.append('Q')
        if self.black_castle_short:
            castling_availability.append('k')
        if self.black_castle_long:
            castling_availability.append('q')
        castling_fen = ''.join(castling_availability) if castling_availability else '-'

        return f"{fen_placement} {active_color} {castling_fen} - 0 1"

class BoardRecognitionHelper:
    def __init__(self, board_detector: BoardDetector, piece_classifier: PieceClassifier):
        self.board_detector = board_detector
        self.piece_classifier = piece_classifier

    def recognize(self, original_image: Picture) -> RecognitionResult:
        result = RecognitionResult(
            board_image=self.board_detector.extract_board_image(original_image),
        )

        squares = []
        ranks = []
        files = []
        for square, rank, file in result.iterate_squares(square_size=128): 
            squares.append(square)
            ranks.append(rank)
            files.append(file)

        class_indexes = self.piece_classifier.classify_pieces(squares)

        for class_index, rank, file in zip(class_indexes, ranks, files):
            result.set_square_class(rank, file, class_index)

        return result

        

    
