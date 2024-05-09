import numpy as np
from chessml.const import PIECE_CLASSES, BOARD_SIZE
import cv2
from typing import Iterator


class BoardRecognitionHelper:
    def __init__(self, board_image: np.ndarray):
        self.board_image = board_image

        self.classified_squares = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.flipped = False
        self.white_castle_short = False
        self.white_castle_long = False
        self.black_castle_short = False
        self.black_castle_long = False
        self.white_to_move = True

    def set_square_class(self, rank_index: int, file_index: int, class_index: int):
        self.classified_squares[rank_index][file_index] = class_index

    def get_fen_placement(self) -> str:
        index_to_piece = {v: k for k, v in PIECE_CLASSES.items() if k is not None}

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
                    piece = index_to_piece[class_index]
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

    def get_board_image(self, size: int) -> np.ndarray:
        height, width, _ = self.board_image.shape

        if height != size or width != size:
            return cv2.resize(
                self.board_image, (size, size), interpolation=cv2.INTER_LINEAR,
            )

        return self.board_image

    def iterate_squares(
        self, square_size: int
    ) -> Iterator[tuple[np.ndarray, int, int]]:

        resized_image = self.get_board_image(BOARD_SIZE * square_size)

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

                yield resized_image[
                    y_start:y_end, x_start:x_end
                ], rank_index, file_index
