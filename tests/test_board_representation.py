import numpy as np
from chess import Board, Piece, square_file, square_rank

from chessml.data.constants import PIECE_CLASSES
from chessml.data.boards.board_representation import FullPosition, OnlyPieces


def test_only_pieces_uses_empty_zero_and_shifted_piece_channels():
    board = Board.empty()
    placements = []

    for square_index, (symbol, class_index) in enumerate(PIECE_CLASSES.items()):
        board.set_piece_at(square_index, Piece.from_symbol(symbol))
        placements.append((square_index, class_index))

    encoded = OnlyPieces()(board)

    assert encoded.shape == (13, 8, 8)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(
        encoded.sum(axis=0), np.ones((8, 8), dtype=np.float32)
    )
    assert encoded[0].sum() == 64 - len(PIECE_CLASSES)

    for square_index, class_index in placements:
        row = 7 - square_rank(square_index)
        column = square_file(square_index)
        assert encoded[class_index + 1, column, row] == 1


def test_full_position_appends_metadata_after_all_thirteen_piece_planes():
    board = Board()
    pieces = OnlyPieces()(board)
    encoded = FullPosition()(board)

    assert encoded.shape == (19, 8, 8)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[:13], pieces)
    assert pieces[0].sum() == 32
    assert pieces[PIECE_CLASSES["K"] + 1].sum() == 1
    np.testing.assert_array_equal(encoded[13], np.zeros((8, 8), dtype=np.float32))
    np.testing.assert_array_equal(
        encoded[14:18], np.ones((4, 8, 8), dtype=np.float32)
    )
    np.testing.assert_array_equal(encoded[18], np.ones((8, 8), dtype=np.float32))
