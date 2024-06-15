from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chess import Board


class BoardsFromFEN(FileLinesDataset):
    """
    Parses FENs from txt-file
    """

    def __init__(self, *args, **kwargs):
        super().__init__(transforms_required=True, *args, **kwargs)
        self.board = Board()

    def transform_line(self, short_fen: str) -> Board:
        # halfmove clock and moves number are ignored
        self.board.set_fen(short_fen)
        return self.board


class ConsecutiveBoardsFromFEN(FileLinesDataset):
    """
    Parses FENs from txt-file
    """

    def __init__(self, *args, **kwargs):
        super().__init__(transforms_required=True, *args, **kwargs)
        self.board_before = Board()
        self.board_after = Board()

    def transform_line(self, line: str) -> Board:
        fen_before, fen_after, elo = line.split("|")
        self.board_before.set_fen(fen_before)
        self.board_after.set_fen(fen_after)
        elo = float(elo)
        return self.board_before, self.board_after, elo
