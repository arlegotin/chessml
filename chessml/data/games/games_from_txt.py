from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.data.games.game import GameResult
from chessml.utils import starts_with_number
from typing import Any, Dict

if 1 == 1:
    raise RuntimeError("GamesFromTXT is deprecated")


class GamesFromTXT(FileLinesDataset):
    """
    Parses games from txt-file.
    See format example: milesh1/35-million-chess-games
    """

    def validate_line(self, line: str):
        """
        Returns False when something's wrong with the game. True otherwise
        """

        # Skipping non-game lines
        assert starts_with_number(line), "line doesn't start with a number"

        # Skipping Fischer chess games because it requires to parse initial possition
        # TODO: parse initial position for Fischer chess
        assert " fen_true " not in line, "Fischer chess are not supported"
        assert " setup_true " not in line, "Fischer chess are not supported"

        # Something wrong with the moves
        assert " blen_true " not in line, "something's wrong with the moves"

    @staticmethod
    def parse_move(move: str) -> str:
        splitted = move.split(".")

        assert splitted[1], f"invalid move: {move}"

        return splitted[1]

    def transform_line(self, line: str) -> Dict[str, Any]:
        left, right = line.replace("\n", "").split(" ### ")

        meta = left.split(" ")

        return {
            "moves": list(map(self.parse_move, filter(None, right.split(" ")))),
            "result": GameResult.from_string(meta[2]),
        }
