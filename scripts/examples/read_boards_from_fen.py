from chessml import script
from chessml.data.boards.boards_from_fen import BoardsFromFEN
from chessml.data.boards.board_representation import FullPosition
from pathlib import Path


@script
def main(args, config):
    boards = BoardsFromFEN(
        path=Path("./datasets/unique_fens.txt"), limit=10, transforms=[FullPosition()],
    )

    for x in boards:
        print(x.shape, "\n")
