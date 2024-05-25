from chessml import script
from chessml.data.games.games_from_pgn import GamesFromPGN
from chessml.data.boards.boards_from_games import BoardsFromGames
from chessml.data.boards.board_representation import OnlyPieces
from pathlib import Path


@script
def main(args, config):
    board_representation = OnlyPieces()

    boards = BoardsFromGames(
        games=GamesFromPGN(path=Path("./datasets/players/Tal.pgn"),),
        transforms=[board_representation],
        limit=10,
    )

    for x in boards:
        print(board_representation(x), "\n")
