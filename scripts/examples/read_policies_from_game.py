from chessml import script
from chessml.data.games.games_from_pgn import GamesFromPGN
from chessml.data.policy.policies_from_games import PoliciesFromGames
from chessml.data.boards.board_representation import OnlyPieces
from pathlib import Path


@script
def main(args, config):
    boards = PoliciesFromGames(
        games=GamesFromPGN(path=Path("./datasets/players/Tal.pgn")),
        transforms=[lambda x: x],
        limit=10,
    )

    for x in boards:
        print(x)
