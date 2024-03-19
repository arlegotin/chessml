from chessml import script
from chessml.data.games.games_from_pgn import GamesFromPGN
from pathlib import Path


@script
def main(args, config):
    games = GamesFromPGN(
        # path=Path("./datasets/players/Tal.pgn"),
        path=Path("./datasets/championships/WorldChamp2021.pgn"),
        # limit=3,
    )

    for game in games:
        print(game)
