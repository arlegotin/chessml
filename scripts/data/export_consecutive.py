from chessml import script, config
from pathlib import Path
from chessml.data.games.games_from_pgn import GamesFromPGN
from chessml.data.boards.boards_from_games import ConsecutiveBoardsFromGames
import logging
import subprocess
from chessml.models.lightning.elo_predictor_model import EloPredictor
from chessml.data.boards.board_representation import OnlyPieces
from chessml.train.standard_training import standard_training
import torch
from tempfile import gettempdir
from tqdm import tqdm

script.add_argument("-l", dest="limit", type=int, default=20_000_000)


@script
def main(args):
    board_representation = OnlyPieces()

    def transform(x):
        current_board, next_board, game, turn = x
        return (
            current_board.fen(),
            next_board.fen(),
            game.white_elo if turn else game.black_elo,
        )

    games = GamesFromPGN(
        paths=[Path("./datasets/lichess_db_standard_rated_2014-09.pgn")]
    )

    dataset = ConsecutiveBoardsFromGames(
        games=games, transforms=[transform], limit=args.limit
    )

    file_with_all_fens = Path(gettempdir()) / "all.txt"

    logging.info(f"filling {file_with_all_fens} with FENs")
    with file_with_all_fens.open("w") as file:
        for f1, f2, elo in tqdm(dataset, total=args.limit):
            file.write(f"{f1}|{f2}|{elo}\n")

    temp_file = Path(gettempdir()) / "temp_fens.txt"
    file_with_unique_fens = Path(config.dataset.path) / "consecutive_fens.txt"

    logging.info(f"extracting unique FENs to {temp_file}")
    subprocess.run(f"sort -u {file_with_all_fens} > {temp_file}", shell=True)

    logging.info(f"shuffling unique FENs to {file_with_unique_fens}")
    subprocess.run(f"shuf {temp_file} -o {file_with_unique_fens}", shell=True)

    file_with_all_fens.unlink()
    temp_file.unlink()
    logging.info(f"{file_with_all_fens} & {temp_file} are removed")
