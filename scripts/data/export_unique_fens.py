from chessml import script
from pathlib import Path
from chessml.data.games.games_from_pgn import GamesFromPGN
from chessml.data.boards.boards_from_games import BoardsFromGames
import logging
import subprocess
import os
from tempfile import gettempdir
from chessml.utils import count_lines_in_file


script.add_argument("-s", dest="section", default="players")
script.add_argument("-f", dest="fens_filename", default="unique_fens.txt")


@script
def main(args, config):
    dir_with_pgns = Path(config.dataset.path) / args.section

    all_files = [
        dir_with_pgns / name for name in sorted(os.listdir(str(dir_with_pgns)))
    ]
    pgn_files = [f for f in all_files if f.suffix == ".pgn"]

    file_with_all_fens = Path(gettempdir()) / "all_fens.txt"

    logging.info(f"filling {file_with_all_fens} with FENs")

    with file_with_all_fens.open("w") as file:
        fens = BoardsFromGames(
            games=GamesFromPGN(paths=pgn_files,),
            transforms=[lambda board: board.fen()],
        )

        for fen in fens:
            fen_parts = fen.split(" ")
            short_fen = " ".join(fen_parts[:4])

            file.write(f"{short_fen}\n")

    logging.info(
        f"{count_lines_in_file(file_with_all_fens)} lines added to {file_with_all_fens}"
    )

    temp_file = Path(gettempdir()) / "temp_fens.txt"
    file_with_unique_fens = Path(config.dataset.path) / args.fens_filename

    logging.info(f"extracting unique FENs to {temp_file}")
    subprocess.run(f"sort -u {file_with_all_fens} > {temp_file}", shell=True)

    logging.info(f"shuffling unique FENs to {file_with_unique_fens}")
    subprocess.run(f"shuf {temp_file} -o {file_with_unique_fens}", shell=True)

    logging.info(
        f"{count_lines_in_file(file_with_unique_fens)} lines added to {file_with_unique_fens}"
    )

    file_with_all_fens.unlink()
    temp_file.unlink()
    logging.info(f"{file_with_all_fens} & {temp_file} are removed")
