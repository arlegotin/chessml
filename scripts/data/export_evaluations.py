from chessml import script
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from pathlib import Path
from stockfish import Stockfish
import logging
from chessml.utils import count_lines_in_file
from tqdm import tqdm

script.add_argument("-d", dest="depth", type=int, default=18)


@script
def main(args):
    logging.info("creating engine")
    stockfish = Stockfish(
        # path="./3rd_party/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64",
        # path="./3rd_party/stockfish_15.1_linux_x64_bmi2/stockfish_15.1_x64_bmi2",
        # path="./3rd_party/stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2",
        # path="./3rd_party/stockfish_15.1_linux_x64_popcnt/stockfish-ubuntu-20.04-x86-64-modern",
        depth=args.depth,
        parameters={"Threads": 4, "Hash": 4096,},
    )

    path_to_fens = Path("./datasets/unique_fens.txt")
    fens = FileLinesDataset(path=path_to_fens)
    number_of_fens = count_lines_in_file(path_to_fens)

    path_to_evaluations = Path(
        f"./datasets/eval_stockfish_{stockfish.get_stockfish_major_version()}_{args.depth}.txt"
    )

    # print(stockfish.get_stockfish_major_version())

    with path_to_evaluations.open("w") as file:
        for fen in tqdm(fens, total=number_of_fens):
            fen += " 0 1"

            if stockfish.is_fen_valid(fen):
                stockfish.set_fen_position(fen)

                eval = stockfish.get_evaluation()
                best_move = stockfish.get_best_move()

                file.write(f"{eval['type']} {eval['value']} {best_move or '-'}\n")
            else:
                logging.warning(f"skip invalid fen '{fen}'")
                file.write("-\n")
