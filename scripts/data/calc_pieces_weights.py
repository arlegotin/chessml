from chessml import script
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from chessml.data.assets import PIECE_CLASSES
from tqdm import tqdm


def count_pieces_and_empty_squares_from_fen(fen):
    piece_counts = Counter()
    for position in fen.split(" ")[0]:  # Split FEN and take the board position part
        if position.isalpha():  # Piece characters
            piece_counts[position] += 1
        elif position.isdigit():  # Numbers indicate empty squares
            piece_counts[None] += int(position)
    return piece_counts


script.add_argument("-l", dest="limit", type=int, default=10_000_000)


@script
def main(args):
    path_to_fens = Path("./datasets/unique_fens.txt")
    fens = FileLinesDataset(path=path_to_fens, limit=args.limit)

    position_counts = []

    for fen in tqdm(fens, total=args.limit):
        position_counts.append(count_pieces_and_empty_squares_from_fen(fen))

    total_counts = defaultdict(list)
    all_classes = set().union(*[counts.keys() for counts in position_counts])

    for counts in position_counts:
        for cls in all_classes:
            total_counts[cls].append(counts.get(cls, 0))

    average_counts = {cls: np.mean(counts) for cls, counts in total_counts.items()}
    print(average_counts)

    total_elements = sum(average_counts.values())
    class_weights = {
        cls: total_elements / count for cls, count in average_counts.items()
    }

    weight_sum = sum(class_weights.values())
    normalized_class_weights = {
        cls: weight / weight_sum for cls, weight in class_weights.items()
    }

    print(PIECE_CLASSES.keys())
    weights = [normalized_class_weights[cls] for cls in PIECE_CLASSES.keys()]

    print(weights)
