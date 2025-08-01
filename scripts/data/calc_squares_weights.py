from chessml import script
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from pathlib import Path
import numpy as np
from tqdm import tqdm


def count_empty_squares_from_fen(fen):
    """Count the number of empty squares in a FEN position."""
    board_position = fen.split(" ")[0]  # Get board position part of FEN
    empty_squares = 0
    
    for char in board_position:
        if char.isdigit():  # Numbers indicate consecutive empty squares
            empty_squares += int(char)
    
    return empty_squares


script.add_argument("-l", dest="limit", type=int, default=10_000_000)


@script
def main(args):
    path_to_fens = Path("./datasets/unique_fens.txt")
    fens = FileLinesDataset(path=path_to_fens, limit=args.limit)

    empty_square_counts = []
    total_positions = 0

    print(f"Processing up to {args.limit} FEN positions...")
    
    for fen in tqdm(fens, total=args.limit):
        empty_squares = count_empty_squares_from_fen(fen)
        empty_square_counts.append(empty_squares)
        total_positions += 1

    # Calculate statistics
    total_empty_squares = sum(empty_square_counts)
    total_squares = total_positions * 64  # Chess board has 64 squares
    
    average_empty_squares_per_position = np.mean(empty_square_counts)
    empty_square_frequency = total_empty_squares / total_squares
    
    print(f"\nResults from {total_positions} positions:")
    print(f"Total empty squares: {total_empty_squares}")
    print(f"Total squares analyzed: {total_squares}")
    print(f"Average empty squares per position: {average_empty_squares_per_position:.2f}")
    print(f"Empty square frequency: {empty_square_frequency:.4f} ({empty_square_frequency*100:.2f}%)")
    
    # Additional statistics
    print(f"\nEmpty squares distribution:")
    print(f"Min empty squares in a position: {min(empty_square_counts)}")
    print(f"Max empty squares in a position: {max(empty_square_counts)}")
    print(f"Standard deviation: {np.std(empty_square_counts):.2f}")
