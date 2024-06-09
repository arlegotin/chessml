from chessml import script
from pathlib import Path
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3
import os
from chessml.data.assets import BOARD_COLORS, PIECE_SETS
import cv2
from chessml.utils import reset_dir


script.add_argument("-s", dest="seed", type=int, default=68)
script.add_argument("-l", type=int, dest="limit", default=64)


@script
def main(args):

    dataset = AugmentedPiecesImages(
        piece_images_3x3=PiecesImages3x3(
            piece_sets=PIECE_SETS,
            board_colors=BOARD_COLORS,
            square_size=64,
            shuffle_seed=args.seed,
            limit=args.limit,
        ),
    )

    output_dir = reset_dir(Path("./output/visualized_augmented_pieces"))

    for i, (picture, name) in enumerate(dataset):
        cv2.imwrite(str(output_dir / f"{i + 1}_{name}.jpg"), picture.cv2)
