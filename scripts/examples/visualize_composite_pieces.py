from chessml import script
from pathlib import Path
from chessml.data.images.pieces_images import generate_pieces_images, CompositePiecesImages
import os
from chessml.const import BOARD_COLORS
import cv2


script.add_argument("-s", dest="seed", type=int, default=66)
script.add_argument("-l", type=int, dest="limit", default=100)


@script
def main(args, config):

    dir_with_piece_sets = Path(config.assets.path) / "piece_png"

    piece_sets = [
        dir_with_piece_sets / name
        for name in sorted(os.listdir(str(dir_with_piece_sets)))
    ]

    p = generate_pieces_images(
        piece_sets=piece_sets,
        # board_colors=BOARD_COLORS,
        board_colors=[
            ("#B58862", "#F0D9B5"),
        ],
        size=128,
    )

    d = CompositePiecesImages(
        piece_images=[x for x in p],
        shuffle_seed=args.seed,
        limit=args.limit,
    )

    for i, (img, name) in enumerate(d):
        print(img.shape, name)

        cv2.imwrite(f"./tmp/{i + 1}.jpg", img)
