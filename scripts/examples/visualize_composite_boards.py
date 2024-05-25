from chessml import script
from pathlib import Path
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.composite_boards_images import CompositeBoardsImages
import os
from chessml.const import BOARD_COLORS
import cv2
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.utils import reset_dir


script.add_argument("-s", dest="seed", type=int, default=80)
script.add_argument("-l", type=int, dest="limit", default=100)


@script
def main(args, config):

    dir_with_piece_sets = Path(config.assets.path) / "piece_png"

    piece_sets = [
        dir_with_piece_sets / name
        for name in sorted(os.listdir(str(dir_with_piece_sets)))
    ]

    d = BoardsImagesFromFENs(
        fens=FileLinesDataset(path=Path("./datasets/unique_fens.txt")),
        piece_sets=piece_sets,
        board_colors=BOARD_COLORS,
        square_size=64,
        shuffle_buffer=32,
        shuffle_seed=args.seed,
    )

    dir_with_bg = Path(config.assets.path) / "bg/512"

    bg_images = [
        cv2.imread(str(dir_with_bg / name))
        for name in sorted(os.listdir(str(dir_with_bg)))
        if name.endswith(".jpg")
    ]

    d2 = CompositeBoardsImages(
        images_with_data=d,
        bg_images=bg_images,
        shuffle_buffer=1,
        shuffle_seed=args.seed,
        # skip_every_nth=2,
        limit=100,
    )

    tmp = reset_dir(Path("./tmp2"))

    for i, (final_image, coords, fen, flipped) in enumerate(d2):
        # print(final_image.shape, coords, fen, flipped)
        # final_image = cv2.resize(final_image, (224, 224))
        cv2.imwrite(
            str(tmp / f"{i + 1}.jpg"),
            final_image,
        )