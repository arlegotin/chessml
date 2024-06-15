from chessml import script, config
from pathlib import Path
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.augmented_boards_images import AugmentedBoardsImages
import os
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, BG_IMAGES, FREE_PIECE_SETS
import cv2
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.utils import reset_dir
from chessml.data.images.picture import Picture


script.add_argument("-s", dest="seed", type=int, default=80)
script.add_argument("-si", dest="size", type=int, default=512)
script.add_argument("-l", type=int, dest="limit", default=64)


@script
def main(args):

    dataset = AugmentedBoardsImages(
        boards_with_data=BoardsImagesFromFENs(
            fens=FileLinesDataset(path=Path(config.dataset.path) / "unique_fens.txt"),
            # piece_sets=PIECE_SETS,
            piece_sets=FREE_PIECE_SETS,
            board_colors=BOARD_COLORS,
            square_size=64,
            shuffle_buffer=1,
            shuffle_seed=args.seed,
        ),
        bg_images=BG_IMAGES,
        shuffle_buffer=1,
        shuffle_seed=args.seed,
        limit=args.limit,
    )

    output_dir = reset_dir(Path("./output/visualized_augmented_boards"))

    for i, (picture, coords, fen, flipped) in enumerate(dataset):
        image = picture.cv2

        # original_height, original_width = image.shape[:2]
        # (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = coords
        # for (x, y) in ((tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)):
        #     cv2.circle(image, (int(x * original_width), int(y * original_height)), 3, (0, 255, 0), -1)

        cv2.imwrite(
            str(output_dir / f"{i + 1}.jpg"), cv2.resize(image, (args.size, args.size))
        )
