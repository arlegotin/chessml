from chessml import script, config
from pathlib import Path
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.augmented_boards_images import AugmentedBoardsImages
import os
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, BG_IMAGES
import cv2
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.utils import reset_dir, write_lines_to_txt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from chessml.data.images.picture import Picture


script.add_argument("-s", dest="seed", type=int, default=81)
script.add_argument("-si", dest="size", type=int, default=512)
script.add_argument("-l", dest="limit", type=int,  default=1_000_000)
script.add_argument("-d", dest="destination", type=str, default="/home/hp/datasets")


@script
def main(args):
    dataset = AugmentedBoardsImages(
        boards_with_data=BoardsImagesFromFENs(
            fens=FileLinesDataset(
                path=Path(config.dataset.path) / "unique_fens.txt",
            ),
            piece_sets=PIECE_SETS,
            board_colors=BOARD_COLORS,
            square_size=64,
            shuffle_buffer=32,
            shuffle_seed=args.seed,
        ),
        bg_images=BG_IMAGES,
        shuffle_buffer=1,
        shuffle_seed=args.seed,
        limit=args.limit,
    )

    destination = reset_dir(Path(args.destination) / f"augmented_boards_{args.size}_{args.limit}")

    def process_one(i, item):
        picture, coords, fen, flipped = item

        (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = coords

        image = cv2.resize(picture.cv2, (args.size, args.size))

        cv2.imwrite(
            str(destination / f"{i + 1}.jpg"),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )

        write_lines_to_txt(destination / f"{i + 1}.txt", [
            f"{tl_x} {tl_y} {tr_x} {tr_y} {br_x} {br_y} {bl_x} {bl_y}",
            fen,
            "1" if flipped else "0",
        ])

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, item in enumerate(tqdm(dataset, total=args.limit)):
            futures.append(executor.submit(process_one, i, item))

        for future in as_completed(futures):
            future.result()
        
