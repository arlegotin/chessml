import logging
from glob import glob
from pathlib import Path

import cv2
from fentoboardimage import fenToImage, loadPiecesFolder
from PIL import Image
from tqdm import tqdm

from chessml import config, script
from chessml.data.assets import (
    BOARD_COLORS,
    INVERTED_PIECE_CLASSES,
    PIECE_CLASSES_NUMBER,
    PIECE_SETS,
)
from chessml.data.boards.board_representation import OnlyPieces
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.picture import Picture
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.lightning.meta_predictor_model import MetaPredictor
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.square_classifier_model import SquareClassifier
from chessml.models.torch.vision_model_adapter import (
    EfficientNetV2Classifier,
    MobileNetV3LargeClassifier,
    MobileNetV3SmallClassifier,
    MobileViTV2FPN,
)
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper
from chessml.utils import reset_dir, write_lines_to_txt

logger = logging.getLogger(__name__)

script.add_argument(
    "-i", dest="input_dir", type=str, default="./test_data/input_frames/book_short"
)
script.add_argument("-ss", dest="square_size", type=int, default=32)
script.add_argument("-d", dest="device", type=str, default="mps")


@script
def main(args):

    board_detector = BoardDetector.load_from_checkpoint(
        "./checkpoints/bd-MobileViTV2FPN-v1.ckpt",
        base_model_class=MobileViTV2FPN,
        map_location=args.device,
    )
    board_detector.eval()

    square_classifier = SquareClassifier.load_from_checkpoint(
        # "./checkpoints/sc-9-bs=64-step=4864.ckpt",
        "./checkpoints/sc-9-bs=64-step=23296.ckpt",
        base_model_class=MobileNetV3SmallClassifier,
        map_location=args.device,
    )
    square_classifier.eval()

    piece_classifier = PieceClassifier.load_from_checkpoint(
        # "./checkpoints/pc-44-bs=128-step=7296.ckpt",
        # "./checkpoints/pc-48-bs=128-step=9216.ckpt",
        "./checkpoints/pc-48-bs=128-step=18944.ckpt",
        # "./checkpoints/pc-48-bs=128-step=15872.ckpt",
        base_model_class=MobileNetV3LargeClassifier,
        map_location=args.device,
    )
    piece_classifier.eval()

    meta_predictor = MetaPredictor.load_from_checkpoint(
        "./checkpoints/mp-MetaPredictor-v1.ckpt",
        input_shape=OnlyPieces().shape,
        map_location=args.device,
    )
    meta_predictor.eval()

    helper = BoardRecognitionHelper(
        board_detector=board_detector,
        square_classifier=square_classifier,
        piece_classifier=piece_classifier,
        meta_predictor=meta_predictor,
    )

    if args.input_dir:
        input_dir = Path(args.input_dir)

        marked_dir = reset_dir(input_dir.parent / f"{input_dir.stem}_marked")
        extracted_dir = reset_dir(input_dir.parent / f"{input_dir.stem}_extracted")
        boards_dir = reset_dir(input_dir.parent / f"{input_dir.stem}_boards")
        squares_dir = reset_dir(input_dir.parent / f"{input_dir.stem}_squares")

        for i, image_path in tqdm(enumerate(sorted(glob(f"{input_dir}/*.png")))):
            image_path = Path(image_path)

            original_image = Picture(image_path)

            marked_image = helper.board_detector.mark_board_on_image(original_image)
            marked_image.pil.save(marked_dir / image_path.name)

            extracted_image = helper.board_detector.extract_board_image(original_image)

            ex = extracted_image.pil.resize(
                (args.square_size * 8, args.square_size * 8)
            )
            ex.save(extracted_dir / image_path.name)

            result = helper.recognize(original_image)

            board_image = fenToImage(
                fen=result.get_fen(),
                squarelength=args.square_size,
                pieceSet=loadPiecesFolder("assets/piece_png/lichess_cburnett"),
                darkColor="#B58862",
                lightColor="#F0D9B5",
                flipped=result.flipped,
            )

            board_image.save(boards_dir / image_path.name)

            write_lines_to_txt(
                boards_dir / f"{image_path.name}.txt", [result.get_fen()]
            )
    else:
        dataset = BoardsImagesFromFENs(
            fens=FileLinesDataset(path=Path(config.dataset.path) / "unique_fens.txt"),
            piece_sets=PIECE_SETS,
            board_colors=BOARD_COLORS,
            square_size=64,
            shuffle_seed=10,
            limit=1024,
        )

        for pic, fen, flipped in dataset:
            result = helper.recognize(pic)
            print("----------")
            print(f"{fen} 0 1", flipped)
            print(result.get_fen(), result.flipped)
