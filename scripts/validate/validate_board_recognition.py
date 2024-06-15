from chessml import script, config
import cv2
from chessml.models.torch.vision_model_adapter import (
    MobileViTV2FPN,
    EfficientNetV2Classifier,
)
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper
from chessml.models.lightning.meta_predictor_model import MetaPredictor
import logging
from chessml.data.assets import PIECE_CLASSES_NUMBER, INVERTED_PIECE_CLASSES
from glob import glob
from fentoboardimage import fenToImage, loadPiecesFolder
from chessml.utils import reset_dir, write_lines_to_txt
from pathlib import Path
from PIL import Image
from chessml.data.images.picture import Picture
from tqdm import tqdm
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.data.assets import BOARD_COLORS, PIECE_SETS
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.boards.board_representation import OnlyPieces

logger = logging.getLogger(__name__)

script.add_argument("-i", dest="input_dir", type=str, default="")
script.add_argument("-ss", dest="square_size", type=int, default=32)
script.add_argument("-d", dest="device", type=str, default="cpu")


@script
def main(args):

    board_detector = BoardDetector.load_from_checkpoint(
        "./checkpoints/bd-skew-40-bs=100-step=5120.ckpt",
        base_model_class=MobileViTV2FPN,
        map_location=args.device,
    )
    board_detector.eval()

    piece_classifier = PieceClassifier.load_from_checkpoint(
        "./checkpoints/pc-37-bs=64-step=35840.ckpt",
        base_model_class=EfficientNetV2Classifier,
        map_location=args.device,
    )
    piece_classifier.eval()

    meta_predictor = MetaPredictor.load_from_checkpoint(
        "./checkpoints/bm-6-bs=1024-step=17920.ckpt",
        input_shape=OnlyPieces().shape,
        map_location=args.device,
    )
    meta_predictor.eval()

    helper = BoardRecognitionHelper(
        board_detector=board_detector,
        piece_classifier=piece_classifier,
        meta_predictor=meta_predictor,
    )

    if args.input_dir:
        input_dir = Path(args.input_dir)

        marked_dir = reset_dir(input_dir.parent / f"{input_dir.stem}_marked")
        extracted_dir = reset_dir(input_dir.parent / f"{input_dir.stem}_extracted")
        boards_dir = reset_dir(input_dir.parent / f"{input_dir.stem}_boards")

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
                flipped=False,
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
