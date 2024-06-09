from chessml import script, config
import cv2
from chessml.models.torch.vision_model_adapter import MobileViTV2FPN, MobileNetV3S50Classifier
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper
import logging
from chessml.data.assets import PIECE_CLASSES_NUMBER, INVERTED_PIECE_CLASSES
from glob import glob
from fentoboardimage import fenToImage, loadPiecesFolder
from chessml.utils import reset_dir
from pathlib import Path
from PIL import Image
from chessml.data.images.picture import Picture
from tqdm import tqdm

logger = logging.getLogger(__name__)

script.add_argument("-i", dest="input_dir", type=str)
script.add_argument("-ss", dest="square_size", type=int, default=32)

@script
def main(args):
    board_detector = BoardDetector.load_from_checkpoint(
        "./checkpoints/bd-skew-40-bs=100-step=5120.ckpt",
        base_model_class=MobileViTV2FPN,
        map_location='cpu',
    )
    board_detector.eval()

    piece_classifier = PieceClassifier.load_from_checkpoint(
        # "./checkpoints/pc-6-bs=64-step=5376.ckpt",
        "./checkpoints/pc-7-bs=64-step=13568.ckpt",
        base_model_class=MobileNetV3S50Classifier,
        map_location='cpu',
    )
    piece_classifier.eval()


    helper = BoardRecognitionHelper(
        board_detector=board_detector,
        piece_classifier=piece_classifier,
    )

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

        ex = extracted_image.pil.resize((args.square_size * 8, args.square_size * 8))
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
