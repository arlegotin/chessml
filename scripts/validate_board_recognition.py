from chessml import script
import cv2
from chessml.models.torch.vision_model_adapter import MobileViTAdapter, EfficientNetV2Adapter
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper
import logging
from chessml.const import PIECE_CLASSES_NUMBER, INVERTED_PIECE_CLASSES
from glob import glob
from fentoboardimage import fenToImage, loadPiecesFolder
from chessml.utils import reset_dir
from pathlib import Path

logger = logging.getLogger(__name__)

@script
def main(args, config):
    board_detector = BoardDetector.load_from_checkpoint(
        "./checkpoints/bd-skew-2-bs=160-step=25600.ckpt",
        base_model_class=MobileViTAdapter,
    )
    board_detector.eval()

    piece_classifier = PieceClassifier.load_from_checkpoint(
        # "./checkpoints/pc-2-bs=160-step=6400.ckpt",
        # base_model_class=MobileViTAdapter,
        "./checkpoints/pc-2-bs=160-step=1024.ckpt",
        base_model_class=EfficientNetV2Adapter,
    )
    piece_classifier.eval()


    helper = BoardRecognitionHelper(
        board_detector=board_detector,
        piece_classifier=piece_classifier,
    )

    name = "book3"

    reset_dir(Path(f"./output/{name}_marked"))
    reset_dir(Path(f"./output/{name}_boards"))

    for i, p in enumerate(sorted(glob(f"./assets/frames_selected/{name}/*.png"))):
        orig_img = cv2.imread(p)
        # orig_img = cv2.resize(orig_img, (1280, 720))
        img, _ = helper.board_detector.mark_board_on_image(orig_img)
        fn = f"{i:04d}.png"
        cv2.imwrite(f"./output/{name}_marked/{fn}", img)
        result = helper.recognize(orig_img)
        
        board_img = fenToImage(
            fen=result.get_fen(),
            squarelength=32,
            pieceSet=loadPiecesFolder("assets/piece_png/lichess_cburnett"),
            darkColor="#B58862",
            lightColor="#F0D9B5",
            flipped=False,
        )

        board_img.save(f"./output/{name}_boards/{fn}")
        print(i)
