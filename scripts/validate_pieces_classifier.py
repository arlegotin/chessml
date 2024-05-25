from chessml import script
import cv2
from chessml.models.torch.vision_model_adapter import VisionModelAdapter
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper
import logging
from chessml.const import PIECE_CLASSES_NUMBER, INVERTED_PIECE_CLASSES

logger = logging.getLogger(__name__)

@script
def main(args, config):
    helper = BoardRecognitionHelper(
        board_image=cv2.imread("./box.png"),
    )

    model = PieceClassifier.load_from_checkpoint(
        "./checkpoints/pc-2-mn=efficientnet_b0-bs=96-step=26624.ckpt",
        base_model_class=VisionModelAdapter,
        base_model_kwargs={"model_name": "efficientnet_b0", "output_features": PIECE_CLASSES_NUMBER,},
    )

    model.eval()

    for square, rank, file in helper.iterate_squares(square_size=128):      
        class_index = model.classify_piece(square)
        print(square.shape, rank, file, class_index, INVERTED_PIECE_CLASSES[class_index])
        cv2.imwrite(f"./tmp/{rank}_{file}_{INVERTED_PIECE_CLASSES[class_index]}.png", square)
        helper.set_square_class(rank, file, class_index)
        
    print(helper.get_fen())
