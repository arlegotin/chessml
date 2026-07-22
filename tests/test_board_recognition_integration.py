from pathlib import Path

import pytest

from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.square_classifier_model import SquareClassifier
from chessml.models.torch.vision_model_adapter import (
    MobileNetV3LargeClassifier,
    MobileNetV3SmallClassifier,
    MobileViTV2FPN,
)
from chessml.models.utils.board_recognition_helper import (
    BoardRecognitionHelper,
    RecognitionFailure,
    RecognitionSuccess,
)


ROOT = Path(__file__).resolve().parents[1]
BOARD_CHECKPOINT = ROOT / "checkpoints/bd-MobileViTV2FPN-v1.ckpt"
SQUARE_CHECKPOINT = ROOT / "checkpoints/sc-9-bs=64-step=23296.ckpt"
PIECE_CHECKPOINT = ROOT / "checkpoints/pc-48-bs=128-step=18944.ckpt"
FRAME = ROOT / "test_data/input_frames/book_long/1.png"

REQUIRED_ASSETS = [
    BOARD_CHECKPOINT,
    SQUARE_CHECKPOINT,
    PIECE_CHECKPOINT,
    FRAME,
]
MISSING_ASSETS = [path for path in REQUIRED_ASSETS if not path.is_file()]

pytestmark = pytest.mark.skipif(
    bool(MISSING_ASSETS),
    reason="requires ignored local inference assets: "
    + ", ".join(str(path.relative_to(ROOT)) for path in MISSING_ASSETS),
)


def test_existing_checkpoints_run_board_recognition_on_cpu():
    board_detector = BoardDetector.load_from_checkpoint(
        BOARD_CHECKPOINT,
        base_model_class=MobileViTV2FPN,
        base_model_kwargs={"pretrained": False},
        map_location="cpu",
        strict=True,
        weights_only=False,
    )
    square_classifier = SquareClassifier.load_from_checkpoint(
        SQUARE_CHECKPOINT,
        base_model_class=MobileNetV3SmallClassifier,
        base_model_kwargs={"pretrained": False},
        map_location="cpu",
        strict=True,
        weights_only=False,
    )
    piece_classifier = PieceClassifier.load_from_checkpoint(
        PIECE_CHECKPOINT,
        base_model_class=MobileNetV3LargeClassifier,
        base_model_kwargs={"pretrained": False},
        map_location="cpu",
        strict=True,
        weights_only=False,
    )

    for model in (board_detector, square_classifier, piece_classifier):
        model.eval()

    result = BoardRecognitionHelper(
        board_detector=board_detector,
        square_classifier=square_classifier,
        piece_classifier=piece_classifier,
    ).recognize(Picture(FRAME))

    assert isinstance(result, (RecognitionSuccess, RecognitionFailure))
