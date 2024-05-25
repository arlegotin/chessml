from chessml import script
import cv2
from chessml.models.torch.vision_model_adapter import VisionModelAdapter, MobileViTAdapter, HRNetAdapter
from chessml.models.lightning.board_detector_model import BoardDetector
import logging
from pathlib import Path
from chessml.utils import reset_dir
import numpy as np

logger = logging.getLogger(__name__)

@script
def main(args, config):
    model = BoardDetector.load_from_checkpoint(
        # "./checkpoints/bd-skew-1-bs=160-step=10240.ckpt",
        "./checkpoints/bd-skew-2-bs=160-step=25600.ckpt",
        base_model_class=MobileViTAdapter,
    )

    model.eval()

    dir = Path("./assets/frames_selected/print1")

    output_dir = reset_dir(Path("./tmp"))

    for img_path in sorted(dir.glob('*.png')):
        img = cv2.imread(str(img_path))
        img, _ = model.mark_board_on_image(img)
        # img, _ = model.extract_board_image(img)
        cv2.imwrite(str(output_dir / img_path.name), img)
