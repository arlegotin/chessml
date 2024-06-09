from chessml import script
import cv2
from chessml.models.lightning.board_detector_model import BoardDetector
import logging
from PIL import Image
from chessml.models.torch.vision_model_adapter import MobileViTV2FPN

logger = logging.getLogger(__name__)

@script
def main(args, config):
    # path = "assets/frames_selected/levy/frame_0016.png"
    path = "assets/frames_selected/book3/frame_0041.png"
    # path = "assets/frames_selected/book/frame_0130.png"
    # path = "assets/frames_selected/some/frame_0010.png"
    # path = "assets/frames_selected/phone/frame_0016.png"
    # path = "tmp/89.jpg"
    # path = "tmp/94.jpg"

    image = Image.open(path)

    model = BoardDetector.load_from_checkpoint(
        # "./checkpoints/bd-skew-29-bs=1024-step=21376.ckpt",
        "./checkpoints/bd-skew-40-bs=100-step=5120.ckpt",
        base_model_class=MobileViTV2FPN,
        # base_model_kwargs={"size": 256},
        map_location='cpu',
    )

    model.eval()

    new_image = model.mark_board_on_image(image)

    new_image.save("box.png")
