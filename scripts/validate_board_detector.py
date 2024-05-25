from chessml import script
import cv2
from chessml.models.torch.vision_model_adapter import VisionModelAdapter
from chessml.models.lightning.board_detector_model import BoardDetector
import logging
from PIL import Image

logger = logging.getLogger(__name__)


# def draw_bounding_box(path, coords):
#     image = cv2.imread(path)
#     # Extract coordinates
#     x1, x2, y1, y2 = coords

#     # Convert coordinates to pixel scale based on image dimensions
#     height, width = image.shape[:2]
#     start_point = (int(x1 * width), int(y1 * height))
#     end_point = (int(x2 * width), int(y2 * height))

#     # Color of the rectangle (B, G, R)
#     color = (0, 255, 0)

#     # Thickness of the rectangle (in pixels)
#     thickness = 2

#     # Using cv2.rectangle to draw the bounding box on a copy of the image
#     image_with_box = cv2.rectangle(
#         image.copy(), start_point, end_point, color, thickness
#     )

#     return image_with_box

@script
def main(args, config):
    # path = "./tmp2/73.jpg"
    path = "/mnt/md0/projects_moved_from_ssd/artem/chessml/tmp2/96.jpg"
    # path = "./87.jpg"
    image = cv2.imread(path)

    model = BoardDetector.load_from_checkpoint(
        # "./checkpoints/bd-new-mn=resnet50-bs=96-step=18688.ckpt", base_model=base_model,
        "./checkpoints/bd-6-mn=efficientnetv2_rw_t_ra2_in1k-bs=320-step=5120.ckpt",
        base_model_class=VisionModelAdapter,
        base_model_kwargs={"model_name": "efficientnetv2_rw_t.ra2_in1k"},
    )

    model.eval()

    b = model.extract_board_image(image)

    if b is None:
        print("No board")

    print("Board", b.shape)

    cv2.imwrite("box2.png", b)
