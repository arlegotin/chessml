from chessml import script
import cv2
from chessml.models.torch.vision_model_adapter import VisionModelAdapter
from chessml.models.lightning.board_detector_model import BoardDetector
import logging

logger = logging.getLogger(__name__)


def draw_bounding_box(path, coords):
    image = cv2.imread(path)
    # Extract coordinates
    x1, x2, y1, y2 = coords

    # Convert coordinates to pixel scale based on image dimensions
    height, width = image.shape[:2]
    start_point = (int(x1 * width), int(y1 * height))
    end_point = (int(x2 * width), int(y2 * height))

    # Color of the rectangle (B, G, R)
    color = (0, 255, 0)

    # Thickness of the rectangle (in pixels)
    thickness = 2

    # Using cv2.rectangle to draw the bounding box on a copy of the image
    image_with_box = cv2.rectangle(
        image.copy(), start_point, end_point, color, thickness
    )

    return image_with_box


def predict_coords(path):
    image = cv2.imread(path)

    model = BoardDetector.load_from_checkpoint(
        # "./checkpoints/bd-new-mn=resnet50-bs=96-step=18688.ckpt", base_model=base_model,
        "./checkpoints/bd-again-mn=efficientnet_b0-bs=96-step=6400.ckpt",
        base_model_class=VisionModelAdapter,
        base_model_kwargs={"model_name": "efficientnet_b0", "output_features": 4,},
    )

    model.eval()

    tensor_image = model.model.preprocess_image(image).unsqueeze(0).to(model.device)

    coords = model(tensor_image)

    return coords[0]


@script
def train(args, config):
    path = "./assets/test/5.png"

    coords = predict_coords(path)

    coords = list(coords.detach().cpu().numpy())

    res = draw_bounding_box(path, coords)

    cv2.imwrite("box.png", res)
