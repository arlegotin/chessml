import numpy as np
import pytest
import torch
from PIL import Image

from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import (
    BoardDetector,
    InvalidBoardGeometryError,
)


class ModelStub(torch.nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.output_features = output_features
        self.preprocessed_images = []

    def preprocess_image(self, image):
        self.preprocessed_images.append(image.copy())
        return torch.zeros((3, image.height, image.width), dtype=torch.float32)

    def forward(self, batch):
        return torch.zeros(
            (batch.shape[0], self.output_features),
            dtype=batch.dtype,
            device=batch.device,
        )


def detector_with_coords(monkeypatch, coords):
    detector = BoardDetector(base_model_class=ModelStub)
    monkeypatch.setattr(
        detector,
        "predict_coords",
        lambda _image: np.asarray(coords, dtype=np.float32),
    )
    return detector


IMAGE = Picture(np.zeros((100, 100, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("image_input", "expected_rgb"),
    [
        (Image.new("RGB", (4, 3), (10, 20, 30)), (10, 20, 30)),
        (Image.new("RGBA", (4, 3), (10, 20, 30, 255)), (10, 20, 30)),
        (Image.new("L", (4, 3), 10), (10, 10, 10)),
        (np.full((3, 4, 3), (30, 20, 10), dtype=np.uint8), (10, 20, 30)),
    ],
    ids=("rgb", "rgba", "grayscale", "opencv-bgr"),
)
def test_predict_coords_normalizes_input_to_rgb(image_input, expected_rgb):
    source = Picture(image_input)
    source_before = source.cv2.copy()
    detector = BoardDetector(base_model_class=ModelStub)

    coords = detector.predict_coords(source)

    assert len(detector.model.preprocessed_images) == 1
    preprocessed = detector.model.preprocessed_images[0]
    assert preprocessed.mode == "RGB"
    assert preprocessed.size == (4, 3)
    assert preprocessed.getpixel((0, 0)) == expected_rgb
    np.testing.assert_array_equal(source.cv2, source_before)
    np.testing.assert_array_equal(coords, np.full(8, 0.5, dtype=np.float32))


@pytest.mark.parametrize(
    ("mode", "color"),
    [
        ("RGB", (0, 0, 0)),
        ("RGBA", (0, 0, 0, 255)),
        ("L", 0),
    ],
)
def test_mark_board_on_image_returns_marked_copy(monkeypatch, mode, color):
    source = Picture(Image.new(mode, (100, 100), color))
    before = np.asarray(source.pil).copy()
    unmarked_rgb = np.asarray(source.as_3_channels.pil).copy()
    detector = detector_with_coords(
        monkeypatch,
        [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
    )

    marked = detector.mark_board_on_image(source)

    assert source.pil.mode == mode
    np.testing.assert_array_equal(np.asarray(source.pil), before)
    assert marked.pil.mode == "RGB"
    assert np.any(np.asarray(marked.pil) != unmarked_rgb)


@pytest.mark.parametrize(
    ("mode", "color", "raw_shape"),
    [
        ("RGBA", (10, 20, 30, 255), (100, 100, 4)),
        ("L", 10, (100, 100)),
    ],
)
def test_extract_board_image_normalizes_file_channels(
    monkeypatch, tmp_path, mode, color, raw_shape
):
    image_path = tmp_path / f"source-{mode}.png"
    Image.new(mode, (100, 100), color).save(image_path)
    source = Picture(image_path)
    assert source.cv2.shape == raw_shape
    detector = detector_with_coords(
        monkeypatch,
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    )

    extracted = detector.extract_board_image(source)

    assert extracted.cv2.shape == (100, 100, 3)


@pytest.mark.parametrize(
    "coords",
    [
        [np.nan, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
        [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9],
        [0.1, 0.1, 0.9, 0.9, 0.1, 0.9, 0.7, 0.1],
        [0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9],
        [0.0, 0.0, 0.001, 0.0, 0.001, 0.001, 0.0, 0.001],
    ],
)
def test_extract_board_image_rejects_unusable_geometry(monkeypatch, coords):
    detector = detector_with_coords(monkeypatch, coords)

    with pytest.raises(InvalidBoardGeometryError):
        detector.extract_board_image(IMAGE)


def test_extract_board_image_scales_rotated_edges_in_pixel_space(monkeypatch):
    source = Picture(np.zeros((800, 1600, 3), dtype=np.uint8))
    detector = detector_with_coords(
        monkeypatch,
        [0.25, 0.125, 0.5, 0.5, 0.40625, 0.75, 0.15625, 0.375],
    )

    extracted = detector.extract_board_image(source)

    assert isinstance(extracted, Picture)
    assert extracted.cv2.shape == (250, 500, 3)
