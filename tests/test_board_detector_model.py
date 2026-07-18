import numpy as np
import pytest
import torch

from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import (
    BoardDetector,
    InvalidBoardGeometryError,
)


class ModelStub(torch.nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.output_features = output_features


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


def test_extract_board_image_accepts_a_valid_quadrilateral(monkeypatch):
    detector = detector_with_coords(
        monkeypatch,
        [0.1, 0.2, 0.9, 0.1, 0.8, 0.9, 0.2, 0.8],
    )

    extracted = detector.extract_board_image(IMAGE)

    assert isinstance(extracted, Picture)
    assert extracted.cv2.shape[0] > 0
    assert extracted.cv2.shape[1] > 0
