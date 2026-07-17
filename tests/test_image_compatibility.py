import numpy as np
from chess import Board
from fentoboardimage import fen_to_image, load_pieces_folder
from PIL import Image

from chessml.data.images.picture import Picture


def test_renderer_api_works_with_pillow():
    image = fen_to_image(
        fen=Board.empty().fen(),
        square_length=16,
        piece_set=lambda _overlay: {},
        dark_color="#B58862",
        light_color="#F0D9B5",
        flipped=False,
    )

    assert callable(load_pieces_folder)
    assert isinstance(image, Image.Image)
    assert image.size == (128, 128)


def test_picture_preserves_pixels_across_pillow_opencv_round_trip():
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [12, 34, 56]],
        ],
        dtype=np.uint8,
    )

    restored = Picture(Picture(Image.fromarray(rgb)).cv2).pil

    np.testing.assert_array_equal(np.asarray(restored), rgb)
