from chess import Board
from fentoboardimage import fen_to_image, load_pieces_folder
from PIL import Image


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
