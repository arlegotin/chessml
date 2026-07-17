from chessml import script
from pathlib import Path
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3
import os
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, FREE_PIECE_SETS
import cv2
from chessml.utils import reset_dir
from fentoboardimage import fen_to_image, load_pieces_folder


@script
def main(args):

    output_dir = reset_dir(Path("./output/visualized_piece_sets"))

    for piece_set in PIECE_SETS:
        img = fen_to_image(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            square_length=64,
            piece_set=load_pieces_folder(str(piece_set)),
            dark_color="#B58862",
            light_color="#F0D9B5",
            flipped=False,
        )

        img.save(str(output_dir / f"{piece_set.stem}.jpg"))
