from chessml import script
from pathlib import Path
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3
import os
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, FREE_PIECE_SETS
import cv2
from chessml.utils import reset_dir
from fentoboardimage import fenToImage, loadPiecesFolder


@script
def main(args):

    output_dir = reset_dir(Path("./output/visualized_piece_sets"))

    for piece_set in PIECE_SETS:
        img = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=64,
            pieceSet=loadPiecesFolder(str(piece_set)),
            darkColor="#B58862",
            lightColor="#F0D9B5",
            flipped=False,
        )

        img.save(str(output_dir / f"{piece_set.stem}.jpg"))
