from chess import Board
from torch_exid import ExtendedIterableDataset
from typing import Iterable, Iterator
from chessml.data.utils.looped_list import LoopedList
from chessml.data.images.picture import Picture
from pathlib import Path
from fentoboardimage import fenToImage, loadPiecesFolder
from typing import Optional
import numpy as np


class BoardsImagesFromFENs(ExtendedIterableDataset):
    def __init__(
        self,
        fens: Iterable[str],
        piece_sets: list[Path],
        board_colors: list[tuple[str, str]],
        square_size: int,
        shuffle_seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms_required=False, shuffle_seed=shuffle_seed, *args, **kwargs
        )
        self.fens = fens

        self.piece_sets = LoopedList(
            list(map(lambda x: loadPiecesFolder(str(x)), piece_sets)),
            shuffle_seed=shuffle_seed,
        )
        self.board_colors = LoopedList(board_colors, shuffle_seed=shuffle_seed + 1)
        self.square_size = square_size

    def generator(self) -> Iterator[tuple[Picture, str, bool]]:
        for i, fen in enumerate(self.fens):
            dark, light = self.board_colors[i]
            piece_set = self.piece_sets[i]
            flipped = i % 2 == 1

            img = fenToImage(
                fen=fen,
                squarelength=self.square_size,
                pieceSet=piece_set,
                darkColor=dark,
                lightColor=light,
                flipped=flipped,
            )

            yield Picture(img), fen, flipped
