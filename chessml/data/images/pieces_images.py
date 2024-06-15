from pathlib import Path
import numpy as np
import cv2
from torch_exid import ExtendedIterableDataset
from typing import Iterable, Iterator, Optional
from chessml.data.utils.looped_list import LoopedList
from chessml.data.images.augment import (
    Augmentator,
    apply_perspective_warp,
)
import random
import itertools
from chessml.data.images.picture import Picture
from chessml import config

# Keys must be the same as in PIECE_CLASSES
piece_file_names = {
    None: None,
    "p": "black/Pawn",
    "r": "black/Rook",
    "n": "black/Knight",
    "b": "black/Bishop",
    "q": "black/Queen",
    "k": "black/King",
    "P": "white/Pawn",
    "R": "white/Rook",
    "N": "white/Knight",
    "B": "white/Bishop",
    "Q": "white/Queen",
    "K": "white/King",
}


def hex_to_bgr(hex_color):
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return rgb[::-1]


class PiecesImages3x3(ExtendedIterableDataset):
    def __init__(
        self,
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

        pieces_pictures_with_names = []

        for piece_set in piece_sets:
            for piece_name, piece_location in piece_file_names.items():
                if piece_name is not None:
                    pieces_pictures_with_names.append(
                        (Picture(piece_set / f"{piece_location}.png"), piece_name)
                    )
                else:
                    empty = Picture(np.zeros((1, 1, 4), dtype=np.uint8))

                    # Adding two empty squares to compensate for the two sides of the pieces
                    # pieces_pictures_with_names.append((empty, piece_name))
                    pieces_pictures_with_names.append((empty, piece_name))

        self.pieces_pictures_with_names = LoopedList(
            pieces_pictures_with_names, shuffle_seed=shuffle_seed
        )

        backgrounds_pictures = []

        for dark, light in board_colors:
            backgrounds_pictures.append(
                (
                    Picture(np.full((1, 1, 3), hex_to_bgr(dark), dtype=np.uint8)),
                    Picture(np.full((1, 1, 3), hex_to_bgr(light), dtype=np.uint8)),
                )
            )

        self.backgrounds_pictures = LoopedList(
            backgrounds_pictures, shuffle_seed=shuffle_seed
        )

        self.square_size = square_size

    def generator(self) -> Iterator[tuple[Picture, str]]:
        for i in itertools.count():
            main_piece, name = self.pieces_pictures_with_names[i]
            dark, light = self.backgrounds_pictures[i]

            squares = []
            for j in range(9):
                """
                (i ^ j) % 2 allows to alternate dark and light squares bot for i and j
                """
                background = cv2.resize(
                    (dark if (i ^ j) % 2 else light).cv2,
                    (self.square_size, self.square_size),
                )

                """
                4 is the index of the main piece
                other pieces are random
                """
                piece = cv2.resize(
                    (
                        main_piece
                        if j == 4
                        else self.pieces_pictures_with_names[(i + 1) * (j + 1)][0]
                    ).cv2,
                    (self.square_size, self.square_size),
                )

                alpha_channel = piece[:, :, 3]
                rgb_channels = piece[:, :, :3]

                alpha_factor = alpha_channel[..., np.newaxis] / 255.0
                foreground = alpha_factor * rgb_channels
                background = (1.0 - alpha_factor) * background

                combined = cv2.add(foreground, background).astype(np.uint8)
                squares.append(combined)

            grid = np.vstack(
                (
                    np.hstack(squares[:3]),
                    np.hstack(squares[3:6]),
                    np.hstack(squares[6:]),
                )
            )

            yield Picture(grid), name


class AugmentedPiecesImages(ExtendedIterableDataset):
    def __init__(
        self,
        piece_images_3x3: Iterable[tuple[Picture, str]],
        shuffle_seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms_required=False, shuffle_seed=shuffle_seed, *args, **kwargs
        )

        self.piece_images_3x3 = piece_images_3x3

        self.augmentator = Augmentator(
            config=config.dataset.augmentations.AugmentedPiecesImages,
            seed=shuffle_seed,
        )

    def generator(self,) -> Iterator[tuple[Picture, str]]:
        crop_delta = 0.1

        for original_picture, piece_name in self.piece_images_3x3:

            square_size = original_picture.cv2.shape[0] // 3

            augmented_image, _ = apply_perspective_warp(
                original_picture.cv2,
                max_skew=0.03,
                max_rotation=3,
                x=square_size,
                y=square_size,
                size=square_size,
            )

            augmented_image = self.augmentator.shift(augmented_image, min_shift=0, max_shift=0.05)
            augmented_image = self.augmentator.center_crop(augmented_image, size=square_size, delta=0.1)
            augmented_image = cv2.resize(augmented_image, (square_size, square_size))

            augmented_image = self.augmentator(augmented_image)

            yield Picture(augmented_image), piece_name
