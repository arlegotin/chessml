from pathlib import Path
import numpy as np
import cv2
from chessml.data.iterable_dataset import ExtendedIterableDataset
from typing import Iterable, Iterator, Optional
from chessml.data.images.augment import (
    Augmentator,
    apply_perspective_warp,
)
import random
from chessml.data.images.picture import Picture
from chessml.data.constants import EMPTY_SQUARE_CHANCE
from chessml import config

# Keys must be the same as in PIECE_CLASSES
PIECE_FILE_NAMES = {
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
        with_empty_squares: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms_required=False, shuffle_seed=shuffle_seed, *args, **kwargs
        )

        self.piece_sets = [
            (
                piece_set.name,
                {
                    piece_name: Picture(piece_set / f"{piece_location}.png")
                    for piece_name, piece_location in PIECE_FILE_NAMES.items()
                },
            )
            for piece_set in piece_sets
        ]
        self.backgrounds = [
            (
                dark,
                light,
                Picture(np.full((1, 1, 3), hex_to_bgr(dark), dtype=np.uint8)),
                Picture(np.full((1, 1, 3), hex_to_bgr(light), dtype=np.uint8)),
            )
            for dark, light in board_colors
        ]
        self.center_labels = list(PIECE_FILE_NAMES)
        if with_empty_squares:
            self.center_labels += [None] * len(PIECE_FILE_NAMES)
        self.empty_picture = Picture(np.zeros((1, 1, 4), dtype=np.uint8))
        self.square_size = square_size

    def generator(self) -> Iterator[tuple[Picture, str | None, str, str, str]]:
        rng = random.Random(self.shuffle_seed)
        piece_labels = tuple(PIECE_FILE_NAMES)

        while True:
            center_labels = self.center_labels[:]
            rng.shuffle(center_labels)

            for center_label in center_labels:
                piece_set_name, pieces = rng.choice(self.piece_sets)
                dark_color, light_color, dark, light = rng.choice(self.backgrounds)
                neighbor_labels = [
                    None
                    if rng.random() < EMPTY_SQUARE_CHANCE
                    else rng.choice(piece_labels)
                    for _ in range(8)
                ]
                labels = neighbor_labels[:4] + [center_label] + neighbor_labels[4:]

                for swap_backgrounds in (False, True):
                    squares = []

                    for index, label in enumerate(labels):
                        background_picture = (
                            dark
                            if (index % 2 == 0) ^ swap_backgrounds
                            else light
                        )
                        background = cv2.resize(
                            background_picture.cv2,
                            (self.square_size, self.square_size),
                        )
                        piece = cv2.resize(
                            (
                                self.empty_picture
                                if label is None
                                else pieces[label]
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

                    yield (
                        Picture(grid),
                        center_label,
                        piece_set_name,
                        dark_color,
                        light_color,
                    )


class AugmentedPiecesImages(ExtendedIterableDataset):
    def __init__(
        self,
        piece_images_3x3: Iterable[tuple[Picture, str | None, str, str, str]],
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

    def generator(self,) -> Iterator[tuple[Picture, str | None, str, str, str]]:

        for original_picture, piece_name, *provenance in self.piece_images_3x3:

            square_size = original_picture.cv2.shape[0] // 3

            augmented_image, _ = apply_perspective_warp(
                original_picture.cv2,
                max_skew=0.03,
                max_rotation=5,
                x=square_size,
                y=square_size,
                size=square_size,
            )

            augmented_image = self.augmentator.shift(augmented_image, min_shift=0, max_shift=0.07)
            augmented_image = self.augmentator.center_crop(augmented_image, size=square_size, delta=0.07)
            augmented_image = cv2.resize(augmented_image, (square_size, square_size))

            augmented_image = self.augmentator(augmented_image)

            yield Picture(augmented_image), piece_name, *provenance
