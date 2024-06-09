from pathlib import Path
import numpy as np
import cv2
from torch_exid import ExtendedIterableDataset
from typing import Iterable, Iterator, Optional
from chessml.data.utils.looped_list import LoopedList
from chessml.data.utils.augment import (
    random_crop,
    add_random_lines,
    add_random_text,
    add_gaussian_noise,
    apply_gaussian_blur,
    add_jpeg_artifacts,
    resolution_jitter,
    add_shift,
    apply_perspective_warp,
    center_crop,
    add_brightness,
    add_contrast,
    add_saturation,
    apply_motion_blur,
)
import random
import itertools
from chessml.data.images.picture import Picture

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
    h = hex_color.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]

def generate_pieces_images(
    piece_sets: list[Path],
    board_colors: list[tuple[str, str]],
    size: int,
) -> Iterator[tuple[Picture, str]]:
    for piece_set in piece_sets:
        for piece_name, piece_location in piece_file_names.items():
            if piece_name is not None:
                path = piece_set / f"{piece_location}.png"
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

                if image is None:
                    raise RuntimeError(f"Coultn't read piece from {path}")
            else:
                image = np.zeros((1, 1, 4), dtype=np.uint8)

            image = cv2.resize(image, (size, size))

            for dark, light in board_colors:
                for background_color in map(hex_to_bgr, [dark, light]):
                    background = np.full((size, size, 3), background_color, dtype=np.uint8)

                    alpha_channel = image[:, :, 3]
                    rgb_channels = image[:, :, :3]

                    alpha_factor = alpha_channel[..., np.newaxis] / 255.0
                    foreground = alpha_factor * rgb_channels
                    background = (1.0 - alpha_factor) * background

                    output_image = cv2.add(foreground, background).astype(np.uint8)

                    yield Picture(output_image), piece_name



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
                    pieces_pictures_with_names.append(
                        (Picture(np.zeros((1, 1, 4), dtype=np.uint8)), piece_name)
                    )

        self.pieces_pictures_with_names = LoopedList(pieces_pictures_with_names, shuffle_seed=shuffle_seed)

        backgrounds_pictures = []

        for dark, light in board_colors:
            backgrounds_pictures.append(
                (
                    Picture(np.full((1, 1, 3), hex_to_bgr(dark), dtype=np.uint8)),
                    Picture(np.full((1, 1, 3), hex_to_bgr(light), dtype=np.uint8)),
                )
            )

        self.backgrounds_pictures = LoopedList(backgrounds_pictures, shuffle_seed=shuffle_seed)

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
                    (main_piece if j == 4 else self.pieces_pictures_with_names[(i + 1)*(j + 1)][0]).cv2,
                    (self.square_size, self.square_size),
                )

                alpha_channel = piece[:, :, 3]
                rgb_channels = piece[:, :, :3]

                alpha_factor = alpha_channel[..., np.newaxis] / 255.0
                foreground = alpha_factor * rgb_channels
                background = (1.0 - alpha_factor) * background

                combined = cv2.add(foreground, background).astype(np.uint8)
                squares.append(combined)

            grid = np.vstack((
                np.hstack(squares[:3]),
                np.hstack(squares[3:6]),
                np.hstack(squares[6:]),
            ))

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

        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            np.random.seed(shuffle_seed)

    def generator(self,) -> Iterator[tuple[Picture, str]]:
        crop_delta = 0.15
        noise_kwargs = {
            "min_mean_scale": 0.0,
            "max_mean_scale": 0.05,
            "min_var_scale": 0.0,
            "max_var_scale": 0.2,
        }
        blur_kwargs = {
            "min_ksize": 0,
            "max_ksize": 6,
        }
        resolution_jitter_kwargs = {
            "min_factor": 0.2,
            "max_factor": 1,
        }
        artifacts_kwargs = {
            "min_quality": 30,
            "max_quality": 95,
        }
        brightness_delta = 0.2
        saturation_delta = 0.2
        contrast_delta = 0.2

        for original_picture, piece_name in self.piece_images_3x3:

            square_size = original_picture.cv2.shape[0] // 3

            augmented_image, _ = apply_perspective_warp(original_picture.cv2, 0.05, 5, square_size, square_size, square_size)
            augmented_image = center_crop(
                augmented_image,
                int(random.uniform(1 - crop_delta, 1 + crop_delta) * square_size),
                int(random.uniform(1 - crop_delta, 1 + crop_delta) * square_size),
            )
            augmented_image = cv2.resize(augmented_image, (square_size, square_size))

            augmented_image = add_brightness(augmented_image, brightness_delta)
            augmented_image = add_saturation(augmented_image, saturation_delta)
            augmented_image = add_contrast(augmented_image, contrast_delta)

            augmented_image = add_gaussian_noise(augmented_image, **noise_kwargs)
            augmented_image = apply_motion_blur(augmented_image, **blur_kwargs)
            augmented_image = apply_gaussian_blur(augmented_image, **blur_kwargs)
            augmented_image = resolution_jitter(augmented_image, **resolution_jitter_kwargs)
            augmented_image = add_jpeg_artifacts(augmented_image, **artifacts_kwargs)

            yield Picture(augmented_image), piece_name

