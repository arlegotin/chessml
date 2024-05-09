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
)
import random
import itertools

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
) -> Iterator[tuple[np.ndarray, str]]:
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

                    yield np.array(output_image)[:, :, ::-1], piece_name


class CompositePiecesImages(ExtendedIterableDataset):
    def __init__(
        self,
        piece_images: list[tuple[np.ndarray, str]],
        shuffle_seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms_required=False, shuffle_seed=shuffle_seed, *args, **kwargs
        )

        self.piece_images = LoopedList(piece_images, shuffle_seed=shuffle_seed)

        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            np.random.seed(shuffle_seed)

    def generator(self,) -> Iterator[tuple[np.ndarray, str]]:
        shift_kwargs = {
            "min_shift": 0.0,
            "max_shift": 0.1,
        }
        noise_kwargs = {
            "min_mean_scale": 0.0,
            "max_mean_scale": 0.05,
            "min_var_scale": 0.0,
            "max_var_scale": 0.2,
        }
        blur_kwargs = {
            "min_ksize": 0,
            "max_ksize": 2,
        }
        resolution_jitter_kwargs = {
            "min_factor": 0.3,
            "max_factor": 1.0,
        }
        artifacts_kwargs = {
            "min_quality": 30,
            "max_quality": 95,
        }

        for i in itertools.count():
            image, piece_name = self.piece_images[i]

            final_image = image.copy()

            final_image = add_shift(final_image, **shift_kwargs)
            final_image = add_gaussian_noise(final_image, **noise_kwargs)
            final_image = apply_gaussian_blur(final_image, **blur_kwargs)
            final_image = resolution_jitter(final_image, **resolution_jitter_kwargs)
            final_image = add_jpeg_artifacts(final_image, **artifacts_kwargs)

            yield final_image, piece_name

