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
            "max_shift": 0.2,
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
            "min_factor": 0.2,
            "max_factor": 1.0,
        }
        artifacts_kwargs = {
            "min_quality": 30,
            "max_quality": 95,
        }

        for i in itertools.count():
            image, piece_name = self.piece_images[i]

            final_image = image.copy()

            final_image = apply_perspective_warp(final_image, max_skew=0.15, max_rotation=15)
            final_image = add_shift(final_image, **shift_kwargs)
            final_image = add_gaussian_noise(final_image, **noise_kwargs)
            final_image = apply_gaussian_blur(final_image, **blur_kwargs)
            final_image = resolution_jitter(final_image, **resolution_jitter_kwargs)
            final_image = add_jpeg_artifacts(final_image, **artifacts_kwargs)

            yield final_image, piece_name


def apply_perspective_warp(image, max_skew: float, max_rotation: float):
    """ Apply a perspective warp and rotation to simulate a 3D effect and calculate new square coordinates. """
    h, w, c = image.shape

    # Generate skew factors
    def gs():
        return random.uniform(-max_skew, max_skew)
    
    skew_x_top = gs() * w
    skew_y_left = gs() * h
    skew_x_bottom = gs() * w
    skew_y_right = gs() * h

    pts1 = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    
    pts2 = np.float32([
        [0 + skew_x_top, 0 + skew_y_left],
        [w - skew_x_top, 0 + skew_y_left],
        [w - skew_x_bottom, h - skew_y_right],
        [0 + skew_x_bottom, h - skew_y_right],
    ])

    fill_color = [0, 0, 255]  # Bright blue

    # Compute the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective warp to the image
    warped_image = cv2.warpPerspective(image, perspective_matrix, (w, h), borderValue=fill_color)

    # Generate a random rotation angle
    angle = random.uniform(-max_rotation, max_rotation)

    # Compute the rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to the warped image
    rotated_image = cv2.warpAffine(warped_image, rotation_matrix, (w, h), borderValue=fill_color)

    noise_bg = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)

    mask = np.all(rotated_image == fill_color, axis=-1)

    mask = np.stack([mask]*3, axis=-1)

    rotated_image = np.where(mask, noise_bg, rotated_image)

    return rotated_image

