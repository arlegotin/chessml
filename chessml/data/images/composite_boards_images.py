from torch_exid import ExtendedIterableDataset
from typing import Iterable, Iterator
from chessml.data.utils.looped_list import LoopedList
from chessml.data.utils.augment import (
    random_crop,
    add_random_lines,
    add_random_text,
    add_gaussian_noise,
    apply_gaussian_blur,
    add_jpeg_artifacts,
)
from typing import Optional
import numpy as np
import random
import cv2


class CompositeBoardsImages(ExtendedIterableDataset):
    def __init__(
        self,
        images_with_data: Iterable[tuple[np.ndarray, str, bool]],
        bg_images: list[np.ndarray],
        shuffle_seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms_required=False, shuffle_seed=shuffle_seed, *args, **kwargs
        )
        self.images_with_data = images_with_data

        self.bg_images = LoopedList(bg_images, shuffle_seed=shuffle_seed)
        self.bg_rotations = LoopedList(
            [
                None,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ],
            shuffle_seed=shuffle_seed + 1,
        )

        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            np.random.seed(shuffle_seed)

    def generator(self,) -> Iterator[tuple[np.ndarray, ...]]:
        board_size_range = (0.2, 1.0)
        bg_crop_kwargs = {
            "min_w": 0.4,
            "max_w": 1.0,
            "min_h": 0.4,
            "max_h": 1.0,
        }
        lines_kwargs = {
            "min_count": 0,
            "max_count": 2,
            "min_thickness_rel": 0.004,
            "max_thickness_rel": 0.008,
        }
        text_kwargs = {
            "min_count": 0,
            "max_count": 2,
            "min_length": 1,
            "max_length": 20,
            "min_font_scale": 0.5,
            "max_font_scale": 2,
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
        artifacts_kwargs = {
            "min_quality": 30,
            "max_quality": 95,
        }

        for i, (board_image, *additional_data) in enumerate(self.images_with_data):
            bg_image = random_crop(self.bg_images[i], **bg_crop_kwargs)

            if i % 2 == 1:
                bg_image = cv2.flip(bg_image, 1)

            rotation = self.bg_rotations[i]
            if rotation is not None:
                bg_image = cv2.rotate(bg_image, rotation)

            bg_h, bg_w = bg_image.shape[:2]
            board_max_size = min(bg_h, bg_w)
            board_size = random.randint(
                round(board_size_range[0] * board_max_size),
                round(board_size_range[1] * board_max_size),
            )
            board_image = cv2.resize(board_image, (board_size, board_size))

            x = random.randint(0, bg_w - board_size)
            y = random.randint(0, bg_h - board_size)

            final_image = bg_image.copy()
            final_image[y : y + board_size, x : x + board_size] = board_image

            final_image = add_random_lines(final_image, **lines_kwargs)
            final_image = add_random_text(final_image, **text_kwargs)
            final_image = add_gaussian_noise(final_image, **noise_kwargs)
            final_image = apply_gaussian_blur(final_image, **blur_kwargs)
            final_image = add_jpeg_artifacts(final_image, **artifacts_kwargs)

            yield (final_image, (x, x + board_size, y, y + board_size)) + tuple(
                additional_data
            )
