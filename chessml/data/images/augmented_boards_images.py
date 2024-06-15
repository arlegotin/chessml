from torch_exid import ExtendedIterableDataset
from typing import Iterable, Iterator
from chessml.data.utils.looped_list import LoopedList
from chessml.data.images.augment import (
    Augmentator,
    apply_perspective_warp,
)
from typing import Optional
import numpy as np
import random
import cv2
from chessml.data.images.picture import Picture
from chessml import config


class AugmentedBoardsImages(ExtendedIterableDataset):
    def __init__(
        self,
        boards_with_data: Iterable[tuple[Picture, str, bool]],
        bg_images: list[Picture],
        shuffle_seed: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            transforms_required=False, shuffle_seed=shuffle_seed, *args, **kwargs
        )
        self.boards_with_data = boards_with_data

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

        self.bg_augmentator = Augmentator(
            config=config.dataset.augmentations.AugmentedBoardsImages.bg,
            seed=shuffle_seed,
        )

        self.board_augmentator = Augmentator(
            config=config.dataset.augmentations.AugmentedBoardsImages.board,
            seed=shuffle_seed,
        )

        self.final_augmentator = Augmentator(
            config=config.dataset.augmentations.AugmentedBoardsImages.final,
            seed=shuffle_seed,
        )

    def generator(self) -> Iterator[tuple[Picture, ...]]:
        for i, (board_picture, *additional_data) in enumerate(self.boards_with_data):
            bg_image = self.bg_augmentator(self.bg_images[i].cv2)

            if i % 2 == 1:
                bg_image = cv2.flip(bg_image, 1)

            rotation = self.bg_rotations[i]
            if rotation is not None:
                bg_image = cv2.rotate(bg_image, rotation)

            bg_h, bg_w = bg_image.shape[:2]
            board_max_size = min(bg_h, bg_w)
            board_size = random.randint(
                round(0.3 * board_max_size),
                round(0.85 * board_max_size),
            )
            board_image = cv2.resize(board_picture.cv2, (board_size, board_size))
            board_image = self.board_augmentator(board_image)

            x = random.randint(0, bg_w - board_size)
            y = random.randint(0, bg_h - board_size)

            board_on_blue_screen = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
            board_on_blue_screen[:] = (255, 0, 0)
            board_on_blue_screen[y : y + board_size, x : x + board_size] = board_image

            board_on_blue_screen, corners = apply_perspective_warp(
                board_on_blue_screen,
                max_skew=0.1,
                max_rotation=15,
                x=x,
                y=y,
                size=board_size,
            )

            blue_mask = (
                (board_on_blue_screen[:, :, 0] == 255)
                & (board_on_blue_screen[:, :, 1] == 0)
                & (board_on_blue_screen[:, :, 2] == 0)
            )
            blue_mask = blue_mask.astype(np.uint8) * 255

            dilated_mask = cv2.dilate(
                blue_mask, np.ones((3, 3), np.uint8), iterations=1
            )
            blurred_mask = cv2.GaussianBlur(dilated_mask, (1, 1), 0)
            _, blue_mask = cv2.threshold(blurred_mask, 1, 255, cv2.THRESH_BINARY)

            mask_inv = cv2.bitwise_not(blue_mask)
            extracrted_board_image = cv2.bitwise_and(
                board_on_blue_screen, board_on_blue_screen, mask=mask_inv
            )
            bg_masked = cv2.bitwise_and(bg_image, bg_image, mask=blue_mask)

            final_image = cv2.add(bg_masked, extracrted_board_image)

            final_h, final_w = final_image.shape[:2]

            corners = [(x / final_w, y / final_h) for (x, y) in corners]

            final_image = self.final_augmentator(final_image)

            yield (Picture(final_image), corners) + tuple(additional_data)
