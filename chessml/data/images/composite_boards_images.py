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
    add_saturation,
    add_brightness,
    add_contrast,
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
        shuffle_seed: int,
        skip_every_nth: Optional[int] = None,
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

        self.skip_every_nth = skip_every_nth

        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            np.random.seed(shuffle_seed)

    def generator(self,) -> Iterator[tuple[np.ndarray, ...]]:
        board_size_range = (0.3, 0.9)
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
            "max_mean_scale": 0.0,
            "min_var_scale": 0.0,
            "max_var_scale": 0.1,
        }
        blur_kwargs = {
            "min_ksize": 0,
            "max_ksize": 2,
        }
        artifacts_kwargs = {
            "min_quality": 30,
            "max_quality": 95,
        }
        brightness_delta = 0.3
        saturation_delta = 0.1
        contrast_delta = 0.1
        max_skew = 0.15
        max_rotation = 20

        for i, (board_image, *additional_data) in enumerate(self.images_with_data):
            bg_image = random_crop(self.bg_images[i], **bg_crop_kwargs)

            bg_image = add_brightness(bg_image, brightness_delta)
            bg_image = add_saturation(bg_image, saturation_delta)
            bg_image = add_contrast(bg_image, contrast_delta)

            if i % 2 == 1:
                bg_image = cv2.flip(bg_image, 1)

            rotation = self.bg_rotations[i]
            if rotation is not None:
                bg_image = cv2.rotate(bg_image, rotation)

            if self.skip_every_nth is not None and i % self.skip_every_nth == 0:
                board_image = np.zeros((0, 0, board_image.shape[2]), dtype=np.uint8)
                board_size = 0
                x = 0
                y = 0
            else:
                bg_h, bg_w = bg_image.shape[:2]
                board_max_size = min(bg_h, bg_w)
                board_size = random.randint(
                    round(board_size_range[0] * board_max_size),
                    round(board_size_range[1] * board_max_size),
                )
                board_image = cv2.resize(board_image, (board_size, board_size))
                board_image = add_brightness(board_image, brightness_delta)
                board_image = add_saturation(board_image, saturation_delta)
                board_image = add_contrast(board_image, contrast_delta)

                x = random.randint(0, bg_w - board_size)
                y = random.randint(0, bg_h - board_size)
                # std_dev = min(bg_h, bg_w) / 10  # Adjust this value to control spread from the center
                # x_mean = bg_w / 2
                # y_mean = bg_h / 2
                # x = int(np.random.normal(x_mean, std_dev)) - board_size // 2
                # y = int(np.random.normal(y_mean, std_dev)) - board_size // 2
                # x = max(0, min(x, bg_w - board_size))
                # y = max(0, min(y, bg_h - board_size))

            final_image = bg_image.copy()

            if board_size > 0:
                final_image[y : y + board_size, x : x + board_size] = board_image

            # top-left 
            # top-right
            # bottom-right
            # bottom-left
            final_image, corners = apply_perspective_warp(final_image, max_skew, max_rotation, x, y, board_size)

            # if board_size == 0:
            #     fh, fw = final_image.shape[:2]
            #     corners = np.float32([
            #         [fw / 2, fh / 2],
            #         [fw / 2, fh / 2],
            #         [fw / 2, fh / 2],
            #         [fw / 2, fh / 2],
            #     ])

            # corners = np.float32([
            #     [x, y],
            #     [x + board_size, y],
            #     [x + board_size, y + board_size],
            #     [x, y + board_size],
            # ])

            final_image = add_random_lines(final_image, **lines_kwargs)
            final_image = add_random_text(final_image, **text_kwargs)
            final_image = add_gaussian_noise(final_image, **noise_kwargs)
            final_image = apply_gaussian_blur(final_image, **blur_kwargs)
            final_image = add_brightness(final_image, brightness_delta)
            final_image = add_saturation(final_image, saturation_delta)
            final_image = add_contrast(final_image, contrast_delta)
            final_image = add_jpeg_artifacts(final_image, **artifacts_kwargs)
            # final_image = draw_corners_on_image(final_image, corners)

            yield (final_image, corners) + tuple(
                additional_data
            )

def draw_corners_on_image(image, corners):
    """ Draw green dots on the specified corner points of the image. """
    for (x, y) in corners:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green color with thickness -1 (filled)
    return image

def apply_perspective_warp(image, max_skew: float, max_rotation: float, x: int, y: int, board_size: int):
    """ Apply a perspective warp and rotation to simulate a 3D effect and calculate new square coordinates. """
    h, w = image.shape[:2]

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

    # Compute the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective warp to the image
    warped_image = cv2.warpPerspective(image, perspective_matrix, (w, h))

    # Generate a random rotation angle
    angle = random.uniform(-max_rotation, max_rotation)

    # Compute the rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to the warped image
    rotated_image = cv2.warpAffine(warped_image, rotation_matrix, (w, h))

    # Convert rotation matrix to 3x3
    rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])

    # Combine the perspective and rotation matrices
    combined_matrix = np.dot(rotation_matrix_3x3, perspective_matrix)

    # Calculate new coordinates of the square's corners
    square_corners = np.float32([
        [x, y],
        [x + board_size, y],
        [x + board_size, y + board_size],
        [x, y + board_size]
    ])
    new_square_corners = cv2.perspectiveTransform(np.array([square_corners]), combined_matrix)[0]

    return rotated_image, new_square_corners