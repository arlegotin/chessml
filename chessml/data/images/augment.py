import random
import cv2
import numpy as np
import string
from typing import Optional


def rand_float(min_val: float, max_val: float, with_negatives: bool = False) -> float:
    if with_negatives and random.random() < 0.5:
        return np.random.uniform(-max_val, -min_val)

    return np.random.uniform(min_val, max_val)


def rand_int(min_val: int, max_val: int, with_negatives: bool = False) -> int:
    return int(rand_float(min_val, max_val, with_negatives))


def rand_text(length: int) -> str:
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def rand_rgb() -> tuple[int, int, int]:
    return tuple(random.randint(0, 255) for _ in range(3))


class Augmentator:
    def __init__(self, config: dict, seed: Optional[int] = None):
        self.config = config

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for name, kwargs in self.config.items():
            # print(name, kwargs)
            image = getattr(self, name)(image, **kwargs)

        return image

    def brightness(
        self, image: np.ndarray, min_delta: float, max_delta: float
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = v.astype(np.int16)
        v += rand_int(255 * min_delta, 255 * max_delta, True)
        np.clip(v, 0, 255, out=v)

        v = v.astype(np.uint8)

        final_hsv = cv2.merge((h, s, v))

        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def saturation(
        self, image: np.ndarray, min_delta: float, max_delta: float
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        s = s.astype(np.int16)
        s += rand_int(255 * min_delta, 255 * max_delta, True)
        np.clip(s, 0, 255, out=s)

        s = s.astype(np.uint8)

        final_hsv = cv2.merge((h, s, v))

        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def contrast(
        self, image: np.ndarray, min_delta: float, max_delta: float
    ) -> np.ndarray:
        contrast_multiplier = 1 + rand_float(min_delta, max_delta, True)

        img_float = image.astype(np.float32)
        mean = np.mean(img_float)
        img_contrasted = (img_float - mean) * contrast_multiplier + mean
        img_contrasted = np.clip(img_contrasted, 0, 255)

        return img_contrasted.astype(np.uint8)

    def crop(
        self, image: np.ndarray, min_w: float, max_w: float, min_h: float, max_h: float
    ) -> np.ndarray:
        height, width, _ = image.shape

        crop_width = rand_int(width * min_w, width * max_w)
        crop_height = rand_int(height * min_h, height * max_h)

        x = rand_int(0, width - crop_width)
        y = rand_int(0, height - crop_height)

        return image[y : y + crop_height, x : x + crop_width]

    def center_crop(self, image: np.ndarray, size: int, delta: float) -> np.ndarray:
        crop_width = int((1 + rand_float(-delta, delta)) * size)
        crop_height = int((1 + rand_float(-delta, delta)) * size)

        height, width = image.shape[:2]

        # Calculate the center of the image
        center_y, center_x = height // 2, width // 2

        # Calculate the coordinates of the top left corner of the crop
        start_x = max(center_x - crop_width // 2, 0)
        start_y = max(center_y - crop_height // 2, 0)

        # Ensure the crop size does not exceed the image dimensions
        end_x = min(start_x + crop_width, width)
        end_y = min(start_y + crop_height, height)

        # Adjust the starting points if the crop size is larger than the remaining space
        start_x = end_x - crop_width if end_x - start_x < crop_width else start_x
        start_y = end_y - crop_height if end_y - start_y < crop_height else start_y

        # Crop the image
        cropped_image = image[start_y:end_y, start_x:end_x]

        return cropped_image

    def jpeg_artifacts(
        self, image: np.ndarray, min_quality: int, max_quality: int
    ) -> np.ndarray:
        quality = rand_int(min_quality, max_quality)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", image, encode_param)
        return cv2.imdecode(encimg, 1)

    def gaussian_blur(
        self, image: np.ndarray, min_radius: int, max_radius: int
    ) -> np.ndarray:
        radius = rand_int(min_radius, max_radius)
        ksize = radius * 2 - 1

        if ksize < 1:
            return image

        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def motion_blur(
        self, image: np.ndarray, min_radius: int, max_radius: int
    ) -> np.ndarray:
        radius = rand_int(min_radius, max_radius)
        ksize = radius * 2 - 1

        if ksize < 1:
            return image

        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = kernel / ksize

        return cv2.filter2D(image, -1, kernel)

    def gaussian_noise(
        self,
        image: np.ndarray,
        min_mean_scale: float,
        max_mean_scale: float,
        min_var_scale: float,
        max_var_scale: float,
    ) -> np.ndarray:
        mean_intensity = np.mean(image)
        intensity_range = np.max(image) - np.min(image)

        min_mean = mean_intensity * min_mean_scale
        max_mean = mean_intensity * max_mean_scale
        min_var = intensity_range * min_var_scale
        max_var = intensity_range * max_var_scale

        mean = rand_float(min_mean, max_mean)
        sigma = rand_float(min_var, max_var) ** 0.5

        gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)

        noisy_image = np.float32(image) + gauss

        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def text(
        self,
        image: np.ndarray,
        min_count: int,
        max_count: int,
        min_length: int,
        max_length: int,
        min_font_scale: float,
        max_font_scale: float,
    ) -> np.ndarray:
        count = rand_int(min_count, max_count)
        output = image.copy()

        for _ in range(count):
            text = rand_text(rand_int(min_length, max_length))
            org = (rand_int(0, image.shape[1]), rand_int(0, image.shape[0]))
            font_scale = rand_float(min_font_scale, max_font_scale)
            font = random.choice(
                [
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cv2.FONT_HERSHEY_PLAIN,
                    cv2.FONT_HERSHEY_DUPLEX,
                    cv2.FONT_HERSHEY_COMPLEX,
                    cv2.FONT_HERSHEY_TRIPLEX,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                    cv2.FONT_ITALIC,
                ]
            )
            cv2.putText(
                output,
                text,
                org,
                font,
                font_scale,
                rand_rgb(),
                int(font_scale * 2),
                cv2.LINE_AA,
            )

        return output

    def lines(
        self,
        image: np.ndarray,
        min_count: int,
        max_count: int,
        min_thickness: float,
        max_thickness: float,
    ) -> np.ndarray:
        count = rand_int(min_count, max_count)
        output = image.copy()
        img_height, img_width = image.shape[:2]
        min_thickness_abs = int(min_thickness * min(img_width, img_height))
        max_thickness_abs = int(max_thickness * min(img_width, img_height))

        for _ in range(count):
            x1, y1 = rand_int(0, img_width), rand_int(0, img_height)
            x2, y2 = rand_int(0, img_width), rand_int(0, img_height)
            thickness = rand_int(min_thickness_abs, max_thickness_abs)

            if thickness > 0:
                cv2.line(output, (x1, y1), (x2, y2), rand_rgb(), thickness)

        return output

    def resolution_jitter(
        self, image: np.ndarray, min_factor: float, max_factor: float
    ) -> np.ndarray:
        factor = rand_float(min_factor, max_factor)

        height, width = image.shape[:2]
        new_dimensions = (int(width * factor), int(height * factor))

        downscaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

        return cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_LINEAR)

    def shift(
        self, image: np.ndarray, min_shift: float, max_shift: float
    ) -> np.ndarray:
        height, width, channels = image.shape

        shift_x = rand_int(width * min_shift, width * max_shift, True)
        shift_y = rand_int(height * min_shift, height * max_shift, True)

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        fill_color = [0, 0, 255]
        shifted_image = cv2.warpAffine(
            image, M, (width, height), borderValue=fill_color
        )

        noise_bg = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

        mask = np.all(shifted_image == fill_color, axis=-1)

        mask = np.stack([mask] * 3, axis=-1)

        result_image = np.where(mask, noise_bg, shifted_image)

        return result_image


def apply_perspective_warp(
    image: np.ndarray,
    max_skew: float,
    max_rotation: float,
    x: int,
    y: int,
    size: int,
):
    """ Apply a perspective warp and rotation to simulate a 3D effect and calculate new square coordinates. """
    h, w = image.shape[:2]

    # Generate skew factors
    def gs():
        return random.uniform(-max_skew, max_skew)

    skew_x_top = gs() * w
    skew_y_left = gs() * h
    skew_x_bottom = gs() * w
    skew_y_right = gs() * h

    # Generate a random rotation angle
    angle = random.uniform(-max_rotation, max_rotation)

    # Compute the rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])

    # Calculate the points after skew
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    pts2 = np.float32(
        [
            [0 + skew_x_top, 0 + skew_y_left],
            [w - skew_x_top, 0 + skew_y_left],
            [w - skew_x_bottom, h - skew_y_right],
            [0 + skew_x_bottom, h - skew_y_right],
        ]
    )

    # Compute the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Combine the rotation and perspective matrices
    combined_matrix = np.dot(rotation_matrix_3x3, perspective_matrix)

    # Apply the combined transformation to the image
    transformed_image = cv2.warpPerspective(
        image,
        combined_matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 0, 0),
    )  # Blue background

    # Calculate new coordinates of the square's corners using the combined transformation matrix
    square_corners = np.float32(
        [
            [x, y],
            [x + size, y],
            [x + size, y + size],
            [x, y + size],
        ]
    )
    new_square_corners = cv2.perspectiveTransform(
        np.array([square_corners]), combined_matrix
    )[0]

    return transformed_image, new_square_corners
