import random
import cv2
import numpy as np
import string
from typing import Optional

def add_brightness(img, delta: float):
    # Calculate the maximum change in brightness
    max_change = int(255 * delta)
    value = np.random.randint(-max_change, max_change)  # Generate random change value
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Properly handle the addition to avoid overflow and underflow:
    # Convert 'v' to int16 to safely add/subtract the random value
    v = v.astype(np.int16)
    v += value
    np.clip(v, 0, 255, out=v)  # Ensure the values stay within the uint8 range

    # Convert 'v' back to uint8 after adjustment
    v = v.astype(np.uint8)

    # Merge the HSV channels back and convert to BGR format
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def add_saturation(img, delta: float):
    # Calculate the maximum change in saturation
    max_change = int(255 * delta)
    value = np.random.randint(-max_change, max_change)  # Generate random change value
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Properly handle the addition to avoid overflow and underflow:
    # Convert 's' to int16 to safely add/subtract the random value
    s = s.astype(np.int16)
    s += value
    np.clip(s, 0, 255, out=s)  # Ensure the values stay within the uint8 range

    # Convert 's' back to uint8 after adjustment
    s = s.astype(np.uint8)

    # Merge the HSV channels back and convert to BGR format
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def add_contrast(img, delta: float):
    # Establish a base contrast multiplier and scale it with delta
    # The multiplier is 1 when delta is 0 (no change)
    # It linearly ranges from 0.5 to 1.5 when delta is 1 (maximum change)
    contrast_multiplier = np.random.uniform(1 - delta, 1 + delta)

    # Apply the contrast adjustment
    img_float = img.astype(np.float32)  # Convert to float for precision
    mean = np.mean(img_float)  # Calculate the mean for scaling around it
    img_contrasted = (img_float - mean) * contrast_multiplier + mean  # Adjust contrast
    img_contrasted = np.clip(img_contrasted, 0, 255)  # Ensure values are within byte range

    # Convert back to unsigned 8-bit integer type
    img = img_contrasted.astype(np.uint8)
    return img

def random_crop(image: np.ndarray, min_w: float, max_w: float, min_h: float, max_h: float):
    height, width, _ = image.shape

    crop_width = random.randint(int(width * min_w), int(width * max_w))
    crop_height = random.randint(int(height * min_h), int(height * max_h))

    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)

    return image[y : y + crop_height, x : x + crop_width]


def add_jpeg_artifacts(image: np.ndarray, min_quality: int, max_quality: int):
    # Randomly choose JPEG quality within specified range to introduce artifacts
    quality = random.randint(min_quality, max_quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def apply_gaussian_blur(image: np.ndarray, min_ksize: int, max_ksize: int):
    # Ensure kernel size is odd
    ksize = random.randint(min_ksize, max_ksize)
    ksize = ksize * 2 - 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0) if ksize > 0 else image
    return blurred

def apply_motion_blur(image: np.ndarray, min_ksize: int, max_ksize: int):
    # Step 1: Create the motion blur kernel
    ksize = random.randint(min_ksize, max_ksize)
    ksize = ksize * 2 - 1

    if ksize < 1:
        return image

    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1)/2), :] = np.ones(ksize)
    kernel = kernel / ksize

    # Step 2: Apply the kernel to the image
    motion_blur = cv2.filter2D(image, -1, kernel)

    return motion_blur


def add_gaussian_noise(
    image: np.ndarray,
    min_mean_scale: float,
    max_mean_scale: float,
    min_var_scale: float,
    max_var_scale: float,
):
    """Add Gaussian noise to an image with relative scaling."""
    if image is None:
        return None

    # Calculate mean and variance relative to the image's characteristics
    mean_intensity = np.mean(image)
    intensity_range = np.max(image) - np.min(image)

    min_mean = mean_intensity * min_mean_scale
    max_mean = mean_intensity * max_mean_scale  # Just an example scaling
    min_var = intensity_range * min_var_scale
    max_var = intensity_range * max_var_scale  # Increase variance for visibility

    mean = random.uniform(min_mean, max_mean)
    var = random.uniform(min_var, max_var)
    sigma = var ** 0.5

    # Generating Gaussian noise
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    # Adding the Gaussian noise to the image
    noisy_image = np.float32(image) + gauss

    # Clipping to avoid overflow and convert to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


def generate_random_text(length: int):
    # Generate a random string of fixed length
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def generate_random_rgb():
    return tuple(random.randint(0, 255) for _ in range(3))


def add_random_rectangles(
    image: np.ndarray, min_count: int, max_count: int, min_size_rel: float, max_size_rel: float
):
    count = random.randint(min_count, max_count)
    output = image.copy()
    img_height, img_width = image.shape[:2]
    min_size_abs = (int(min_size_rel * img_width), int(min_size_rel * img_height))
    max_size_abs = (int(max_size_rel * img_width), int(max_size_rel * img_height))

    for _ in range(count):
        x1 = random.randint(0, img_width - max_size_abs[0])
        y1 = random.randint(0, img_height - max_size_abs[1])
        width = random.randint(min_size_abs[0], max_size_abs[0])
        height = random.randint(min_size_abs[1], max_size_abs[1])
        cv2.rectangle(
            output, (x1, y1), (x1 + width, y1 + height), generate_random_rgb(), -1
        )
    return output


def add_random_circles(
    image: np.ndarray, min_count: int, max_count: int, min_radius_rel: float, max_radius_rel: float
):
    count = random.randint(min_count, max_count)
    output = image.copy()
    img_height, img_width = image.shape[:2]
    min_radius_abs = int(min_radius_rel * min(img_width, img_height))
    max_radius_abs = int(max_radius_rel * min(img_width, img_height))

    for _ in range(count):
        center = (random.randint(0, img_width), random.randint(0, img_height))
        radius = random.randint(min_radius_abs, max_radius_abs)
        cv2.circle(output, center, radius, generate_random_rgb(), -1)
    return output


def add_random_text(
    image: np.ndarray,
    min_count: int,
    max_count: int,
    min_length: int,
    max_length: int,
    min_font_scale: float,
    max_font_scale: float,
):
    count = random.randint(min_count, max_count)
    output = image.copy()

    for _ in range(count):
        text = generate_random_text(random.randint(min_length, max_length))
        org = (random.randint(0, image.shape[1]), random.randint(0, image.shape[0]))
        font_scale = random.uniform(min_font_scale, max_font_scale)
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
            generate_random_rgb(),
            int(font_scale * 2),
            cv2.LINE_AA,
        )
    return output


def add_random_lines(
    image: np.ndarray,
    min_count: int,
    max_count: int,
    min_thickness_rel: float,
    max_thickness_rel: float,
):
    count = random.randint(min_count, max_count)
    output = image.copy()
    img_height, img_width = image.shape[:2]
    min_thickness_abs = int(min_thickness_rel * min(img_width, img_height))
    max_thickness_abs = int(max_thickness_rel * min(img_width, img_height))

    for _ in range(count):
        x1, y1 = random.randint(0, img_width), random.randint(0, img_height)
        x2, y2 = random.randint(0, img_width), random.randint(0, img_height)
        thickness = random.randint(min_thickness_abs, max_thickness_abs)
        
        if thickness > 0:
            cv2.line(output, (x1, y1), (x2, y2), generate_random_rgb(), thickness)
    return output

def resolution_jitter(image: np.ndarray, min_factor: float, max_factor: float):
    factor = random.uniform(min_factor, max_factor)

    height, width = image.shape[:2]
    new_dimensions = (int(width * factor), int(height * factor))

    downscaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    upscaled = cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_LINEAR)

    return upscaled

def add_shift(image: np.ndarray, min_shift: float, max_shift: float) -> np.ndarray:
    height, width, channels = image.shape

    # Calculate random shifts in the x and y directions
    shift_x = random.randint(int(width * min_shift), int(width * max_shift)) * random.choice([-1, 1])
    shift_y = random.randint(int(height * min_shift), int(height * max_shift)) * random.choice([-1, 1])

    # Create a transformation matrix for the shift
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Shift the image using a distinctive fill color (e.g., bright blue)
    fill_color = [0, 0, 255]  # Bright blue
    shifted_image = cv2.warpAffine(image, M, (width, height), borderValue=fill_color)

    # Generate a random colorful noise background
    noise_bg = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

    # Create a mask where the shifted image has the fill color
    mask = np.all(shifted_image == fill_color, axis=-1)

    # Convert mask to 3 channels
    mask = np.stack([mask]*3, axis=-1)

    # Use the mask to combine the shifted image with the noise background
    result_image = np.where(mask, noise_bg, shifted_image)

    return result_image

def apply_perspective_warp(image: np.ndarray, max_skew: float, max_rotation: float, x: int, y: int, board_size: int):
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

    # Combine the rotation and perspective matrices
    combined_matrix = np.dot(rotation_matrix_3x3, perspective_matrix)

    # Apply the combined transformation to the image
    transformed_image = cv2.warpPerspective(image, combined_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0)) # Blue background

    # Calculate new coordinates of the square's corners using the combined transformation matrix
    square_corners = np.float32([
        [x, y],
        [x + board_size, y],
        [x + board_size, y + board_size],
        [x, y + board_size]
    ])
    new_square_corners = cv2.perspectiveTransform(np.array([square_corners]), combined_matrix)[0]

    return transformed_image, new_square_corners

def center_crop(image, crop_width: int, crop_height: Optional[int] = None):
    if crop_height is None:
        crop_height = crop_width

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
