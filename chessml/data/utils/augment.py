import random
import cv2
import numpy as np
import string


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
