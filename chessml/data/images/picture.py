from pathlib import Path
from typing import Optional, Union, Self
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
from functools import cached_property
import cv2


def CV2toPIL(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def PILtoCV2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


class Picture:
    def __init__(self, image_input: Union[Path, str, Image.Image, np.ndarray]):
        if isinstance(image_input, (str, Path)):
            self.path = Path(image_input)
            if not self.path.is_file():
                raise ValueError(f"Provided path '{self.path}' is not a valid file.")
            self._pil = None
            self._cv2 = None
        elif isinstance(image_input, Image.Image):
            self.path = None
            self._pil = image_input
            self._cv2 = None
        elif isinstance(image_input, np.ndarray):
            self.path = None
            self._pil = None
            self._cv2 = image_input
        else:
            raise TypeError(
                "Input must be a Path, str, PIL Image, or cv2 image (numpy ndarray)."
            )

    @cached_property
    def pil(self) -> Image.Image:
        if self._pil is not None:
            return self._pil

        if self._cv2 is not None:
            return CV2toPIL(self._cv2)

        if self.path is not None:
            try:
                return Image.open(str(self.path.resolve()))
            except Exception as e:
                raise ValueError(f"Cannot open image file at '{self.path}': {e}")

        raise ValueError("No valid image source provided")

    @cached_property
    def cv2(self) -> np.ndarray:
        if self._cv2 is not None:
            return self._cv2

        if self._pil is not None:
            return PILtoCV2(self._pil)

        if self.path is not None:
            cv2_image = cv2.imread(str(self.path.resolve()), cv2.IMREAD_UNCHANGED)

            # if cv2_image.dtype != np.uint8:
            #     cv2_image = cv2_image.astype(np.uint8)

            if cv2_image is None:
                raise ValueError(f"Cannot read image file at '{self.path}'")
            return cv2_image

        raise ValueError("No valid image source provided")

    @property
    def as_3_channels(self) -> Self:
        image = self.cv2

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return Picture(image)

    @property
    def normalized(self) -> Self:
        sigma = 0.001
        v = np.median(self.cv2)

        # Apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(self.cv2, lower, upper)

        return Picture(edged)

    @property
    def bw(self) -> Self:
        ycrcb = cv2.cvtColor(self.cv2, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]

        return Picture(cv2.merge([y_channel, y_channel, y_channel]))


def autocorrect_brightness_contrast(image, clip_hist_percent=1):
    """
    Autocorrects brightness and contrast of an image using OpenCV.
    
    Parameters:
    - image: Input image in OpenCV format (numpy array).
    - clip_hist_percent: Percentage of histogram to clip for contrast stretching (default: 1%).
    
    Returns:
    - corrected_image: Brightness and contrast corrected image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    max_value = accumulator[-1]
    clip_hist_percent *= max_value / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while minimum_gray < hist_size and accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while maximum_gray >= 0 and accumulator[maximum_gray] >= (
        max_value - clip_hist_percent
    ):
        maximum_gray -= 1

    # Calculate alpha and beta values
    if maximum_gray <= minimum_gray:  # Adjusted to <= to handle edge cases
        alpha = 1.0
        beta = 0.0
    else:
        alpha = 255.0 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

    # Apply the contrast and brightness adjustment
    corrected_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return corrected_image
