from pathlib import Path
from typing import Optional, Union
from PIL import Image
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
            raise TypeError("Input must be a Path, str, PIL Image, or cv2 image (numpy ndarray).")

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
