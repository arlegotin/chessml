import cv2
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Callable, Optional
from chessml.utils import read_lines_from_txt
import numpy as np
import os

class ImagesWithTxt(Dataset):
    def __init__(
        self,
        path_to_dir: str,
        read_image: Callable[[Path], Any],
        preprocess_image: Callable[[Any], Any],
        preprocess_txt: Callable[[list[str]], Any],
        offset: int = 0,
        limit: Optional[int] = None,
    ):
        images_count = sum(1 for entry in os.scandir(path_to_dir) if entry.is_file() and entry.name.lower().endswith('.jpg'))
        txt_count = sum(1 for entry in os.scandir(path_to_dir) if entry.is_file() and entry.name.lower().endswith('.txt'))
        
        assert images_count == txt_count, f"Number of images and text files must be the same, but got {images_count} vs {txt_count}"

        self.path_to_dir = path_to_dir
        self.read_image = read_image
        self.preprocess_image = preprocess_image
        self.preprocess_txt = preprocess_txt

        self.offset = offset
        self.limit = images_count if limit is None else min(images_count, limit)

    def __len__(self):
        return self.limit - self.offset

    def __getitem__(self, idx):
        index = self.offset + idx + 1
        image_path = f"{self.path_to_dir}/{index}.jpg"
        text_path = f"{self.path_to_dir}/{index}.txt"

        image = self.read_image(image_path)
        image = self.preprocess_image(image)

        txt = read_lines_from_txt(text_path)
        txt = self.preprocess_txt(txt)

        return image, txt

