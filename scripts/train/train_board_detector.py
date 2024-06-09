from chessml import script, config
from chessml.models.torch.vision_model_adapter import MobileViTV2FPN
from chessml.models.lightning.board_detector_model import BoardDetector
from pathlib import Path
import logging
import os
from chessml.data.utils.images_with_txt_dataset import ImagesWithTxt
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, BG_IMAGES
import cv2
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.train.standard_training import standard_training
import torch
from time import time
from PIL import Image
from typing import Callable, Optional, Any
from torch.utils.data import Dataset
from functools import partial
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.augmented_boards_images import AugmentedBoardsImages
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.data.images.picture import Picture

logger = logging.getLogger(__name__)

def use_pregenerated_dataset(
    path_to_dir: str,
    preprocess_image: Callable[[str], Any],
    offset: int = 0,
    limit: Optional[int] = None,
    **kwargs,
) -> Dataset:
    def preprocess_txt(data):
        coords_str, *_ = data
        coords = list(map(float, coords_str.split()))
        
        return torch.tensor(
            coords,
            dtype=torch.float32,
        )

    return ImagesWithTxt(
        path_to_dir=path_to_dir,
        read_image=lambda p: Picture(p),
        preprocess_image=preprocess_image,
        preprocess_txt=preprocess_txt,
        offset=offset,
        limit=limit,
    )

def use_dynamic_dataset(
    preprocess_image: Callable[[str], Any],
    shuffle_seed: int,
    offset: int = 0,
    limit: Optional[int] = None,
    **kwargs,
) -> Dataset:

    return AugmentedBoardsImages(
        boards_with_data=BoardsImagesFromFENs(
            fens=FileLinesDataset(
                path=Path(config.dataset.path) / "unique_fens.txt",
            ),
            piece_sets=PIECE_SETS,
            board_colors=BOARD_COLORS,
            square_size=64,
            shuffle_seed=shuffle_seed,
        ),
        bg_images=BG_IMAGES,
        transforms=[
            lambda x: (preprocess_image(x[0]), torch.tensor(x[1], dtype=torch.float32))
        ],
        shuffle_seed=shuffle_seed,
        **kwargs,
    )

script.add_argument("-bs", dest="batch_size", type=int, default=100)
script.add_argument("-vb", dest="val_batches", type=int, default=512)
script.add_argument("-vi", dest="val_interval", type=int, default=1024)
script.add_argument("-s", dest="seed", type=int, default=65)
script.add_argument("-dt", dest="dataset_type", choices=["pregenerated", "dynamic"], default="pregenerated")
script.add_argument("-dp", dest="dataset_path", type=str, default="/home/hp/datasets/augmented_boards_512")
script.add_argument("-m", dest="model", choices=["MobileViTV2FPN"], default="MobileViTV2FPN")
script.add_argument("-c", dest="checkpoint", type=str, default=None)

@script
def train(args):

    base_model_class = {
        "MobileViTV2FPN": MobileViTV2FPN,
    }[args.model]

    if args.checkpoint is not None:
        model = BoardDetector.load_from_checkpoint(
            args.checkpoint,
            base_model_class=base_model_class,
        )
    else:
        model = BoardDetector(
            base_model_class=base_model_class,
        )

    if args.dataset_type == "pregenerated":
        make_dataset = partial(
            use_pregenerated_dataset,
            path_to_dir=args.dataset_path,
        )
    elif args.dataset_type == "dynamic":
        make_dataset = partial(
            use_dynamic_dataset,
            shuffle_seed=args.seed,
        )

    make_dataset = partial(
        make_dataset,
        preprocess_image=lambda p: model.model.preprocess_image(p.pil),
    )

    standard_training(
        model=model,
        make_dataset=make_dataset,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        val_interval=args.val_interval,
        checkpoint_name=f"bd-m={args.model}-v1-bs={args.batch_size}-{{step}}",
    )
