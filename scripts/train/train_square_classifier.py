from chessml import script, config
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.torch.vision_model_adapter import EfficientNetV2Classifier, MobileNetV3LargeClassifier, MobileViTSClassifier, EfficientNetB3Classifier
from pathlib import Path
import logging
import os
import torch
from torch import Tensor
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, PIECE_CLASSES
from chessml.train.standard_training import standard_training
from chessml.data.utils.csv_dataset import CSVDataset
from chessml.data.images.picture import Picture
from typing import Callable

logger = logging.getLogger(__name__)
m = 2
script.add_argument("-bs", dest="batch_size", type=int, default=int(64 * m))
script.add_argument("-vb", dest="val_batches", type=int, default=int(1024 // m))
script.add_argument("-vi", dest="val_interval", type=int, default=int(256 // m))
script.add_argument("-s", dest="seed", type=int, default=69)

# pc-44-bs=128-step=7296.ckpt

class SquareClassifierDataset(CSVDataset):
    def __init__(self, preprocess_image: Callable[[Picture], Tensor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_image = preprocess_image

    def __getitem__(self, idx):
        path, piece_name = super().__getitem__(idx)
        print(path, piece_name)
        quit()
        
        # if piece_name is None:  # or any other condition
        #     # Recursively get the next item
        #     return self.__getitem__((idx + 1) % len(self))
            
        picture = Picture(path)
        piece_class = PIECE_CLASSES[piece_name or None]
        return self.preprocess_image(picture.bw.pil), torch.tensor(piece_class, dtype=torch.long)


@script
def train(args):
    path_to_csv = Path(config.dataset.path_to_big) / "square_classifier" / "meta.csv"

    model = PieceClassifier(base_model_class=MobileNetV3LargeClassifier)
    # model = PieceClassifier(base_model_class=MobileViTSClassifier)
    # model = PieceClassifier(base_model_class=EfficientNetB3Classifier)

    def make_dataset(limit: int = None, offset: int = 0, **kwargs):
        return SquareClassifierDataset(
            path=path_to_csv,
            limit=limit,
            offset=offset,
            preprocess_image=model.model.preprocess_image,
        )

    standard_training(
        model=model,
        make_dataset=make_dataset,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        val_interval=args.val_interval,
        checkpoint_name=f"pc-44-bs={args.batch_size}-{{step}}",
        num_workers=71,
    )
