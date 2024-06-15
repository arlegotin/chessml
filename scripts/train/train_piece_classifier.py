from chessml import script, config
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.torch.vision_model_adapter import EfficientNetV2Classifier
from pathlib import Path
import logging
import os
import torch
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, PIECE_CLASSES
from chessml.train.standard_training import standard_training

logger = logging.getLogger(__name__)
m = 1
script.add_argument("-bs", dest="batch_size", type=int, default=int(64 * m))
script.add_argument("-vb", dest="val_batches", type=int, default=int(2048 // m))
script.add_argument("-vi", dest="val_interval", type=int, default=int(1024 // m))
script.add_argument("-s", dest="seed", type=int, default=69)

"""
next:
- no weight (pc-33-bs=64-step=15360 nice but empty squares are recognized as pieces sometimes)
- label smoothing 0.1
- RAdam
"""


@script
def train(args):

    model = PieceClassifier(base_model_class=EfficientNetV2Classifier)

    def make_dataset(offset: int = 0, **kwargs):
        def innrtrnfrm(img, piece_name):
            piece_class = PIECE_CLASSES[piece_name]
            return (
                model.model.preprocess_image(img.bw.pil),
                torch.tensor(piece_class, dtype=torch.long),
            )

        return AugmentedPiecesImages(
            piece_images_3x3=PiecesImages3x3(
                piece_sets=PIECE_SETS,
                board_colors=BOARD_COLORS,
                square_size=64,
                shuffle_seed=args.seed + offset,
            ),
            transforms=[lambda x: innrtrnfrm(*x)],
            shuffle_seed=args.seed + offset,
            **kwargs,
        )

    standard_training(
        model=model,
        make_dataset=make_dataset,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        val_interval=args.val_interval,
        checkpoint_name=f"pc-37-bs={args.batch_size}-{{step}}",
    )
