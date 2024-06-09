from chessml import script, config
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.torch.vision_model_adapter import MobileNetV3S50Classifier
from pathlib import Path
import logging
import os
import torch
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, PIECE_CLASSES
from chessml.train.standard_training import standard_training

logger = logging.getLogger(__name__)

script.add_argument("-bs", dest="batch_size", type=int, default=64)
script.add_argument("-vb", dest="val_batches", type=int, default=2048)
script.add_argument("-vi", dest="val_interval", type=int, default=32 * 8)
script.add_argument("-sb", dest="shuffle_buffer", type=int, default=1)
script.add_argument("-s", dest="seed", type=int, default=67)


@script
def train(args):

    model = PieceClassifier(
        base_model_class=MobileNetV3S50Classifier,
    )

    def make_dataset(**kwargs):
        def innrtrnfrm(img, piece_name):
            piece_class = PIECE_CLASSES[piece_name]
            return model.model.preprocess_image(img.pil), torch.tensor(piece_class, dtype=torch.long)

        return AugmentedPiecesImages(
            piece_images_3x3=PiecesImages3x3(
                piece_sets=PIECE_SETS,
                board_colors=BOARD_COLORS,
                square_size=64,
                shuffle_seed=args.seed,
            ),
            transforms=[lambda x: innrtrnfrm(*x)],
            shuffle_seed=args.seed,
            **kwargs,
        )

    standard_training(
        model=model,
        make_dataset=make_dataset,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        val_interval=args.val_interval,
        checkpoint_name=f"pc-7-bs={args.batch_size}-{{step}}",
    )
