from chessml import script
from chessml.models.torch.vision_model_adapter import MobileViTAdapter, EfficientNetV2Adapter
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from pathlib import Path
import logging
import os
import torch
from chessml.data.images.pieces_images import generate_pieces_images, CompositePiecesImages
from chessml.const import BOARD_COLORS, PIECE_CLASSES
from chessml.train.standard_training import standard_training

logger = logging.getLogger(__name__)


script.add_argument("-bs", dest="batch_size", type=int, default=160)
script.add_argument("-vb", dest="val_batches", type=int, default=128)
script.add_argument("-vi", dest="val_interval", type=int, default=128)
script.add_argument("-sb", dest="shuffle_buffer", type=int, default=1)
script.add_argument("-s", dest="seed", type=int, default=65)


@script
def train(args, config):

    dir_with_piece_sets = Path(config.assets.path) / "piece_png"
    piece_sets = [
        dir_with_piece_sets / name
        for name in sorted(os.listdir(str(dir_with_piece_sets)))
    ]

    piece_images = [x for x in generate_pieces_images(
        piece_sets=piece_sets,
        board_colors=BOARD_COLORS,
        size=128,
    )]

    logger.info(f"{len(piece_images)} pure piece images generated")

    model = PieceClassifier(
        # base_model_class=MobileViTAdapter,
        base_model_class=EfficientNetV2Adapter,
    )

    def make_dataset(**kwargs):
        def innrtrnfrm(img, piece_name):
            piece_class = PIECE_CLASSES[piece_name]
            return model.model.preprocess_image(img), torch.tensor(piece_class, dtype=torch.long)

        return CompositePiecesImages(
            piece_images=piece_images,
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
        shuffle_buffer=args.shuffle_buffer,
        checkpoint_name=f"pc-2-bs={args.batch_size}-{{step}}",
        config=config,
    )
