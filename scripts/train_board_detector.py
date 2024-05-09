from chessml import script
from chessml.models.torch.vision_model_adapter import VisionModelAdapter
from chessml.models.lightning.board_detector_model import BoardDetector
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import logging
import os
import torch
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.composite_boards_images import CompositeBoardsImages
from chessml.const import BOARD_COLORS
import cv2
from chessml.data.utils.file_lines_dataset import FileLinesDataset

logger = logging.getLogger(__name__)


script.add_argument("-mn", dest="model_name", default="efficientnet_b0")
script.add_argument("-bs", dest="batch_size", type=int, default=96)
script.add_argument("-vb", dest="val_batches", type=int, default=128)
script.add_argument("-vi", dest="val_interval", type=int, default=256)
script.add_argument("-ci", dest="checkpoint_interval", type=int, default=256)
script.add_argument("-ss", dest="shuffle_seed", type=int, default=67)
script.add_argument("-me", dest="max_epochs", type=int, default=1500)
script.add_argument("-s", dest="seed", type=int, default=65)


@script
def train(args, config):

    # dir_with_pgns = Path(config.dataset.path) / "players"

    # pgn_files = [
    #     dir_with_pgns / name
    #     for name in sorted(os.listdir(str(dir_with_pgns)))
    #     if name.endswith(".pgn")
    # ]

    dir_with_piece_sets = Path(config.assets.path) / "piece_png"

    piece_sets = [
        dir_with_piece_sets / name
        for name in sorted(os.listdir(str(dir_with_piece_sets)))
    ]

    dir_with_bg = Path(config.assets.path) / "bg"

    bg_images = [
        cv2.imread(str(dir_with_bg / name))
        for name in sorted(os.listdir(str(dir_with_bg)))
        if name.endswith(".jpg")
    ]

    model = BoardDetector(
        base_model_class=VisionModelAdapter,
        base_model_kwargs={"model_name": args.model_name, "output_features": 4,},
    )

    def make_ds(**kwargs):
        def innrtrnfrm(img, coords, fen, flipped):
            x1, x2, y1, y2 = coords
            original_height, original_width = img.shape[:2]
            relative_coords = torch.tensor(
                [
                    (x1 / original_width),
                    (x2 / original_width),
                    (y1 / original_height),
                    (y2 / original_height),
                ]
            )
            return model.model.preprocess_image(img), relative_coords.flatten()

        return CompositeBoardsImages(
            images_with_data=BoardsImagesFromFENs(
                # fens=BoardsFromGames(
                #     games=GamesFromPGN(paths=pgn_files),
                #     transforms=[lambda board: board.fen()],
                #     shuffle_buffer=8,
                #     shuffle_seed=args.seed,
                # ),
                fens=FileLinesDataset(path=Path("./datasets/unique_fens.txt")),
                piece_sets=piece_sets,
                board_colors=BOARD_COLORS,
                square_size=64,
                shuffle_seed=args.seed,
            ),
            bg_images=bg_images,
            transforms=[lambda x: innrtrnfrm(*x)],
            **kwargs,
        )

    val_dataloader = DataLoader(
        make_ds(
            shuffle_seed=args.shuffle_seed, limit=args.batch_size * args.val_batches,
        ),
        batch_size=args.batch_size,
    )

    train_dataloader = DataLoader(
        make_ds(
            offset=args.batch_size * args.val_batches,
            shuffle_buffer=1,
            shuffle_seed=args.shuffle_seed,
        ),
        batch_size=args.batch_size,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=TensorBoardLogger(config.logs.tensorboard_path),
        callbacks=[
            ModelCheckpoint(
                dirpath=config.checkpoints.path,
                every_n_train_steps=args.checkpoint_interval,
                filename=f"bd-again-mn={args.model_name}-bs={args.batch_size}-{{step}}",
                save_top_k=3,
                monitor="train_loss",
            ),
        ],
        val_check_interval=args.val_interval,
        check_val_every_n_epoch=None,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )
