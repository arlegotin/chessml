from chessml import script
from chessml.models.torch.vision_model_adapter import VisionModelAdapter, MobileViTAdapter, HRNetAdapter
from chessml.models.lightning.board_detector_model import BoardDetector
from pathlib import Path
import logging
import os
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.composite_boards_images import CompositeBoardsImages
from chessml.const import BOARD_COLORS
import cv2
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.train.standard_training import standard_training

logger = logging.getLogger(__name__)

# efficientnet_b0
# resnet50
# efficientnetv2_rw_t.ra2_in1k
# mobilenetv3_large_100.miil_in21k

script.add_argument("-bs", dest="batch_size", type=int, default=160)
script.add_argument("-vb", dest="val_batches", type=int, default=128)
script.add_argument("-vi", dest="val_interval", type=int, default=512)
script.add_argument("-sb", dest="shuffle_buffer", type=int, default=1)
script.add_argument("-s", dest="seed", type=int, default=65)


@script
def train(args, config):

    model = BoardDetector(
        base_model_class=MobileViTAdapter,
    )

    dir_with_piece_sets = Path(config.assets.path) / "piece_png"
    piece_sets = [
        dir_with_piece_sets / name
        for name in sorted(os.listdir(str(dir_with_piece_sets)))
    ]

    dir_with_bg = Path(config.assets.path) / "bg/512"
    bg_images = [
        cv2.imread(str(dir_with_bg / name))
        for name in sorted(os.listdir(str(dir_with_bg)))
        if name.endswith(".jpg")
    ]

    def make_dataset(**kwargs):
        return CompositeBoardsImages(
            images_with_data=BoardsImagesFromFENs(
                fens=FileLinesDataset(path=Path("./datasets/unique_fens.txt")),
                piece_sets=piece_sets,
                board_colors=BOARD_COLORS,
                square_size=64,
                shuffle_seed=args.seed,
            ),
            bg_images=bg_images,
            # skip_every_nth=11,
            transforms=[lambda x: model.preprocess_input(x[0], x[1])],
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
        checkpoint_name=f"bd-skew-2-bs={args.batch_size}-{{step}}",
        config=config,
    )
