from glob import glob
from pathlib import Path

from tqdm import tqdm

from chessml import config, script
from chessml.data.assets import BOARD_COLORS, PIECE_SETS
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.picture import Picture
from chessml.data.utils.file_lines_dataset import FileLinesDataset
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.square_classifier_model import SquareClassifier
from chessml.models.torch.vision_model_adapter import (
    MobileNetV3LargeClassifier,
    MobileNetV3SmallClassifier,
    MobileViTV2FPN,
)
from chessml.models.utils.board_recognition_helper import (
    BoardRecognitionHelper,
    RecognitionFailure,
)
from chessml.utils import reset_dir, write_lines_to_txt


script.add_argument(
    "-i",
    dest="input_dir",
    type=str,
    default="./test_data/input_frames/book_long",
)
script.add_argument("-ss", dest="square_size", type=int, default=32)
script.add_argument("-d", dest="device", type=str, default="mps")


@script
def main(args):
    board_detector = BoardDetector.load_from_checkpoint(
        "./checkpoints/bd-MobileViTV2FPN-v1.ckpt",
        base_model_class=MobileViTV2FPN,
        base_model_kwargs={"pretrained": False},
        map_location=args.device,
        strict=True,
        weights_only=False,
    )
    square_classifier = SquareClassifier.load_from_checkpoint(
        "./checkpoints/sc-9-bs=64-step=23296.ckpt",
        base_model_class=MobileNetV3SmallClassifier,
        base_model_kwargs={"pretrained": False},
        map_location=args.device,
        strict=True,
        weights_only=False,
    )
    piece_classifier = PieceClassifier.load_from_checkpoint(
        "./checkpoints/pc-48-bs=128-step=18944.ckpt",
        base_model_class=MobileNetV3LargeClassifier,
        base_model_kwargs={"pretrained": False},
        map_location=args.device,
        strict=True,
        weights_only=False,
    )

    for model in (board_detector, square_classifier, piece_classifier):
        model.eval()

    helper = BoardRecognitionHelper(
        board_detector=board_detector,
        square_classifier=square_classifier,
        piece_classifier=piece_classifier,
    )

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = reset_dir(
            input_dir.parent / f"{input_dir.stem}_recognized"
        )

        for image_name in tqdm(sorted(glob(f"{input_dir}/*.png"))):
            image_path = Path(image_name)
            result = helper.recognize(Picture(image_path))
            text_path = output_dir / f"{image_path.name}.txt"

            if isinstance(result, RecognitionFailure):
                write_lines_to_txt(
                    text_path,
                    [f"failure:{result.reason.name}"],
                )
                continue

            result.board_image.pil.resize(
                (args.square_size * 8, args.square_size * 8)
            ).save(output_dir / image_path.name)
            write_lines_to_txt(text_path, [result.source_placement])
        return

    dataset = BoardsImagesFromFENs(
        fens=FileLinesDataset(path=Path(config.dataset.path) / "unique_fens.txt"),
        piece_sets=PIECE_SETS,
        board_colors=BOARD_COLORS,
        square_size=64,
        shuffle_seed=10,
        limit=1024,
    )

    for picture, fen, flipped in dataset:
        expected = fen.split()[0]
        if flipped:
            expected = "/".join(
                row[::-1] for row in reversed(expected.split("/"))
            )
        result = helper.recognize(picture)
        print("----------")
        print("expected source placement:", expected)
        if isinstance(result, RecognitionFailure):
            print("recognition failure:", result.reason.name)
        else:
            print("recognized source placement:", result.source_placement)
