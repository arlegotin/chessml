import csv
import logging
from pathlib import Path
import random

from tqdm import tqdm

from chessml import config, script
from chessml.data.assets import BOARD_COLORS, PIECE_SETS
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3

logger = logging.getLogger(__name__)

script.add_argument("-l", dest="limit", type=int, default=2**20)
script.add_argument("-e", dest="with_empty_squares", action="store_true")
script.add_argument("-s", dest="seed", type=int, default=70)


def validation_count(length: int) -> int:
    if length < 2:
        raise ValueError("at least two items are required")
    return max(1, round(length / 5))


def split_sources(values, rng: random.Random):
    values = sorted(values)
    rng.shuffle(values)
    count = validation_count(len(values))
    return values[count:], values[:count]


def write_split(
    dataset_dir: Path,
    split: str,
    piece_sets: list[Path],
    board_colors: list[tuple[str, str]],
    limit: int,
    seed: int,
    with_empty_squares: bool,
) -> Path:
    images_dir = dataset_dir / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / f"{split}.csv"
    dataset = AugmentedPiecesImages(
        piece_images_3x3=PiecesImages3x3(
            piece_sets=piece_sets,
            board_colors=board_colors,
            square_size=64,
            shuffle_seed=seed,
            with_empty_squares=with_empty_squares,
            limit=limit,
        ),
        shuffle_seed=seed,
        limit=limit,
    )

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["image_path", "piece_name", "piece_set", "dark_color", "light_color"]
        )
        rows = tqdm(dataset, desc=f"Generating {split} dataset", total=limit)
        for index, (picture, piece_name, piece_set, dark_color, light_color) in enumerate(rows):
            target = int(piece_name is not None) if with_empty_squares else piece_name
            image_path = images_dir / f"{index}_{target}.png"
            picture.pil.save(image_path)
            writer.writerow(
                [str(image_path), target, piece_set, dark_color, light_color]
            )

    return csv_path


def generate_classifier_dataset(
    dataset_dir: Path,
    piece_sets: list[Path],
    board_colors: list[tuple[str, str]],
    limit: int,
    seed: int,
    with_empty_squares: bool,
) -> tuple[Path, Path]:
    validation_limit = validation_count(limit)
    rng = random.Random(seed)
    training_piece_sets, validation_piece_sets = split_sources(piece_sets, rng)
    training_colors, validation_colors = split_sources(board_colors, rng)

    train_csv = write_split(
        dataset_dir,
        "train",
        training_piece_sets,
        training_colors,
        limit - validation_limit,
        seed,
        with_empty_squares,
    )
    validation_csv = write_split(
        dataset_dir,
        "validation",
        validation_piece_sets,
        validation_colors,
        validation_limit,
        (seed + 1) % 2**32,
        with_empty_squares,
    )
    return train_csv, validation_csv


@script
def main(args):
    dataset_name = "square_classifier" if args.with_empty_squares else "piece_classifier"
    dataset_dir = Path(config.dataset.path_to_big) / dataset_name
    paths = generate_classifier_dataset(
        dataset_dir=dataset_dir,
        piece_sets=PIECE_SETS,
        board_colors=BOARD_COLORS,
        limit=args.limit,
        seed=args.seed,
        with_empty_squares=args.with_empty_squares,
    )
    logger.info("Datasets generated at %s and %s", *paths)
