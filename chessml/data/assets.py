from pathlib import Path
import os

from chessml import config
from chessml.data.constants import (
    BOARD_COLORS,
    BOARD_SIZE,
    EMPTY_SQUARE_CHANCE,
    FREE_PIECE_SETS_NAMES,
    INVERTED_PIECE_CLASSES,
    PIECE_CLASSES,
    PIECE_CLASSES_NUMBER,
    PIECE_SYMBOLS,
    PIECE_WEIGHTS,
)
from chessml.data.images.picture import Picture

PIECE_SETS: list[Path] = [
    Path(config.assets.path) / "piece_png" / name
    for name in sorted(os.listdir(str(Path(config.assets.path) / "piece_png")))
    if not name.startswith("_")
    and (Path(config.assets.path) / "piece_png" / name).is_dir()
]

FREE_PIECE_SETS: list[Path] = [p for p in PIECE_SETS if p.name in FREE_PIECE_SETS_NAMES]

BG_IMAGES: list[Picture] = [
    Picture(Path(config.assets.path) / "bg" / "512" / name).as_3_channels
    for name in sorted(os.listdir(str(Path(config.assets.path) / "bg" / "512")))
    if name.endswith(".jpg")
]
