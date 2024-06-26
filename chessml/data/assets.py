from chessml import config
from pathlib import Path
from chessml.data.images.picture import Picture
import os

BOARD_COLORS = [
    # (dark, light)
    ("#B58862", "#F0D9B5"),
    ("#9A5824", "#D6A059"),
    ("#775E30", "#977D50"),
    ("#88653D", "#B9AC9D"),
    ("#835835", "#B9945E"),
    ("#B6703C", "#E0BF92"),
    ("#9A6147", "#E1CAA1"),
    ("#92694A", "#F5EAD6"),
    ("#BD8814", "#CFCFC7"),
    ("#8CA1AD", "#DEE3E6"),
    ("#5E7489", "#8498AD"),
    ("#4675AB", "#C3CDD7"),
    ("#6D7C97", "#CFD2E3"),
    ("#9DA2AA", "#E5E5DF"),
    ("#C1C18E", "#ECECEC"),
    ("#85A666", "#FFFFDC"),
    ("#627B64", "#7D967F"),
    ("#58935D", "#F1F6B3"),
    ("#847B6A", "#AEA795"),
    ("#878787", "#ABABAB"),
    ("#747474", "#AEAEAE"),
    ("#7D498D", "#9E90B0"),
    ("#967BB0", "#E7DBF0"),
    ("#F37A7A", "#EFF0C3"),
    
    ("#D8A46D", "#ECCBA5"),
    ("#C2AD9B", "#DDCEC0"),
    ("#BA5745", "#F5DBC3"),
    ("#B49960", "#E3E0C3"),
    ("#E2E1E1", "#E6E5E5"),
    ("#D18715", "#F9E4AD"),
    ("#6C6360", "#CCCDCD"),
    ("#A8A9A8", "#D8D9D8"),
    ("#AE773B", "#ADADAD"),
    ("#303030", "#C74C51"),
    ("#FAD9E1", "#FEFFFE"),
    ("#4B7399", "#EAE9D2"),
    ("#734024", "#F0D0AE"),
    ("#5D301F", "#BD9055"),
    ("#32684C", "#E9E8E7"),
    ("#606160", "#E8E8E8"),
    ("#C9C8C8", "#EBEAEA"),
    ("#8476B9", "#F0F1F0"),
    ("#68635E", "#C7C2AD"),
    ("#6A9B41", "#F3F3F4"),
    ("#C5703C", "#EBC69B"),
    ("#383736", "#AFACA6"),
    ("#828785", "#E1EAEB"),
    ("#C4D7E4", "#F0F1F0"),
    ("#6C4E36", "#BEA37F"),
    ("#86A7BA", "#D9E4E7"),
    ("#B88761", "#EDD6AF"),
    ("#2F3542", "#7E8797"),
    ("#8E6747", "#CBAF7F"),
    ("#779954", "#E9EDCC"),
]

PIECE_SETS: list[Path] = [
    Path(config.assets.path) / "piece_png" / name
    for name in sorted(os.listdir(str(Path(config.assets.path) / "piece_png")))
    if not name.startswith("_")
]

FREE_PIECE_SETS_NAMES: list[str] = list(
    map(
        lambda name: f"lichess_{name}",
        ["cburnett", "chessnut", "pirouetti", "merida", "mpchess"],
    )
)

FREE_PIECE_SETS: list[Path] = [p for p in PIECE_SETS if p.name in FREE_PIECE_SETS_NAMES]

BG_IMAGES: list[Picture] = [
    Picture(Path(config.assets.path) / "bg" / "512" / name).as_3_channels
    for name in sorted(os.listdir(str(Path(config.assets.path) / "bg" / "512")))
    if name.endswith(".jpg")
]

PIECE_CLASSES = {
    None: 0,
    "p": 1,
    "r": 2,
    "n": 3,
    "b": 4,
    "q": 5,
    "k": 6,
    "P": 7,
    "R": 8,
    "N": 9,
    "B": 10,
    "Q": 11,
    "K": 12,
}

# Calculated with scripts/data/calc_pieces_weights.py
PIECE_WEIGHTS = [
    0.0020529834340402738,
    0.016855757893023245,
    0.061449736976114035,
    0.09922613973323134,
    0.08763822838222773,
    0.14438239584336376,
    0.08915722673948553,
    0.016769470342390635,
    0.06122205994982844,
    0.1016601086933676,
    0.08675515858412415,
    0.14367350668931772,
    0.08915722673948553,
]

INVERTED_PIECE_CLASSES = {value: key for key, value in PIECE_CLASSES.items()}

PIECE_CLASSES_NUMBER = len(PIECE_CLASSES)

# We assume that all games are classical chess games
BOARD_SIZE = 8
