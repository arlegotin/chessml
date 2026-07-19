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

FREE_PIECE_SETS_NAMES: list[str] = list(
    map(
        lambda name: f"lichess_{name}",
        ["cburnett", "chessnut", "pirouetti", "merida", "mpchess"],
    )
)

PIECE_CLASSES = {
    "p": 0,
    "r": 1,
    "n": 2,
    "b": 3,
    "q": 4,
    "k": 5,
    "P": 6,
    "R": 7,
    "N": 8,
    "B": 9,
    "Q": 10,
    "K": 11,
}

PIECE_SYMBOLS = {
  "p": "♟",  # black pawn
  "r": "♜",  # black rook
  "n": "♞",  # black knight
  "b": "♝",  # black bishop
  "q": "♛",  # black queen
  "k": "♚",  # black king
  "P": "♙",  # white pawn
  "R": "♖",  # white rook
  "N": "♘",  # white knight
  "B": "♗",  # white bishop
  "Q": "♕",  # white queen
  "K": "♔",  # white king
}

# Calculated with scripts/data/calc_pieces_weights.py
PIECE_WEIGHTS = [
  0.016890433673548793,
  0.061576151795682516,
  0.09943026842715445,
  0.08781851834559319,
  0.14467942029647896,
  0.08934064159667002,
  0.01680396861157633,
  0.061348006390659855,
  0.10186924456489752,
  0.08693363189025578,
  0.14396907281081261,
  0.08934064159667002,
]

EMPTY_SQUARE_CHANCE = 0.6786

INVERTED_PIECE_CLASSES = {value: key for key, value in PIECE_CLASSES.items()}

PIECE_CLASSES_NUMBER = len(PIECE_CLASSES)

# We assume that all games are classical chess games
BOARD_SIZE = 8
