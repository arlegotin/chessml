from __future__ import annotations

from pathlib import Path

from fentoboardimage import fen_to_image
from PIL import Image, ImageDraw

from benchmarks.board_recognition import (
    BenchmarkValidationError,
    _validate_recipe_template,
)


PIECE_PATH_BY_SYMBOL = {
    "p": "black/Pawn.png",
    "r": "black/Rook.png",
    "n": "black/Knight.png",
    "b": "black/Bishop.png",
    "q": "black/Queen.png",
    "k": "black/King.png",
    "P": "white/Pawn.png",
    "R": "white/Rook.png",
    "N": "white/Knight.png",
    "B": "white/Bishop.png",
    "Q": "white/Queen.png",
    "K": "white/King.png",
}


def _by_id(spec: dict, key: str, identifier: str) -> dict:
    try:
        return next(item for item in spec[key] if item["id"] == identifier)
    except (KeyError, StopIteration, TypeError) as error:
        raise BenchmarkValidationError(f"unknown {key} id: {identifier!r}") from error


def _make_piece_loader(directory: Path):
    pieces = {}
    for symbol, relative_path in PIECE_PATH_BY_SYMBOL.items():
        try:
            with Image.open(directory / relative_path) as source:
                source.load()
                pieces[symbol] = source.convert("RGBA")
        except OSError as error:
            raise BenchmarkValidationError(
                f"could not load piece image: {directory / relative_path}"
            ) from error

    def load(board: Image.Image) -> dict[str, Image.Image]:
        if board.width != board.height or board.width % 8:
            raise BenchmarkValidationError("piece loader requires a square 8x8 board")
        length = board.width // 8
        return {
            symbol: image.resize(
                (length, length), resample=Image.Resampling.LANCZOS
            )
            for symbol, image in pieces.items()
        }

    return load


def draw_ui_recipe(
    recipe: dict, canvas_size: list[int], coordinate_space: list[int]
) -> Image.Image:
    if (
        not isinstance(canvas_size, (list, tuple))
        or len(canvas_size) != 2
        or any(type(value) is not int or value <= 0 for value in canvas_size)
    ):
        raise BenchmarkValidationError("canvas size must contain two positive integers")
    if coordinate_space != [1000, 1000]:
        raise BenchmarkValidationError("coordinate space must be [1000, 1000]")
    _validate_recipe_template(recipe, "render recipe")

    width, height = canvas_size
    image = Image.new("RGB", (width, height), recipe["background"])
    draw = ImageDraw.Draw(image)
    for shape in recipe["shapes"]:
        if shape["kind"] == "polygon":
            points = [
                (x * (width - 1) // 1000, y * (height - 1) // 1000)
                for x, y in shape["points"]
            ]
            draw.polygon(points, fill=shape["fill"])
            continue
        left, top, right, bottom = shape["box"]
        box = (
            left * width // 1000,
            top * height // 1000,
            right * width // 1000 - 1,
            bottom * height // 1000 - 1,
        )
        if box[0] > box[2] or box[1] > box[3]:
            raise BenchmarkValidationError("scaled recipe box is empty")
        getattr(draw, shape["kind"])(box, fill=shape["fill"])
    return image


def _render_board(
    case: dict, spec: dict, repo_root: Path, square_size: int
) -> Image.Image:
    position = _by_id(spec, "positions", case["base_position"])
    piece_set = _by_id(spec, "piece_sets", case["piece_set"])
    palette = _by_id(spec, "palettes", case["palette"])
    try:
        directory = (repo_root / piece_set["path"]).resolve(strict=True)
    except OSError as error:
        raise BenchmarkValidationError("piece-set path does not resolve") from error
    return fen_to_image(
        fen=position["fen"],
        square_length=square_size,
        piece_set=_make_piece_loader(directory),
        dark_color=palette["dark"],
        light_color=palette["light"],
        arrows=None,
        last_move=None,
        coordinates=None,
        flipped=case["view"] == "black_bottom",
    )


def _apply_quality(image: Image.Image, quality_id: str, spec: dict) -> Image.Image:
    quality = _by_id(spec, "qualities", quality_id)
    if quality_id == "clean":
        return image.copy()
    numerator, denominator = quality["downsample"]
    smaller = (
        image.width * numerator // denominator,
        image.height * numerator // denominator,
    )
    return image.resize(
        smaller, getattr(Image.Resampling, quality["downsample_filter"])
    ).resize(
        image.size, getattr(Image.Resampling, quality["upsample_filter"])
    )


def render_case(case: dict, spec: dict, repo_root: Path) -> Image.Image:
    recipes = spec["ui_recipes"]
    if case["board_present"]:
        square_size = 64 if case["layout"] == "board_crop" else 60
        board = _render_board(case, spec, repo_root, square_size)
        if case["layout"] == "board_crop":
            canvas = board
        else:
            canvas = draw_ui_recipe(
                recipes["positive_desktop"],
                case["image_size"],
                recipes["coordinate_space"],
            )
            canvas.paste(board, tuple(case["board_box"][:2]))
    else:
        canvas = draw_ui_recipe(
            recipes["negative_templates"][case["negative_template"]],
            case["image_size"],
            recipes["coordinate_space"],
        )

    image = _apply_quality(canvas, case["quality"], spec).convert(
        case["image_mode"]
    )
    image.load()
    if image.size != tuple(case["image_size"]):
        raise BenchmarkValidationError("rendered image size differs from case plan")
    return image.copy()
