from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import zlib
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import chess
from PIL import Image


PIECE_SYMBOLS = frozenset("prnbqkPRNBQK")
PIECE_NAMES = ("Pawn", "Rook", "Knight", "Bishop", "Queen", "King")
EXPECTED_SOURCE_SPEC_SHA256 = "8393ff76e9f10b4bbe24b558f9b2d0032c493ebb6fe1781a4515b4790018756b"
ALLOWED_PIECE_PATHS = tuple(
    f"{color}/{piece}.png"
    for color in ("black", "white")
    for piece in PIECE_NAMES
)
PREDICTION_OUTCOMES = frozenset({"success", "failure", "error"})
FAILURE_REASONS = frozenset(
    {"NO_BOARD", "INVALID_GEOMETRY", "DECODING_FAILED", "INVALID_PLACEMENT"}
)


def _ratio(correct: int, total: int) -> dict:
    return {
        "correct": correct,
        "total": total,
        "value": None if total == 0 else correct / total,
    }


GROUP_FIELDS = (
    "suite",
    "split",
    "base_position",
    "piece_set",
    "palette",
    "view",
    "layout",
    "quality",
    "channel_mode",
    "negative_template",
)

_EXPECTED_DEPENDENCIES = {
    "Pillow": "12.3.0",
    "chess": "1.11.2",
    "fentoboardimage": "1.4.1",
    "python": "3.14.6",
}
_EXPECTED_COUNTS = {
    "all": 568,
    "input_contract_negative": 8,
    "input_contract_positive": 16,
    "main_negative": 32,
    "main_positive": 512,
    "negative": 40,
    "positive": 528,
}
_EXPECTED_CASE_MATRIX = {
    "input_contract": {
        "channel_modes": ["RGBA", "L"],
        "layout_id": "board_crop",
        "negative_template_ids": [
            "lobby_cards",
            "modal_dialog",
            "analysis_panels",
            "profile_cards",
        ],
        "palette_id": "brown",
        "piece_set_id": "lichess_chessnut",
        "position_ids": ["pos_01", "pos_06", "pos_11", "pos_16"],
        "quality_id": "clean",
    },
    "main": {
        "channel_mode": "RGB",
        "layout_ids": ["board_crop", "desktop_ui"],
        "palette_index_formula": "(position_index + piece_set_index) % 4",
        "quality_ids": ["clean", "browser_scaled"],
    },
    "views": ["white_bottom", "black_bottom"],
}
_EXPECTED_LIMITATIONS = [
    "Variants share 16 base positions; 568 is not an independent-position count.",
    "Current-checkpoint holdout cannot be proved because its training manifest is unavailable.",
    "This synthetic corpus does not measure accuracy on unseen chess sites.",
    "P2-01 remains open until an independently captured and manually labeled online-screenshot suite exists.",
]
_EXPECTED_RENDERER = {
    "piece_resampling": "LANCZOS",
    "png_compress_level": 9,
    "png_optimize": False,
}
_EXPECTED_PIECE_SETS = {
    "lichess_chessnut": {
        "license": "Apache-2.0",
        "path": "assets/piece_png/lichess_chessnut",
        "tree_sha256": "ab9628638c8266b8f7e9a3f36f9d518c5de59042bf0c0c5a09c6b8384654ef3b",
    },
    "lichess_celtic": {
        "license": "MIT",
        "path": "assets/piece_png/lichess_celtic",
        "tree_sha256": "5fc39e440f5aba5cff5cebce938c6fa32c1f6ac8ef5b3060861f2fea0a4ec397",
    },
    "lichess_fantasy": {
        "license": "MIT",
        "path": "assets/piece_png/lichess_fantasy",
        "tree_sha256": "d8b894b8f0952cc6c0db654cc47681f818f4c7f7ae00814b54dbff5e1aab52cd",
    },
    "lichess_spatial": {
        "license": "MIT",
        "path": "assets/piece_png/lichess_spatial",
        "tree_sha256": "fc033d2aff36ff3f0edee801ac2aa01b1a1d0c2b20cb95dd2fd1065c48646d0d",
    },
}


class BenchmarkValidationError(ValueError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _validate_json_native(value: object, active: set[int] | None = None) -> None:
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return
    if value_type is float:
        if math.isfinite(value):
            return
        raise BenchmarkValidationError("value is not canonical JSON")
    if value_type not in {list, dict}:
        raise BenchmarkValidationError("value is not canonical JSON")

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise BenchmarkValidationError("value is not canonical JSON")
    active.add(identity)
    try:
        if value_type is list:
            for item in value:
                _validate_json_native(item, active)
        else:
            if any(type(key) is not str for key in value):
                raise BenchmarkValidationError("value is not canonical JSON")
            for item in value.values():
                _validate_json_native(item, active)
    finally:
        active.remove(identity)


def _same_json_value(actual: object, expected: object) -> bool:
    _validate_json_native(actual)
    _validate_json_native(expected)
    try:
        return canonical_json_bytes(actual) == canonical_json_bytes(expected)
    except (TypeError, ValueError, UnicodeError) as error:
        raise BenchmarkValidationError("value is not canonical JSON") from error


def expand_placement(placement: str) -> tuple:
    if not isinstance(placement, str):
        raise BenchmarkValidationError("placement must be a string")
    rows = placement.split("/")
    if len(rows) != 8:
        raise BenchmarkValidationError("placement must contain eight ranks")
    squares: list[str | None] = []
    for row in rows:
        expanded: list[str | None] = []
        for symbol in row:
            if symbol in "12345678":
                expanded.extend([None] * int(symbol))
            elif symbol in PIECE_SYMBOLS:
                expanded.append(symbol)
            else:
                raise BenchmarkValidationError(f"invalid placement symbol: {symbol!r}")
        if len(expanded) != 8:
            raise BenchmarkValidationError("every rank must expand to eight squares")
        squares.extend(expanded)
    result = tuple(squares)
    if compress_placement(result) != placement:
        raise BenchmarkValidationError("placement must use canonical FEN compression")
    return result


def compress_placement(squares: Sequence[str | None]) -> str:
    if len(squares) != 64:
        raise BenchmarkValidationError("source grid must contain 64 squares")
    rows: list[str] = []
    for start in range(0, 64, 8):
        row: list[str] = []
        empty = 0
        for symbol in squares[start : start + 8]:
            if symbol is None:
                empty += 1
            else:
                if symbol not in PIECE_SYMBOLS:
                    raise BenchmarkValidationError(f"invalid piece symbol: {symbol!r}")
                if empty:
                    row.append(str(empty))
                    empty = 0
                row.append(symbol)
        if empty:
            row.append(str(empty))
        rows.append("".join(row))
    return "/".join(rows)


def rotate_placement(placement: str) -> str:
    return compress_placement(tuple(reversed(expand_placement(placement))))


def _score_group(cases: Sequence[dict], prediction_by_id: dict[str, dict]) -> dict:
    counts = {
        "all": len(cases),
        "positive": 0,
        "negative": 0,
        "success": 0,
        "failure": 0,
        "error": 0,
    }
    positive_successes = 0
    exact_positive_placements = 0
    correct_positive_squares = 0
    occupancy_tp = 0
    occupancy_fp = 0
    occupancy_fn = 0
    correct_symbols = 0
    both_occupied = 0
    negative_failures = 0
    negative_no_board = 0
    positive_no_board = 0
    failure_reasons = {reason: 0 for reason in sorted(FAILURE_REASONS)}
    execution_error_count = 0

    for case in cases:
        prediction = prediction_by_id[case["id"]]
        outcome = prediction["outcome"]
        counts[outcome] += 1
        if outcome == "failure":
            failure_reasons[prediction["reason"]] += 1
        elif outcome == "error":
            execution_error_count += 1

        if case["board_present"]:
            counts["positive"] += 1
            expected = expand_placement(case["source_placement"])
            if outcome == "success":
                positive_successes += 1
                predicted = expand_placement(prediction["source_placement"])
                exact_positive_placements += expected == predicted
                correct_positive_squares += sum(
                    expected_symbol == predicted_symbol
                    for expected_symbol, predicted_symbol in zip(expected, predicted)
                )
                for expected_symbol, predicted_symbol in zip(expected, predicted):
                    if expected_symbol is not None and predicted_symbol is not None:
                        occupancy_tp += 1
                        both_occupied += 1
                        correct_symbols += expected_symbol == predicted_symbol
                    elif predicted_symbol is not None:
                        occupancy_fp += 1
                    elif expected_symbol is not None:
                        occupancy_fn += 1
            else:
                occupancy_fn += sum(symbol is not None for symbol in expected)
                if outcome == "failure" and prediction["reason"] == "NO_BOARD":
                    positive_no_board += 1
        else:
            counts["negative"] += 1
            if outcome == "failure":
                negative_failures += 1
                negative_no_board += prediction["reason"] == "NO_BOARD"

    positive_count = counts["positive"]
    negative_count = counts["negative"]
    return {
        "counts": counts,
        "positive_success_rate": _ratio(positive_successes, positive_count),
        "exact_placement": _ratio(exact_positive_placements, positive_count),
        "square_accuracy": _ratio(correct_positive_squares, positive_count * 64),
        "occupancy": {"tp": occupancy_tp, "fp": occupancy_fp, "fn": occupancy_fn},
        "occupancy_precision": _ratio(occupancy_tp, occupancy_tp + occupancy_fp),
        "occupancy_recall": _ratio(occupancy_tp, occupancy_tp + occupancy_fn),
        "conditional_piece_accuracy": _ratio(correct_symbols, both_occupied),
        "negative_any_rejection_rate": _ratio(negative_failures, negative_count),
        "negative_no_board_rate": _ratio(negative_no_board, negative_count),
        "positive_false_no_board_rate": _ratio(positive_no_board, positive_count),
        "failure_reasons": failure_reasons,
        "execution_error_count": execution_error_count,
    }


def score_predictions(cases: Sequence[dict], predictions: Sequence[dict]) -> dict:
    if any(not isinstance(prediction, dict) for prediction in predictions):
        raise BenchmarkValidationError("predictions must be objects")
    case_ids = _unique_ids(cases, "case")
    prediction_ids = _unique_ids(predictions, "prediction")
    if set(prediction_ids) != set(case_ids):
        raise BenchmarkValidationError("prediction IDs must equal case IDs")

    prediction_by_id = {prediction["id"]: prediction for prediction in predictions}
    for prediction in predictions:
        outcome = prediction.get("outcome")
        if not isinstance(outcome, str) or outcome not in PREDICTION_OUTCOMES:
            raise BenchmarkValidationError("invalid prediction outcome")
        if outcome == "success":
            expand_placement(prediction.get("source_placement"))
        elif outcome == "failure":
            reason = prediction.get("reason")
            if not isinstance(reason, str) or reason not in FAILURE_REASONS:
                raise BenchmarkValidationError("invalid prediction failure reason")
        elif not isinstance(prediction.get("error_type"), str) or not isinstance(
            prediction.get("error_message"), str
        ):
            raise BenchmarkValidationError("invalid prediction execution error")

    groups = {}
    for field in GROUP_FIELDS:
        values = sorted({case[field] for case in cases if field in case})
        if values:
            groups[field] = {
                value: _score_group(
                    [
                        case
                        for case in cases
                        if field in case and case[field] == value
                    ],
                    prediction_by_id,
                )
                for value in values
            }
    return {
        "overall": _score_group(cases, prediction_by_id),
        "groups": groups,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(b"\0")
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise BenchmarkValidationError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise BenchmarkValidationError(f"invalid JSON constant: {value}")


def load_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise BenchmarkValidationError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{path} must contain a JSON object")
    return value


def piece_set_tree_sha256(directory: Path) -> str:
    if directory.is_symlink() or not directory.is_dir():
        raise BenchmarkValidationError(f"{directory} must be a real directory")
    for color in ("black", "white"):
        color_dir = directory / color
        if color_dir.is_symlink() or not color_dir.is_dir():
            raise BenchmarkValidationError(f"{color_dir} must be a real directory")
    pngs = tuple(
        path for path in directory.rglob("*") if path.suffix.lower() == ".png"
    )
    if any(path.is_symlink() or not path.is_file() for path in pngs):
        raise BenchmarkValidationError(f"{directory} contains a non-regular PNG")
    allowed = set(ALLOWED_PIECE_PATHS)
    actual = {path.relative_to(directory).as_posix() for path in pngs}
    if actual != allowed:
        raise BenchmarkValidationError(
            f"{directory} PNG inventory differs: "
            f"missing={sorted(allowed - actual)}, extra={sorted(actual - allowed)}"
        )
    digest = hashlib.sha256()
    for relative in sorted(ALLOWED_PIECE_PATHS):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((directory / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_source_spec(path: Path) -> dict:
    return load_json(path)


def source_spec_sha256(spec: object) -> str:
    return hashlib.sha256(canonical_json_bytes(spec)).hexdigest()


def validate_piece_set(directory: Path, expected_tree_sha256: str) -> None:
    actual_tree_sha256 = piece_set_tree_sha256(directory)
    if actual_tree_sha256 != expected_tree_sha256:
        raise BenchmarkValidationError(
            f"{directory} tree hash mismatch: "
            f"expected {expected_tree_sha256}, got {actual_tree_sha256}"
        )

    file_digests: set[str] = set()
    for relative in ALLOWED_PIECE_PATHS:
        path = directory / relative
        digest = file_sha256(path)
        if digest in file_digests:
            raise BenchmarkValidationError(
                f"{directory} contains duplicate file bytes: {relative}"
            )
        file_digests.add(digest)
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (100, 100):
                    raise BenchmarkValidationError(
                        f"{path} must be exactly 100x100 pixels"
                    )
                if image.mode != "RGBA":
                    raise BenchmarkValidationError(f"{path} must use RGBA mode")
                if image.getchannel("A").getextrema()[1] == 0:
                    raise BenchmarkValidationError(f"{path} must have nonempty alpha")
        except BenchmarkValidationError:
            raise
        except OSError as error:
            raise BenchmarkValidationError(f"could not decode PNG: {path}") from error


def _records(spec: dict, key: str) -> list[dict]:
    value = spec.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise BenchmarkValidationError(f"{key} must be a list of objects")
    return value


def _unique_ids(records: Iterable[dict], label: str) -> list[str]:
    ids: list[str] = []
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise BenchmarkValidationError(f"invalid {label} id")
        if identifier in ids:
            raise BenchmarkValidationError(f"duplicate {label} id: {identifier!r}")
        ids.append(identifier)
    return ids


def _rgb(value: object, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, str)
        or len(value) != 7
        or not value.startswith("#")
        or any(symbol not in "0123456789abcdefABCDEF" for symbol in value[1:])
    ):
        raise BenchmarkValidationError(f"invalid {label} color")
    try:
        return tuple(bytes.fromhex(value[1:]))  # type: ignore[return-value]
    except ValueError as error:
        raise BenchmarkValidationError(f"invalid {label} color") from error


def _coordinates(value: object, length: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise BenchmarkValidationError(f"{label} must contain {length} coordinates")
    if any(type(coordinate) is not int for coordinate in value):
        raise BenchmarkValidationError(f"{label} coordinate integer required")
    if any(coordinate < 0 or coordinate > 1000 for coordinate in value):
        raise BenchmarkValidationError(f"{label} coordinate range is 0..1000")
    return value


def _validate_recipe_template(template: object, label: str) -> None:
    if not isinstance(template, dict) or set(template) != {"background", "shapes"}:
        raise BenchmarkValidationError(f"{label} template keys differ")
    _rgb(template["background"], f"{label} background")
    shapes = template["shapes"]
    if not isinstance(shapes, list):
        raise BenchmarkValidationError(f"{label} shapes must be a list")
    for index, shape in enumerate(shapes):
        shape_label = f"{label} shape {index}"
        if not isinstance(shape, dict):
            raise BenchmarkValidationError(f"{shape_label} must be an object")
        kind = shape.get("kind")
        if not isinstance(kind, str) or kind not in {"rectangle", "ellipse", "polygon"}:
            raise BenchmarkValidationError(f"invalid {shape_label} shape kind")
        expected_keys = (
            {"kind", "fill", "points"}
            if kind == "polygon"
            else {"kind", "fill", "box"}
        )
        if set(shape) != expected_keys:
            raise BenchmarkValidationError(f"{shape_label} geometry keys differ")
        _rgb(shape["fill"], f"{shape_label} fill")
        if kind == "polygon":
            points = shape["points"]
            if not isinstance(points, list) or len(points) < 3:
                raise BenchmarkValidationError(
                    f"{shape_label} must contain at least three points"
                )
            for point in points:
                _coordinates(point, 2, shape_label)
        else:
            box = _coordinates(shape["box"], 4, shape_label)
            if box[0] >= box[2] or box[1] >= box[3]:
                raise BenchmarkValidationError(f"{shape_label} coordinate order invalid")


def _validate_ui_recipes(spec: dict, negative_template_ids: set[str]) -> None:
    recipes = spec.get("ui_recipes")
    expected_keys = {"coordinate_space", "negative_templates", "positive_desktop"}
    if not isinstance(recipes, dict) or set(recipes) != expected_keys:
        raise BenchmarkValidationError("UI recipe keys differ")
    coordinate_space = _coordinates(recipes["coordinate_space"], 2, "coordinate space")
    if coordinate_space != [1000, 1000]:
        raise BenchmarkValidationError("coordinate space must be [1000, 1000]")
    negative = recipes["negative_templates"]
    if not isinstance(negative, dict) or set(negative) != negative_template_ids:
        raise BenchmarkValidationError("negative recipe keys differ")
    for template_id, template in negative.items():
        _validate_recipe_template(template, f"negative recipe {template_id}")
    _validate_recipe_template(recipes["positive_desktop"], "positive desktop recipe")


def _validate_layouts(layouts: list[dict]) -> None:
    for layout in layouts:
        canvas = layout.get("canvas_size")
        box = layout.get("board_box")
        if (
            not isinstance(canvas, list)
            or len(canvas) != 2
            or any(type(value) is not int or value <= 0 for value in canvas)
        ):
            raise BenchmarkValidationError("layout canvas_size integers invalid")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(value) is not int for value in box)
        ):
            raise BenchmarkValidationError("layout board_box integers invalid")
        if not (0 <= box[0] < box[2] <= canvas[0] and 0 <= box[1] < box[3] <= canvas[1]):
            raise BenchmarkValidationError("layout board_box geometry invalid")


def _validate_palettes(palettes: list[dict]) -> None:
    for palette in palettes:
        light = _rgb(palette.get("light"), f"palette {palette.get('id')} light")
        dark = _rgb(palette.get("dark"), f"palette {palette.get('id')} dark")
        if math.dist(light, dark) < 64:
            raise BenchmarkValidationError(f"palette {palette.get('id')} RGB contrast too low")
        light_luma = 0.2126 * light[0] + 0.7152 * light[1] + 0.0722 * light[2]
        dark_luma = 0.2126 * dark[0] + 0.7152 * dark[1] + 0.0722 * dark[2]
        if abs(light_luma - dark_luma) < 40:
            raise BenchmarkValidationError(
                f"palette {palette.get('id')} luminance contrast too low"
            )


def _validate_positions(positions: list[dict]) -> None:
    placements: list[str] = []
    rotated_placements: set[str] = set()
    piece_counts: list[int] = []
    all_symbols: set[str] = set()

    for position in positions:
        fen = position.get("fen")
        if not isinstance(fen, str):
            raise BenchmarkValidationError("position FEN must be a string")
        fields = fen.split(" ")
        if len(fields) != 6 or any(not field for field in fields):
            raise BenchmarkValidationError(f"malformed FEN for {position.get('id')}")
        placement = fields[0]
        squares = expand_placement(placement)
        if squares.count("K") != 1 or squares.count("k") != 1:
            raise BenchmarkValidationError(
                f"position {position.get('id')} must contain exactly one king per color"
            )
        source = position.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("kind"), str):
            raise BenchmarkValidationError(f"invalid source for {position.get('id')}")
        if "record" in source and f"{source['record']} 0 1" != fen:
            raise BenchmarkValidationError(
                f"position {position.get('id')} source.record does not match FEN"
            )
        try:
            board = chess.Board(fen)
        except ValueError as error:
            raise BenchmarkValidationError(
                f"malformed FEN for {position.get('id')}"
            ) from error
        if not board.is_valid():
            raise BenchmarkValidationError(f"invalid FEN for {position.get('id')}")
        if board.fen(en_passant="fen") != fen:
            raise BenchmarkValidationError(f"noncanonical FEN for {position.get('id')}")
        if placement in placements:
            raise BenchmarkValidationError(
                f"duplicate placement for {position.get('id')}"
            )
        if placement in rotated_placements:
            raise BenchmarkValidationError(
                f"rotated duplicate placement for {position.get('id')}"
            )
        placements.append(placement)
        rotated_placements.add(rotate_placement(placement))
        symbols = {symbol for symbol in squares if symbol is not None}
        all_symbols.update(symbols)
        piece_counts.append(sum(symbol is not None for symbol in squares))

    if not PIECE_SYMBOLS.issubset(all_symbols):
        raise BenchmarkValidationError("positions must contain all twelve piece symbols")
    for low, high in ((2, 8), (9, 20), (21, 32)):
        if not any(low <= count <= high for count in piece_counts):
            raise BenchmarkValidationError(
                f"positions lack required piece-count bucket {low}..{high}"
            )
    development = [
        index
        for index, position in enumerate(positions)
        if position.get("split") == "development"
    ]
    if development != [0, 5, 10, 15]:
        raise BenchmarkValidationError("development indexes must be [0, 5, 10, 15]")


def validate_source_spec(
    spec: dict, repo_root: Path, *, verify_local_assets: bool
) -> None:
    if not isinstance(spec, dict):
        raise BenchmarkValidationError("source specification must be an object")
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise BenchmarkValidationError("schema_version must be exactly 1")
    if spec.get("benchmark_version") != "online-render-v1":
        raise BenchmarkValidationError("benchmark_version differs")
    if spec.get("claim") != "synthetic_online_render_regression_only":
        raise BenchmarkValidationError("claim differs")
    if spec.get("dependencies") != _EXPECTED_DEPENDENCIES:
        raise BenchmarkValidationError("dependencies differ")
    if type(spec.get("base_position_count")) is not int:
        raise BenchmarkValidationError("base_position_count must be an integer")
    if spec["base_position_count"] != 16:
        raise BenchmarkValidationError("base_position_count must be exactly 16")
    if spec.get("limitations") != _EXPECTED_LIMITATIONS:
        raise BenchmarkValidationError("limitations differ")
    if spec.get("case_matrix") != _EXPECTED_CASE_MATRIX:
        raise BenchmarkValidationError("case_matrix differs")
    expected_counts = spec.get("expected_counts")
    if not isinstance(expected_counts, dict) or any(
        type(value) is not int for value in expected_counts.values()
    ):
        raise BenchmarkValidationError("expected_counts values must be integers")
    if expected_counts != _EXPECTED_COUNTS:
        raise BenchmarkValidationError("expected_counts differ")
    if spec.get("renderer") != _EXPECTED_RENDERER:
        raise BenchmarkValidationError("renderer differs")

    layouts = _records(spec, "layouts")
    palettes = _records(spec, "palettes")
    piece_sets = _records(spec, "piece_sets")
    positions = _records(spec, "positions")
    qualities = _records(spec, "qualities")
    negative_templates = _records(spec, "negative_templates")
    _unique_ids(layouts, "layout")
    _unique_ids(palettes, "palette")
    piece_set_ids = _unique_ids(piece_sets, "piece set")
    _unique_ids(positions, "position")
    _unique_ids(qualities, "quality")
    negative_template_ids = set(_unique_ids(negative_templates, "negative template"))

    if len(positions) != spec["base_position_count"]:
        raise BenchmarkValidationError("base_position_count does not match positions")
    if set(piece_set_ids) != set(_EXPECTED_PIECE_SETS):
        raise BenchmarkValidationError("piece-set ids differ")
    for piece_set in piece_sets:
        expected = _EXPECTED_PIECE_SETS[piece_set["id"]]
        if piece_set.get("license") != expected["license"]:
            raise BenchmarkValidationError(
                f"piece set {piece_set['id']} license differs"
            )
        if piece_set.get("tree_sha256") != expected["tree_sha256"]:
            raise BenchmarkValidationError(
                f"piece set {piece_set['id']} tree_sha256 differs"
            )
        if piece_set.get("path") != expected["path"]:
            raise BenchmarkValidationError(
                f"piece-set path differs for {piece_set['id']}"
            )

    _validate_layouts(layouts)
    _validate_palettes(palettes)
    _validate_positions(positions)
    _validate_ui_recipes(spec, negative_template_ids)

    actual_spec_sha256 = source_spec_sha256(spec)
    if actual_spec_sha256 != EXPECTED_SOURCE_SPEC_SHA256:
        raise BenchmarkValidationError(
            "source specification digest mismatch: "
            f"expected {EXPECTED_SOURCE_SPEC_SHA256}, got {actual_spec_sha256}"
        )

    if not verify_local_assets:
        return
    if platform.python_version() != _EXPECTED_DEPENDENCIES["python"]:
        raise BenchmarkValidationError(
            "python runtime version mismatch: "
            f"expected {_EXPECTED_DEPENDENCIES['python']}, "
            f"got {platform.python_version()}"
        )
    for distribution in ("Pillow", "chess", "fentoboardimage"):
        try:
            installed = version(distribution)
        except Exception as error:
            raise BenchmarkValidationError(
                f"could not determine installed package version for {distribution}"
            ) from error
        if installed != _EXPECTED_DEPENDENCIES[distribution]:
            raise BenchmarkValidationError(
                f"installed package version mismatch for {distribution}: "
                f"expected {_EXPECTED_DEPENDENCIES[distribution]}, got {installed}"
            )

    resolved_root = repo_root.resolve(strict=True)
    for piece_set in piece_sets:
        directory = repo_root / piece_set["path"]
        try:
            resolved_directory = directory.resolve(strict=True)
        except OSError as error:
            raise BenchmarkValidationError(
                f"piece-set path does not resolve: {directory}"
            ) from error
        if not resolved_directory.is_relative_to(resolved_root):
            raise BenchmarkValidationError(f"piece-set path escapes repository: {directory}")
        validate_piece_set(directory, piece_set["tree_sha256"])


def build_case_plan(spec: dict) -> list[dict]:
    validate_source_spec(spec, Path(), verify_local_assets=False)
    matrix = spec["case_matrix"]
    layouts = {layout["id"]: layout for layout in spec["layouts"]}
    positions = {position["id"]: position for position in spec["positions"]}
    templates = {
        template["id"]: template for template in spec["negative_templates"]
    }
    palette_ids = [palette["id"] for palette in spec["palettes"]]
    cases: list[dict] = []

    for position_index, position in enumerate(spec["positions"]):
        canonical = position["fen"].split()[0]
        for piece_set_index, piece_set in enumerate(spec["piece_sets"]):
            palette = palette_ids[(position_index + piece_set_index) % 4]
            for view in matrix["views"]:
                source = canonical if view == "white_bottom" else rotate_placement(canonical)
                for layout_id in matrix["main"]["layout_ids"]:
                    layout = layouts[layout_id]
                    x0, y0, x1, y1 = layout["board_box"]
                    for quality_id in matrix["main"]["quality_ids"]:
                        case_id = (
                            f"main_pos_{position['id']}_{piece_set['id']}_{view}_"
                            f"{layout_id}_{quality_id}_rgb"
                        )
                        cases.append(
                            {
                                "id": case_id,
                                "path": f"images/{case_id}.png",
                                "suite": "main",
                                "split": position["split"],
                                "board_present": True,
                                "base_position": position["id"],
                                "canonical_placement": canonical,
                                "source_placement": source,
                                "view": view,
                                "piece_set": piece_set["id"],
                                "palette": palette,
                                "layout": layout_id,
                                "quality": quality_id,
                                "channel_mode": "RGB",
                                "image_mode": "RGB",
                                "image_size": list(layout["canvas_size"]),
                                "board_box": list(layout["board_box"]),
                                "corners": [
                                    [x0, y0],
                                    [x1 - 1, y0],
                                    [x1 - 1, y1 - 1],
                                    [x0, y1 - 1],
                                ],
                            }
                        )

    for template in spec["negative_templates"]:
        for layout_id in matrix["main"]["layout_ids"]:
            layout = layouts[layout_id]
            for quality_id in matrix["main"]["quality_ids"]:
                case_id = (
                    f"main_neg_{template['id']}_{layout_id}_{quality_id}_rgb"
                )
                cases.append(
                    {
                        "id": case_id,
                        "path": f"images/{case_id}.png",
                        "suite": "main",
                        "split": template["split"],
                        "board_present": False,
                        "negative_template": template["id"],
                        "layout": layout_id,
                        "quality": quality_id,
                        "channel_mode": "RGB",
                        "image_mode": "RGB",
                        "image_size": list(layout["canvas_size"]),
                    }
                )

    input_matrix = matrix["input_contract"]
    layout_id = input_matrix["layout_id"]
    layout = layouts[layout_id]
    x0, y0, x1, y1 = layout["board_box"]
    for position_id in input_matrix["position_ids"]:
        position = positions[position_id]
        canonical = position["fen"].split()[0]
        for view in matrix["views"]:
            source = canonical if view == "white_bottom" else rotate_placement(canonical)
            for channel_mode in input_matrix["channel_modes"]:
                case_id = f"input_pos_{position_id}_{view}_{channel_mode.lower()}"
                cases.append(
                    {
                        "id": case_id,
                        "path": f"images/{case_id}.png",
                        "suite": "input_contract",
                        "split": position["split"],
                        "board_present": True,
                        "base_position": position_id,
                        "canonical_placement": canonical,
                        "source_placement": source,
                        "view": view,
                        "piece_set": input_matrix["piece_set_id"],
                        "palette": input_matrix["palette_id"],
                        "layout": layout_id,
                        "quality": input_matrix["quality_id"],
                        "channel_mode": channel_mode,
                        "image_mode": channel_mode,
                        "image_size": list(layout["canvas_size"]),
                        "board_box": list(layout["board_box"]),
                        "corners": [
                            [x0, y0],
                            [x1 - 1, y0],
                            [x1 - 1, y1 - 1],
                            [x0, y1 - 1],
                        ],
                    }
                )

    for template_id in input_matrix["negative_template_ids"]:
        template = templates[template_id]
        for channel_mode in input_matrix["channel_modes"]:
            case_id = f"input_neg_{template_id}_{channel_mode.lower()}"
            cases.append(
                {
                    "id": case_id,
                    "path": f"images/{case_id}.png",
                    "suite": "input_contract",
                    "split": template["split"],
                    "board_present": False,
                    "negative_template": template_id,
                    "layout": layout_id,
                    "quality": input_matrix["quality_id"],
                    "channel_mode": channel_mode,
                    "image_mode": channel_mode,
                    "image_size": list(layout["canvas_size"]),
                }
            )

    return sorted(cases, key=lambda case: case["id"])


def validate_case_plan(plan: object, spec: dict) -> None:
    if not isinstance(plan, list) or not _same_json_value(plan, build_case_plan(spec)):
        raise BenchmarkValidationError("case plan differs from semantic source matrix")


def runtime_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "Pillow": version("Pillow"),
        "chess": version("chess"),
        "fentoboardimage": version("fenToBoardImage"),
        "zlib_build": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
    }


def _reject_symlink_components(path: Path, label: str) -> None:
    candidate = path if path.is_absolute() else Path.cwd() / path
    current = Path(candidate.parts[0])
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            raise BenchmarkValidationError(f"{label} path contains a symlink")


def _validated_manifest_path(dataset_dir: Path) -> Path:
    manifest_path = dataset_dir / "manifest.json"
    _reject_symlink_components(dataset_dir, "dataset")
    _reject_symlink_components(manifest_path, "manifest")
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise BenchmarkValidationError("dataset path must be a real directory")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BenchmarkValidationError("manifest path must be a regular file")
    return manifest_path


def _read_manifest_bytes(manifest_path: Path) -> bytes:
    try:
        return manifest_path.read_bytes()
    except OSError as error:
        raise BenchmarkValidationError("could not read manifest") from error


def _reject_digest_identity_in_dataset(dataset_dir: Path, digest_path: Path) -> None:
    try:
        digest_stat = os.lstat(digest_path)
    except OSError as error:
        raise BenchmarkValidationError("could not inspect manifest digest") from error
    digest_identity = digest_stat.st_dev, digest_stat.st_ino

    def reject_walk_error(error: OSError) -> None:
        raise BenchmarkValidationError("could not traverse dataset") from error

    for root, directories, files in os.walk(
        dataset_dir, followlinks=False, onerror=reject_walk_error
    ):
        for name in directories + files:
            try:
                entry_stat = os.lstat(Path(root) / name)
            except OSError as error:
                raise BenchmarkValidationError("could not traverse dataset") from error
            if (entry_stat.st_dev, entry_stat.st_ino) == digest_identity:
                raise BenchmarkValidationError(
                    "manifest digest filesystem identity must be distinct from the dataset"
                )


def _verify_manifest_bytes(
    dataset_dir: Path,
    source_spec: dict,
    encoded_manifest: bytes,
    *,
    verify_images: bool,
) -> dict:
    try:
        manifest = json.loads(
            encoded_manifest,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except BenchmarkValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise BenchmarkValidationError("invalid JSON in manifest") from error
    if not isinstance(manifest, dict):
        raise BenchmarkValidationError("manifest must contain a JSON object")
    try:
        canonical = canonical_json_bytes(manifest)
    except (TypeError, ValueError, UnicodeError) as error:
        raise BenchmarkValidationError("manifest is not canonical JSON") from error
    if encoded_manifest != canonical:
        raise BenchmarkValidationError("manifest encoding is not canonical JSON")

    expected_plan = build_case_plan(source_spec)
    top_level_fields = {
        "schema_version",
        "benchmark_version",
        "claim",
        "base_position_count",
        "limitations",
        "source_spec_sha256",
        "generation_runtime",
        "expected_counts",
        "cases",
    }
    if set(manifest) != top_level_fields:
        raise BenchmarkValidationError("manifest top-level fields differ")
    expected_metadata = {
        "schema_version": source_spec["schema_version"],
        "benchmark_version": source_spec["benchmark_version"],
        "claim": source_spec["claim"],
        "base_position_count": source_spec["base_position_count"],
        "limitations": source_spec["limitations"],
        "source_spec_sha256": source_spec_sha256(source_spec),
        "expected_counts": source_spec["expected_counts"],
    }
    actual_metadata = {field: manifest[field] for field in expected_metadata}
    if not _same_json_value(actual_metadata, expected_metadata):
        raise BenchmarkValidationError("manifest source-owned metadata differs")

    runtime = manifest["generation_runtime"]
    runtime_keys = {
        "python",
        "python_implementation",
        "Pillow",
        "chess",
        "fentoboardimage",
        "zlib_build",
        "zlib_runtime",
    }
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_keys
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise BenchmarkValidationError("generation runtime metadata invalid")

    actual_cases = manifest["cases"]
    if not isinstance(actual_cases, list) or len(actual_cases) != len(expected_plan):
        raise BenchmarkValidationError("manifest case inventory differs")
    hash_pattern = re.compile(r"[0-9a-f]{64}")
    for actual, expected in zip(actual_cases, expected_plan):
        if not isinstance(actual, dict) or set(actual) != set(expected) | {
            "file_sha256",
            "pixel_sha256",
        }:
            raise BenchmarkValidationError("manifest case fields differ")
        if any(
            not isinstance(actual[field], str)
            or hash_pattern.fullmatch(actual[field]) is None
            for field in ("file_sha256", "pixel_sha256")
        ):
            raise BenchmarkValidationError("manifest generated hash invalid")
        semantic_case = {
            key: value
            for key, value in actual.items()
            if key not in {"file_sha256", "pixel_sha256"}
        }
        if not _same_json_value(semantic_case, expected):
            raise BenchmarkValidationError("manifest case differs from source plan")

    if not verify_images:
        return manifest

    expected_paths = {case["path"] for case in expected_plan}
    actual_paths: set[str] = set()

    def reject_walk_error(error: OSError) -> None:
        raise BenchmarkValidationError("could not traverse dataset") from error

    for root, directories, files in os.walk(
        dataset_dir, followlinks=False, onerror=reject_walk_error
    ):
        root_path = Path(root)
        for name in directories:
            if (root_path / name).is_symlink():
                raise BenchmarkValidationError("dataset contains symlinked directory")
        for name in files:
            path = root_path / name
            if path.is_symlink():
                raise BenchmarkValidationError("dataset contains symlinked file")
            if path.suffix.lower() == ".png":
                if not path.is_file():
                    raise BenchmarkValidationError("dataset PNG must be a regular file")
                actual_paths.add(path.relative_to(dataset_dir).as_posix())
    if actual_paths != expected_paths:
        raise BenchmarkValidationError("dataset PNG inventory differs")

    resolved_dataset = dataset_dir.resolve(strict=True)
    for case in actual_cases:
        relative = PurePosixPath(case["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != case["path"]
            or relative.suffix != ".png"
        ):
            raise BenchmarkValidationError("unsafe manifest image path")
        image_path = dataset_dir.joinpath(*relative.parts)
        try:
            resolved_image = image_path.resolve(strict=True)
        except OSError as error:
            raise BenchmarkValidationError("manifest image path does not resolve") from error
        if (
            not resolved_image.is_relative_to(resolved_dataset)
            or image_path.is_symlink()
            or not image_path.is_file()
        ):
            raise BenchmarkValidationError("manifest image path escapes dataset")
        if file_sha256(image_path) != case["file_sha256"]:
            raise BenchmarkValidationError("manifest image file hash mismatch")
        try:
            with Image.open(image_path) as image:
                image.load()
                if image.format != "PNG":
                    raise BenchmarkValidationError("manifest image must encode PNG")
                if image.mode != case["image_mode"]:
                    raise BenchmarkValidationError("manifest image mode differs")
                if list(image.size) != case["image_size"]:
                    raise BenchmarkValidationError("manifest image size differs")
                if pixel_sha256(image) != case["pixel_sha256"]:
                    raise BenchmarkValidationError("manifest image pixel hash mismatch")
        except BenchmarkValidationError:
            raise
        except OSError as error:
            raise BenchmarkValidationError("could not decode manifest image") from error
    return manifest


def verify_unanchored_manifest(
    dataset_dir: Path,
    source_spec: dict,
    *,
    verify_images: bool = True,
) -> dict:
    manifest_path = _validated_manifest_path(dataset_dir)
    return _verify_manifest_bytes(
        dataset_dir,
        source_spec,
        _read_manifest_bytes(manifest_path),
        verify_images=verify_images,
    )


def verify_manifest(
    dataset_dir: Path,
    source_spec: dict,
    digest_path: Path,
    *,
    verify_images: bool = True,
) -> dict:
    manifest_path = _validated_manifest_path(dataset_dir)
    _reject_symlink_components(digest_path, "digest")
    if digest_path.is_symlink() or not digest_path.is_file():
        raise BenchmarkValidationError("digest path must be a regular file")
    try:
        resolved_dataset = dataset_dir.resolve(strict=True)
        resolved_digest = digest_path.resolve(strict=True)
    except OSError as error:
        raise BenchmarkValidationError(
            "dataset or manifest digest does not resolve"
        ) from error
    if resolved_digest == resolved_dataset or resolved_digest.is_relative_to(
        resolved_dataset
    ):
        raise BenchmarkValidationError(
            "manifest digest must be outside the dataset"
        )
    _reject_digest_identity_in_dataset(dataset_dir, digest_path)
    try:
        digest_bytes = digest_path.read_bytes()
    except OSError as error:
        raise BenchmarkValidationError("could not read manifest digest") from error
    if re.fullmatch(rb"[0-9a-f]{64}  manifest\.json\n", digest_bytes) is None:
        raise BenchmarkValidationError("manifest digest format differs")
    encoded_manifest = _read_manifest_bytes(manifest_path)
    if hashlib.sha256(encoded_manifest).hexdigest().encode("ascii") != digest_bytes[:64]:
        raise BenchmarkValidationError("manifest digest mismatch")
    return _verify_manifest_bytes(
        dataset_dir,
        source_spec,
        encoded_manifest,
        verify_images=verify_images,
    )
