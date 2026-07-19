import builtins
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import zlib
from collections import Counter
from functools import lru_cache
from importlib.metadata import version
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
from PIL import Image

import benchmarks.board_recognition as oracle
from benchmarks.board_recognition import (
    ALLOWED_PIECE_PATHS,
    BenchmarkValidationError,
    EXPECTED_SOURCE_SPEC_SHA256,
    canonical_json_bytes,
    compress_placement,
    expand_placement,
    file_sha256,
    load_json,
    load_source_spec,
    piece_set_tree_sha256,
    pixel_sha256,
    rotate_placement,
    score_predictions,
    source_spec_sha256,
    validate_piece_set,
    validate_source_spec,
)


ASYMMETRIC = "r3k2r/8/8/3p4/8/2N5/8/R3K2R"
EXPECTED = "8/8/8/3p4/4P3/8/8/4K2k"
WRONG_PIECE = "8/8/8/3q4/4P3/8/8/4K2k"
MISSING_PIECE = "8/8/8/8/4P3/8/8/4K2k"
EXTRA_PIECE = "N7/8/8/3p4/4P3/8/8/4K2k"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = ROOT / "benchmarks/board_recognition_v1.json"
EVALUATOR_PATH = ROOT / "scripts/validate/evaluate_board_recognition_benchmark.py"


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location(
        "_board_recognition_benchmark_evaluator", EVALUATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator():
    return _load_evaluator_module()


def cases():
    return [
        {
            "id": "positive",
            "board_present": True,
            "source_placement": EXPECTED,
            "suite": "main",
            "split": "development",
        },
        {
            "id": "negative",
            "board_present": False,
            "suite": "main",
            "split": "development",
            "negative_template": "dialog",
        },
    ]


def test_placement_round_trip_and_rotation_are_source_ordered():
    squares = expand_placement(ASYMMETRIC)
    assert len(squares) == 64
    assert compress_placement(squares) == ASYMMETRIC
    assert rotate_placement(rotate_placement(ASYMMETRIC)) == ASYMMETRIC
    assert rotate_placement(ASYMMETRIC) == "R2K3R/8/5N2/8/4p3/8/8/r2k3r"


@pytest.mark.parametrize(
    "placement",
    [
        None,
        "",
        "8/8",
        "9/8/8/8/8/8/8/8",
        "44/8/8/8/8/8/8/8",
        "11111111/8/8/8/8/8/8/8",
        "K/8/8/8/8/8/8/8",
        "x7/8/8/8/8/8/8/8",
    ],
)
def test_placement_parser_rejects_non_64_square_inputs(placement):
    with pytest.raises(BenchmarkValidationError):
        expand_placement(placement)


def test_one_wrong_symbol_fails_exact_board_without_hiding_occupancy():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "success", "source_placement": WRONG_PIECE},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["exact_placement"]["value"] == 0
    assert report["overall"]["square_accuracy"]["value"] == pytest.approx(63 / 64)
    assert report["overall"]["occupancy_precision"]["value"] == 1
    assert report["overall"]["occupancy_recall"]["value"] == 1
    assert report["overall"]["conditional_piece_accuracy"] == {
        "correct": 3,
        "total": 4,
        "value": 0.75,
    }


def test_occupied_to_empty_error_reduces_recall_not_precision():
    report = score_predictions(
        cases(),
        [
            {
                "id": "positive",
                "outcome": "success",
                "source_placement": MISSING_PIECE,
            },
            {
                "id": "negative",
                "outcome": "failure",
                "reason": "INVALID_GEOMETRY",
            },
        ],
    )
    assert report["overall"]["occupancy_precision"]["value"] == 1
    assert report["overall"]["occupancy_recall"]["value"] == 0.75
    assert report["overall"]["negative_any_rejection_rate"]["value"] == 1
    assert report["overall"]["negative_no_board_rate"]["value"] == 0


def test_empty_to_occupied_error_reduces_precision_not_recall():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "success", "source_placement": EXTRA_PIECE},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["square_accuracy"]["value"] == pytest.approx(63 / 64)
    assert report["overall"]["occupancy_precision"]["value"] == 0.8
    assert report["overall"]["occupancy_recall"]["value"] == 1
    assert report["overall"]["occupancy"] == {"tp": 4, "fp": 1, "fn": 0}


def test_failure_on_positive_penalizes_every_square_and_no_board_false_rejection():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "failure", "reason": "NO_BOARD"},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["square_accuracy"] == {
        "correct": 0,
        "total": 64,
        "value": 0,
    }
    assert report["overall"]["occupancy_recall"]["value"] == 0
    assert report["overall"]["conditional_piece_accuracy"]["value"] is None
    assert report["overall"]["positive_false_no_board_rate"]["value"] == 1


def test_rotated_prediction_fails_exact_placement():
    report = score_predictions(
        cases(),
        [
            {
                "id": "positive",
                "outcome": "success",
                "source_placement": rotate_placement(EXPECTED),
            },
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["exact_placement"]["value"] == 0


def test_success_prediction_on_negative_has_zero_rejection():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "success", "source_placement": EXPECTED},
            {"id": "negative", "outcome": "success", "source_placement": EXPECTED},
        ],
    )
    assert report["overall"]["negative_any_rejection_rate"]["value"] == 0
    assert report["overall"]["negative_no_board_rate"]["value"] == 0


def test_error_prediction_penalizes_positive_squares_and_counts_execution_error():
    report = score_predictions(
        cases(),
        [
            {
                "id": "positive",
                "outcome": "error",
                "error_type": "RuntimeError",
                "error_message": "checkpoint failed",
            },
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["square_accuracy"] == {
        "correct": 0,
        "total": 64,
        "value": 0,
    }
    assert report["overall"]["occupancy"]["fn"] == 4
    assert report["overall"]["execution_error_count"] == 1
    assert report["overall"]["counts"]["error"] == 1


@pytest.mark.parametrize(
    "predictions",
    [
        [{"id": "positive", "outcome": "success", "source_placement": EXPECTED}],
        [
            {"id": "positive", "outcome": "success", "source_placement": EXPECTED},
            {"id": "positive", "outcome": "failure", "reason": "NO_BOARD"},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
        [
            {"id": "positive", "outcome": "success", "source_placement": EXPECTED},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
            {"id": "extra", "outcome": "failure", "reason": "NO_BOARD"},
        ],
        [
            {"id": "positive", "outcome": "unknown"},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    ],
    ids=["missing", "duplicate", "extra", "invalid"],
)
def test_prediction_inventory_and_outcomes_are_validated(predictions):
    with pytest.raises(BenchmarkValidationError):
        score_predictions(cases(), predictions)


@pytest.mark.parametrize(
    "prediction",
    [
        {"id": "positive", "outcome": "success"},
        {"id": "positive", "outcome": "failure", "reason": "OTHER"},
        {"id": "positive", "outcome": "error", "error_type": "RuntimeError"},
        {
            "id": "positive",
            "outcome": "error",
            "error_type": 1,
            "error_message": "failed",
        },
    ],
    ids=["success", "failure", "missing-error-message", "non-string-error-type"],
)
def test_prediction_payload_is_validated_for_each_outcome(prediction):
    with pytest.raises(BenchmarkValidationError):
        score_predictions(
            cases(),
            [prediction, {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"}],
        )


def test_noncanonical_success_prediction_is_invalid():
    with pytest.raises(BenchmarkValidationError, match="canonical"):
        score_predictions(
            cases(),
            [
                {
                    "id": "positive",
                    "outcome": "success",
                    "source_placement": "44/8/8/8/8/8/8/8",
                },
                {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
            ],
        )


def test_prediction_report_groups_results_by_suite_and_split():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "success", "source_placement": EXPECTED},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["groups"]["suite"]["main"]["counts"]["all"] == 2
    assert report["groups"]["split"]["development"]["counts"]["all"] == 2


def test_prediction_report_has_exact_schema_and_counts_all_failure_reasons():
    reason_cases = [
        {"id": "no-board", "board_present": False},
        {"id": "invalid-geometry", "board_present": False},
        {"id": "decoding-failed", "board_present": False},
        {"id": "invalid-placement", "board_present": False},
    ]
    report = score_predictions(
        reason_cases,
        [
            {"id": "no-board", "outcome": "failure", "reason": "NO_BOARD"},
            {
                "id": "invalid-geometry",
                "outcome": "failure",
                "reason": "INVALID_GEOMETRY",
            },
            {
                "id": "decoding-failed",
                "outcome": "failure",
                "reason": "DECODING_FAILED",
            },
            {
                "id": "invalid-placement",
                "outcome": "failure",
                "reason": "INVALID_PLACEMENT",
            },
        ],
    )

    assert set(report) == {"overall", "groups"}
    overall = report["overall"]
    assert set(overall) == {
        "counts",
        "positive_success_rate",
        "exact_placement",
        "square_accuracy",
        "occupancy",
        "occupancy_precision",
        "occupancy_recall",
        "conditional_piece_accuracy",
        "negative_any_rejection_rate",
        "negative_no_board_rate",
        "positive_false_no_board_rate",
        "failure_reasons",
        "execution_error_count",
    }
    assert overall["counts"] == {
        "all": 4,
        "positive": 0,
        "negative": 4,
        "success": 0,
        "failure": 4,
        "error": 0,
    }
    assert overall["occupancy"] == {"tp": 0, "fp": 0, "fn": 0}
    for metric in (
        "positive_success_rate",
        "exact_placement",
        "square_accuracy",
        "occupancy_precision",
        "occupancy_recall",
        "conditional_piece_accuracy",
        "negative_any_rejection_rate",
        "negative_no_board_rate",
        "positive_false_no_board_rate",
    ):
        assert set(overall[metric]) == {"correct", "total", "value"}
    assert overall["failure_reasons"] == {
        "DECODING_FAILED": 1,
        "INVALID_GEOMETRY": 1,
        "INVALID_PLACEMENT": 1,
        "NO_BOARD": 1,
    }


def test_prediction_rejection_edge_semantics():
    report = score_predictions(
        [
            {
                "id": "positive",
                "board_present": True,
                "source_placement": EXPECTED,
            },
            {"id": "negative", "board_present": False},
        ],
        [
            {
                "id": "positive",
                "outcome": "failure",
                "reason": "INVALID_GEOMETRY",
            },
            {
                "id": "negative",
                "outcome": "error",
                "error_type": "RuntimeError",
                "error_message": "checkpoint failed",
            },
        ],
    )

    assert report["overall"]["negative_any_rejection_rate"] == {
        "correct": 0,
        "total": 1,
        "value": 0,
    }
    assert report["overall"]["negative_no_board_rate"] == {
        "correct": 0,
        "total": 1,
        "value": 0,
    }
    assert report["overall"]["positive_false_no_board_rate"] == {
        "correct": 0,
        "total": 1,
        "value": 0,
    }
    assert report["overall"]["execution_error_count"] == 1


def test_prediction_groups_cover_all_fields_filter_sort_and_ignore_input_order():
    grouped_cases = [
        {
            "id": "z-positive",
            "board_present": True,
            "source_placement": EXPECTED,
            "suite": "z-suite",
            "split": "evaluation",
            "base_position": "pos-z",
            "piece_set": "set-z",
            "palette": "z-palette",
            "view": "white_bottom",
            "layout": "z-layout",
            "quality": "z-quality",
            "channel_mode": "RGB",
        },
        {
            "id": "z-negative",
            "board_present": False,
            "suite": "z-suite",
            "split": "evaluation",
            "base_position": "pos-z",
            "piece_set": "set-z",
            "palette": "z-palette",
            "view": "white_bottom",
            "layout": "z-layout",
            "quality": "z-quality",
            "channel_mode": "RGB",
            "negative_template": "z-template",
        },
        {
            "id": "a-negative",
            "board_present": False,
            "suite": "a-suite",
            "split": "development",
            "base_position": "pos-a",
            "piece_set": "set-a",
            "palette": "a-palette",
            "view": "black_bottom",
            "layout": "a-layout",
            "quality": "a-quality",
            "channel_mode": "L",
            "negative_template": "a-template",
        },
    ]
    grouped_predictions = [
        {
            "id": "z-positive",
            "outcome": "success",
            "source_placement": EXPECTED,
        },
        {"id": "z-negative", "outcome": "failure", "reason": "NO_BOARD"},
        {
            "id": "a-negative",
            "outcome": "error",
            "error_type": "RuntimeError",
            "error_message": "checkpoint failed",
        },
    ]
    report = score_predictions(grouped_cases, grouped_predictions)
    expected_group_values = {
        "suite": ["a-suite", "z-suite"],
        "split": ["development", "evaluation"],
        "base_position": ["pos-a", "pos-z"],
        "piece_set": ["set-a", "set-z"],
        "palette": ["a-palette", "z-palette"],
        "view": ["black_bottom", "white_bottom"],
        "layout": ["a-layout", "z-layout"],
        "quality": ["a-quality", "z-quality"],
        "channel_mode": ["L", "RGB"],
        "negative_template": ["a-template", "z-template"],
    }

    assert set(report["groups"]) == {
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
    }
    for field, values in expected_group_values.items():
        assert list(report["groups"][field]) == values
    assert {
        value: group["counts"]["all"]
        for value, group in report["groups"]["negative_template"].items()
    } == {"a-template": 1, "z-template": 1}
    assert score_predictions(
        list(reversed(grouped_cases)), list(reversed(grouped_predictions))
    ) == report

    ungrouped_report = score_predictions(
        [{"id": "plain-negative", "board_present": False}],
        [{"id": "plain-negative", "outcome": "failure", "reason": "NO_BOARD"}],
    )
    assert ungrouped_report["groups"] == {}


def test_canonical_json_is_stable_and_has_one_newline():
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_file_and_decoded_pixel_hashes_have_independent_fixtures(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    assert file_sha256(payload) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

    image = Image.new("RGB", (2, 1), (1, 2, 3))
    expected = hashlib.sha256(b"RGB\0" + b"2x1\0" + bytes((1, 2, 3, 1, 2, 3)))
    assert pixel_sha256(image) == expected.hexdigest()


def _source_spec():
    return load_source_spec(SOURCE_SPEC)


def _set_path(value, path, replacement):
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


def _validate_mutation(path, replacement, match):
    spec = copy.deepcopy(_source_spec())
    _set_path(spec, path, replacement)
    with pytest.raises(BenchmarkValidationError, match=match):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_version_1_source_spec_is_semantically_valid():
    spec = _source_spec()
    assert source_spec_sha256(spec) == EXPECTED_SOURCE_SPEC_SHA256
    validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_provisioned_version_1_assets_match_reviewed_hashes():
    if not (ROOT / "assets/piece_png/lichess_chessnut").is_dir():
        pytest.skip("requires ignored, locally provisioned benchmark assets")
    validate_source_spec(_source_spec(), ROOT, verify_local_assets=True)


@pytest.mark.parametrize(
    ("path", "replacement", "match"),
    [
        (("schema_version",), 2, "schema_version"),
        (("benchmark_version",), "online-render-v2", "benchmark_version"),
        (("claim",), "broader_claim", "claim"),
        (("dependencies", "Pillow"), "0.0.0", "dependencies"),
        (("base_position_count",), 17, "base_position_count"),
        (("limitations",), ["changed"], "limitations"),
        (("case_matrix", "main", "channel_mode"), "RGBA", "case_matrix"),
        (("case_matrix", "main", "palette_index_formula"), "1 / 0", "case_matrix"),
        (("expected_counts", "all"), 569, "expected_counts"),
        (("renderer", "png_compress_level"), 8, "renderer"),
        (("piece_sets", 0, "license"), "MIT", "license"),
        (("piece_sets", 0, "tree_sha256"), "0" * 64, "tree_sha256"),
        (("piece_sets", 0, "path"), "assets/piece_png/other", "piece-set path"),
    ],
)
def test_frozen_source_boundaries_fail_before_the_digest(path, replacement, match):
    _validate_mutation(path, replacement, match)


def test_source_spec_rejects_boolean_where_an_integer_is_required():
    _validate_mutation(("expected_counts", "all"), True, "integer")


def test_source_spec_rejects_duplicate_ids():
    spec = copy.deepcopy(_source_spec())
    spec["palettes"][1]["id"] = spec["palettes"][0]["id"]
    with pytest.raises(BenchmarkValidationError, match="duplicate palette id"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_malformed_fen():
    _validate_mutation(("positions", 0, "fen"), "not a fen", "FEN")


def test_source_spec_reaches_python_chess_validity_gate():
    _validate_mutation(
        ("positions", 0, "fen"),
        "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
        "invalid FEN",
    )


def test_source_spec_rejects_noncanonical_full_fen():
    spec = _source_spec()
    fen = spec["positions"][0]["fen"].replace(" 0 1", " 00 1")
    _validate_mutation(("positions", 0, "fen"), fen, "noncanonical FEN")


def test_source_spec_rejects_noncanonical_placement():
    _validate_mutation(
        ("positions", 0, "fen"),
        "11111111/8/8/8/8/8/8/8 w - - 0 1",
        "canonical FEN compression",
    )


def test_source_spec_rejects_source_record_mismatch():
    _validate_mutation(
        ("positions", 1, "source", "record"),
        "8/8/8/8/8/8/8/8 w - -",
        "source.record",
    )


def test_source_spec_rejects_missing_king():
    _validate_mutation(
        ("positions", 0, "fen"),
        "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
        "king",
    )


def test_source_spec_rejects_direct_duplicate_placement():
    spec = copy.deepcopy(_source_spec())
    spec["positions"][2]["fen"] = spec["positions"][1]["fen"]
    spec["positions"][2]["source"]["record"] = spec["positions"][1]["source"][
        "record"
    ]
    with pytest.raises(BenchmarkValidationError, match="duplicate placement"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_rotated_duplicate_placement():
    spec = copy.deepcopy(_source_spec())
    placement = rotate_placement(spec["positions"][0]["fen"].split()[0])
    spec["positions"][1]["fen"] = f"{placement} w - - 0 1"
    spec["positions"][1]["source"]["record"] = f"{placement} w - -"
    with pytest.raises(BenchmarkValidationError, match="rotated duplicate placement"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_low_contrast_palette():
    spec = copy.deepcopy(_source_spec())
    spec["palettes"][0]["light"] = "#111111"
    spec["palettes"][0]["dark"] = "#101010"
    with pytest.raises(BenchmarkValidationError, match="contrast"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_reaches_luminance_contrast_gate():
    spec = copy.deepcopy(_source_spec())
    spec["palettes"][0]["light"] = "#FF0000"
    spec["palettes"][0]["dark"] = "#0000FF"
    with pytest.raises(BenchmarkValidationError, match="luminance contrast"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_hex_color_whitespace_as_validation_error():
    spec = copy.deepcopy(_source_spec())
    spec["palettes"][0]["light"] = "#00 00 "
    with pytest.raises(BenchmarkValidationError, match="color"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_missing_piece_symbol_bucket():
    spec = copy.deepcopy(_source_spec())
    for position in spec["positions"]:
        fields = position["fen"].split()
        fields[0] = fields[0].replace("Q", "R").replace("q", "r")
        position["fen"] = " ".join(fields)
        if "record" in position["source"]:
            position["source"]["record"] = " ".join(fields[:4])
    with pytest.raises(BenchmarkValidationError, match="all twelve piece symbols"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_missing_low_piece_count_bucket():
    spec = copy.deepcopy(_source_spec())
    for position in spec["positions"]:
        board = chess.Board(position["fen"])
        while len(board.piece_map()) < 9:
            for square in chess.SQUARES:
                if board.piece_at(square) is not None or chess.square_rank(square) in (0, 7):
                    continue
                for color in (chess.WHITE, chess.BLACK):
                    candidate = board.copy(stack=False)
                    candidate.set_piece_at(square, chess.Piece(chess.PAWN, color))
                    if candidate.is_valid():
                        board = candidate
                        break
                else:
                    continue
                break
            else:
                raise AssertionError("could not construct a valid nine-piece position")
        position["fen"] = board.fen(en_passant="fen")
        if "record" in position["source"]:
            position["source"]["record"] = " ".join(position["fen"].split()[:4])
    with pytest.raises(BenchmarkValidationError, match="piece-count bucket 2..8"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_wrong_development_position_indexes():
    _validate_mutation(("positions", 1, "split"), "development", "development indexes")


def test_source_spec_rejects_unexpected_recipe_key():
    spec = copy.deepcopy(_source_spec())
    spec["ui_recipes"]["extra"] = {}
    with pytest.raises(BenchmarkValidationError, match="recipe keys"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_unexpected_template_key():
    spec = copy.deepcopy(_source_spec())
    spec["ui_recipes"]["negative_templates"]["lobby_cards"]["caption"] = "x"
    with pytest.raises(BenchmarkValidationError, match="template keys"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_negative_recipe_inventory_drift():
    spec = copy.deepcopy(_source_spec())
    del spec["ui_recipes"]["negative_templates"]["lobby_cards"]
    with pytest.raises(BenchmarkValidationError, match="negative recipe keys"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_invalid_recipe_color():
    _validate_mutation(
        (
            "ui_recipes",
            "negative_templates",
            "lobby_cards",
            "shapes",
            0,
            "fill",
        ),
        "brown",
        "color",
    )


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ([True, 0, 1000, 90], "coordinate integer"),
        ([-1, 0, 1000, 90], "coordinate range"),
        ([1000, 0, 0, 90], "coordinate order"),
    ],
)
def test_source_spec_rejects_invalid_recipe_coordinates(replacement, match):
    _validate_mutation(
        (
            "ui_recipes",
            "negative_templates",
            "lobby_cards",
            "shapes",
            0,
            "box",
        ),
        replacement,
        match,
    )


def test_source_spec_rejects_invalid_recipe_kind():
    _validate_mutation(
        (
            "ui_recipes",
            "negative_templates",
            "lobby_cards",
            "shapes",
            0,
            "kind",
        ),
        "line",
        "shape kind",
    )


def test_source_spec_rejects_non_string_recipe_kind_as_validation_error():
    _validate_mutation(
        (
            "ui_recipes",
            "negative_templates",
            "lobby_cards",
            "shapes",
            0,
            "kind",
        ),
        ["rectangle"],
        "shape kind",
    )


def test_source_spec_rejects_wrong_shape_geometry_keys():
    spec = copy.deepcopy(_source_spec())
    shape = spec["ui_recipes"]["negative_templates"]["lobby_cards"]["shapes"][0]
    shape["points"] = [[0, 0], [1, 1], [2, 2]]
    with pytest.raises(BenchmarkValidationError, match="geometry keys"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


@pytest.mark.parametrize("mutation", ["valid_color", "shape_order"])
def test_well_formed_recipe_changes_reach_the_source_digest_gate(mutation):
    spec = copy.deepcopy(_source_spec())
    template = spec["ui_recipes"]["negative_templates"]["lobby_cards"]
    if mutation == "valid_color":
        template["shapes"][0]["fill"] = "#181614"
    else:
        template["shapes"][:2] = reversed(template["shapes"][:2])
    with pytest.raises(BenchmarkValidationError, match="source specification digest"):
        validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_source_spec_rejects_coordinate_space_drift_before_digest():
    _validate_mutation(
        ("ui_recipes", "coordinate_space"),
        [999, 1000],
        "coordinate space must be",
    )


def test_load_json_rejects_nested_duplicate_key(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"outer":{"same":1,"same":2}}', encoding="utf-8")
    with pytest.raises(BenchmarkValidationError, match="duplicate JSON key"):
        load_json(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity"])
def test_load_json_rejects_nonstandard_constants(tmp_path, constant):
    path = tmp_path / "constant.json"
    path.write_text(f'{{"bad":{constant}}}', encoding="utf-8")
    with pytest.raises(BenchmarkValidationError, match="invalid JSON constant"):
        load_json(path)


def test_load_json_rejects_non_object_root(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(BenchmarkValidationError, match="JSON object"):
        load_json(path)


def _expected_tree_hash(directory):
    digest = hashlib.sha256()
    for relative in sorted(ALLOWED_PIECE_PATHS):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((directory / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _make_piece_tree(directory):
    for index, relative in enumerate(ALLOWED_PIECE_PATHS, 1):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGBA",
            (100, 100),
            (index, index * 7 % 256, index * 17 % 256, 255),
        ).save(path)
    return _expected_tree_hash(directory)


def test_piece_set_accepts_exact_valid_tree(tmp_path):
    directory = tmp_path / "pieces"
    expected = _make_piece_tree(directory)
    assert piece_set_tree_sha256(directory) == expected
    validate_piece_set(directory, expected)


def test_piece_set_rejects_wrong_expected_hash(tmp_path):
    directory = tmp_path / "pieces"
    _make_piece_tree(directory)
    with pytest.raises(BenchmarkValidationError, match="tree hash"):
        validate_piece_set(directory, "0" * 64)


@pytest.mark.parametrize("mutation", ["missing", "extra", "nested", "uppercase"])
def test_piece_set_rejects_inventory_drift(tmp_path, mutation):
    directory = tmp_path / "pieces"
    _make_piece_tree(directory)
    pawn = directory / "black/Pawn.png"
    if mutation == "missing":
        pawn.unlink()
    elif mutation == "extra":
        Image.new("RGBA", (100, 100), (1, 2, 3, 255)).save(
            directory / "black/extra.png"
        )
    elif mutation == "nested":
        nested = directory / "black/nested/extra.png"
        nested.parent.mkdir()
        Image.new("RGBA", (100, 100), (1, 2, 3, 255)).save(nested)
    else:
        pawn.rename(directory / "black/PAWN.PNG")
    with pytest.raises(BenchmarkValidationError, match="PNG inventory"):
        piece_set_tree_sha256(directory)


def test_piece_set_rejects_symlinked_color_directory(tmp_path):
    directory = tmp_path / "pieces"
    _make_piece_tree(directory)
    color = directory / "black"
    target = tmp_path / "real-black"
    color.rename(target)
    color.symlink_to(target, target_is_directory=True)
    with pytest.raises(BenchmarkValidationError, match="real directory"):
        piece_set_tree_sha256(directory)


def test_piece_set_rejects_symlinked_png(tmp_path):
    directory = tmp_path / "pieces"
    _make_piece_tree(directory)
    png = directory / "black/Pawn.png"
    target = tmp_path / "real-pawn.png"
    png.rename(target)
    png.symlink_to(target)
    with pytest.raises(BenchmarkValidationError, match="non-regular PNG"):
        piece_set_tree_sha256(directory)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("undecodable", "decode"),
        ("size", "100x100"),
        ("mode", "RGBA"),
        ("alpha", "alpha"),
        ("duplicate", "duplicate file bytes"),
    ],
)
def test_piece_set_rejects_invalid_decoded_assets(tmp_path, mutation, match):
    directory = tmp_path / "pieces"
    _make_piece_tree(directory)
    pawn = directory / "black/Pawn.png"
    if mutation == "undecodable":
        pawn.write_bytes(b"not a PNG")
    elif mutation == "size":
        Image.new("RGBA", (99, 100), (1, 2, 3, 255)).save(pawn)
    elif mutation == "mode":
        Image.new("RGB", (100, 100), (1, 2, 3)).save(pawn)
    elif mutation == "alpha":
        Image.new("RGBA", (100, 100), (1, 2, 3, 0)).save(pawn)
    else:
        (directory / "black/Rook.png").write_bytes(pawn.read_bytes())
    expected = _expected_tree_hash(directory)
    with pytest.raises(BenchmarkValidationError, match=match):
        validate_piece_set(directory, expected)


def test_local_asset_preflight_rejects_wrong_python_runtime(monkeypatch):
    monkeypatch.setattr(oracle.platform, "python_version", lambda: "0.0.0")
    with pytest.raises(BenchmarkValidationError, match="python runtime version"):
        validate_source_spec(_source_spec(), ROOT, verify_local_assets=True)


def test_local_asset_preflight_rejects_wrong_installed_package(monkeypatch):
    monkeypatch.setattr(oracle, "version", lambda _distribution: "0.0.0")
    with pytest.raises(BenchmarkValidationError, match="installed package version"):
        validate_source_spec(_source_spec(), ROOT, verify_local_assets=True)


POSITIVE_CASE_FIELDS = {
    "id",
    "path",
    "suite",
    "split",
    "board_present",
    "base_position",
    "canonical_placement",
    "source_placement",
    "view",
    "piece_set",
    "palette",
    "layout",
    "quality",
    "channel_mode",
    "image_mode",
    "image_size",
    "board_box",
    "corners",
}
NEGATIVE_CASE_FIELDS = {
    "id",
    "path",
    "suite",
    "split",
    "board_present",
    "negative_template",
    "layout",
    "quality",
    "channel_mode",
    "image_mode",
    "image_size",
}


def _case_plan():
    return oracle.build_case_plan(_source_spec())


def test_case_plan_has_exact_counts_semantic_ids_paths_and_fields():
    plan = _case_plan()

    assert len(plan) == 568
    assert [case["id"] for case in plan] == sorted(case["id"] for case in plan)
    assert len({case["id"] for case in plan}) == 568
    assert len({case["path"] for case in plan}) == 568
    assert sum(case["board_present"] for case in plan) == 528
    assert sum(not case["board_present"] for case in plan) == 40
    assert Counter((case["suite"], case["board_present"]) for case in plan) == {
        ("main", True): 512,
        ("main", False): 32,
        ("input_contract", True): 16,
        ("input_contract", False): 8,
    }

    for case in plan:
        assert set(case) == (
            POSITIVE_CASE_FIELDS if case["board_present"] else NEGATIVE_CASE_FIELDS
        )
        assert case["path"] == f"images/{case['id']}.png"
        if case["suite"] == "main" and case["board_present"]:
            expected_id = (
                f"main_pos_{case['base_position']}_{case['piece_set']}_"
                f"{case['view']}_{case['layout']}_{case['quality']}_rgb"
            )
        elif case["suite"] == "main":
            expected_id = (
                f"main_neg_{case['negative_template']}_{case['layout']}_"
                f"{case['quality']}_rgb"
            )
        elif case["board_present"]:
            expected_id = (
                f"input_pos_{case['base_position']}_{case['view']}_"
                f"{case['channel_mode'].lower()}"
            )
        else:
            expected_id = (
                f"input_neg_{case['negative_template']}_"
                f"{case['channel_mode'].lower()}"
            )
        assert case["id"] == expected_id


def test_case_plan_main_matrix_palette_formula_and_balance_are_exact():
    spec = _source_spec()
    plan = _case_plan()
    main_positive = [
        case for case in plan if case["suite"] == "main" and case["board_present"]
    ]
    position_ids = [position["id"] for position in spec["positions"]]
    piece_set_ids = [piece_set["id"] for piece_set in spec["piece_sets"]]
    palette_ids = [palette["id"] for palette in spec["palettes"]]
    views = spec["case_matrix"]["views"]
    layout_ids = spec["case_matrix"]["main"]["layout_ids"]
    quality_ids = spec["case_matrix"]["main"]["quality_ids"]
    expected_matrix = set(
        product(position_ids, piece_set_ids, views, layout_ids, quality_ids)
    )
    actual_matrix = [
        (
            case["base_position"],
            case["piece_set"],
            case["view"],
            case["layout"],
            case["quality"],
        )
        for case in main_positive
    ]

    assert len(actual_matrix) == len(set(actual_matrix)) == 512
    assert set(actual_matrix) == expected_matrix
    for case in main_positive:
        position_index = position_ids.index(case["base_position"])
        piece_set_index = piece_set_ids.index(case["piece_set"])
        assert case["palette"] == palette_ids[
            (position_index + piece_set_index) % 4
        ]

    dimensions = {
        "base_position": position_ids,
        "piece_set": piece_set_ids,
        "view": views,
        "layout": layout_ids,
        "quality": quality_ids,
    }
    for field, values in dimensions.items():
        for value in values:
            palette_counts = Counter(
                case["palette"] for case in main_positive if case[field] == value
            )
            assert set(palette_counts) == set(palette_ids)
            assert len(set(palette_counts.values())) == 1


def test_case_plan_positive_labels_are_bound_to_source_and_view():
    spec = _source_spec()
    positions = {position["id"]: position for position in spec["positions"]}

    for case in (case for case in _case_plan() if case["board_present"]):
        position = positions[case["base_position"]]
        canonical_placement = position["fen"].split()[0]
        expected = canonical_placement
        if case["view"] == "black_bottom":
            expected = compress_placement(
                tuple(reversed(expand_placement(canonical_placement)))
            )
        assert case["split"] == position["split"]
        assert case["canonical_placement"] == canonical_placement
        assert case["source_placement"] == expected


def _cross(first, second, third):
    return (second[0] - first[0]) * (third[1] - second[1]) - (
        second[1] - first[1]
    ) * (third[0] - second[0])


def _twice_polygon_area(points):
    return sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )


def test_case_plan_geometry_uses_ordered_in_bounds_pixel_centers():
    expected_geometry = {
        "board_crop": (
            [512, 512],
            [0, 0, 512, 512],
            [[0, 0], [511, 0], [511, 511], [0, 511]],
        ),
        "desktop_ui": (
            [960, 540],
            [24, 30, 504, 510],
            [[24, 30], [503, 30], [503, 509], [24, 509]],
        ),
    }

    for case in (case for case in _case_plan() if case["board_present"]):
        image_size, board_box, corners = expected_geometry[case["layout"]]
        assert case["image_size"] == image_size
        assert case["board_box"] == board_box
        assert case["corners"] == corners
        assert corners[0][0] < corners[1][0]
        assert corners[0][1] == corners[1][1]
        assert corners[1][1] < corners[2][1]
        assert corners[1][0] == corners[2][0]
        assert corners[2][0] > corners[3][0]
        assert corners[2][1] == corners[3][1]
        assert corners[3][1] > corners[0][1]
        assert corners[3][0] == corners[0][0]
        assert all(
            _cross(corners[index - 2], corners[index - 1], corners[index]) > 0
            for index in range(4)
        )
        assert _twice_polygon_area(corners) > 0
        assert all(
            0 <= x < image_size[0] and 0 <= y < image_size[1]
            for x, y in corners
        )


def test_case_plan_suites_inherit_frozen_modes_groups_and_splits():
    spec = _source_spec()
    plan = _case_plan()
    position_splits = {
        position["id"]: position["split"] for position in spec["positions"]
    }
    template_splits = {
        template["id"]: template["split"]
        for template in spec["negative_templates"]
    }
    main = [case for case in plan if case["suite"] == "main"]
    input_cases = [case for case in plan if case["suite"] == "input_contract"]
    input_matrix = spec["case_matrix"]["input_contract"]

    assert {(case["channel_mode"], case["image_mode"]) for case in main} == {
        ("RGB", "RGB")
    }
    assert {
        (
            case["negative_template"],
            case["layout"],
            case["quality"],
        )
        for case in main
        if not case["board_present"]
    } == set(
        product(
            template_splits,
            spec["case_matrix"]["main"]["layout_ids"],
            spec["case_matrix"]["main"]["quality_ids"],
        )
    )
    assert {
        (case["base_position"], case["view"], case["channel_mode"])
        for case in input_cases
        if case["board_present"]
    } == set(
        product(
            input_matrix["position_ids"],
            spec["case_matrix"]["views"],
            input_matrix["channel_modes"],
        )
    )
    assert {
        (case["negative_template"], case["channel_mode"])
        for case in input_cases
        if not case["board_present"]
    } == set(
        product(
            input_matrix["negative_template_ids"],
            input_matrix["channel_modes"],
        )
    )
    assert {case["channel_mode"] for case in input_cases} == {"RGBA", "L"}
    assert {case["image_mode"] for case in input_cases} == {"RGBA", "L"}
    assert {case["layout"] for case in input_cases} == {"board_crop"}
    assert {case["quality"] for case in input_cases} == {"clean"}
    assert {
        case["palette"] for case in input_cases if case["board_present"]
    } == {"brown"}
    assert {
        case["piece_set"] for case in input_cases if case["board_present"]
    } == {"lichess_chessnut"}

    for case in plan:
        expected_split = (
            position_splits[case["base_position"]]
            if case["board_present"]
            else template_splits[case["negative_template"]]
        )
        assert case["split"] == expected_split


def test_case_plan_validator_rejects_wrong_count_and_palette_formula():
    plan = _case_plan()
    for path, replacement in (
        (("expected_counts", "all"), 569),
        (("case_matrix", "main", "palette_index_formula"), "0"),
    ):
        spec = copy.deepcopy(_source_spec())
        _set_path(spec, path, replacement)
        with pytest.raises(BenchmarkValidationError):
            oracle.validate_case_plan(plan, spec)


def test_case_plan_validator_rejects_invalid_view_duplicate_id_and_unsafe_path():
    spec = _source_spec()
    for mutation in ("view", "duplicate", "path"):
        plan = _case_plan()
        if mutation == "view":
            next(case for case in plan if case["board_present"])["view"] = "sideways"
        elif mutation == "duplicate":
            plan[1]["id"] = plan[0]["id"]
        else:
            plan[0]["path"] = "../escape.png"
        with pytest.raises(BenchmarkValidationError):
            oracle.validate_case_plan(plan, spec)


@pytest.mark.parametrize("mutation", ["board-present", "geometry", "non-json"])
def test_case_plan_validator_preserves_json_types(mutation):
    plan = _case_plan()
    positive = next(case for case in plan if case["board_present"])
    if mutation == "board-present":
        positive["board_present"] = 1
    elif mutation == "geometry":
        positive["corners"][0][0] = 0.0
    else:
        positive["path"] = object()

    with pytest.raises(BenchmarkValidationError):
        oracle.validate_case_plan(plan, _source_spec())


@pytest.mark.parametrize("field", ["corners", "image_size"])
def test_case_plan_validator_requires_json_native_lists(field):
    plan = _case_plan()
    positive = next(case for case in plan if case["board_present"])
    if field == "corners":
        positive[field][0] = tuple(positive[field][0])
    else:
        positive[field] = tuple(positive[field])

    with pytest.raises(BenchmarkValidationError, match="canonical JSON"):
        oracle.validate_case_plan(plan, _source_spec())


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({1: "value"}, {"1": "value"}),
        ({"1": "value"}, {1: "value"}),
        (["value"], ("value",)),
    ],
)
def test_json_native_comparison_rejects_non_json_values_on_either_side(
    actual, expected
):
    with pytest.raises(BenchmarkValidationError, match="canonical JSON"):
        oracle._same_json_value(actual, expected)


def test_case_plan_runtime_fingerprint_has_only_reproducible_values():
    assert oracle.runtime_fingerprint() == {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "Pillow": version("Pillow"),
        "chess": version("chess"),
        "fentoboardimage": version("fenToBoardImage"),
        "zlib_build": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
    }


MANIFEST_FIELDS = {
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


def _manifest_fixture():
    spec = _source_spec()
    cases_with_hashes = [
        {
            **case,
            "file_sha256": "1" * 64,
            "pixel_sha256": "2" * 64,
        }
        for case in _case_plan()
    ]
    return {
        "schema_version": spec["schema_version"],
        "benchmark_version": spec["benchmark_version"],
        "claim": spec["claim"],
        "base_position_count": spec["base_position_count"],
        "limitations": copy.deepcopy(spec["limitations"]),
        "source_spec_sha256": source_spec_sha256(spec),
        "generation_runtime": oracle.runtime_fingerprint(),
        "expected_counts": copy.deepcopy(spec["expected_counts"]),
        "cases": cases_with_hashes,
    }


def _write_manifest(tmp_path, manifest=None):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    manifest = _manifest_fixture() if manifest is None else manifest
    encoded = canonical_json_bytes(manifest)
    (dataset_dir / "manifest.json").write_bytes(encoded)
    digest_path = tmp_path / "manifest.sha256"
    digest_path.write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  manifest.json\n",
        encoding="ascii",
    )
    return dataset_dir, digest_path, manifest


def test_manifest_without_images_is_accepted_against_the_source_anchor(tmp_path):
    dataset_dir, digest_path, manifest = _write_manifest(tmp_path)

    assert set(manifest) == MANIFEST_FIELDS
    assert len(manifest["cases"]) == 568
    assert oracle.verify_manifest(
        dataset_dir, _source_spec(), digest_path, verify_images=False
    ) == manifest


def test_manifest_requires_an_existing_digest(tmp_path):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    digest_path.unlink()

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize(
    "digest",
    [
        f"{'A' * 64}  manifest.json\n",
        f"{'0' * 64} manifest.json\n",
        f"{'0' * 64}  other.json\n",
        f"{'0' * 64}  manifest.json",
        f"{'0' * 64}  manifest.json\n\n",
    ],
    ids=["uppercase", "one-space", "filename", "newline", "extra-line"],
)
def test_manifest_digest_format_is_exact(tmp_path, digest):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    digest_path.write_text(digest, encoding="ascii")

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


def test_manifest_digest_must_match_the_encoded_manifest(tmp_path):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    digest_path.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize("encoding", ["pretty", "extra-newline"])
def test_refreshed_digest_does_not_anchor_noncanonical_manifest_encoding(
    tmp_path, encoding
):
    dataset_dir, digest_path, manifest = _write_manifest(tmp_path)
    if encoding == "pretty":
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    else:
        encoded = canonical_json_bytes(manifest) + b"\n"
    (dataset_dir / "manifest.json").write_bytes(encoded)
    digest_path.write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  manifest.json\n",
        encoding="ascii",
    )

    with pytest.raises(BenchmarkValidationError, match="canonical"):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


def test_manifest_digest_mismatch_precedes_manifest_parsing(tmp_path):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    (dataset_dir / "manifest.json").write_bytes(b"not JSON\n")
    digest_path.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")

    with pytest.raises(BenchmarkValidationError, match="manifest digest mismatch"):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize("location", ["root", "images", "lexical-alias"])
def test_manifest_digest_must_resolve_outside_dataset(
    tmp_path, monkeypatch, location
):
    dataset_dir, outside_digest, _manifest = _write_manifest(tmp_path)
    if location == "root":
        digest_path = dataset_dir / "manifest.sha256"
    elif location == "images":
        (dataset_dir / "images").mkdir()
        digest_path = dataset_dir / "images/manifest.sha256"
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        digest_path = outside / ".." / dataset_dir.name / "manifest.sha256"
    digest_path.write_bytes(outside_digest.read_bytes())
    manifest_path = dataset_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    digest_before = digest_path.read_bytes()
    inventory_before = sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    real_read_bytes = Path.read_bytes

    def forbid_digest_read(path):
        if path == digest_path:
            raise AssertionError("contained digest was read before rejection")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_digest_read)

    with pytest.raises(BenchmarkValidationError, match="outside.*dataset"):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )

    assert real_read_bytes(manifest_path) == manifest_before
    assert real_read_bytes(digest_path) == digest_before
    assert sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    ) == inventory_before


def test_manifest_digest_must_not_share_filesystem_identity_with_dataset(
    tmp_path, monkeypatch
):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir()
    digest_alias = metadata_dir / "anchor.dat"
    os.link(digest_path, digest_alias)
    assert not digest_alias.is_symlink()
    assert digest_alias.suffix != ".png"
    assert digest_alias.lstat().st_dev == digest_path.lstat().st_dev
    assert digest_alias.lstat().st_ino == digest_path.lstat().st_ino

    paths_before = sorted(
        path.relative_to(dataset_dir).as_posix() for path in dataset_dir.rglob("*")
    )
    real_read_bytes = Path.read_bytes
    bytes_before = {
        path.relative_to(dataset_dir).as_posix(): real_read_bytes(path)
        for path in dataset_dir.rglob("*")
        if path.is_file()
    }
    digest_before = real_read_bytes(digest_path)

    def forbid_digest_read(path):
        if path == digest_path:
            raise AssertionError("external digest was read before hard-link rejection")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_digest_read)

    with pytest.raises(BenchmarkValidationError, match="filesystem identity.*dataset"):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )

    assert real_read_bytes(digest_path) == digest_before
    assert sorted(
        path.relative_to(dataset_dir).as_posix() for path in dataset_dir.rglob("*")
    ) == paths_before
    assert {
        path.relative_to(dataset_dir).as_posix(): real_read_bytes(path)
        for path in dataset_dir.rglob("*")
        if path.is_file()
    } == bytes_before


def test_manifest_digest_identity_scan_converts_entry_lstat_errors(
    tmp_path, monkeypatch
):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    unreadable_entry = dataset_dir / "metadata.dat"
    unreadable_entry.write_bytes(b"metadata")
    real_lstat = os.lstat

    def failing_lstat(path, *args, **kwargs):
        if Path(path) == unreadable_entry:
            raise PermissionError("permission denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(oracle.os, "lstat", failing_lstat)

    with pytest.raises(BenchmarkValidationError, match="traverse"):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


def test_manifest_digest_rejects_crlf_raw_bytes(tmp_path):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    digest_path.write_bytes(digest_path.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(BenchmarkValidationError, match="format"):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("schema_version", "change"),
        ("benchmark_version", "change"),
        ("claim", "change"),
        ("source_spec_sha256", "change"),
        ("base_position_count", "omit"),
        ("base_position_count", "change"),
        ("limitations", "omit"),
        ("limitations", "change"),
        ("expected_counts", "change"),
    ],
)
def test_manifest_source_owned_metadata_cannot_drift(tmp_path, field, mutation):
    manifest = _manifest_fixture()
    if mutation == "omit":
        del manifest[field]
    elif field in {"schema_version", "base_position_count"}:
        manifest[field] += 1
    elif field == "limitations":
        manifest[field] = ["changed"]
    elif field == "expected_counts":
        manifest[field]["all"] += 1
    else:
        manifest[field] = "changed"
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


def test_manifest_top_level_fields_are_exact(tmp_path):
    manifest = _manifest_fixture()
    manifest["extra"] = "not source-owned"
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize(
    "mutation",
    ["metadata", "nested-metadata", "board-present", "geometry"],
)
def test_manifest_preserves_json_types(tmp_path, mutation):
    manifest = _manifest_fixture()
    if mutation == "metadata":
        manifest["base_position_count"] = 16.0
    elif mutation == "nested-metadata":
        manifest["expected_counts"]["all"] = 568.0
    else:
        positive = next(case for case in manifest["cases"] if case["board_present"])
        if mutation == "board-present":
            positive["board_present"] = 1
        else:
            positive["board_box"][0] = 0.0
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_manifest_case_inventory_is_exact(tmp_path, mutation):
    manifest = _manifest_fixture()
    if mutation == "missing":
        manifest["cases"].pop()
    elif mutation == "extra":
        extra = copy.deepcopy(manifest["cases"][0])
        extra["id"] = "extra"
        extra["path"] = "images/extra.png"
        manifest["cases"].append(extra)
    else:
        manifest["cases"].append(copy.deepcopy(manifest["cases"][0]))
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


def test_manifest_semantic_label_is_reconstructed_from_source(tmp_path):
    manifest = _manifest_fixture()
    case = next(case for case in manifest["cases"] if case["board_present"])
    case["source_placement"] = rotate_placement(case["source_placement"])
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


def test_manifest_relative_case_path_is_source_owned(tmp_path):
    manifest = _manifest_fixture()
    manifest["cases"][0]["path"] = "../escape.png"
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize("field", ["file_sha256", "pixel_sha256"])
def test_manifest_generated_hashes_are_lowercase_64_hex(tmp_path, field):
    manifest = _manifest_fixture()
    manifest["cases"][0][field] = "A" * 64
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "empty", "non-string"])
def test_manifest_generation_runtime_has_exact_nonempty_string_values(
    tmp_path, mutation
):
    manifest = _manifest_fixture()
    runtime = manifest["generation_runtime"]
    if mutation == "missing":
        runtime.pop("python")
    elif mutation == "extra":
        runtime["host"] = "machine"
    elif mutation == "empty":
        runtime["python"] = ""
    else:
        runtime["python"] = 1
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


@pytest.mark.parametrize("target", ["dataset", "manifest", "digest"])
def test_manifest_verifier_rejects_symlinked_boundaries(tmp_path, target):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)
    if target == "dataset":
        real_dataset = tmp_path / "real-dataset"
        dataset_dir.rename(real_dataset)
        dataset_dir.symlink_to(real_dataset, target_is_directory=True)
    elif target == "manifest":
        manifest_path = dataset_dir / "manifest.json"
        real_manifest = dataset_dir / "real-manifest.json"
        manifest_path.rename(real_manifest)
        manifest_path.symlink_to(real_manifest)
    else:
        real_digest = dataset_dir / "real-manifest.sha256"
        digest_path.rename(real_digest)
        digest_path.symlink_to(real_digest)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(
            dataset_dir, _source_spec(), digest_path, verify_images=False
        )


def test_manifest_verifier_rejects_symlinked_dataset_and_manifest_ancestor(
    tmp_path,
):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    dataset_dir, digest_path, _manifest = _write_manifest(real_root)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(BenchmarkValidationError, match="symlink"):
        oracle.verify_manifest(
            linked_root / dataset_dir.name,
            _source_spec(),
            digest_path,
            verify_images=False,
        )


def test_manifest_verifier_rejects_symlinked_digest_ancestor(tmp_path):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    dataset_dir, digest_path, _manifest = _write_manifest(real_root)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(BenchmarkValidationError, match="symlink"):
        oracle.verify_manifest(
            dataset_dir,
            _source_spec(),
            linked_root / digest_path.name,
            verify_images=False,
        )


def test_manifest_verifier_accepts_relative_manifest_paths(tmp_path, monkeypatch):
    dataset_dir, digest_path, manifest = _write_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert oracle.verify_manifest(
        Path(dataset_dir.name),
        _source_spec(),
        Path(digest_path.name),
        verify_images=False,
    ) == manifest


def test_manifest_image_verification_requires_the_planned_png_inventory(tmp_path):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)

    with pytest.raises(BenchmarkValidationError):
        oracle.verify_manifest(dataset_dir, _source_spec(), digest_path)


def test_manifest_image_inventory_rejects_walk_errors(tmp_path, monkeypatch):
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path)

    def failing_walk(_directory, *, followlinks, onerror=None):
        assert followlinks is False
        if onerror is None:
            raise AssertionError("os.walk errors would be ignored")
        onerror(OSError("permission denied"))
        yield

    monkeypatch.setattr(oracle.os, "walk", failing_walk)

    with pytest.raises(BenchmarkValidationError, match="traverse"):
        oracle.verify_manifest(dataset_dir, _source_spec(), digest_path)


def _encoded_test_image(mode, size, image_format="PNG"):
    color = {"RGB": (1, 2, 3), "RGBA": (1, 2, 3, 255), "L": 1}[mode]
    output = io.BytesIO()
    Image.new(mode, size, color).save(output, format=image_format)
    return output.getvalue()


def _decoded_pixel_sha256(encoded):
    with Image.open(io.BytesIO(encoded)) as image:
        image.load()
        return pixel_sha256(image)


@lru_cache(maxsize=None)
def _cached_png(mode, size):
    encoded = _encoded_test_image(mode, size)
    return encoded, _decoded_pixel_sha256(encoded)


def _bind_case_image(dataset_dir, case, encoded):
    (dataset_dir / case["path"]).write_bytes(encoded)
    case["file_sha256"] = hashlib.sha256(encoded).hexdigest()
    case["pixel_sha256"] = _decoded_pixel_sha256(encoded)


@pytest.fixture
def full_image_manifest(tmp_path):
    manifest = _manifest_fixture()
    dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)
    (dataset_dir / "images").mkdir()
    for case in manifest["cases"]:
        encoded, pixel_hash = _cached_png(
            case["image_mode"], tuple(case["image_size"])
        )
        (dataset_dir / case["path"]).write_bytes(encoded)
        case["file_sha256"] = hashlib.sha256(encoded).hexdigest()
        case["pixel_sha256"] = pixel_hash
    dataset_dir, digest_path, manifest = _write_manifest(tmp_path, manifest)
    return tmp_path, dataset_dir, digest_path, manifest


def test_manifest_image_verifier_accepts_full_planned_inventory(
    full_image_manifest,
):
    _tmp_path, dataset_dir, digest_path, manifest = full_image_manifest

    assert oracle.verify_manifest(dataset_dir, _source_spec(), digest_path) == manifest


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("non-png", "encode PNG"),
        ("decode", "decode"),
        ("mode", "mode"),
        ("size", "size"),
        ("file-hash", "file hash"),
        ("pixel-hash", "pixel hash"),
        ("extra", "inventory"),
        ("missing", "inventory"),
        ("nested-symlink", "symlink"),
    ],
)
def test_manifest_image_verifier_rejects_artifact_mutations(
    full_image_manifest, mutation, match
):
    tmp_path, dataset_dir, _digest_path, manifest = full_image_manifest
    case = next(case for case in manifest["cases"] if case["image_mode"] == "RGB")
    image_path = dataset_dir / case["path"]

    if mutation == "non-png":
        _bind_case_image(
            dataset_dir,
            case,
            _encoded_test_image("RGB", tuple(case["image_size"]), "JPEG"),
        )
    elif mutation == "decode":
        encoded = b"not an image"
        image_path.write_bytes(encoded)
        case["file_sha256"] = hashlib.sha256(encoded).hexdigest()
    elif mutation == "mode":
        _bind_case_image(
            dataset_dir,
            case,
            _encoded_test_image("L", tuple(case["image_size"])),
        )
    elif mutation == "size":
        width, height = case["image_size"]
        _bind_case_image(
            dataset_dir,
            case,
            _encoded_test_image("RGB", (width - 1, height)),
        )
    elif mutation == "file-hash":
        case["file_sha256"] = "0" * 64
    elif mutation == "pixel-hash":
        case["pixel_sha256"] = "0" * 64
    elif mutation == "extra":
        encoded, _pixel_hash = _cached_png("RGB", (1, 1))
        (dataset_dir / "images/extra.png").write_bytes(encoded)
    elif mutation == "missing":
        image_path.unlink()
    else:
        real_directory = tmp_path / "real-images"
        real_directory.mkdir()
        (dataset_dir / "images/nested").symlink_to(
            real_directory, target_is_directory=True
        )

    _dataset_dir, digest_path, _manifest = _write_manifest(tmp_path, manifest)
    with pytest.raises(BenchmarkValidationError, match=match):
        oracle.verify_manifest(dataset_dir, _source_spec(), digest_path)


def test_evaluator_module_import_defers_chessml(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "chessml" or name.startswith("chessml."):
            raise AssertionError("evaluator imported chessml at module load")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _load_evaluator_module()


def test_evaluator_readme_labels_write_before_benchmark_commands():
    section = (ROOT / "README.md").read_text(encoding="utf-8").split(
        "### Online board-recognition benchmark", 1
    )[1]
    introduction, command_block = section.split("```bash", 1)
    introduction = " ".join(introduction.split())

    assert (
        "Ordinary users should use the read-only `--preflight` and `--verify` "
        "commands. The shown `--write` line is maintainer-only:"
        in introduction
    )
    assert command_block.split("```", 1)[0].strip().splitlines() == [
        "uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight",
        "uv run --locked python -m scripts.data.generate_board_recognition_benchmark --write datasets/board_recognition_benchmark/v1",
        "uv run --locked python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256",
    ]


def test_runner_maps_typed_results_continues_after_errors_and_scores(
    evaluator, tmp_path
):
    runner_cases = [
        {
            "id": "d-success",
            "path": "images/d-success.png",
            "board_present": True,
            "source_placement": EXPECTED,
        },
        {
            "id": "c-success",
            "path": "images/c-success.png",
            "board_present": True,
            "source_placement": EXPECTED,
        },
        {
            "id": "b-failure",
            "path": "images/b-failure.png",
            "board_present": False,
        },
        {
            "id": "a-error",
            "path": "images/a-error.png",
            "board_present": True,
            "source_placement": EXPECTED,
        },
    ]
    original_cases = copy.deepcopy(runner_cases)

    class Picture:
        constructed = []

        def __init__(self, path):
            self.path = path
            self.side_effects = []
            self.constructed.append(self)

        def mark_board_on_image(self):
            self.side_effects.append("mark")
            raise AssertionError("input picture was marked")

        def save(self, *_args, **_kwargs):
            self.side_effects.append("save")
            raise AssertionError("input picture was saved")

    class BoardImage:
        def __init__(self):
            self.side_effects = []

        def mark_board_on_image(self):
            self.side_effects.append("mark")
            raise AssertionError("recognized board image was marked")

        def save(self, *_args, **_kwargs):
            self.side_effects.append("save")
            raise AssertionError("recognized board image was saved")

    class RecognitionSuccess:
        def __init__(self):
            self.source_placement = EXPECTED
            self.board_image = BoardImage()

    class RecognitionFailure:
        def __init__(self):
            self.reason = SimpleNamespace(name="NO_BOARD")

    class CaseExplosion(RuntimeError):
        pass

    class Helper:
        def __init__(self):
            self.calls = []
            self.board_images = []

        def recognize(self, picture):
            self.calls.append(picture)
            if picture.path.stem == "a-error":
                raise CaseExplosion("case exploded")
            if picture.path.stem == "b-failure":
                return RecognitionFailure()
            result = RecognitionSuccess()
            self.board_images.append(result.board_image)
            return result

    helper = Helper()
    predictions = evaluator.run_predictions(
        tmp_path,
        runner_cases,
        helper,
        picture_cls=Picture,
        success_cls=RecognitionSuccess,
        failure_cls=RecognitionFailure,
    )

    assert predictions == [
        {
            "id": "a-error",
            "outcome": "error",
            "error_type": "CaseExplosion",
            "error_message": "case exploded",
        },
        {"id": "b-failure", "outcome": "failure", "reason": "NO_BOARD"},
        {
            "id": "c-success",
            "outcome": "success",
            "source_placement": EXPECTED,
        },
        {
            "id": "d-success",
            "outcome": "success",
            "source_placement": EXPECTED,
        },
    ]
    assert [picture.path for picture in Picture.constructed] == [
        tmp_path / "images/a-error.png",
        tmp_path / "images/b-failure.png",
        tmp_path / "images/c-success.png",
        tmp_path / "images/d-success.png",
    ]
    assert helper.calls == Picture.constructed
    assert len({id(picture) for picture in Picture.constructed}) == len(runner_cases)
    assert len({id(image) for image in helper.board_images}) == 2
    assert all(picture.side_effects == [] for picture in Picture.constructed)
    assert all(image.side_effects == [] for image in helper.board_images)
    assert not any(
        picture is image
        for picture in Picture.constructed
        for image in helper.board_images
    )
    assert runner_cases == original_cases
    scored = score_predictions(runner_cases, predictions)
    assert scored["overall"]["execution_error_count"] == 1


def test_runner_turns_invalid_typed_and_unknown_results_into_execution_errors(
    evaluator, tmp_path
):
    class Picture:
        def __init__(self, path):
            self.path = path

    class RecognitionSuccess:
        def __init__(self, placement):
            self.source_placement = placement
            self.board_image = object()

    class RecognitionFailure:
        def __init__(self, reason):
            self.reason = SimpleNamespace(name=reason)

    class Helper:
        def __init__(self):
            self.results = iter(
                [
                    RecognitionSuccess("8/8"),
                    RecognitionFailure("OTHER"),
                    SimpleNamespace(
                        source_placement=EXPECTED, board_image=object()
                    ),
                ]
            )

        def recognize(self, _picture):
            return next(self.results)

    predictions = evaluator.run_predictions(
        tmp_path,
        [
            {"id": "a-invalid-success", "path": "images/a.png"},
            {"id": "b-invalid-failure", "path": "images/b.png"},
            {"id": "c-unknown", "path": "images/c.png"},
        ],
        Helper(),
        picture_cls=Picture,
        success_cls=RecognitionSuccess,
        failure_cls=RecognitionFailure,
    )

    assert [prediction["id"] for prediction in predictions] == [
        "a-invalid-success",
        "b-invalid-failure",
        "c-unknown",
    ]
    assert all(prediction["outcome"] == "error" for prediction in predictions)
    assert all(
        prediction["error_type"] == "InvalidRecognitionResult"
        for prediction in predictions
    )
    assert all(prediction["error_message"] for prediction in predictions)


def test_evaluator_report_writer_uses_canonical_json(evaluator, tmp_path):
    report_path = tmp_path / "report.json"
    report = {"z": 1, "a": [2]}

    evaluator.write_report(report_path, report)

    assert report_path.read_bytes() == canonical_json_bytes(report)
    assert list(tmp_path.iterdir()) == [report_path]


def test_evaluator_report_writer_default_keeps_mkstemp_descriptor(
    evaluator, tmp_path, monkeypatch
):
    report_path = tmp_path / "report.json"
    report = {"report": True}

    def forbid_path_reopen(_path, _encoded):
        raise AssertionError("default report write reopened the mkstemp path")

    monkeypatch.setattr(Path, "write_bytes", forbid_path_reopen)

    evaluator.write_report(report_path, report)

    assert report_path.read_bytes() == canonical_json_bytes(report)


@pytest.mark.parametrize("replacement_kind", ["file", "symlink"])
def test_evaluator_report_writer_leaves_replaced_temp_path_untouched(
    evaluator, tmp_path, replacement_kind
):
    report_path = tmp_path / "report.json"
    temp_paths = []
    replacement_target = tmp_path / "replacement-target"

    def replacing_write(temp_path, _encoded):
        temp_paths.append(temp_path)
        temp_path.unlink()
        if replacement_kind == "file":
            temp_path.write_bytes(b"replacement")
        else:
            replacement_target.write_bytes(b"replacement")
            temp_path.symlink_to(replacement_target)

    with pytest.raises(BenchmarkValidationError, match="owned regular file"):
        evaluator.write_report(
            report_path, {"report": True}, write_fn=replacing_write
        )

    assert not os.path.lexists(report_path)
    assert os.path.lexists(temp_paths[0])
    if replacement_kind == "file":
        assert temp_paths[0].read_bytes() == b"replacement"
    else:
        assert temp_paths[0].is_symlink()
        assert temp_paths[0].readlink() == replacement_target


@pytest.mark.parametrize("write_kind", ["short", "altered"])
def test_evaluator_report_writer_rejects_noncanonical_temp_bytes(
    evaluator, tmp_path, write_kind
):
    report_path = tmp_path / "report.json"
    temp_paths = []

    def invalid_write(temp_path, encoded):
        temp_paths.append(temp_path)
        value = encoded[:-1] if write_kind == "short" else b"[" + encoded[1:]
        temp_path.write_bytes(value)

    with pytest.raises(BenchmarkValidationError, match="canonical content"):
        evaluator.write_report(
            report_path, {"report": True}, write_fn=invalid_write
        )

    assert temp_paths
    assert not os.path.lexists(temp_paths[0])
    assert not os.path.lexists(report_path)


@pytest.mark.parametrize("target", ["file", "dangling-symlink"])
def test_evaluator_report_writer_refuses_existing_targets_unchanged(
    evaluator, tmp_path, target
):
    report_path = tmp_path / "report.json"
    if target == "file":
        report_path.write_bytes(b"sentinel")
    else:
        report_path.symlink_to(tmp_path / "missing-target")

    with pytest.raises(FileExistsError):
        evaluator.write_report(report_path, {"new": "report"})

    if target == "file":
        assert report_path.read_bytes() == b"sentinel"
    else:
        assert report_path.is_symlink()
        assert report_path.readlink() == tmp_path / "missing-target"


@pytest.mark.parametrize("parent_kind", ["symlink", "file", "missing"])
def test_evaluator_report_writer_rejects_nonreal_parent(
    evaluator, tmp_path, parent_kind
):
    parent = tmp_path / "report-parent"
    if parent_kind == "symlink":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        parent.symlink_to(real_parent, target_is_directory=True)
    elif parent_kind == "file":
        parent.write_bytes(b"not a directory")

    with pytest.raises(BenchmarkValidationError, match="parent"):
        evaluator.write_report(parent / "report.json", {"report": True})


def test_evaluator_report_writer_rejects_symlinked_parent_ancestor(
    evaluator, tmp_path
):
    real_root = tmp_path / "real-root"
    real_parent = real_root / "reports"
    real_parent.mkdir(parents=True)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(BenchmarkValidationError, match="symlink"):
        evaluator.write_report(
            linked_root / "reports/report.json", {"report": True}
        )

    assert list(real_parent.iterdir()) == []


def test_evaluator_report_writer_accepts_relative_report_path(
    evaluator, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    parent = Path("reports")
    parent.mkdir()
    report_path = parent / "report.json"
    report = {"report": True}

    evaluator.write_report(report_path, report)

    assert report_path.read_bytes() == canonical_json_bytes(report)


@pytest.mark.parametrize("failure_point", ["write", "rename"])
def test_evaluator_report_writer_cleans_temp_after_injected_failure(
    evaluator, tmp_path, failure_point
):
    report_path = tmp_path / "report.json"
    temp_paths = []

    def write_fn(temp_path, encoded):
        temp_paths.append(temp_path)
        if failure_point == "write":
            raise OSError("injected write failure")
        temp_path.write_bytes(encoded)

    def rename_fn(source, destination):
        assert failure_point == "rename"
        assert source == temp_paths[0]
        assert destination == report_path
        raise OSError("injected rename failure")

    with pytest.raises(OSError, match=failure_point):
        evaluator.write_report(
            report_path,
            {"report": True},
            write_fn=write_fn,
            rename_fn=rename_fn,
        )

    assert temp_paths
    assert not os.path.lexists(temp_paths[0])
    assert not os.path.lexists(report_path)
    assert list(tmp_path.iterdir()) == []


def test_evaluator_report_writer_rechecks_target_before_rename(
    evaluator, tmp_path
):
    report_path = tmp_path / "report.json"
    temp_paths = []

    def racing_write(temp_path, encoded):
        temp_paths.append(temp_path)
        temp_path.write_bytes(encoded)
        report_path.write_bytes(b"racing sentinel")

    with pytest.raises(FileExistsError):
        evaluator.write_report(
            report_path, {"report": True}, write_fn=racing_write
        )

    assert report_path.read_bytes() == b"racing sentinel"
    assert not os.path.lexists(temp_paths[0])


def _evaluator_cli_args(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_path = tmp_path / "source.json"
    source_path.write_text("{}\n", encoding="utf-8")
    digest_path = tmp_path / "manifest.sha256"
    digest_path.write_text(f"{'a' * 64}  manifest.json\n", encoding="ascii")
    checkpoints = {
        "board_detector": tmp_path / "board-detector.ckpt",
        "square_classifier": tmp_path / "square-classifier.ckpt",
        "piece_classifier": tmp_path / "piece-classifier.ckpt",
    }
    for name, path in checkpoints.items():
        path.write_bytes(name.encode("ascii"))
    report_path = tmp_path / "report.json"
    args = [
        "--dataset",
        str(dataset_dir),
        "--source-spec",
        str(source_path),
        "--manifest-digest",
        str(digest_path),
        "--board-detector-checkpoint",
        str(checkpoints["board_detector"]),
        "--square-classifier-checkpoint",
        str(checkpoints["square_classifier"]),
        "--piece-classifier-checkpoint",
        str(checkpoints["piece_classifier"]),
        "--report",
        str(report_path),
        "--device",
        "cpu",
    ]
    return args, dataset_dir, source_path, digest_path, checkpoints, report_path


@pytest.mark.parametrize(
    ("execution_error", "expected_status"), [(False, 0), (True, 1)]
)
def test_evaluator_cli_verifies_before_loading_and_publishes_complete_report(
    evaluator, tmp_path, monkeypatch, execution_error, expected_status
):
    args, dataset_dir, source_path, digest_path, checkpoints, report_path = (
        _evaluator_cli_args(tmp_path)
    )
    source = {
        "benchmark_version": "online-render-v1",
        "claim": "synthetic-regression-only",
        "base_position_count": 1,
        "limitations": ["not unseen-site accuracy"],
    }
    manifest = {
        "cases": [
            {
                "id": "negative",
                "path": "images/negative.png",
                "board_present": False,
            }
        ]
    }
    events = []

    class Picture:
        def __init__(self, path):
            self.path = path

    class RecognitionSuccess:
        pass

    class RecognitionFailure:
        reason = SimpleNamespace(name="NO_BOARD")

    class Helper:
        def recognize(self, picture):
            events.append(("recognize", picture.path))
            if execution_error:
                raise RuntimeError("model exploded")
            return RecognitionFailure()

    def load_source(path):
        events.append(("source", path))
        return source

    def verify(dataset, loaded_source, digest):
        events.append(("verify", dataset, loaded_source, digest))
        return manifest

    def helper_factory(board, square, piece, device):
        events.append(("load", board, square, piece, device))
        return Helper(), Picture, RecognitionSuccess, RecognitionFailure

    fingerprint = {
        "python": "3.test",
        "torch": "torch-test",
        "torchvision": "torchvision-test",
        "timm": "timm-test",
        "lightning": "lightning-test",
        "opencv-python": "opencv-test",
        "numpy": "numpy-test",
    }
    monkeypatch.setattr(evaluator, "load_source_spec", load_source)
    monkeypatch.setattr(evaluator, "verify_manifest", verify)
    monkeypatch.setattr(evaluator, "load_helper", helper_factory)
    monkeypatch.setattr(evaluator, "inference_fingerprint", lambda: fingerprint)

    assert evaluator.main(args) == expected_status

    assert events[0] == ("source", source_path)
    assert events[1] == ("verify", dataset_dir, source, digest_path)
    assert events[2] == (
        "load",
        checkpoints["board_detector"],
        checkpoints["square_classifier"],
        checkpoints["piece_classifier"],
        "cpu",
    )
    report = json.loads(report_path.read_bytes())
    assert set(report) == {
        "benchmark_version",
        "claim",
        "base_position_count",
        "limitations",
        "manifest_sha256",
        "checkpoints",
        "device",
        "inference_fingerprint",
        "predictions",
        "scores",
    }
    assert {
        key: report[key]
        for key in (
            "benchmark_version",
            "claim",
            "base_position_count",
            "limitations",
        )
    } == source
    assert report["manifest_sha256"] == "a" * 64
    assert report["checkpoints"] == {
        name: {"basename": path.name, "sha256": file_sha256(path)}
        for name, path in checkpoints.items()
    }
    assert report["device"] == "cpu"
    assert report["inference_fingerprint"] == fingerprint
    assert report["scores"] == score_predictions(
        manifest["cases"], report["predictions"]
    )
    assert report["scores"]["overall"]["execution_error_count"] == int(
        execution_error
    )
    assert report_path.read_bytes() == canonical_json_bytes(report)


def test_evaluator_cli_rejects_report_inside_dataset_before_model_loading(
    evaluator, tmp_path, monkeypatch
):
    args, dataset_dir, _source_path, _digest_path, _checkpoints, _report_path = (
        _evaluator_cli_args(tmp_path)
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    report_path = outside / ".." / dataset_dir.name / "report.json"
    args[args.index("--report") + 1] = str(report_path)
    events = []

    def verify(*_args, **_kwargs):
        events.append("verify")
        return {"cases": []}

    def forbidden(stage):
        def fail(*_args, **_kwargs):
            events.append(stage)
            raise AssertionError(f"{stage} ran before report containment rejection")

        return fail

    monkeypatch.setattr(evaluator, "load_source_spec", lambda _path: {})
    monkeypatch.setattr(evaluator, "verify_manifest", verify)
    monkeypatch.setattr(evaluator, "load_helper", forbidden("load"))
    monkeypatch.setattr(evaluator, "run_predictions", forbidden("inference"))
    monkeypatch.setattr(evaluator, "write_report", forbidden("publication"))

    with pytest.raises(BenchmarkValidationError, match="outside.*dataset"):
        evaluator.main(args)

    assert events == ["verify"]
    assert not os.path.lexists(report_path)


def test_evaluator_cli_tampered_corpus_prevents_model_loading(
    evaluator, tmp_path, monkeypatch
):
    args, _dataset_dir, _source_path, _digest_path, _checkpoints, report_path = (
        _evaluator_cli_args(tmp_path)
    )
    loaded = False

    def reject_manifest(*_args, **_kwargs):
        raise BenchmarkValidationError("tampered corpus")

    def helper_factory(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("model helper loaded before corpus verification")

    monkeypatch.setattr(evaluator, "load_source_spec", lambda _path: {})
    monkeypatch.setattr(evaluator, "verify_manifest", reject_manifest)
    monkeypatch.setattr(evaluator, "load_helper", helper_factory)

    with pytest.raises(BenchmarkValidationError, match="tampered corpus"):
        evaluator.main(args)

    assert loaded is False
    assert not os.path.lexists(report_path)
