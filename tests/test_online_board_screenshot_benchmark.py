from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib
from collections import Counter
from io import BytesIO
from itertools import product
from operator import itemgetter
from pathlib import Path

import pytest
from PIL import Image

import benchmarks.board_recognition as board_recognition
import benchmarks.captured_board_recognition as captured
from benchmarks.board_recognition import (
    BenchmarkValidationError,
    build_case_plan,
    canonical_json_bytes,
    rotate_placement,
    score_predictions,
    validate_source_spec,
)


ROOT = Path(__file__).resolve().parents[1]
ZERO_SHA256 = "0" * 64
TOP_LEVEL_KEYS = {
    "schema_version",
    "benchmark_version",
    "claim",
    "base_position_count",
    "limitations",
    "expected_counts",
    "acceptance_policy",
    "capture_protocol",
    "allowed_sources",
    "capture_conditions",
    "evidence",
    "cases",
}
COMMON_CASE_KEYS = {
    "id",
    "path",
    "suite",
    "split",
    "board_present",
    "site",
    "viewport",
    "layout",
    "quality",
    "channel_mode",
    "image_mode",
    "image_size",
    "source_url",
    "page_kind",
}
POSITIVE_ONLY_KEYS = {
    "base_position",
    "fen",
    "canonical_placement",
    "source_placement",
    "view",
    "piece_set",
    "board_theme",
    "board_rect",
}
NEGATIVE_ONLY_KEYS = {"negative_page"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REVIEW_ISOLATION = [
    "reviewer_a_received_only_opaque_images_and_schema",
    "reviewer_b_received_only_opaque_images_and_schema",
]


def _png_chunk(chunk_type, payload):
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _png_chunks(encoded):
    chunks = []
    offset = len(PNG_SIGNATURE)
    while offset < len(encoded):
        length = struct.unpack(">I", encoded[offset : offset + 4])[0]
        end = offset + 12 + length
        chunks.append(
            (
                encoded[offset + 4 : offset + 8],
                encoded[offset + 8 : offset + 8 + length],
                encoded[offset:end],
            )
        )
        offset = end
    return chunks


def _rebuild_png(chunks):
    return PNG_SIGNATURE + b"".join(
        _png_chunk(chunk_type, payload) for chunk_type, payload, _raw in chunks
    )


def _insert_valid_crc_chunk_before_iend(encoded, chunk_type, payload):
    chunks = _png_chunks(encoded)
    return PNG_SIGNATURE + b"".join(
        _png_chunk(chunk_type, payload)
        if existing_type == b"IEND"
        else raw
        for existing_type, _existing_payload, raw in chunks
    ) + next(raw for existing_type, _payload, raw in chunks if existing_type == b"IEND")


def _image_bytes(mode="RGB", size=(8, 8), color=None):
    if color is None:
        color = {"RGB": (1, 2, 3), "RGBA": (1, 2, 3, 255), "L": 1}[mode]
    output = BytesIO()
    Image.new(mode, size, color).save(output, format="PNG")
    return output.getvalue()


def _pixel_hash(encoded):
    with Image.open(BytesIO(encoded)) as image:
        image.load()
        return board_recognition.pixel_sha256(image)


def _write_evidence_bytes(bundle, name, encoded):
    record = bundle["spec"]["evidence"][name]
    path = bundle["root"].joinpath(*record["path"].split("/"))
    path.write_bytes(encoded)
    record["sha256"] = hashlib.sha256(encoded).hexdigest()
    return record["sha256"]


def _write_evidence_json(bundle, name):
    return _write_evidence_bytes(
        bundle, name, canonical_json_bytes(bundle["documents"][name])
    )


@pytest.fixture
def valid_spec():
    spec = captured.build_capture_plan()
    for record in spec["evidence"].values():
        records = record if isinstance(record, list) else [record]
        for item in records:
            item["sha256"] = "1" * 64
    for case in spec["cases"]:
        if case["board_present"]:
            side = 400 if case["viewport"] == "desktop_clean" else 300
            case["board_rect"] = [10, 10, side, side]
    return spec


@pytest.fixture
def rgb_png():
    return _image_bytes()


@pytest.fixture
def captured_evidence(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    spec = captured.build_capture_plan()
    cases = sorted(spec["cases"], key=itemgetter("id"))
    for case in cases:
        if case["board_present"]:
            side = 400 if case["viewport"] == "desktop_clean" else 300
            case["board_rect"] = [10, 10, side, side]

    license_by_path = {}
    for record in spec["evidence"]["licenses"]:
        encoded = f"frozen evidence for {record['path']}\n".encode()
        path = root.joinpath(*record["path"].split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        record["sha256"] = hashlib.sha256(encoded).hexdigest()
        license_by_path[record["path"]] = copy.deepcopy(record)

    assets = {
        site: {
            "piece_asset_id": f"{site}-piece-v1",
            "board_asset_id": f"{site}-board-v1",
            "piece_asset_url_prefix": f"https://assets.example/{site}/pieces/",
            "board_style_signature": f"{site}-brown-v1",
            "upstream_revision": f"{site}-revision-v1",
        }
        for site in captured.SITES
    }
    rights_paths = {
        "lichess": {
            "project_copying": "benchmarks/licenses/lichess-lila-COPYING.md",
            "piece_license": "benchmarks/licenses/lichess-chessnut-LICENSE.txt",
            "terms": "benchmarks/licenses/lichess-terms-2026-07-19.txt",
            "privacy": "benchmarks/licenses/lichess-privacy-2026-07-19.txt",
        },
        "pychess": {
            "project_copying": "benchmarks/licenses/pychess-variants-COPYING.md",
            "piece_license": "benchmarks/licenses/pychess-firi-LICENSE.txt",
            "terms": "benchmarks/licenses/pychess-terms-2026-05-22.md",
            "privacy": "benchmarks/licenses/pychess-privacy-2026-05-22.md",
        },
    }
    rights = {
        "schema_version": 1,
        "benchmark_version": captured.CAPTURED_VERSION,
        "checked_at_utc": "2026-07-19T12:00:00Z",
        "sites": {
            site: assets[site]
            | {
                field: license_by_path[path]
                for field, path in rights_paths[site].items()
            }
            for site in captured.SITES
        },
    }

    receipts = {
        "schema_version": 1,
        "benchmark_version": captured.CAPTURED_VERSION,
        "captures": [],
    }
    for case in cases:
        viewport = captured.VIEWPORTS[case["viewport"]]
        receipt = {
            "id": case["id"],
            "png_sha256": hashlib.sha256(case["id"].encode()).hexdigest(),
            "captured_at_utc": "2026-07-19T12:00:00Z",
            "final_url": case["source_url"],
            "title": f"Frozen {case['site']} page",
            "site": case["site"],
            "page_kind": case["page_kind"],
            "viewport": case["viewport"],
            "viewport_size": [viewport["width"], viewport["height"]],
            "dpr": viewport["dpr"],
            "scroll": [0, 0],
            "visibility_state": "visible",
            "navigation_state": "complete",
            "user_agent": "frozen-test-agent",
            "platform": "frozen-test-platform",
            "language": "en-US",
        }
        if case["board_present"]:
            receipt.update(
                {
                    field: case[field]
                    for field in (
                        "fen",
                        "view",
                        "board_rect",
                        "piece_set",
                        "board_theme",
                    )
                }
                | assets[case["site"]]
            )
        else:
            receipt["no_complete_board"] = True
        receipts["captures"].append(receipt)

    def annotations(reviewer):
        return {
            "schema_version": 1,
            "benchmark_version": captured.CAPTURED_VERSION,
            "reviewer": reviewer,
            "isolation": "opaque_images_only",
            "cases": [
                {
                    "id": case["id"],
                    "board_present": case["board_present"],
                    **(
                        {"source_placement": case["source_placement"]}
                        if case["board_present"]
                        else {}
                    ),
                    "full_resolution_pass": True,
                    "privacy_pass": True,
                    "capture_condition_pass": True,
                }
                for case in cases
            ],
        }

    documents = {
        "receipts": receipts,
        "annotations_a": annotations("A"),
        "annotations_b": annotations("B"),
        "rights": rights,
    }
    bundle = {"root": root, "spec": spec, "documents": documents}
    for name in ("receipts", "annotations_a", "annotations_b", "rights"):
        path = root.joinpath(*spec["evidence"][name]["path"].split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_evidence_json(bundle, name)

    documents["review"] = {
        "schema_version": 1,
        "benchmark_version": captured.CAPTURED_VERSION,
        "annotation_sha256": {
            "a": spec["evidence"]["annotations_a"]["sha256"],
            "b": spec["evidence"]["annotations_b"]["sha256"],
        },
        "case_ids": [case["id"] for case in cases],
        "review_isolation": list(REVIEW_ISOLATION),
        "a_b_agreement": True,
        "controlled_input_agreement": True,
        "model_output_used": False,
    }
    review_path = root.joinpath(*spec["evidence"]["review"]["path"].split("/"))
    review_path.parent.mkdir(parents=True, exist_ok=True)
    _write_evidence_json(bundle, "review")

    attribution = b"frozen attribution evidence\n"
    attribution_path = root.joinpath(
        *spec["evidence"]["attribution"]["path"].split("/")
    )
    attribution_path.write_bytes(attribution)
    spec["evidence"]["attribution"]["sha256"] = hashlib.sha256(
        attribution
    ).hexdigest()
    return bundle


@pytest.fixture
def verified_image_dataset(tmp_path, rgb_png):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    (dataset / "manifest.json").write_bytes(b"{}\n")
    image_path = images / "case.png"
    image_path.write_bytes(rgb_png)
    case = {
        "id": "case",
        "path": "images/case.png",
        "file_sha256": hashlib.sha256(rgb_png).hexdigest(),
        "pixel_sha256": _pixel_hash(rgb_png),
        "image_mode": "RGB",
        "image_size": [8, 8],
    }
    return dataset, case, rgb_png


def _positive(spec):
    return next(case for case in spec["cases"] if case["board_present"])


def _negative(spec):
    return next(case for case in spec["cases"] if not case["board_present"])


def _perfect_predictions(spec):
    return [
        {
            "id": case["id"],
            "outcome": "success",
            "source_placement": case["source_placement"],
        }
        if case["board_present"]
        else {"id": case["id"], "outcome": "failure", "reason": "NO_BOARD"}
        for case in spec["cases"]
    ]


def test_capture_plan_is_exact_deterministic_40_case_matrix():
    first = captured.build_capture_plan()
    second = captured.build_capture_plan()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert set(first) == TOP_LEVEL_KEYS
    assert len(first["cases"]) == 40
    assert Counter(case["board_present"] for case in first["cases"]) == {
        True: 32,
        False: 8,
    }
    positives = [case for case in first["cases"] if case["board_present"]]
    assert {
        (case["site"], case["base_position"], case["view"], case["viewport"])
        for case in positives
    } == set(
        product(
            ("lichess", "pychess"),
            ("start", "najdorf", "nimzo", "pawn_endgame"),
            ("white_bottom", "black_bottom"),
            ("desktop_clean", "compact_scaled"),
        )
    )


def test_capture_plan_locks_source_policy_placeholders_fields_and_order():
    plan = captured.build_capture_plan()
    assert plan["schema_version"] == 1
    assert plan["benchmark_version"] == "online-screenshot-v1"
    assert plan["claim"] == "captured_online_screenshot_acceptance"
    assert plan["base_position_count"] == 4
    assert plan["limitations"] == [
        "This suite covers anonymous Lichess and PyChess pages only.",
        "The compact condition is native responsive browser scaling, not postprocessing.",
        "Empty and invalid complete boards are outside online-screenshot-v1.",
        "No benchmark image or label may be used for training, calibration, checkpoint selection, or threshold selection.",
    ]
    assert plan["expected_counts"] == {"all": 40, "positive": 32, "negative": 8}
    assert plan["acceptance_policy"] == captured.CAPTURED_ACCEPTANCE_POLICY
    assert plan["capture_protocol"] == captured.CAPTURE_PROTOCOL
    assert plan["allowed_sources"] == {
        "lichess": {
            "editor_url": "https://lichess.org/editor",
            "negative_urls": [
                "https://lichess.org/terms-of-service",
                "https://lichess.org/privacy",
            ],
            "piece_set": "lichess_chessnut",
            "board_theme": "brown",
        },
        "pychess": {
            "editor_url": "https://www.pychess.org/editor/chess",
            "negative_urls": [
                "https://www.pychess.org/terms",
                "https://www.pychess.org/privacy",
            ],
            "piece_set": "pychess_firi",
            "board_theme": "brown",
        },
    }
    assert plan["capture_conditions"] == captured.VIEWPORTS

    evidence = plan["evidence"]
    assert set(evidence) == {
        "receipts",
        "annotations_a",
        "annotations_b",
        "review",
        "attribution",
        "rights",
        "licenses",
    }
    assert all(
        record["sha256"] == ZERO_SHA256
        for value in evidence.values()
        for record in (value if isinstance(value, list) else [value])
    )
    assert [record["path"] for record in evidence["licenses"]] == sorted(
        captured.LICENSE_PATHS
    )

    expected_ids = [
        "capture_"
        + hashlib.sha256(f"online-screenshot-v1:{index:02d}".encode("ascii"))
        .hexdigest()[:16]
        for index in range(40)
    ]
    assert [case["id"] for case in plan["cases"]] == expected_ids
    for case in plan["cases"]:
        expected_keys = COMMON_CASE_KEYS | (
            POSITIVE_ONLY_KEYS if case["board_present"] else NEGATIVE_ONLY_KEYS
        )
        assert set(case) == expected_keys
        assert case["path"] == f"images/{case['id']}.png"
        assert case["suite"] == "captured"
        assert case["split"] == "acceptance"
        assert case["layout"] == "full_page"
        assert case["channel_mode"] == case["image_mode"] == "RGB"
        if case["board_present"]:
            assert case["fen"] == captured.POSITIONS[case["base_position"]]
            assert case["canonical_placement"] == case["fen"].split()[0]
            assert case["source_placement"] == (
                case["canonical_placement"]
                if case["view"] == "white_bottom"
                else rotate_placement(case["canonical_placement"])
            )
            assert case["board_rect"] == [0, 0, 0, 0]


def test_captured_source_is_explicit_and_dispatches_without_synthetic_assets(valid_spec):
    validate_source_spec(
        valid_spec, Path("/does/not/exist"), verify_local_assets=False
    )
    assert build_case_plan(valid_spec) == sorted(
        valid_spec["cases"], key=itemgetter("id")
    )


def test_captured_case_plan_accepts_explicit_case_order_only_after_validation(valid_spec):
    valid_spec["cases"].reverse()
    assert captured.build_captured_case_plan(valid_spec) == sorted(
        valid_spec["cases"], key=itemgetter("id")
    )
    valid_spec["expected_counts"]["all"] = 39
    with pytest.raises(BenchmarkValidationError, match="expected_counts"):
        captured.build_captured_case_plan(valid_spec)


def test_captured_source_rejects_matrix_orientation_and_field_leakage(valid_spec):
    mutated = copy.deepcopy(valid_spec)
    positive = _positive(mutated)
    positive["source_placement"] = positive["canonical_placement"]
    positive["view"] = "black_bottom"
    with pytest.raises(BenchmarkValidationError, match="source placement"):
        validate_source_spec(mutated, ROOT, verify_local_assets=False)

    mutated = copy.deepcopy(valid_spec)
    negative = _negative(mutated)
    negative["piece_set"] = "lichess_chessnut"
    with pytest.raises(BenchmarkValidationError, match="negative fields"):
        validate_source_spec(mutated, ROOT, verify_local_assets=False)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("schema_version", True, "schema_version"),
        ("benchmark_version", "online-screenshot-v2", "benchmark_version"),
        ("claim", "synthetic_online_render_regression_only", "claim"),
        ("base_position_count", 5, "base_position_count"),
        ("limitations", [], "limitations"),
        ("expected_counts", {"all": 39, "positive": 32, "negative": 7}, "expected_counts"),
        ("acceptance_policy", {}, "acceptance_policy"),
        ("capture_protocol", {}, "capture_protocol"),
        ("allowed_sources", {}, "allowed_sources"),
        ("capture_conditions", {}, "capture_conditions"),
    ],
)
def test_captured_source_rejects_locked_top_level_values(
    valid_spec, field, replacement, match
):
    valid_spec[field] = replacement
    with pytest.raises(BenchmarkValidationError, match=match):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_captured_source_rejects_top_level_key_drift(valid_spec, mutation):
    if mutation == "missing":
        del valid_spec["limitations"]
    else:
        valid_spec["unexpected"] = True
    with pytest.raises(BenchmarkValidationError, match="source keys"):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("evidence-key", "evidence keys"),
        ("record-key", "evidence record"),
        ("unsafe-path", "evidence path"),
        ("wrong-path", "evidence path"),
        ("zero-hash", "evidence sha256"),
        ("uppercase-hash", "evidence sha256"),
        ("license-order", "license"),
        ("license-empty", "license"),
    ],
)
def test_captured_source_rejects_evidence_contract(valid_spec, mutation, match):
    evidence = valid_spec["evidence"]
    if mutation == "evidence-key":
        evidence["other"] = {"path": "benchmarks/other", "sha256": "1" * 64}
    elif mutation == "record-key":
        evidence["receipts"]["extra"] = True
    elif mutation == "unsafe-path":
        evidence["receipts"]["path"] = "../receipts.json"
    elif mutation == "wrong-path":
        evidence["receipts"]["path"] = "benchmarks/other.json"
    elif mutation == "zero-hash":
        evidence["receipts"]["sha256"] = ZERO_SHA256
    elif mutation == "uppercase-hash":
        evidence["receipts"]["sha256"] = "A" * 64
    elif mutation == "license-order":
        evidence["licenses"][:2] = reversed(evidence["licenses"][:2])
    else:
        evidence["licenses"] = []
    with pytest.raises(BenchmarkValidationError, match=match):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("id", "../capture", "case id"),
        ("id", "capture_0000000000000000", "case id"),
        ("path", "../image.png", "case path"),
        ("path", "images/other.png", "case path"),
        ("suite", "main", "suite"),
        ("split", "development", "split"),
        ("layout", "board_crop", "layout"),
        ("quality", "browser_scaled", "quality"),
        ("channel_mode", "RGBA", "channel_mode"),
        ("image_mode", "L", "image_mode"),
        ("image_size", [1, 1], "image_size"),
        ("page_kind", "privacy", "page_kind"),
        ("piece_set", "pychess_firi", "piece_set"),
        ("board_theme", "blue", "board_theme"),
        ("site", "pychess", "site"),
        ("viewport", "compact_scaled", "viewport"),
    ],
)
def test_captured_source_rejects_positive_case_contract(
    valid_spec, field, replacement, match
):
    _positive(valid_spec)[field] = replacement
    with pytest.raises(BenchmarkValidationError, match=match):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://lichess.org/editor",
        "https://evil.example/editor",
        "https://lichess.org/analysis",
        "https://lichess.org/editor?fen=start",
        "https://lichess.org/editor#board",
        "https://user@lichess.org/editor",
    ],
)
def test_captured_source_rejects_unsafe_or_unapproved_positive_urls(valid_spec, url):
    _positive(valid_spec)["source_url"] = url
    with pytest.raises(BenchmarkValidationError, match="source URL"):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://lichess.org/terms-of-service",
        "https://evil.example/terms-of-service",
        "https://lichess.org/terms",
        "https://lichess.org/terms-of-service?x=1",
        "https://lichess.org/terms-of-service#x",
    ],
)
def test_captured_source_rejects_unsafe_or_unapproved_negative_urls(valid_spec, url):
    _negative(valid_spec)["source_url"] = url
    with pytest.raises(BenchmarkValidationError, match="source URL"):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


@pytest.mark.parametrize(
    ("fen", "match"),
    [
        ("not a FEN", "FEN"),
        ("8/8/8/8/8/8/8/K7 w - - 0 1", "invalid FEN"),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 00 1",
            "noncanonical FEN",
        ),
    ],
)
def test_captured_source_rejects_invalid_or_noncanonical_full_fen(
    valid_spec, fen, match
):
    positive = _positive(valid_spec)
    positive["fen"] = fen
    with pytest.raises(BenchmarkValidationError, match=match):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


def test_captured_source_rejects_duplicate_and_rotated_base_positions(valid_spec):
    duplicate = copy.deepcopy(valid_spec)
    start_fen = captured.POSITIONS["start"]
    start = start_fen.split()[0]
    for case in duplicate["cases"]:
        if case.get("base_position") == "najdorf":
            case["fen"] = start_fen
            case["canonical_placement"] = start
            case["source_placement"] = (
                start if case["view"] == "white_bottom" else rotate_placement(start)
            )
    with pytest.raises(BenchmarkValidationError, match="duplicate position"):
        captured.validate_captured_source_spec(
            duplicate, ROOT, verify_local_assets=False
        )

    rotated = copy.deepcopy(valid_spec)
    base_fen = captured.POSITIONS["pawn_endgame"]
    rotated_placement = rotate_placement(base_fen.split()[0])
    rotated_fen = f"{rotated_placement} w - - 0 1"
    for case in rotated["cases"]:
        if case.get("base_position") == "nimzo":
            case["fen"] = rotated_fen
            case["canonical_placement"] = rotated_placement
            case["source_placement"] = (
                rotated_placement
                if case["view"] == "white_bottom"
                else rotate_placement(rotated_placement)
            )
    with pytest.raises(BenchmarkValidationError, match="rotated duplicate position"):
        captured.validate_captured_source_spec(
            rotated, ROOT, verify_local_assets=False
        )


@pytest.mark.parametrize(
    "rect",
    [
        [0, 0, 0, 0],
        [10, 10, 100, 99],
        [-1, 10, 100, 100],
        [10, 10, True, 100],
        [1400, 10, 100, 100],
        [10, 10, 100],
    ],
)
def test_captured_source_rejects_invalid_board_rectangles(valid_spec, rect):
    _positive(valid_spec)["board_rect"] = rect
    with pytest.raises(BenchmarkValidationError, match="board_rect"):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


def test_captured_source_rejects_positive_negative_field_sets(valid_spec):
    missing_positive = copy.deepcopy(valid_spec)
    del _positive(missing_positive)["fen"]
    with pytest.raises(BenchmarkValidationError, match="positive fields"):
        captured.validate_captured_source_spec(
            missing_positive, ROOT, verify_local_assets=False
        )

    leaked_positive = copy.deepcopy(valid_spec)
    _positive(leaked_positive)["negative_page"] = "/privacy"
    with pytest.raises(BenchmarkValidationError, match="positive fields"):
        captured.validate_captured_source_spec(
            leaked_positive, ROOT, verify_local_assets=False
        )

    missing_negative = copy.deepcopy(valid_spec)
    del _negative(missing_negative)["negative_page"]
    with pytest.raises(BenchmarkValidationError, match="negative fields"):
        captured.validate_captured_source_spec(
            missing_negative, ROOT, verify_local_assets=False
        )


def test_captured_source_rejects_case_count_and_matrix_cell_drift(valid_spec):
    missing = copy.deepcopy(valid_spec)
    missing["cases"].pop()
    with pytest.raises(BenchmarkValidationError, match="case count"):
        captured.validate_captured_source_spec(
            missing, ROOT, verify_local_assets=False
        )

    duplicate = copy.deepcopy(valid_spec)
    positives = [case for case in duplicate["cases"] if case["board_present"]]
    kept = {"id": positives[0]["id"], "path": positives[0]["path"]}
    positives[0].clear()
    positives[0].update(copy.deepcopy(positives[4]))
    positives[0].update(kept)
    with pytest.raises(BenchmarkValidationError, match="positive matrix"):
        captured.validate_captured_source_spec(
            duplicate, ROOT, verify_local_assets=False
        )


def test_captured_source_rejects_duplicate_ids_and_paths(valid_spec):
    valid_spec["cases"][1]["id"] = valid_spec["cases"][0]["id"]
    with pytest.raises(BenchmarkValidationError, match="duplicate case id"):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )

    valid_spec = copy.deepcopy(valid_spec)
    valid_spec["cases"][1]["id"] = captured.build_capture_plan()["cases"][1]["id"]
    valid_spec["cases"][1]["path"] = valid_spec["cases"][0]["path"]
    with pytest.raises(BenchmarkValidationError, match="duplicate case path"):
        captured.validate_captured_source_spec(
            valid_spec, ROOT, verify_local_assets=False
        )


def test_capture_plan_placeholders_are_not_a_final_source():
    with pytest.raises(BenchmarkValidationError, match="evidence sha256"):
        captured.validate_captured_source_spec(
            captured.build_capture_plan(), ROOT, verify_local_assets=False
        )


def test_captured_acceptance_returns_auditable_ordered_perfect_result(valid_spec):
    scores = score_predictions(valid_spec["cases"], _perfect_predictions(valid_spec))
    assert captured.evaluate_captured_acceptance(
        captured.CAPTURED_ACCEPTANCE_POLICY, scores
    ) == {
        "passed": True,
        "requirements": [
            {
                "metric": "exact_placement",
                "required": {"correct": 32, "total": 32},
                "actual": {"correct": 32, "total": 32},
                "passed": True,
            },
            {
                "metric": "negative_no_board_rate",
                "required": {"correct": 8, "total": 8},
                "actual": {"correct": 8, "total": 8},
                "passed": True,
            },
            {
                "metric": "positive_false_no_board_rate",
                "required": {"correct": 0, "total": 32},
                "actual": {"correct": 0, "total": 32},
                "passed": True,
            },
            {
                "metric": "execution_error_count",
                "required": 0,
                "actual": 0,
                "passed": True,
            },
        ],
    }


@pytest.mark.parametrize(
    ("metric", "replacement"),
    [
        ("exact_placement", {"correct": 31, "total": 32}),
        ("negative_no_board_rate", {"correct": 7, "total": 8}),
        ("positive_false_no_board_rate", {"correct": 1, "total": 32}),
        ("execution_error_count", 1),
    ],
)
def test_captured_acceptance_fails_each_strict_requirement(
    valid_spec, metric, replacement
):
    scores = score_predictions(valid_spec["cases"], _perfect_predictions(valid_spec))
    if metric == "execution_error_count":
        scores["overall"][metric] = replacement
    else:
        scores["overall"][metric].update(replacement)
    result = captured.evaluate_captured_acceptance(
        captured.CAPTURED_ACCEPTANCE_POLICY, scores
    )
    assert result["passed"] is False
    failed = [item["metric"] for item in result["requirements"] if not item["passed"]]
    assert failed == [metric]


def test_captured_scores_group_by_site_and_viewport(valid_spec):
    scores = score_predictions(valid_spec["cases"], _perfect_predictions(valid_spec))
    assert list(scores["groups"]["site"]) == ["lichess", "pychess"]
    assert list(scores["groups"]["viewport"]) == [
        "compact_scaled",
        "desktop_clean",
    ]
    for field in ("site", "viewport"):
        for group in scores["groups"][field].values():
            assert group["counts"]["all"] == 20
            assert group["exact_placement"] == {
                "correct": 16,
                "total": 16,
                "value": 1,
            }
            assert group["negative_no_board_rate"] == {
                "correct": 4,
                "total": 4,
                "value": 1,
            }


def test_browser_png_accepts_only_ihdr_contiguous_idat_iend(rgb_png):
    chunks = _png_chunks(rgb_png)
    rebuilt = []
    for chunk_type, payload, raw in chunks:
        if chunk_type == b"IDAT":
            middle = len(payload) // 2
            rebuilt.extend(
                [
                    _png_chunk(b"IDAT", payload[:middle]),
                    _png_chunk(b"IDAT", payload[middle:]),
                ]
            )
        else:
            rebuilt.append(raw)
    encoded = PNG_SIGNATURE + b"".join(rebuilt)
    assert captured.validate_browser_png(
        encoded, mode="RGB", size=(8, 8)
    ) == _pixel_hash(rgb_png)


@pytest.mark.parametrize("chunk_type", [b"tEXt", b"eXIf", b"iCCP", b"vpAg"])
def test_browser_png_rejects_known_and_unknown_ancillary_chunks(
    rgb_png, chunk_type
):
    mutated = _insert_valid_crc_chunk_before_iend(
        rgb_png, chunk_type, b"sensitive"
    )
    with pytest.raises(BenchmarkValidationError, match="PNG chunk sequence"):
        captured.validate_browser_png(mutated, mode="RGB", size=(8, 8))


@pytest.mark.parametrize(
    "mutation",
    [
        "signature",
        "length",
        "crc",
        "duplicate-ihdr",
        "misordered-ihdr",
        "no-idat",
        "fragmented-idat",
        "nonzero-iend",
        "trailing-bytes",
        "decode-failure",
    ],
)
def test_browser_png_rejects_malformed_chunk_streams(rgb_png, mutation):
    chunks = _png_chunks(rgb_png)
    ihdr = next(raw for kind, _payload, raw in chunks if kind == b"IHDR")
    idat_payload = next(
        payload for kind, payload, _raw in chunks if kind == b"IDAT"
    )
    iend = next(raw for kind, _payload, raw in chunks if kind == b"IEND")
    if mutation == "signature":
        encoded = b"not a png" + rgb_png[len(PNG_SIGNATURE) :]
    elif mutation == "length":
        encoded = PNG_SIGNATURE + struct.pack(">I", len(rgb_png)) + b"IHDR"
    elif mutation == "crc":
        encoded = bytearray(rgb_png)
        crc_offset = len(PNG_SIGNATURE) + 8 + 13
        encoded[crc_offset] ^= 1
        encoded = bytes(encoded)
    elif mutation == "duplicate-ihdr":
        encoded = PNG_SIGNATURE + ihdr + ihdr + _png_chunk(b"IDAT", idat_payload) + iend
    elif mutation == "misordered-ihdr":
        encoded = PNG_SIGNATURE + _png_chunk(b"IDAT", idat_payload) + ihdr + iend
    elif mutation == "no-idat":
        encoded = PNG_SIGNATURE + ihdr + iend
    elif mutation == "fragmented-idat":
        middle = len(idat_payload) // 2
        encoded = (
            PNG_SIGNATURE
            + ihdr
            + _png_chunk(b"IDAT", idat_payload[:middle])
            + _png_chunk(b"IHDR", b"\0" * 13)
            + _png_chunk(b"IDAT", idat_payload[middle:])
            + iend
        )
    elif mutation == "nonzero-iend":
        encoded = PNG_SIGNATURE + ihdr + _png_chunk(b"IDAT", idat_payload) + _png_chunk(b"IEND", b"x")
    elif mutation == "trailing-bytes":
        encoded = rgb_png + b"trailing"
    else:
        replacement = b"not-a-zlib-stream" if mutation == "decode-failure" else idat_payload
        encoded = PNG_SIGNATURE + ihdr + _png_chunk(b"IDAT", replacement) + iend
    with pytest.raises(BenchmarkValidationError):
        captured.validate_browser_png(encoded, mode="RGB", size=(8, 8))


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_browser_png_rejects_non_rgb_modes(mode):
    encoded = _image_bytes(mode=mode)
    with pytest.raises(BenchmarkValidationError, match="mode"):
        captured.validate_browser_png(encoded, mode="RGB", size=(8, 8))


def test_browser_png_rejects_wrong_size(rgb_png):
    with pytest.raises(BenchmarkValidationError, match="size"):
        captured.validate_browser_png(rgb_png, mode="RGB", size=(9, 8))


@pytest.mark.parametrize("mutation", ["compressed-trailing", "decompressed-trailing"])
def test_browser_png_rejects_hidden_idat_payload(rgb_png, mutation):
    chunks = _png_chunks(rgb_png)
    compressed = b"".join(
        payload for chunk_type, payload, _raw in chunks if chunk_type == b"IDAT"
    )
    if mutation == "compressed-trailing":
        replacement = compressed + b"SENSITIVE-BYTES"
    else:
        replacement = zlib.compress(zlib.decompress(compressed) + b"SENSITIVE-BYTES")
    rebuilt = []
    replaced = False
    for chunk_type, payload, raw in chunks:
        if chunk_type == b"IDAT":
            if not replaced:
                rebuilt.append(_png_chunk(b"IDAT", replacement))
                replaced = True
        else:
            rebuilt.append(raw)
    with pytest.raises(BenchmarkValidationError, match="IDAT"):
        captured.validate_browser_png(
            PNG_SIGNATURE + b"".join(rebuilt), mode="RGB", size=(8, 8)
        )


def test_captured_evidence_accepts_exact_hash_bound_documents(captured_evidence):
    result = captured.validate_captured_evidence(
        captured_evidence["spec"], captured_evidence["root"]
    )
    assert result == {
        name: captured_evidence["documents"][name]
        for name in (
            "receipts",
            "annotations_a",
            "annotations_b",
            "review",
            "rights",
        )
    }
    captured.validate_captured_source_spec(
        captured_evidence["spec"],
        captured_evidence["root"],
        verify_local_assets=True,
    )


@pytest.mark.parametrize(
    "name", ["receipts", "annotations_a", "annotations_b", "review", "rights"]
)
def test_captured_evidence_requires_canonical_json(captured_evidence, name):
    encoded = b" " + canonical_json_bytes(captured_evidence["documents"][name])
    _write_evidence_bytes(captured_evidence, name, encoded)
    with pytest.raises(BenchmarkValidationError, match="canonical JSON"):
        captured.validate_captured_evidence(
            captured_evidence["spec"], captured_evidence["root"]
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("unsafe-path", "evidence path"),
        ("missing", "evidence"),
        ("symlink", "symlink"),
        ("hardlink", "identity"),
        ("attribution-hash", "hash"),
        ("license-hash", "hash"),
    ],
)
def test_captured_evidence_rejects_unsafe_or_unbound_files(
    captured_evidence, tmp_path, mutation, match
):
    spec = captured_evidence["spec"]
    root = captured_evidence["root"]
    receipts = root.joinpath(*spec["evidence"]["receipts"]["path"].split("/"))
    if mutation == "unsafe-path":
        spec["evidence"]["receipts"]["path"] = "../receipts.json"
    elif mutation == "missing":
        receipts.unlink()
    elif mutation == "symlink":
        target = tmp_path / "receipts.json"
        receipts.replace(target)
        receipts.symlink_to(target)
    elif mutation == "hardlink":
        annotations_a = root.joinpath(
            *spec["evidence"]["annotations_a"]["path"].split("/")
        )
        annotations_b = root.joinpath(
            *spec["evidence"]["annotations_b"]["path"].split("/")
        )
        annotations_b.unlink()
        os.link(annotations_a, annotations_b)
        spec["evidence"]["annotations_b"]["sha256"] = spec["evidence"][
            "annotations_a"
        ]["sha256"]
    elif mutation == "attribution-hash":
        path = root.joinpath(*spec["evidence"]["attribution"]["path"].split("/"))
        path.write_bytes(b"changed attribution\n")
    else:
        record = spec["evidence"]["licenses"][0]
        root.joinpath(*record["path"].split("/")).write_bytes(b"changed license\n")
    with pytest.raises(BenchmarkValidationError, match=match):
        captured.validate_captured_evidence(spec, root)


def test_captured_evidence_never_follows_swapped_parent_directory(
    captured_evidence, tmp_path, monkeypatch
):
    root = captured_evidence["root"]
    benchmarks = root / "benchmarks"
    moved_benchmarks = tmp_path / "moved-benchmarks"
    outside_benchmarks = tmp_path / "outside-benchmarks"
    shutil.copytree(benchmarks, outside_benchmarks)
    real_open = os.open
    swapped = False

    def swap_parent_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        candidate = Path(path)
        component_swap = candidate == Path("benchmarks") and not swapped
        if component_swap:
            benchmarks.rename(moved_benchmarks)
            benchmarks.symlink_to(outside_benchmarks, target_is_directory=True)
            swapped = True
        try:
            descriptor = real_open(path, flags, *args, **kwargs)
        except OSError:
            if component_swap:
                benchmarks.unlink()
                moved_benchmarks.rename(benchmarks)
                swapped = False
            raise
        return descriptor

    monkeypatch.setattr(captured.os, "open", swap_parent_before_open)
    try:
        with pytest.raises(BenchmarkValidationError, match="symlink"):
            captured.validate_captured_evidence(
                captured_evidence["spec"], captured_evidence["root"]
            )
    finally:
        if swapped:
            benchmarks.unlink()
            moved_benchmarks.rename(benchmarks)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("receipt-order", "receipt"),
        ("receipt-fields", "receipt"),
        ("receipt-source", "receipt"),
        ("negative-claim", "receipt"),
        ("annotation-inventory", "annotation"),
        ("annotation-fields", "annotation"),
        ("annotation-full-resolution", "full_resolution"),
        ("annotation-privacy", "privacy"),
        ("annotation-capture", "capture_condition"),
        ("annotation-disagreement", "agreement"),
        ("annotation-agreed-wrong", "consensus"),
        ("review-binding", "annotation"),
        ("review-ids", "case"),
        ("review-isolation", "isolation"),
        ("review-a-b", "a_b_agreement"),
        ("review-controlled", "controlled_input_agreement"),
        ("review-model", "model_output_used"),
        ("rights-fields", "rights"),
        ("rights-site-fields", "rights"),
        ("rights-license", "license"),
        ("rights-valid-record-swap", "license"),
        ("rights-asset", "asset"),
    ],
)
def test_captured_evidence_rejects_schema_inventory_and_agreement_drift(
    captured_evidence, mutation, match
):
    documents = captured_evidence["documents"]
    if mutation.startswith("receipt") or mutation == "negative-claim":
        captures = documents["receipts"]["captures"]
        if mutation == "receipt-order":
            captures.reverse()
        elif mutation == "receipt-fields":
            captures[0]["extra"] = True
        elif mutation == "receipt-source":
            captures[0]["final_url"] = "https://example.invalid/"
        else:
            next(
                receipt
                for receipt in captures
                if "no_complete_board" in receipt
            )["no_complete_board"] = False
        _write_evidence_json(captured_evidence, "receipts")
    elif mutation == "annotation-agreed-wrong":
        for name in ("annotations_a", "annotations_b"):
            annotation = next(
                case
                for case in documents[name]["cases"]
                if case["board_present"]
            )
            annotation["board_present"] = False
            del annotation["source_placement"]
            digest = _write_evidence_json(captured_evidence, name)
            documents["review"]["annotation_sha256"][name[-1]] = digest
        _write_evidence_json(captured_evidence, "review")
    elif mutation.startswith("annotation"):
        annotations = documents["annotations_b"]
        if mutation == "annotation-inventory":
            annotations["cases"].pop()
        elif mutation == "annotation-fields":
            annotations["cases"][0]["extra"] = True
        elif mutation == "annotation-disagreement":
            next(
                case for case in annotations["cases"] if case["board_present"]
            )["source_placement"] = "8/8/8/8/8/8/8/8"
        else:
            field = {
                "annotation-full-resolution": "full_resolution_pass",
                "annotation-privacy": "privacy_pass",
                "annotation-capture": "capture_condition_pass",
            }[mutation]
            annotations["cases"][0][field] = False
        digest = _write_evidence_json(captured_evidence, "annotations_b")
        documents["review"]["annotation_sha256"]["b"] = digest
        _write_evidence_json(captured_evidence, "review")
    elif mutation.startswith("review"):
        review = documents["review"]
        if mutation == "review-binding":
            review["annotation_sha256"]["a"] = "f" * 64
        elif mutation == "review-ids":
            review["case_ids"].pop()
        elif mutation == "review-isolation":
            review["review_isolation"].reverse()
        elif mutation == "review-a-b":
            review["a_b_agreement"] = False
        elif mutation == "review-controlled":
            review["controlled_input_agreement"] = False
        else:
            review["model_output_used"] = True
        _write_evidence_json(captured_evidence, "review")
    else:
        rights = documents["rights"]
        if mutation == "rights-fields":
            rights["extra"] = True
        elif mutation == "rights-site-fields":
            rights["sites"]["lichess"]["extra"] = True
        elif mutation == "rights-license":
            rights["sites"]["lichess"]["privacy"] = {
                "path": "benchmarks/licenses/not-in-source.txt",
                "sha256": "f" * 64,
            }
        elif mutation == "rights-valid-record-swap":
            rights["sites"]["lichess"]["terms"] = copy.deepcopy(
                rights["sites"]["lichess"]["privacy"]
            )
        else:
            rights["sites"]["lichess"]["piece_asset_id"] = "changed"
        _write_evidence_json(captured_evidence, "rights")
    with pytest.raises(BenchmarkValidationError, match=match):
        captured.validate_captured_evidence(
            captured_evidence["spec"], captured_evidence["root"]
        )


@pytest.mark.parametrize("require_minimal_png", [False, True])
def test_verified_image_bytes_accepts_exact_inventory_and_bytes(
    verified_image_dataset, require_minimal_png
):
    dataset, case, encoded = verified_image_dataset
    assert board_recognition.load_verified_image_bytes(
        dataset, [case], require_minimal_png=require_minimal_png
    ) == {"case": encoded}


@pytest.mark.parametrize("path_change", ["unlink", "replace"])
def test_verified_image_bytes_retains_original_after_path_changes(
    verified_image_dataset, path_change
):
    dataset, case, encoded = verified_image_dataset
    retained = board_recognition.load_verified_image_bytes(
        dataset, [case], require_minimal_png=False
    )
    image_path = dataset / case["path"]
    image_path.unlink()
    if path_change == "replace":
        image_path.write_bytes(_image_bytes(color=(9, 8, 7)))
    assert retained["case"] is encoded or retained["case"] == encoded
    assert _pixel_hash(retained["case"]) == case["pixel_sha256"]


def test_verified_image_bytes_rejects_mutation_before_loading(
    verified_image_dataset,
):
    dataset, case, _encoded = verified_image_dataset
    (dataset / case["path"]).write_bytes(_image_bytes(color=(9, 8, 7)))
    with pytest.raises(BenchmarkValidationError, match="file hash"):
        board_recognition.load_verified_image_bytes(
            dataset, [case], require_minimal_png=False
        )


@pytest.mark.parametrize("swap", ["symlink", "file"])
def test_verified_image_bytes_rejects_path_swap_before_open(
    verified_image_dataset, tmp_path, monkeypatch, swap
):
    dataset, case, _encoded = verified_image_dataset
    image_path = dataset / case["path"]
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(_image_bytes(color=(9, 8, 7)))
    real_open = os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path).name == image_path.name and not swapped:
            swapped = True
            image_path.unlink()
            if swap == "symlink":
                image_path.symlink_to(replacement)
            else:
                image_path.write_bytes(replacement.read_bytes())
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(board_recognition.os, "open", swap_before_open)
    with pytest.raises(BenchmarkValidationError):
        board_recognition.load_verified_image_bytes(
            dataset, [case], require_minimal_png=False
        )


def test_verified_image_bytes_pins_images_directory_during_parent_swap(
    verified_image_dataset, tmp_path, monkeypatch
):
    dataset, case, encoded = verified_image_dataset
    images = dataset / "images"
    moved_images = tmp_path / "moved-images"
    outside_images = tmp_path / "outside-images"
    outside_images.mkdir()
    outside_image = outside_images / "case.png"
    outside_image.write_bytes(encoded)
    real_open = os.open
    swapped = False
    opened_outside = False

    def swap_parent_before_image_open(path, flags, *args, **kwargs):
        nonlocal swapped, opened_outside
        if Path(path).name == "case.png" and not swapped:
            swapped = True
            images.rename(moved_images)
            images.symlink_to(outside_images, target_is_directory=True)
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path).name == "case.png":
            opened = os.fstat(descriptor)
            outside = os.stat(outside_image)
            opened_outside = (opened.st_dev, opened.st_ino) == (
                outside.st_dev,
                outside.st_ino,
            )
        return descriptor

    monkeypatch.setattr(board_recognition.os, "open", swap_parent_before_image_open)
    retained = board_recognition.load_verified_image_bytes(
        dataset, [case], require_minimal_png=False
    )
    assert retained == {"case": encoded}
    assert opened_outside is False


@pytest.mark.parametrize("require_minimal_png", [False, True])
@pytest.mark.parametrize(
    "mutation", ["extra-png", "extra-directory", "symlink", "missing", "hardlink"]
)
def test_verified_image_bytes_rejects_inventory_and_identity_drift(
    verified_image_dataset, tmp_path, require_minimal_png, mutation
):
    dataset, case, encoded = verified_image_dataset
    if mutation == "extra-png":
        (dataset / "images/extra.png").write_bytes(encoded)
    elif mutation == "extra-directory":
        extra = dataset / "extra"
        extra.mkdir()
        (extra / "file.txt").write_text("extra")
    elif mutation == "symlink":
        target = tmp_path / "target"
        target.write_text("target")
        (dataset / "extra-link").symlink_to(target)
    elif mutation == "missing":
        (dataset / case["path"]).unlink()
    else:
        os.link(dataset / case["path"], tmp_path / "image-hardlink.png")
    with pytest.raises(BenchmarkValidationError):
        board_recognition.load_verified_image_bytes(
            dataset, [case], require_minimal_png=require_minimal_png
        )


@pytest.mark.parametrize(
    "imports",
    [
        "import benchmarks.board_recognition; import benchmarks.captured_board_recognition",
        "import benchmarks.captured_board_recognition; import benchmarks.board_recognition",
    ],
)
def test_board_recognition_import_order_is_cycle_free(imports):
    completed = subprocess.run(
        [sys.executable, "-c", imports],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="session")
def captured_pngs():
    return {
        case["id"]: _image_bytes(
            size=tuple(case["image_size"]),
            color=(index + 1, (index * 7 + 3) % 256, (index * 13 + 5) % 256),
        )
        for index, case in enumerate(
            sorted(captured.build_capture_plan()["cases"], key=itemgetter("id"))
        )
    }


def _materialize_capture(bundle, capture_dir, captured_pngs):
    images = capture_dir / "images"
    images.mkdir(parents=True)
    receipts = {
        receipt["id"]: receipt
        for receipt in bundle["documents"]["receipts"]["captures"]
    }
    for identifier, encoded in captured_pngs.items():
        (images / f"{identifier}.png").write_bytes(encoded)
        receipts[identifier]["png_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_evidence_json(bundle, "receipts")


def _source_path(root):
    return root / "benchmarks/board_recognition_online_screenshot_v1.json"


@pytest.fixture
def capture_fixture(captured_evidence, tmp_path, captured_pngs):
    capture_dir = tmp_path / "capture"
    _materialize_capture(captured_evidence, capture_dir, captured_pngs)
    plan = captured.build_capture_plan()
    plan_path = tmp_path / "capture-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan))
    return {
        "bundle": captured_evidence,
        "capture_plan": plan,
        "capture_plan_path": plan_path,
        "receipts_path": captured_evidence["root"].joinpath(
            *captured.EVIDENCE_PATHS["receipts"].split("/")
        ),
        "capture_dir": capture_dir,
    }


@pytest.fixture
def assembly_fixture(capture_fixture):
    bundle = capture_fixture["bundle"]
    root = bundle["root"]
    review_path = root.joinpath(*captured.EVIDENCE_PATHS["review"].split("/"))
    review_path.unlink()
    arguments = {
        "capture_plan_path": capture_fixture["capture_plan_path"],
        "receipts_path": capture_fixture["receipts_path"],
        "capture_dir": capture_fixture["capture_dir"],
        "annotations_a_path": root.joinpath(
            *captured.EVIDENCE_PATHS["annotations_a"].split("/")
        ),
        "annotations_b_path": root.joinpath(
            *captured.EVIDENCE_PATHS["annotations_b"].split("/")
        ),
        "attribution_path": root.joinpath(
            *captured.EVIDENCE_PATHS["attribution"].split("/")
        ),
        "rights_path": root.joinpath(*captured.EVIDENCE_PATHS["rights"].split("/")),
        "license_paths": [
            root.joinpath(*relative.split("/"))
            for relative in sorted(captured.LICENSE_PATHS)
        ],
        "review_path": review_path,
        "source_spec_path": _source_path(root),
        "repo_root": root,
    }
    return {"bundle": bundle, "arguments": arguments}


@pytest.fixture
def reviewed_fixture(captured_evidence, tmp_path, captured_pngs):
    root = captured_evidence["root"]
    capture_dir = tmp_path / "reviewed-capture"
    _materialize_capture(captured_evidence, capture_dir, captured_pngs)
    source_spec_path = _source_path(root)
    source_spec_path.write_bytes(canonical_json_bytes(captured_evidence["spec"]))
    return {
        "source_spec_path": source_spec_path,
        "capture_dir": capture_dir,
        "destination": root / "benchmarks/board_recognition_online_screenshot_v1",
        "manifest_digest_path": (
            root / "benchmarks/board_recognition_online_screenshot_v1_manifest.sha256"
        ),
        "freeze_digest_path": (
            root / "benchmarks/board_recognition_online_screenshot_v1_freeze.sha256"
        ),
        "repo_root": root,
    }


def _rewrite_receipts(bundle):
    _write_evidence_json(bundle, "receipts")
    source = _source_path(bundle["root"])
    if source.exists():
        source.write_bytes(canonical_json_bytes(bundle["spec"]))


def _capture_receipt(bundle, identifier):
    return next(
        receipt
        for receipt in bundle["documents"]["receipts"]["captures"]
        if receipt["id"] == identifier
    )


def _same_size_cases(spec):
    first = spec["cases"][0]
    second = next(
        case
        for case in spec["cases"][1:]
        if case["image_size"] == first["image_size"]
    )
    return first, second


def _reencode_png(encoded, compress_level):
    output = BytesIO()
    with Image.open(BytesIO(encoded)) as image:
        image.load()
        image.save(output, format="PNG", compress_level=compress_level)
    return output.getvalue()


def _freeze_input_bytes(arguments):
    spec = json.loads(arguments["source_spec_path"].read_text())
    paths = [arguments["source_spec_path"]]
    for name, record in spec["evidence"].items():
        records = record if name == "licenses" else [record]
        paths.extend(
            arguments["repo_root"].joinpath(*item["path"].split("/"))
            for item in records
        )
    paths.extend((arguments["capture_dir"] / case["path"]) for case in spec["cases"])
    return {path: path.read_bytes() for path in paths}


def test_write_capture_plan_publishes_canonical_nonexistent_file(tmp_path):
    destination = tmp_path / "capture-plan.json"
    result = captured.write_capture_plan(destination)
    assert result == captured.build_capture_plan()
    assert destination.read_bytes() == canonical_json_bytes(result)
    with pytest.raises((FileExistsError, BenchmarkValidationError)):
        captured.write_capture_plan(destination)
    assert destination.read_bytes() == canonical_json_bytes(result)


def test_write_capture_plan_target_race_preserves_racer_and_cleans_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "capture-plan.json"
    real_publish = captured._publish_new_file

    def race(*args, **kwargs):
        destination.write_bytes(b"racer")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(captured, "_publish_new_file", race)
    with pytest.raises((FileExistsError, BenchmarkValidationError, OSError)):
        captured.write_capture_plan(destination)
    assert destination.read_bytes() == b"racer"
    assert list(tmp_path.iterdir()) == [destination]


def test_write_capture_plan_rejects_temp_source_swap_at_publication(
    tmp_path, monkeypatch
):
    destination = tmp_path / "capture-plan.json"
    real_publish = captured._publish_new_file

    def swap_source(temp_path, target, *args, **kwargs):
        temp_path.unlink()
        temp_path.write_bytes(b"evil\n")
        return real_publish(temp_path, target, *args, **kwargs)

    monkeypatch.setattr(captured, "_publish_new_file", swap_source)

    with pytest.raises(BenchmarkValidationError, match="published output identity"):
        captured.write_capture_plan(destination)

    assert destination.read_bytes() == b"evil\n"


def test_write_capture_plan_rejects_parent_move_during_publication(
    tmp_path, monkeypatch
):
    parent = tmp_path / "output"
    parent.mkdir()
    moved_parent = tmp_path / "moved-output"
    destination = parent / "capture-plan.json"
    real_publish = captured._publish_new_file

    def move_parent(source, target, *args, **kwargs):
        parent.rename(moved_parent)
        parent.symlink_to(moved_parent, target_is_directory=True)
        return real_publish(source, target, *args, **kwargs)

    monkeypatch.setattr(captured, "_publish_new_file", move_parent)

    with pytest.raises(BenchmarkValidationError, match="output parent"):
        captured._write_new_file(destination, b"canonical\n")

    assert not (moved_parent / destination.name).exists()


def test_write_capture_plan_rejects_symlinked_parent(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(BenchmarkValidationError, match="symlink"):
        captured.write_capture_plan(linked_parent / "capture-plan.json")
    assert list(real_parent.iterdir()) == []


def test_write_capture_plan_cleans_owned_output_after_postcondition_failure(
    tmp_path, monkeypatch
):
    destination = tmp_path / "capture-plan.json"

    def fail_verification(_path):
        raise BenchmarkValidationError("injected plan verification failure")

    monkeypatch.setattr(captured, "_load_capture_plan", fail_verification)
    with pytest.raises(BenchmarkValidationError, match="injected plan verification"):
        captured.write_capture_plan(destination)
    assert not os.path.lexists(destination)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("platform", "function_name", "expected_at_fdcwd"),
    [("darwin", "renameatx_np", -2), ("linux", "renameat2", -100)],
)
def test_directory_noreplace_rename_uses_platform_at_fdcwd(
    tmp_path, monkeypatch, platform, function_name, expected_at_fdcwd
):
    calls = []

    class Rename:
        def __call__(self, *arguments):
            calls.append(arguments)
            return 0

    rename = Rename()
    library = type("Library", (), {function_name: rename})()
    monkeypatch.setattr(captured.sys, "platform", platform)
    monkeypatch.setattr(captured.ctypes, "CDLL", lambda *_args, **_kwargs: library)
    captured._rename_directory_noreplace(tmp_path / "source", tmp_path / "target")
    assert calls[0][0] == expected_at_fdcwd
    assert calls[0][2] == expected_at_fdcwd


def test_verify_capture_staging_accepts_exact_once_read_inventory(capture_fixture):
    result = captured.verify_capture_staging(
        capture_fixture["capture_plan"],
        capture_fixture["receipts_path"],
        capture_fixture["capture_dir"],
    )
    assert len(result["buffers"]) == 40
    assert len(set(result["file_sha256"].values())) == 40
    assert len(set(result["pixel_sha256"].values())) == 40


def test_verify_capture_staging_opens_each_source_png_exactly_once(
    capture_fixture, monkeypatch
):
    real_open = os.open
    opened_pngs = []

    def counting_open(path, flags, *args, **kwargs):
        if Path(path).suffix == ".png":
            opened_pngs.append(Path(path).name)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(captured.os, "open", counting_open)
    captured.verify_capture_staging(
        capture_fixture["capture_plan"],
        capture_fixture["receipts_path"],
        capture_fixture["capture_dir"],
    )
    assert Counter(opened_pngs) == Counter(
        {Path(case["path"]).name: 1 for case in capture_fixture["capture_plan"]["cases"]}
    )


@pytest.mark.parametrize("duplicate_kind", ["encoded", "pixels"])
def test_verify_capture_staging_rejects_duplicate_encoded_or_pixel_images(
    capture_fixture, duplicate_kind
):
    bundle = capture_fixture["bundle"]
    first, second = _same_size_cases(bundle["spec"])
    first_path = capture_fixture["capture_dir"] / first["path"]
    second_path = capture_fixture["capture_dir"] / second["path"]
    encoded = first_path.read_bytes()
    replacement = encoded if duplicate_kind == "encoded" else _reencode_png(encoded, 9)
    assert duplicate_kind == "encoded" or replacement != encoded
    second_path.write_bytes(replacement)
    _capture_receipt(bundle, second["id"])["png_sha256"] = hashlib.sha256(
        replacement
    ).hexdigest()
    _rewrite_receipts(bundle)
    with pytest.raises(BenchmarkValidationError, match="duplicate"):
        captured.verify_capture_staging(
            capture_fixture["capture_plan"],
            capture_fixture["receipts_path"],
            capture_fixture["capture_dir"],
        )


def test_assemble_reviewed_source_uses_consensus_then_controlled_input(
    assembly_fixture
):
    arguments = assembly_fixture["arguments"]
    result = captured.assemble_reviewed_source(**arguments)
    source = result["source_spec"]
    review = result["review"]
    assert arguments["source_spec_path"].read_bytes() == canonical_json_bytes(source)
    assert arguments["review_path"].read_bytes() == canonical_json_bytes(review)
    assert review["a_b_agreement"] is True
    assert review["controlled_input_agreement"] is True
    assert review["model_output_used"] is False
    captured.validate_captured_source_spec(
        source, arguments["repo_root"], verify_local_assets=True
    )


def test_assemble_reviewed_source_rejects_a_b_before_controlled_comparison(
    assembly_fixture
):
    bundle = assembly_fixture["bundle"]
    arguments = assembly_fixture["arguments"]
    positive = next(
        case
        for case in bundle["documents"]["annotations_b"]["cases"]
        if case["board_present"]
    )
    positive["source_placement"] = "8/8/8/8/8/8/8/8"
    _write_evidence_json(bundle, "annotations_b")
    with pytest.raises(BenchmarkValidationError, match="A/B"):
        captured.assemble_reviewed_source(**arguments)
    assert not arguments["review_path"].exists()
    assert not arguments["source_spec_path"].exists()


def test_assemble_reviewed_source_rejects_rights_asset_mismatch(assembly_fixture):
    bundle = assembly_fixture["bundle"]
    arguments = assembly_fixture["arguments"]
    bundle["documents"]["rights"]["sites"]["lichess"][
        "piece_asset_id"
    ] = "mismatched-asset"
    _write_evidence_json(bundle, "rights")
    with pytest.raises(BenchmarkValidationError, match="asset"):
        captured.assemble_reviewed_source(**arguments)
    assert not arguments["review_path"].exists()
    assert not arguments["source_spec_path"].exists()


@pytest.mark.parametrize("output", ["review_path", "source_spec_path"])
def test_assemble_reviewed_source_refuses_preexisting_output_unchanged(
    assembly_fixture, output
):
    arguments = assembly_fixture["arguments"]
    arguments[output].write_bytes(b"preexisting")
    with pytest.raises((FileExistsError, BenchmarkValidationError)):
        captured.assemble_reviewed_source(**arguments)
    assert arguments[output].read_bytes() == b"preexisting"
    other = "source_spec_path" if output == "review_path" else "review_path"
    assert not os.path.lexists(arguments[other])


@pytest.mark.parametrize("failure", ["source-write", "source-race"])
def test_assemble_reviewed_source_transaction_cleans_only_owned_outputs(
    assembly_fixture, monkeypatch, failure
):
    arguments = assembly_fixture["arguments"]
    sentinel = arguments["repo_root"] / "unrelated-sentinel"
    sentinel.write_bytes(b"keep")
    inputs = {
        path: path.read_bytes()
        for name, path in arguments.items()
        if name.endswith("_path")
        and name not in {"review_path", "source_spec_path"}
        and path.is_file()
    }
    real_write = captured._write_new_file

    def fail_source(path, encoded):
        if path == arguments["source_spec_path"]:
            if failure == "source-race":
                path.write_bytes(b"racer")
                return real_write(path, encoded)
            raise OSError("injected source write failure")
        return real_write(path, encoded)

    monkeypatch.setattr(captured, "_write_new_file", fail_source)
    with pytest.raises((FileExistsError, OSError, BenchmarkValidationError)):
        captured.assemble_reviewed_source(**arguments)
    assert not arguments["review_path"].exists()
    if failure == "source-race":
        assert arguments["source_spec_path"].read_bytes() == b"racer"
    else:
        assert not arguments["source_spec_path"].exists()
    assert sentinel.read_bytes() == b"keep"
    assert {path: path.read_bytes() for path in inputs} == inputs


def test_freezer_preserves_every_png_byte(reviewed_fixture):
    result = captured.freeze_captured_corpus(**reviewed_fixture)
    for case in result["manifest"]["cases"]:
        source = reviewed_fixture["capture_dir"] / case["path"]
        frozen = reviewed_fixture["destination"] / case["path"]
        assert frozen.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "failure", ["image-copy", "manifest-write", "digest-write", "freeze-write", "rename"]
)
def test_freeze_captured_corpus_failure_cleans_only_invocation_owned_artifacts(
    reviewed_fixture, monkeypatch, failure
):
    sentinel = reviewed_fixture["repo_root"] / "unrelated-sentinel"
    sentinel.write_bytes(b"keep")
    inputs = _freeze_input_bytes(reviewed_fixture)
    real_staging_write = captured._write_staging_file
    real_new_file = captured._write_new_file

    def fail_staging(path, encoded):
        is_manifest = path.name == "manifest.json"
        if (failure == "manifest-write" and is_manifest) or (
            failure == "image-copy" and path.suffix == ".png"
        ):
            raise OSError(f"injected {failure}")
        return real_staging_write(path, encoded)

    def fail_new(path, encoded):
        if failure == "digest-write" and path == reviewed_fixture["manifest_digest_path"]:
            raise OSError("injected digest-write")
        if failure == "freeze-write" and path == reviewed_fixture["freeze_digest_path"]:
            raise OSError("injected freeze-write")
        return real_new_file(path, encoded)

    def fail_rename(_source, _destination):
        raise OSError("injected rename")

    monkeypatch.setattr(captured, "_write_staging_file", fail_staging)
    monkeypatch.setattr(captured, "_write_new_file", fail_new)
    if failure == "rename":
        monkeypatch.setattr(captured, "_rename_directory_noreplace", fail_rename)
    with pytest.raises(OSError, match="injected"):
        captured.freeze_captured_corpus(**reviewed_fixture)
    assert not os.path.lexists(reviewed_fixture["destination"])
    assert not os.path.lexists(reviewed_fixture["manifest_digest_path"])
    assert not os.path.lexists(reviewed_fixture["freeze_digest_path"])
    assert _freeze_input_bytes(reviewed_fixture) == inputs
    assert sentinel.read_bytes() == b"keep"
    assert not any(
        child.name.startswith(f".{reviewed_fixture['destination'].name}.staging-")
        for child in reviewed_fixture["destination"].parent.iterdir()
    )


@pytest.mark.parametrize("target", ["dataset", "manifest-digest", "freeze-digest"])
def test_freeze_captured_corpus_target_races_preserve_racer(
    reviewed_fixture, monkeypatch, target
):
    destination = reviewed_fixture["destination"]
    manifest_digest = reviewed_fixture["manifest_digest_path"]
    freeze_digest = reviewed_fixture["freeze_digest_path"]
    real_rename = captured._rename_directory_noreplace
    real_write = captured._write_new_file

    def race_rename(source, target_path):
        if target == "dataset":
            target_path.mkdir()
            (target_path / "racer").write_bytes(b"keep")
        return real_rename(source, target_path)

    def race_write(path, encoded):
        if (target == "manifest-digest" and path == manifest_digest) or (
            target == "freeze-digest" and path == freeze_digest
        ):
            path.write_bytes(b"racer")
        return real_write(path, encoded)

    monkeypatch.setattr(captured, "_rename_directory_noreplace", race_rename)
    monkeypatch.setattr(captured, "_write_new_file", race_write)
    with pytest.raises((FileExistsError, BenchmarkValidationError, OSError)):
        captured.freeze_captured_corpus(**reviewed_fixture)
    if target == "dataset":
        assert (destination / "racer").read_bytes() == b"keep"
    else:
        assert not os.path.lexists(destination)
    if target == "manifest-digest":
        assert manifest_digest.read_bytes() == b"racer"
    else:
        assert not os.path.lexists(manifest_digest)
    if target == "freeze-digest":
        assert freeze_digest.read_bytes() == b"racer"
    else:
        assert not os.path.lexists(freeze_digest)


@pytest.mark.parametrize(
    "output", ["destination", "manifest_digest_path", "freeze_digest_path"]
)
def test_freeze_captured_corpus_refuses_preexisting_output_unchanged(
    reviewed_fixture, output
):
    path = reviewed_fixture[output]
    if output == "destination":
        path.mkdir()
        (path / "sentinel").write_bytes(b"preexisting")
    else:
        path.write_bytes(b"preexisting")
    with pytest.raises((FileExistsError, BenchmarkValidationError)):
        captured.freeze_captured_corpus(**reviewed_fixture)
    if output == "destination":
        assert (path / "sentinel").read_bytes() == b"preexisting"
    else:
        assert path.read_bytes() == b"preexisting"
    for key in ("destination", "manifest_digest_path", "freeze_digest_path"):
        if key != output:
            assert not os.path.lexists(reviewed_fixture[key])


@pytest.mark.parametrize("output", ["dataset", "manifest-digest", "freeze-digest"])
def test_freeze_captured_corpus_rejects_symlinked_output_parent(
    reviewed_fixture, tmp_path, output
):
    root = reviewed_fixture["repo_root"]
    real_parent = root / f"real-{output}"
    real_parent.mkdir()
    linked_parent = root / f"linked-{output}"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    key = {
        "dataset": "destination",
        "manifest-digest": "manifest_digest_path",
        "freeze-digest": "freeze_digest_path",
    }[output]
    reviewed_fixture[key] = linked_parent / Path(reviewed_fixture[key]).name
    with pytest.raises(BenchmarkValidationError, match="symlink"):
        captured.freeze_captured_corpus(**reviewed_fixture)
    assert list(real_parent.iterdir()) == []
    for output_key in ("destination", "manifest_digest_path", "freeze_digest_path"):
        assert not os.path.lexists(reviewed_fixture[output_key])


@pytest.mark.parametrize("duplicate_kind", ["encoded", "pixels"])
def test_freeze_captured_corpus_rejects_duplicates_before_destination(
    reviewed_fixture, captured_evidence, duplicate_kind
):
    first, second = _same_size_cases(captured_evidence["spec"])
    first_path = reviewed_fixture["capture_dir"] / first["path"]
    second_path = reviewed_fixture["capture_dir"] / second["path"]
    encoded = first_path.read_bytes()
    replacement = encoded if duplicate_kind == "encoded" else _reencode_png(encoded, 9)
    assert duplicate_kind == "encoded" or replacement != encoded
    second_path.write_bytes(replacement)
    _capture_receipt(captured_evidence, second["id"])["png_sha256"] = hashlib.sha256(
        replacement
    ).hexdigest()
    _rewrite_receipts(captured_evidence)
    with pytest.raises(BenchmarkValidationError, match="duplicate"):
        captured.freeze_captured_corpus(**reviewed_fixture)
    assert not reviewed_fixture["destination"].exists()
    assert not reviewed_fixture["manifest_digest_path"].exists()
    assert not reviewed_fixture["freeze_digest_path"].exists()


def test_freeze_captured_corpus_writes_retained_buffer_after_source_swap(
    reviewed_fixture, monkeypatch
):
    spec = json.loads(reviewed_fixture["source_spec_path"].read_text())
    case = spec["cases"][0]
    source = reviewed_fixture["capture_dir"] / case["path"]
    original = source.read_bytes()
    replacement = _image_bytes(
        size=tuple(case["image_size"]), color=(250, 251, 252)
    )
    real_snapshot = captured._snapshot_capture_buffers

    def swap_after_read(*args, **kwargs):
        snapshot = real_snapshot(*args, **kwargs)
        source.unlink()
        source.write_bytes(replacement)
        return snapshot

    monkeypatch.setattr(captured, "_snapshot_capture_buffers", swap_after_read)
    result = captured.freeze_captured_corpus(**reviewed_fixture)
    frozen = reviewed_fixture["destination"] / case["path"]
    manifest_case = next(
        item for item in result["manifest"]["cases"] if item["id"] == case["id"]
    )
    assert source.read_bytes() == replacement
    assert frozen.read_bytes() == original
    assert manifest_case["file_sha256"] == hashlib.sha256(original).hexdigest()


def _expected_commitment_paths(arguments, spec):
    root = arguments["repo_root"]
    paths = {
        arguments["source_spec_path"].relative_to(root).as_posix(),
        arguments["manifest_digest_path"].relative_to(root).as_posix(),
        (arguments["destination"] / "manifest.json").relative_to(root).as_posix(),
    }
    paths.update(
        (arguments["destination"] / case["path"]).relative_to(root).as_posix()
        for case in spec["cases"]
    )
    for name, record in spec["evidence"].items():
        records = record if name == "licenses" else [record]
        paths.update(item["path"] for item in records)
    return paths


def test_freeze_commitment_is_exact_canonical_and_read_only(reviewed_fixture):
    result = captured.freeze_captured_corpus(**reviewed_fixture)
    before = {
        path.relative_to(reviewed_fixture["repo_root"]).as_posix(): path.read_bytes()
        for path in reviewed_fixture["repo_root"].rglob("*")
        if path.is_file()
    }
    digest = captured.verify_freeze_commitment(
        reviewed_fixture["source_spec_path"],
        reviewed_fixture["destination"],
        reviewed_fixture["manifest_digest_path"],
        reviewed_fixture["freeze_digest_path"],
        reviewed_fixture["repo_root"],
    )
    after = {
        path.relative_to(reviewed_fixture["repo_root"]).as_posix(): path.read_bytes()
        for path in reviewed_fixture["repo_root"].rglob("*")
        if path.is_file()
    }
    encoded = reviewed_fixture["freeze_digest_path"].read_bytes()
    lines = encoded.decode("ascii").splitlines()
    paths = [line[66:] for line in lines]
    spec = json.loads(reviewed_fixture["source_spec_path"].read_text())
    assert digest == result["freeze_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert set(paths) == _expected_commitment_paths(reviewed_fixture, spec)
    assert reviewed_fixture["freeze_digest_path"].relative_to(
        reviewed_fixture["repo_root"]
    ).as_posix() not in paths
    assert after == before


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "unsorted", "unsafe", "symlink", "hardlink"]
)
def test_freeze_commitment_rejects_inventory_and_identity_drift(
    reviewed_fixture, tmp_path, mutation
):
    captured.freeze_captured_corpus(**reviewed_fixture)
    freeze = reviewed_fixture["freeze_digest_path"]
    lines = freeze.read_bytes().splitlines(keepends=True)
    if mutation == "missing":
        freeze.write_bytes(b"".join(lines[1:]))
    elif mutation == "extra":
        sentinel = reviewed_fixture["repo_root"] / "sentinel"
        sentinel.write_bytes(b"sentinel")
        freeze.write_bytes(
            b"".join(lines)
            + f"{hashlib.sha256(b'sentinel').hexdigest()}  sentinel\n".encode()
        )
    elif mutation == "unsorted":
        freeze.write_bytes(b"".join(reversed(lines)))
    elif mutation == "unsafe":
        digest, _path = lines[0].decode("ascii").rstrip("\n").split("  ", 1)
        freeze.write_text(f"{digest}  ../outside\n", encoding="ascii")
    elif mutation == "symlink":
        source = reviewed_fixture["source_spec_path"]
        copy_path = tmp_path / "source-copy.json"
        copy_path.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(copy_path)
    else:
        source = reviewed_fixture["source_spec_path"]
        alias = tmp_path / "source-hardlink.json"
        os.link(source, alias)
    with pytest.raises(BenchmarkValidationError):
        captured.verify_freeze_commitment(
            reviewed_fixture["source_spec_path"],
            reviewed_fixture["destination"],
            reviewed_fixture["manifest_digest_path"],
            reviewed_fixture["freeze_digest_path"],
            reviewed_fixture["repo_root"],
        )


def _load_freezer_cli():
    return importlib.import_module(
        "scripts.data.freeze_online_board_screenshot_benchmark"
    )


def _cli_mode_arguments(root):
    common = {
        "plan": root / "capture-plan.json",
        "receipts": root / "receipts.json",
        "capture": root / "capture",
        "annotations_a": root / "annotations-a.json",
        "annotations_b": root / "annotations-b.json",
        "attribution": root / "ATTRIBUTION.md",
        "rights": root / "RIGHTS.json",
        "licenses": root / "benchmarks/licenses",
        "review": root / "review.json",
        "source": root / "source.json",
        "dataset": root / "dataset",
        "manifest_digest": root / "manifest.sha256",
        "freeze_digest": root / "freeze.sha256",
    }
    common["licenses"].mkdir(parents=True)
    return common, {
        "write": ["--write-plan", str(common["plan"])],
        "captures": [
            "--verify-captures",
            "--capture-plan",
            str(common["plan"]),
            "--receipts",
            str(common["receipts"]),
            "--capture-dir",
            str(common["capture"]),
        ],
        "assemble": [
            "--assemble-source",
            "--capture-plan",
            str(common["plan"]),
            "--receipts",
            str(common["receipts"]),
            "--capture-dir",
            str(common["capture"]),
            "--annotations-a",
            str(common["annotations_a"]),
            "--annotations-b",
            str(common["annotations_b"]),
            "--attribution",
            str(common["attribution"]),
            "--rights",
            str(common["rights"]),
            "--licenses-dir",
            str(common["licenses"]),
            "--review-output",
            str(common["review"]),
            "--source-output",
            str(common["source"]),
        ],
        "freeze": [
            "--freeze",
            "--source-spec",
            str(common["source"]),
            "--capture-dir",
            str(common["capture"]),
            "--dataset",
            str(common["dataset"]),
            "--manifest-digest",
            str(common["manifest_digest"]),
            "--freeze-digest",
            str(common["freeze_digest"]),
        ],
        "verify": [
            "--verify",
            "--source-spec",
            str(common["source"]),
            "--dataset",
            str(common["dataset"]),
            "--manifest-digest",
            str(common["manifest_digest"]),
            "--freeze-digest",
            str(common["freeze_digest"]),
        ],
    }


def test_cli_modes_dispatch_all_five_exact_success_paths(tmp_path, monkeypatch):
    cli = _load_freezer_cli()
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    paths, modes = _cli_mode_arguments(tmp_path)
    called = []

    monkeypatch.setattr(
        cli.captured,
        "write_capture_plan",
        lambda path: called.append(("write", path)) or {"cases": [None] * 40},
    )
    monkeypatch.setattr(
        cli.captured,
        "_load_capture_plan",
        lambda _path: captured.build_capture_plan(),
    )
    monkeypatch.setattr(
        cli.captured,
        "verify_capture_staging",
        lambda *args: called.append(("captures", args))
        or {"buffers": {str(index): b"x" for index in range(40)}},
    )
    monkeypatch.setattr(
        cli.captured,
        "assemble_reviewed_source",
        lambda *args: called.append(("assemble", args))
        or {"source_spec": {"expected_counts": captured.EXPECTED_COUNTS}},
    )
    monkeypatch.setattr(
        cli.captured,
        "freeze_captured_corpus",
        lambda *args: called.append(("freeze", args))
        or {
            "manifest": {"expected_counts": captured.EXPECTED_COUNTS},
            "freeze_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        cli.captured,
        "verify_freeze_commitment",
        lambda *args: called.append(("verify", args)) or "b" * 64,
    )
    for name in ("write", "captures", "assemble", "freeze", "verify"):
        assert cli.main(modes[name]) == 0
    assert [item[0] for item in called] == [
        "write",
        "captures",
        "assemble",
        "freeze",
        "verify",
    ]
    assemble_args = next(item[1] for item in called if item[0] == "assemble")
    assert assemble_args[7] == [
        tmp_path.joinpath(*relative.split("/"))
        for relative in sorted(captured.LICENSE_PATHS)
    ]
    assert paths["licenses"].resolve() == (tmp_path / "benchmarks/licenses").resolve()


@pytest.mark.parametrize(
    ("mode", "missing_option"),
    [
        ("captures", "--capture-plan"),
        ("captures", "--receipts"),
        ("captures", "--capture-dir"),
        ("assemble", "--capture-plan"),
        ("assemble", "--receipts"),
        ("assemble", "--capture-dir"),
        ("assemble", "--annotations-a"),
        ("assemble", "--annotations-b"),
        ("assemble", "--attribution"),
        ("assemble", "--rights"),
        ("assemble", "--licenses-dir"),
        ("assemble", "--review-output"),
        ("assemble", "--source-output"),
        ("freeze", "--source-spec"),
        ("freeze", "--capture-dir"),
        ("freeze", "--dataset"),
        ("freeze", "--manifest-digest"),
        ("freeze", "--freeze-digest"),
        ("verify", "--source-spec"),
        ("verify", "--dataset"),
        ("verify", "--manifest-digest"),
        ("verify", "--freeze-digest"),
    ],
)
def test_cli_mode_rejects_each_missing_dependency(
    tmp_path, monkeypatch, mode, missing_option
):
    cli = _load_freezer_cli()
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    _paths, modes = _cli_mode_arguments(tmp_path)
    arguments = list(modes[mode])
    index = arguments.index(missing_option)
    del arguments[index : index + 2]
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("mode", "irrelevant"),
    [
        ("write", ["--capture-dir", "unused"]),
        ("captures", ["--source-spec", "unused"]),
        ("assemble", ["--dataset", "unused"]),
        ("freeze", ["--capture-plan", "unused"]),
        ("verify", ["--capture-dir", "unused"]),
    ],
)
def test_cli_mode_rejects_mode_irrelevant_arguments(
    tmp_path, monkeypatch, mode, irrelevant
):
    cli = _load_freezer_cli()
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    _paths, modes = _cli_mode_arguments(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(modes[mode] + irrelevant)
    assert error.value.code == 2


def test_cli_mode_requires_exact_licenses_directory(tmp_path, monkeypatch):
    cli = _load_freezer_cli()
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    _paths, modes = _cli_mode_arguments(tmp_path)
    arguments = modes["assemble"]
    index = arguments.index("--licenses-dir") + 1
    wrong = tmp_path / "other-licenses"
    wrong.mkdir()
    arguments[index] = str(wrong)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
