from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import struct
import sys
import tempfile
import zlib
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Sequence
from urllib.parse import urlsplit

import chess
from PIL import Image

from benchmarks.board_recognition import (
    BenchmarkValidationError,
    _open_pinned_directory,
    _open_relative_regular_file,
    _reject_symlink_components,
    _same_json_value,
    canonical_json_bytes,
    load_verified_image_bytes,
    pixel_sha256,
    rotate_placement,
    runtime_fingerprint,
    source_spec_sha256,
    verify_manifest,
    verify_unanchored_manifest,
)


CAPTURED_VERSION = "online-screenshot-v1"
CAPTURED_CLAIM = "captured_online_screenshot_acceptance"
POSITIONS = {
    "start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "najdorf": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "nimzo": "rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7",
    "pawn_endgame": "8/5k2/8/3p4/3P4/4K3/8/8 w - - 0 1",
}
SITES = {
    "lichess": {
        "editor_url": "https://lichess.org/editor",
        "negative_urls": (
            "https://lichess.org/terms-of-service",
            "https://lichess.org/privacy",
        ),
        "piece_set": "lichess_chessnut",
        "board_theme": "brown",
    },
    "pychess": {
        "editor_url": "https://www.pychess.org/editor/chess",
        "negative_urls": (
            "https://www.pychess.org/terms",
            "https://www.pychess.org/privacy",
        ),
        "piece_set": "pychess_firi",
        "board_theme": "brown",
    },
}
VIEWPORTS = {
    "desktop_clean": {
        "width": 1440,
        "height": 900,
        "dpr": 1,
        "quality": "clean",
    },
    "compact_scaled": {
        "width": 800,
        "height": 600,
        "dpr": 1,
        "quality": "browser_scaled",
    },
}
CAPTURED_ACCEPTANCE_POLICY = {
    "exact_placement": {"correct": 32, "total": 32},
    "negative_no_board_rate": {"correct": 8, "total": 8},
    "positive_false_no_board_rate": {"correct": 0, "total": 32},
    "execution_error_count": 0,
}
LIMITATIONS = [
    "This suite covers anonymous Lichess and PyChess pages only.",
    "The compact condition is native responsive browser scaling, not postprocessing.",
    "Empty and invalid complete boards are outside online-screenshot-v1.",
    "No benchmark image or label may be used for training, calibration, checkpoint selection, or threshold selection.",
]
EXPECTED_COUNTS = {"all": 40, "positive": 32, "negative": 8}
EVIDENCE_PATHS = {
    "receipts": "benchmarks/board_recognition_online_screenshot_v1_capture_receipts.json",
    "annotations_a": "benchmarks/board_recognition_online_screenshot_v1_annotations_a.json",
    "annotations_b": "benchmarks/board_recognition_online_screenshot_v1_annotations_b.json",
    "review": "benchmarks/board_recognition_online_screenshot_v1_review.json",
    "attribution": "benchmarks/board_recognition_online_screenshot_v1_ATTRIBUTION.md",
    "rights": "benchmarks/board_recognition_online_screenshot_v1_RIGHTS.json",
}
CAPTURE_PROTOCOL = {
    "browser_context": "fresh_anonymous_isolated",
    "capture_mode": "visible_viewport",
    "full_page": False,
    "manual_sequential": True,
    "max_concurrency": 1,
    "automated_retry": False,
    "postprocessing": "none",
    "dom_reads": ["controlled_fen", "orientation", "style", "receipt_metadata"],
}
LICENSE_PATHS = [
    "benchmarks/licenses/AGPL-3.0-or-later.txt",
    "benchmarks/licenses/Apache-2.0.txt",
    "benchmarks/licenses/CC-BY-4.0.txt",
    "benchmarks/licenses/GPL-3.0-or-later.txt",
    "benchmarks/licenses/chessgroundx-GPL-3.0.txt",
    "benchmarks/licenses/lichess-chessnut-LICENSE.txt",
    "benchmarks/licenses/lichess-lila-COPYING.md",
    "benchmarks/licenses/lichess-privacy-2026-07-19.txt",
    "benchmarks/licenses/lichess-terms-2026-07-19.txt",
    "benchmarks/licenses/pychess-firi-LICENSE.txt",
    "benchmarks/licenses/pychess-privacy-2026-05-22.md",
    "benchmarks/licenses/pychess-terms-2026-05-22.md",
    "benchmarks/licenses/pychess-variants-COPYING.md",
]

_ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"capture_[0-9a-f]{16}")
_VIEWS = ("white_bottom", "black_bottom")
_TOP_LEVEL_KEYS = {
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
_EVIDENCE_KEYS = set(EVIDENCE_PATHS) | {"licenses"}
_COMMON_CASE_KEYS = {
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
_POSITIVE_ONLY_KEYS = {
    "base_position",
    "fen",
    "canonical_placement",
    "source_placement",
    "view",
    "piece_set",
    "board_theme",
    "board_rect",
}
_NEGATIVE_ONLY_KEYS = {"negative_page"}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_REVIEW_ISOLATION = [
    "reviewer_a_received_only_opaque_images_and_schema",
    "reviewer_b_received_only_opaque_images_and_schema",
]
_RECEIPT_COMMON_KEYS = {
    "id",
    "png_sha256",
    "captured_at_utc",
    "final_url",
    "title",
    "site",
    "page_kind",
    "viewport",
    "viewport_size",
    "dpr",
    "scroll",
    "visibility_state",
    "navigation_state",
    "user_agent",
    "platform",
    "language",
}
_RECEIPT_POSITIVE_KEYS = {
    "fen",
    "view",
    "board_rect",
    "piece_set",
    "board_theme",
    "piece_asset_id",
    "board_asset_id",
    "piece_asset_url_prefix",
    "board_style_signature",
    "upstream_revision",
}
_RECEIPT_NEGATIVE_KEYS = {"no_complete_board"}
_ASSET_IDENTITY_KEYS = (
    "piece_asset_id",
    "board_asset_id",
    "piece_asset_url_prefix",
    "board_style_signature",
    "upstream_revision",
)
_RIGHTS_EVIDENCE_KEYS = (
    "project_copying",
    "piece_license",
    "terms",
    "privacy",
)
_RIGHTS_PATHS = {
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
_ALLOWED_SOURCES = {
    site: {
        key: list(value) if key == "negative_urls" else value
        for key, value in settings.items()
    }
    for site, settings in SITES.items()
}


def _opaque_id(index: int) -> str:
    value = hashlib.sha256(f"{CAPTURED_VERSION}:{index:02d}".encode("ascii"))
    return f"capture_{value.hexdigest()[:16]}"


def _page_kind(path: str) -> str:
    return "terms" if "terms" in path else "privacy"


def _build_cases() -> list[dict]:
    cases = []
    for site_name, site in SITES.items():
        for base_position, fen in POSITIONS.items():
            canonical = fen.split()[0]
            for view in _VIEWS:
                source = (
                    canonical
                    if view == "white_bottom"
                    else rotate_placement(canonical)
                )
                for viewport_name, viewport in VIEWPORTS.items():
                    identifier = _opaque_id(len(cases))
                    cases.append(
                        {
                            "id": identifier,
                            "path": f"images/{identifier}.png",
                            "suite": "captured",
                            "split": "acceptance",
                            "board_present": True,
                            "site": site_name,
                            "viewport": viewport_name,
                            "layout": "full_page",
                            "quality": viewport["quality"],
                            "channel_mode": "RGB",
                            "image_mode": "RGB",
                            "image_size": [
                                viewport["width"] * viewport["dpr"],
                                viewport["height"] * viewport["dpr"],
                            ],
                            "source_url": site["editor_url"],
                            "page_kind": "editor",
                            "base_position": base_position,
                            "fen": fen,
                            "canonical_placement": canonical,
                            "source_placement": source,
                            "view": view,
                            "piece_set": site["piece_set"],
                            "board_theme": site["board_theme"],
                            "board_rect": [0, 0, 0, 0],
                        }
                    )
        for source_url in site["negative_urls"]:
            negative_page = urlsplit(source_url).path
            for viewport_name, viewport in VIEWPORTS.items():
                identifier = _opaque_id(len(cases))
                cases.append(
                    {
                        "id": identifier,
                        "path": f"images/{identifier}.png",
                        "suite": "captured",
                        "split": "acceptance",
                        "board_present": False,
                        "site": site_name,
                        "viewport": viewport_name,
                        "layout": "full_page",
                        "quality": viewport["quality"],
                        "channel_mode": "RGB",
                        "image_mode": "RGB",
                        "image_size": [
                            viewport["width"] * viewport["dpr"],
                            viewport["height"] * viewport["dpr"],
                        ],
                        "source_url": source_url,
                        "page_kind": _page_kind(negative_page),
                        "negative_page": negative_page,
                    }
                )
    return cases


def build_capture_plan() -> dict:
    evidence = {
        name: {"path": path, "sha256": _ZERO_SHA256}
        for name, path in EVIDENCE_PATHS.items()
    }
    evidence["licenses"] = [
        {"path": path, "sha256": _ZERO_SHA256} for path in sorted(LICENSE_PATHS)
    ]
    return {
        "schema_version": 1,
        "benchmark_version": CAPTURED_VERSION,
        "claim": CAPTURED_CLAIM,
        "base_position_count": len(POSITIONS),
        "limitations": list(LIMITATIONS),
        "expected_counts": dict(EXPECTED_COUNTS),
        "acceptance_policy": copy.deepcopy(CAPTURED_ACCEPTANCE_POLICY),
        "capture_protocol": copy.deepcopy(CAPTURE_PROTOCOL),
        "allowed_sources": copy.deepcopy(_ALLOWED_SOURCES),
        "capture_conditions": copy.deepcopy(VIEWPORTS),
        "evidence": evidence,
        "cases": _build_cases(),
    }


def _safe_path(value: object) -> bool:
    if not isinstance(value, str) or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _validate_hash_record(record: object, expected_path: str, label: str) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise BenchmarkValidationError(f"{label} evidence record keys differ")
    if not _safe_path(record["path"]) or record["path"] != expected_path:
        raise BenchmarkValidationError(f"{label} evidence path differs")
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or digest == _ZERO_SHA256
    ):
        raise BenchmarkValidationError(f"{label} evidence sha256 invalid")


def _validate_evidence(evidence: object) -> None:
    if not isinstance(evidence, dict) or set(evidence) != _EVIDENCE_KEYS:
        raise BenchmarkValidationError("evidence keys differ")
    for name, path in EVIDENCE_PATHS.items():
        _validate_hash_record(evidence[name], path, name)
    licenses = evidence["licenses"]
    if not isinstance(licenses, list) or not licenses:
        raise BenchmarkValidationError("license evidence must be a nonempty list")
    if any(not isinstance(record, dict) for record in licenses):
        raise BenchmarkValidationError("license evidence records must be objects")
    paths = [record.get("path") for record in licenses]
    if paths != sorted(LICENSE_PATHS):
        raise BenchmarkValidationError("license evidence paths or order differ")
    for record, path in zip(licenses, sorted(LICENSE_PATHS), strict=True):
        _validate_hash_record(record, path, "license")


def _validate_idat_stream(ihdr: bytes, compressed: bytes, size: Sequence[int]) -> None:
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    if (width, height) != tuple(size):
        raise BenchmarkValidationError("browser PNG size differs")
    if (bit_depth, color_type) != (8, 2):
        raise BenchmarkValidationError("browser PNG mode must be 8-bit RGB")
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise BenchmarkValidationError("browser PNG IHDR methods invalid")

    passes = (
        ((0, 0, 1, 1),)
        if interlace == 0
        else (
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        )
    )
    layouts = []
    for start_x, start_y, step_x, step_y in passes:
        pass_width = 0 if width <= start_x else (width - start_x + step_x - 1) // step_x
        pass_height = (
            0 if height <= start_y else (height - start_y + step_y - 1) // step_y
        )
        if pass_width and pass_height:
            layouts.append((pass_width * 3, pass_height))
    expected_length = sum((row_bytes + 1) * rows for row_bytes, rows in layouts)

    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(compressed, expected_length + 1)
        if decompressor.unconsumed_tail or len(decoded) > expected_length:
            raise BenchmarkValidationError("PNG IDAT scanline extent differs")
        decoded += decompressor.flush()
    except zlib.error as error:
        raise BenchmarkValidationError("PNG IDAT zlib stream invalid") from error
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or len(decoded) != expected_length
    ):
        raise BenchmarkValidationError("PNG IDAT stream is not exact")
    offset = 0
    for row_bytes, rows in layouts:
        for _row in range(rows):
            if decoded[offset] > 4:
                raise BenchmarkValidationError("PNG IDAT filter byte invalid")
            offset += row_bytes + 1


def validate_browser_png(
    encoded: bytes, *, mode: str, size: Sequence[int]
) -> str:
    if type(encoded) is not bytes or not encoded.startswith(_PNG_SIGNATURE):
        raise BenchmarkValidationError("invalid PNG signature")
    if mode != "RGB":
        raise BenchmarkValidationError("browser PNG mode must be RGB")
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes))
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise BenchmarkValidationError("browser PNG size is invalid")

    chunks = []
    idat_payloads = []
    ihdr = None
    offset = len(_PNG_SIGNATURE)
    while offset < len(encoded):
        if len(encoded) - offset < 12:
            raise BenchmarkValidationError("PNG chunk length exceeds input")
        length = struct.unpack(">I", encoded[offset : offset + 4])[0]
        chunk_type = encoded[offset + 4 : offset + 8]
        payload_end = offset + 8 + length
        end = payload_end + 4
        if end > len(encoded):
            raise BenchmarkValidationError("PNG chunk length exceeds input")
        payload = encoded[offset + 8 : payload_end]
        expected_crc = struct.unpack(">I", encoded[payload_end:end])[0]
        if zlib.crc32(chunk_type + payload) & 0xFFFFFFFF != expected_crc:
            raise BenchmarkValidationError("PNG chunk CRC differs")
        chunks.append((chunk_type, length))
        if chunk_type == b"IHDR" and ihdr is None:
            ihdr = payload
        elif chunk_type == b"IDAT":
            idat_payloads.append(payload)
        if chunk_type == b"IEND" and end != len(encoded):
            raise BenchmarkValidationError("PNG contains bytes after IEND")
        offset = end

    chunk_types = [chunk_type for chunk_type, _length in chunks]
    if (
        len(chunks) < 3
        or chunk_types[0] != b"IHDR"
        or chunk_types[-1] != b"IEND"
        or any(chunk_type != b"IDAT" for chunk_type in chunk_types[1:-1])
    ):
        raise BenchmarkValidationError("PNG chunk sequence must be IHDR, IDAT+, IEND")
    if chunks[0][1] != 13:
        raise BenchmarkValidationError("PNG IHDR length must be 13")
    if chunks[-1][1] != 0:
        raise BenchmarkValidationError("PNG IEND length must be zero")
    _validate_idat_stream(ihdr, b"".join(idat_payloads), size)

    try:
        with Image.open(BytesIO(encoded)) as image:
            if image.format != "PNG":
                raise BenchmarkValidationError("browser image must encode PNG")
            image.verify()
        with Image.open(BytesIO(encoded)) as image:
            image.load()
            if image.format != "PNG":
                raise BenchmarkValidationError("browser image must encode PNG")
            if image.mode != "RGB":
                raise BenchmarkValidationError("browser PNG mode must be RGB")
            if image.size != tuple(size):
                raise BenchmarkValidationError("browser PNG size differs")
            return pixel_sha256(image)
    except BenchmarkValidationError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise BenchmarkValidationError("could not decode browser PNG") from error


def _no_duplicate_json_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise BenchmarkValidationError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise BenchmarkValidationError(f"invalid JSON constant: {value}")


def _decode_canonical_evidence(encoded: bytes, label: str) -> dict:
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_no_duplicate_json_object,
            parse_constant=_reject_json_constant,
        )
    except BenchmarkValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise BenchmarkValidationError(f"invalid JSON in {label} evidence") from error
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} evidence must be a JSON object")
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise BenchmarkValidationError(
            f"{label} evidence is not canonical JSON"
        ) from error
    if encoded != canonical:
        raise BenchmarkValidationError(f"{label} evidence is not canonical JSON")
    return value


def _validate_rights(rights: dict, license_records: list[dict]) -> dict[str, dict]:
    if set(rights) != {
        "schema_version",
        "benchmark_version",
        "checked_at_utc",
        "sites",
    }:
        raise BenchmarkValidationError("rights evidence fields differ")
    if type(rights["schema_version"]) is not int or rights["schema_version"] != 1:
        raise BenchmarkValidationError("rights schema_version differs")
    if rights["benchmark_version"] != CAPTURED_VERSION:
        raise BenchmarkValidationError("rights benchmark_version differs")
    if not isinstance(rights["checked_at_utc"], str) or not rights["checked_at_utc"]:
        raise BenchmarkValidationError("rights checked_at_utc invalid")
    sites = rights["sites"]
    if not isinstance(sites, dict) or set(sites) != set(SITES):
        raise BenchmarkValidationError("rights site inventory differs")
    expected_fields = set(_ASSET_IDENTITY_KEYS) | set(_RIGHTS_EVIDENCE_KEYS)
    for site_name, site in sites.items():
        if not isinstance(site, dict) or set(site) != expected_fields:
            raise BenchmarkValidationError("rights site fields differ")
        if any(
            not isinstance(site[field], str) or not site[field]
            for field in _ASSET_IDENTITY_KEYS
        ):
            raise BenchmarkValidationError("rights asset identity invalid")
        for field in _RIGHTS_EVIDENCE_KEYS:
            record = site[field]
            expected = next(
                candidate
                for candidate in license_records
                if candidate["path"] == _RIGHTS_PATHS[site_name][field]
            )
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or not _same_json_value(record, expected)
            ):
                raise BenchmarkValidationError("rights license evidence differs")
    return sites


def _validate_receipts(receipts: dict, spec: dict, rights: dict[str, dict]) -> None:
    if set(receipts) != {"schema_version", "benchmark_version", "captures"}:
        raise BenchmarkValidationError("receipt evidence fields differ")
    if type(receipts["schema_version"]) is not int or receipts["schema_version"] != 1:
        raise BenchmarkValidationError("receipt schema_version differs")
    if receipts["benchmark_version"] != CAPTURED_VERSION:
        raise BenchmarkValidationError("receipt benchmark_version differs")
    captures = receipts["captures"]
    cases = sorted(spec["cases"], key=lambda case: case["id"])
    if not isinstance(captures, list) or [
        capture.get("id") if isinstance(capture, dict) else None
        for capture in captures
    ] != [case["id"] for case in cases]:
        raise BenchmarkValidationError("receipt case inventory or order differs")

    for receipt, case in zip(captures, cases, strict=True):
        expected_keys = _RECEIPT_COMMON_KEYS | (
            _RECEIPT_POSITIVE_KEYS
            if case["board_present"]
            else _RECEIPT_NEGATIVE_KEYS
        )
        if set(receipt) != expected_keys:
            raise BenchmarkValidationError("receipt fields differ")
        if (
            not isinstance(receipt["png_sha256"], str)
            or _SHA256_RE.fullmatch(receipt["png_sha256"]) is None
        ):
            raise BenchmarkValidationError("receipt PNG hash invalid")
        if any(
            not isinstance(receipt[field], str) or not receipt[field]
            for field in (
                "captured_at_utc",
                "title",
                "user_agent",
                "platform",
                "language",
            )
        ):
            raise BenchmarkValidationError("receipt browser metadata invalid")
        if (
            receipt["final_url"] != case["source_url"]
            or receipt["site"] != case["site"]
            or receipt["page_kind"] != case["page_kind"]
            or receipt["viewport"] != case["viewport"]
        ):
            raise BenchmarkValidationError("receipt source metadata differs")
        viewport = VIEWPORTS[case["viewport"]]
        if (
            not _same_json_value(
                receipt["viewport_size"],
                [viewport["width"], viewport["height"]],
            )
            or not _same_json_value(receipt["dpr"], viewport["dpr"])
            or receipt["visibility_state"] != "visible"
            or receipt["navigation_state"] != "complete"
        ):
            raise BenchmarkValidationError("receipt capture condition differs")
        scroll = receipt["scroll"]
        if (
            not isinstance(scroll, list)
            or len(scroll) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in scroll
            )
        ):
            raise BenchmarkValidationError("receipt scroll metadata invalid")
        if case["board_present"]:
            for field in ("fen", "view", "board_rect", "piece_set", "board_theme"):
                if not _same_json_value(receipt[field], case[field]):
                    raise BenchmarkValidationError("receipt board metadata differs")
            if any(
                receipt[field] != rights[case["site"]][field]
                for field in _ASSET_IDENTITY_KEYS
            ):
                raise BenchmarkValidationError("receipt asset identity differs")
        elif receipt["no_complete_board"] is not True:
            raise BenchmarkValidationError("negative receipt board claim differs")


def _validate_annotations(
    annotations: dict, reviewer: str, cases: list[dict]
) -> list[dict]:
    if set(annotations) != {
        "schema_version",
        "benchmark_version",
        "reviewer",
        "isolation",
        "cases",
    }:
        raise BenchmarkValidationError("annotation evidence fields differ")
    if (
        type(annotations["schema_version"]) is not int
        or annotations["schema_version"] != 1
        or annotations["benchmark_version"] != CAPTURED_VERSION
        or annotations["reviewer"] != reviewer
        or annotations["isolation"] != "opaque_images_only"
    ):
        raise BenchmarkValidationError("annotation header differs")
    records = annotations["cases"]
    if not isinstance(records, list) or [
        record.get("id") if isinstance(record, dict) else None for record in records
    ] != [case["id"] for case in cases]:
        raise BenchmarkValidationError("annotation case inventory or order differs")

    consensus = []
    for record, case in zip(records, cases, strict=True):
        present = record.get("board_present")
        if type(present) is not bool:
            raise BenchmarkValidationError("annotation board_present invalid")
        expected_keys = {
            "id",
            "board_present",
            "full_resolution_pass",
            "privacy_pass",
            "capture_condition_pass",
        } | ({"source_placement"} if present else set())
        if set(record) != expected_keys:
            raise BenchmarkValidationError("annotation case fields differ")
        for field in (
            "full_resolution_pass",
            "privacy_pass",
            "capture_condition_pass",
        ):
            if record[field] is not True:
                raise BenchmarkValidationError(f"annotation {field} must be true")
        if present and not isinstance(record["source_placement"], str):
            raise BenchmarkValidationError("annotation source placement invalid")
        consensus.append(
            {
                "id": record["id"],
                "board_present": record["board_present"],
                **(
                    {"source_placement": record["source_placement"]}
                    if present
                    else {}
                ),
            }
        )
    return consensus


def _validate_review(review: dict, spec: dict, case_ids: list[str]) -> None:
    if set(review) != {
        "schema_version",
        "benchmark_version",
        "annotation_sha256",
        "case_ids",
        "review_isolation",
        "a_b_agreement",
        "controlled_input_agreement",
        "model_output_used",
    }:
        raise BenchmarkValidationError("review evidence fields differ")
    if type(review["schema_version"]) is not int or review["schema_version"] != 1:
        raise BenchmarkValidationError("review schema_version differs")
    if review["benchmark_version"] != CAPTURED_VERSION:
        raise BenchmarkValidationError("review benchmark_version differs")
    expected_hashes = {
        "a": spec["evidence"]["annotations_a"]["sha256"],
        "b": spec["evidence"]["annotations_b"]["sha256"],
    }
    if not _same_json_value(review["annotation_sha256"], expected_hashes):
        raise BenchmarkValidationError("review annotation hash binding differs")
    if not _same_json_value(review["case_ids"], case_ids):
        raise BenchmarkValidationError("review case inventory differs")
    if not _same_json_value(review["review_isolation"], _REVIEW_ISOLATION):
        raise BenchmarkValidationError("review isolation differs")
    if review["a_b_agreement"] is not True:
        raise BenchmarkValidationError("review a_b_agreement must be true")
    if review["controlled_input_agreement"] is not True:
        raise BenchmarkValidationError(
            "review controlled_input_agreement must be true"
        )
    if review["model_output_used"] is not False:
        raise BenchmarkValidationError("review model_output_used must be false")


def validate_captured_evidence(spec: dict, repo_root: Path) -> dict:
    validate_captured_source_spec(spec, repo_root, verify_local_assets=False)
    repo_root = Path(repo_root)
    raw = {}
    identities = set()
    records = [
        (name, spec["evidence"][name])
        for name in EVIDENCE_PATHS
    ] + [
        (f"license:{index}", record)
        for index, record in enumerate(spec["evidence"]["licenses"])
    ]
    repo_fd = _open_pinned_directory(repo_root, "evidence repository")
    try:
        for name, record in records:
            descriptor = _open_relative_regular_file(
                repo_fd, PurePosixPath(record["path"]), "evidence file"
            )
            try:
                opened = os.fstat(descriptor)
                identity = opened.st_dev, opened.st_ino
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or identity in identities
                ):
                    raise BenchmarkValidationError(
                        "evidence file must have a distinct regular-file identity"
                    )
                blocks = []
                while block := os.read(descriptor, 1024 * 1024):
                    blocks.append(block)
                encoded = b"".join(blocks)
            except OSError as error:
                raise BenchmarkValidationError("could not read evidence file") from error
            finally:
                os.close(descriptor)
            if (
                hashlib.sha256(encoded).hexdigest()
                != record["sha256"]
            ):
                raise BenchmarkValidationError(f"{name} evidence hash mismatch")
            identities.add(identity)
            if name in EVIDENCE_PATHS:
                raw[name] = encoded
    finally:
        os.close(repo_fd)

    documents = {
        name: _decode_canonical_evidence(raw[name], name)
        for name in ("receipts", "annotations_a", "annotations_b", "review", "rights")
    }
    rights = _validate_rights(documents["rights"], spec["evidence"]["licenses"])
    _validate_receipts(documents["receipts"], spec, rights)
    cases = sorted(spec["cases"], key=lambda case: case["id"])
    annotations_a = _validate_annotations(documents["annotations_a"], "A", cases)
    annotations_b = _validate_annotations(documents["annotations_b"], "B", cases)
    if not _same_json_value(annotations_a, annotations_b):
        raise BenchmarkValidationError("annotation A/B agreement differs")
    for annotation, case in zip(annotations_a, cases, strict=True):
        expected = {
            "id": case["id"],
            "board_present": case["board_present"],
            **(
                {"source_placement": case["source_placement"]}
                if case["board_present"]
                else {}
            ),
        }
        if not _same_json_value(annotation, expected):
            raise BenchmarkValidationError(
                "annotation consensus differs from source cases"
            )
    _validate_review(documents["review"], spec, [case["id"] for case in cases])
    return documents


def _validate_fen(fen: object) -> chess.Board:
    if not isinstance(fen, str):
        raise BenchmarkValidationError("position FEN must be a string")
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise BenchmarkValidationError("malformed position FEN") from error
    if not board.is_valid():
        raise BenchmarkValidationError("invalid FEN")
    if board.fen(en_passant="fen") != fen:
        raise BenchmarkValidationError("noncanonical FEN")
    return board


def _validate_board_rect(rect: object, image_size: list[int]) -> None:
    if not isinstance(rect, list) or len(rect) != 4:
        raise BenchmarkValidationError("board_rect must have four coordinates")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in rect
    ):
        raise BenchmarkValidationError("board_rect coordinates invalid")
    x, y, width, height = rect
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or width != height
        or x + width > image_size[0]
        or y + height > image_size[1]
    ):
        raise BenchmarkValidationError("board_rect geometry invalid")


def _positive_key(case: dict) -> tuple:
    return (
        case["site"],
        case["base_position"],
        case["view"],
        case["viewport"],
    )


def _negative_key(case: dict) -> tuple:
    return case["site"], case["negative_page"], case["viewport"]


def _validate_position_inventory(fens: dict[str, str]) -> None:
    placements = set()
    rotated = set()
    for fen in fens.values():
        placement = fen.split()[0]
        if placement in placements:
            raise BenchmarkValidationError("duplicate position placement")
        if placement in rotated:
            raise BenchmarkValidationError("rotated duplicate position placement")
        placements.add(placement)
        rotated.add(rotate_placement(placement))
    if not _same_json_value(fens, POSITIONS):
        raise BenchmarkValidationError("base position FENs differ")


def _validate_cases(cases: object) -> None:
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise BenchmarkValidationError("cases must be a list of objects")
    if len(cases) != EXPECTED_COUNTS["all"]:
        raise BenchmarkValidationError("case count differs")
    if Counter(case.get("board_present") for case in cases) != Counter(
        {True: EXPECTED_COUNTS["positive"], False: EXPECTED_COUNTS["negative"]}
    ):
        raise BenchmarkValidationError("positive and negative case counts differ")

    ids = [case.get("id") for case in cases]
    if any(not isinstance(identifier, str) for identifier in ids):
        raise BenchmarkValidationError("case id invalid")
    if len(set(ids)) != len(ids):
        raise BenchmarkValidationError("duplicate case id")
    if any(_ID_RE.fullmatch(identifier) is None for identifier in ids):
        raise BenchmarkValidationError("case id invalid")
    expected_cases = _build_cases()
    expected_by_id = {case["id"]: case for case in expected_cases}
    if set(ids) != set(expected_by_id):
        raise BenchmarkValidationError("case id binding differs")

    paths = [case.get("path") for case in cases]
    if any(not isinstance(path, str) for path in paths):
        raise BenchmarkValidationError("case path invalid")
    if len(set(paths)) != len(paths):
        raise BenchmarkValidationError("duplicate case path")
    for case in cases:
        if not _safe_path(case["path"]):
            raise BenchmarkValidationError("case path invalid")

    positive_keys = []
    negative_keys = []
    fens = {}
    for case in cases:
        present = case.get("board_present")
        if type(present) is not bool:
            raise BenchmarkValidationError("board_present must be a boolean")
        expected_fields = _COMMON_CASE_KEYS | (
            _POSITIVE_ONLY_KEYS if present else _NEGATIVE_ONLY_KEYS
        )
        label = "positive" if present else "negative"
        if set(case) != expected_fields:
            raise BenchmarkValidationError(f"{label} fields differ")
        if case["suite"] != "captured":
            raise BenchmarkValidationError("suite differs")
        if case["split"] != "acceptance":
            raise BenchmarkValidationError("split differs")
        if case["layout"] != "full_page":
            raise BenchmarkValidationError("layout differs")
        if case["channel_mode"] != "RGB":
            raise BenchmarkValidationError("channel_mode differs")
        if case["image_mode"] != "RGB":
            raise BenchmarkValidationError("image_mode differs")

        site_name = case["site"]
        expected_case = expected_by_id[case["id"]]
        if site_name != expected_case["site"]:
            raise BenchmarkValidationError("site differs")
        viewport_name = case["viewport"]
        if viewport_name != expected_case["viewport"]:
            raise BenchmarkValidationError("viewport differs")
        site = SITES[site_name]
        viewport = VIEWPORTS[viewport_name]
        if case["quality"] != viewport["quality"]:
            raise BenchmarkValidationError("quality differs")
        image_size = [
            viewport["width"] * viewport["dpr"],
            viewport["height"] * viewport["dpr"],
        ]
        if not _same_json_value(case["image_size"], image_size):
            raise BenchmarkValidationError("image_size differs")

        if present:
            board = _validate_fen(case["fen"])
            canonical = board.board_fen()
            if case["canonical_placement"] != canonical:
                raise BenchmarkValidationError("canonical placement differs")
            view = case["view"]
            if view not in _VIEWS:
                raise BenchmarkValidationError("view differs")
            expected_source = (
                canonical if view == "white_bottom" else rotate_placement(canonical)
            )
            if case["source_placement"] != expected_source:
                raise BenchmarkValidationError("source placement differs")
            base_position = case["base_position"]
            if base_position not in POSITIONS:
                raise BenchmarkValidationError("base_position differs")
            if base_position in fens and fens[base_position] != case["fen"]:
                raise BenchmarkValidationError(
                    "base position FEN differs across cases"
                )
            fens[base_position] = case["fen"]
            if case["source_url"] != site["editor_url"]:
                raise BenchmarkValidationError("source URL differs")
            if case["page_kind"] != "editor":
                raise BenchmarkValidationError("page_kind differs")
            if case["piece_set"] != site["piece_set"]:
                raise BenchmarkValidationError("piece_set differs")
            if case["board_theme"] != site["board_theme"]:
                raise BenchmarkValidationError("board_theme differs")
            _validate_board_rect(case["board_rect"], image_size)
            positive_keys.append(_positive_key(case))
        else:
            negative_page = case["negative_page"]
            allowed_urls = {
                urlsplit(url).path: url for url in site["negative_urls"]
            }
            if negative_page not in allowed_urls:
                raise BenchmarkValidationError("negative_page differs")
            if case["source_url"] != allowed_urls[negative_page]:
                raise BenchmarkValidationError("source URL differs")
            if case["page_kind"] != _page_kind(negative_page):
                raise BenchmarkValidationError("page_kind differs")
            negative_keys.append(_negative_key(case))

    _validate_position_inventory(fens)
    expected_positive = Counter(
        _positive_key(case) for case in expected_cases if case["board_present"]
    )
    if Counter(positive_keys) != expected_positive:
        raise BenchmarkValidationError(
            "positive matrix differs (site/base_position/view/viewport)"
        )
    expected_negative = Counter(
        _negative_key(case) for case in expected_cases if not case["board_present"]
    )
    if Counter(negative_keys) != expected_negative:
        raise BenchmarkValidationError(
            "negative matrix differs (site/negative_page/viewport)"
        )
    expected_ids = {
        (_positive_key(case) if case["board_present"] else _negative_key(case)):
        case["id"]
        for case in expected_cases
    }
    for case in cases:
        key = _positive_key(case) if case["board_present"] else _negative_key(case)
        if case["id"] != expected_ids[key]:
            raise BenchmarkValidationError("case id binding differs")
        if case["path"] != f"images/{case['id']}.png":
            raise BenchmarkValidationError("case path invalid")


def validate_captured_source_spec(
    spec: dict, repo_root: Path, *, verify_local_assets: bool
) -> None:
    if not isinstance(spec, dict):
        raise BenchmarkValidationError("source specification must be an object")
    if set(spec) != _TOP_LEVEL_KEYS:
        raise BenchmarkValidationError("source keys differ")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise BenchmarkValidationError("schema_version must be exactly 1")
    if spec["benchmark_version"] != CAPTURED_VERSION:
        raise BenchmarkValidationError("benchmark_version differs")
    if spec["claim"] != CAPTURED_CLAIM:
        raise BenchmarkValidationError("claim differs")
    if (
        type(spec["base_position_count"]) is not int
        or spec["base_position_count"] != len(POSITIONS)
    ):
        raise BenchmarkValidationError("base_position_count differs")
    for field, expected in (
        ("limitations", LIMITATIONS),
        ("expected_counts", EXPECTED_COUNTS),
        ("acceptance_policy", CAPTURED_ACCEPTANCE_POLICY),
        ("capture_protocol", CAPTURE_PROTOCOL),
        ("allowed_sources", _ALLOWED_SOURCES),
        ("capture_conditions", VIEWPORTS),
    ):
        if not _same_json_value(spec[field], expected):
            raise BenchmarkValidationError(f"{field} differs")
    _validate_evidence(spec["evidence"])
    _validate_cases(spec["cases"])
    if verify_local_assets:
        validate_captured_evidence(spec, repo_root)


def build_captured_case_plan(spec: dict) -> list[dict]:
    validate_captured_source_spec(spec, Path(), verify_local_assets=False)
    return sorted(spec["cases"], key=lambda case: case["id"])


def evaluate_captured_acceptance(policy: dict, scores: dict) -> dict:
    overall = scores["overall"]
    actual = {
        metric: {
            "correct": overall[metric]["correct"],
            "total": overall[metric]["total"],
        }
        for metric in (
            "exact_placement",
            "negative_no_board_rate",
            "positive_false_no_board_rate",
        )
    } | {"execution_error_count": overall["execution_error_count"]}
    requirements = [
        {
            "metric": metric,
            "required": required,
            "actual": actual[metric],
            "passed": actual[metric] == required,
        }
        for metric, required in policy.items()
    ]
    return {
        "passed": all(item["passed"] for item in requirements),
        "requirements": requirements,
    }


_SOURCE_SPEC_PATH = "benchmarks/board_recognition_online_screenshot_v1.json"


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_capture_plan(capture_plan: dict) -> None:
    if not _same_json_value(capture_plan, build_capture_plan()):
        raise BenchmarkValidationError("capture plan differs from the locked plan")


def _read_regular_file(path: Path, label: str) -> tuple[bytes, tuple[int, int]]:
    path = _absolute_path(Path(path))
    _reject_symlink_components(path, label)
    parent_fd = _open_pinned_directory(path.parent, f"{label} parent")
    descriptor = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise BenchmarkValidationError(
                f"{label} must have a distinct regular-file identity"
            )
        blocks = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        return b"".join(blocks), (opened.st_dev, opened.st_ino)
    except BenchmarkValidationError:
        raise
    except OSError as error:
        raise BenchmarkValidationError(f"could not read {label}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _read_canonical_json_file(
    path: Path, label: str
) -> tuple[dict, bytes, tuple[int, int]]:
    encoded, identity = _read_regular_file(path, label)
    return _decode_canonical_evidence(encoded, label), encoded, identity


def _load_capture_plan(path: Path) -> dict:
    capture_plan, _encoded, _identity = _read_canonical_json_file(
        path, "capture plan"
    )
    _validate_capture_plan(capture_plan)
    return capture_plan


def _validate_new_destination(path: Path, label: str) -> Path:
    path = _absolute_path(Path(path))
    if not path.name:
        raise BenchmarkValidationError(f"{label} path is invalid")
    _reject_symlink_components(path.parent, f"{label} parent")
    parent_fd = _open_pinned_directory(path.parent, f"{label} parent")
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return path
        except OSError as error:
            raise BenchmarkValidationError(f"could not inspect {label}") from error
        raise FileExistsError(f"{label} already exists: {path}")
    finally:
        os.close(parent_fd)


def _owned_regular_file(path: Path, identity: tuple[int, int]) -> bool:
    try:
        current = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(current.st_mode) and (
        current.st_dev,
        current.st_ino,
    ) == identity


def _remove_owned_file(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is not None and _owned_regular_file(path, identity):
        path.unlink()


def _entry_is_owned_regular_file(
    directory_fd: int, name: str, identity: tuple[int, int]
) -> bool:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(current.st_mode) and (
        current.st_dev,
        current.st_ino,
    ) == identity


def _remove_owned_entry(
    directory_fd: int, name: str, identity: tuple[int, int] | None
) -> None:
    if identity is not None and _entry_is_owned_regular_file(
        directory_fd, name, identity
    ):
        os.unlink(name, dir_fd=directory_fd)


def _require_absent_entry(directory_fd: int, name: str, label: str) -> None:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise BenchmarkValidationError(f"could not inspect {label}") from error
    raise FileExistsError(f"{label} already exists")


def _require_pinned_parent(path: Path, directory_fd: int, label: str) -> None:
    _reject_symlink_components(path, label)
    current_fd = _open_pinned_directory(path, label)
    try:
        current = os.fstat(current_fd)
        pinned = os.fstat(directory_fd)
        if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
            raise BenchmarkValidationError(f"{label} identity changed")
    finally:
        os.close(current_fd)


def _open_owned_temp(directory_fd: int, destination_name: str) -> tuple[int, str]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(100):
        name = f".{destination_name}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=directory_fd), name
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique output temp file")


def _publish_new_file(
    owned_temp: Path,
    destination: Path,
    *,
    parent_fd: int | None = None,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    owned_temp = _absolute_path(Path(owned_temp))
    destination = _absolute_path(Path(destination))
    close_parent = parent_fd is None
    if owned_temp.parent != destination.parent:
        raise BenchmarkValidationError("published files must share one parent")
    if parent_fd is None:
        _reject_symlink_components(destination.parent, "output parent")
        parent_fd = _open_pinned_directory(destination.parent, "output parent")
    try:
        os.link(
            owned_temp.name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if expected_identity is not None and not _entry_is_owned_regular_file(
            parent_fd, destination.name, expected_identity
        ):
            raise BenchmarkValidationError("published output identity differs")
    finally:
        if close_parent:
            os.close(parent_fd)


def _write_new_file(
    path: Path,
    encoded: bytes,
    *,
    write_fn=None,
    publish_fn=None,
    pre_publish_fn=None,
    post_publish_fn=None,
) -> tuple[int, int]:
    if type(encoded) is not bytes:
        raise BenchmarkValidationError("published content must be bytes")
    path = _validate_new_destination(path, "output")
    parent_fd = _open_pinned_directory(path.parent, "output parent")
    descriptor = None
    try:
        _require_pinned_parent(path.parent, parent_fd, "output parent")
        _require_absent_entry(parent_fd, path.name, "output")
        descriptor, temp_name = _open_owned_temp(parent_fd, path.name)
        temp_path = path.parent / temp_name
        opened = os.fstat(descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        raise
    identity = opened.st_dev, opened.st_ino
    published = False
    try:
        if write_fn is None:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("could not write complete output")
                remaining = remaining[written:]
        else:
            write_fn(temp_path, encoded)
        os.fsync(descriptor)
        if not _entry_is_owned_regular_file(parent_fd, temp_name, identity):
            raise BenchmarkValidationError("output temp path is not owned")
        os.lseek(descriptor, 0, os.SEEK_SET)
        actual = b"".join(iter(lambda: os.read(descriptor, 1024 * 1024), b""))
        if actual != encoded:
            raise BenchmarkValidationError("output temp bytes differ")
        _require_pinned_parent(path.parent, parent_fd, "output parent")
        _require_absent_entry(parent_fd, path.name, "output")
        if pre_publish_fn is not None:
            pre_publish_fn()
            _require_pinned_parent(path.parent, parent_fd, "output parent")
            _require_absent_entry(parent_fd, path.name, "output")
        publisher = publish_fn or _publish_new_file
        if publisher is _publish_new_file:
            publisher(
                temp_path,
                path,
                parent_fd=parent_fd,
                expected_identity=identity,
            )
        else:
            publisher(temp_path, path)
        if not _entry_is_owned_regular_file(parent_fd, path.name, identity):
            raise BenchmarkValidationError("published output identity differs")
        published = True
        _require_pinned_parent(path.parent, parent_fd, "output parent")
        if post_publish_fn is not None:
            post_publish_fn()
            _require_pinned_parent(path.parent, parent_fd, "output parent")
        if not _entry_is_owned_regular_file(parent_fd, path.name, identity):
            raise BenchmarkValidationError("published output identity differs")
        _remove_owned_entry(parent_fd, temp_name, identity)
        return identity
    except BaseException:
        if published:
            _remove_owned_entry(parent_fd, path.name, identity)
        raise
    finally:
        _remove_owned_entry(parent_fd, temp_name, identity)
        os.close(descriptor)
        os.close(parent_fd)


def _write_staging_file(path: Path, encoded: bytes) -> None:
    if type(encoded) is not bytes:
        raise BenchmarkValidationError("staging content must be bytes")
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("could not write complete staging file")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _owned_directory_identity(path: Path) -> tuple[int, int]:
    current = os.lstat(path)
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise BenchmarkValidationError("owned path is not a real directory")
    return current.st_dev, current.st_ino


def _remove_owned_directory(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISLNK(current.st_mode)
        and stat.S_ISDIR(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        shutil.rmtree(path)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    source = _absolute_path(Path(source))
    destination = _absolute_path(Path(destination))
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        at_fdcwd = -2
        rename = library.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        at_fdcwd = -100
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise BenchmarkValidationError(
                "platform has no no-replace directory rename"
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            1,
        )
    else:
        raise BenchmarkValidationError(
            "platform has no supported no-replace directory rename"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), os.fspath(destination)
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def write_capture_plan(path: Path) -> dict:
    plan = build_capture_plan()
    identity = _write_new_file(path, canonical_json_bytes(plan))
    try:
        if not _same_json_value(_load_capture_plan(path), plan):
            raise BenchmarkValidationError("published capture plan differs")
        return plan
    except BaseException:
        _remove_owned_file(Path(path), identity)
        raise


def _receipt_bound_spec(
    spec: dict,
    receipts: dict,
    rights: dict[str, dict] | None,
) -> dict:
    captures = receipts.get("captures") if isinstance(receipts, dict) else None
    if not isinstance(captures, list) or any(
        not isinstance(receipt, dict) for receipt in captures
    ):
        raise BenchmarkValidationError("receipt captures must be objects")
    bound = copy.deepcopy(spec)
    bound_cases = sorted(bound["cases"], key=lambda case: case["id"])
    for receipt, case in zip(captures, bound_cases, strict=False):
        if case["board_present"]:
            _validate_board_rect(receipt.get("board_rect"), case["image_size"])
            if case["board_rect"] == [0, 0, 0, 0]:
                case["board_rect"] = copy.deepcopy(receipt["board_rect"])
            if any(
                not isinstance(receipt.get(field), str) or not receipt[field]
                for field in _ASSET_IDENTITY_KEYS
            ):
                raise BenchmarkValidationError("receipt asset identity invalid")
    if rights is None:
        rights = {}
        for site_name in SITES:
            receipt = next(
                (
                    item
                    for item in captures
                    if item.get("site") == site_name and "piece_asset_id" in item
                ),
                {},
            )
            rights[site_name] = {
                field: receipt.get(field) for field in _ASSET_IDENTITY_KEYS
            }
    _validate_receipts(receipts, bound, rights)
    return bound


def _snapshot_capture_buffers(
    spec: dict,
    receipts: dict,
    capture_dir: Path,
    rights: dict[str, dict] | None = None,
) -> dict:
    bound = _receipt_bound_spec(spec, receipts, rights)
    cases = sorted(bound["cases"], key=lambda case: case["id"])
    receipts_by_id = {receipt["id"]: receipt for receipt in receipts["captures"]}
    capture_fd = _open_pinned_directory(capture_dir, "capture directory")
    images_fd = None
    try:
        try:
            with os.scandir(capture_fd) as entries:
                root_entries = {
                    entry.name: entry.stat(follow_symlinks=False) for entry in entries
                }
        except OSError as error:
            raise BenchmarkValidationError(
                "could not inspect capture inventory"
            ) from error
        if set(root_entries) != {"images"} or not stat.S_ISDIR(
            root_entries["images"].st_mode
        ):
            raise BenchmarkValidationError("capture inventory differs")
        try:
            images_fd = os.open(
                "images",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=capture_fd,
            )
        except OSError as error:
            raise BenchmarkValidationError("could not open capture images") from error
        opened_images = os.fstat(images_fd)
        listed_images = root_entries["images"]
        if (opened_images.st_dev, opened_images.st_ino) != (
            listed_images.st_dev,
            listed_images.st_ino,
        ):
            raise BenchmarkValidationError("capture images identity changed")
        try:
            with os.scandir(images_fd) as entries:
                image_entries = {
                    entry.name: entry.stat(follow_symlinks=False) for entry in entries
                }
        except OSError as error:
            raise BenchmarkValidationError(
                "could not inspect capture images"
            ) from error
        expected_names = {PurePosixPath(case["path"]).name for case in cases}
        if set(image_entries) != expected_names:
            raise BenchmarkValidationError("capture image inventory differs")
        listed_identities = set()
        for entry in image_entries.values():
            identity = entry.st_dev, entry.st_ino
            if (
                not stat.S_ISREG(entry.st_mode)
                or entry.st_nlink != 1
                or identity in listed_identities
            ):
                raise BenchmarkValidationError(
                    "capture image must have a distinct regular-file identity"
                )
            listed_identities.add(identity)

        buffers = {}
        file_hashes = {}
        pixel_hashes = {}
        opened_identities = set()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        for case in cases:
            name = PurePosixPath(case["path"]).name
            try:
                descriptor = os.open(name, flags, dir_fd=images_fd)
            except OSError as error:
                raise BenchmarkValidationError("could not open capture PNG") from error
            try:
                opened = os.fstat(descriptor)
                listed = image_entries[name]
                identity = opened.st_dev, opened.st_ino
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or identity != (listed.st_dev, listed.st_ino)
                    or identity in opened_identities
                ):
                    raise BenchmarkValidationError("capture PNG identity changed")
                opened_identities.add(identity)
                blocks = []
                while block := os.read(descriptor, 1024 * 1024):
                    blocks.append(block)
                encoded = b"".join(blocks)
            except OSError as error:
                raise BenchmarkValidationError("could not read capture PNG") from error
            finally:
                os.close(descriptor)
            file_digest = hashlib.sha256(encoded).hexdigest()
            if file_digest != receipts_by_id[case["id"]]["png_sha256"]:
                raise BenchmarkValidationError("capture PNG receipt hash mismatch")
            pixel_digest = validate_browser_png(
                encoded, mode=case["image_mode"], size=case["image_size"]
            )
            buffers[case["id"]] = encoded
            file_hashes[case["id"]] = file_digest
            pixel_hashes[case["id"]] = pixel_digest
        if len(set(file_hashes.values())) != len(cases):
            raise BenchmarkValidationError("duplicate encoded capture PNG")
        if len(set(pixel_hashes.values())) != len(cases):
            raise BenchmarkValidationError("duplicate decoded capture pixels")
        return {
            "cases": cases,
            "buffers": buffers,
            "file_sha256": file_hashes,
            "pixel_sha256": pixel_hashes,
        }
    finally:
        if images_fd is not None:
            os.close(images_fd)
        os.close(capture_fd)


def verify_capture_staging(
    capture_plan: dict, receipts_path: Path, capture_dir: Path
) -> dict:
    _validate_capture_plan(capture_plan)
    receipts, receipts_bytes, _identity = _read_canonical_json_file(
        receipts_path, "capture receipts"
    )
    result = _snapshot_capture_buffers(capture_plan, receipts, capture_dir)
    result.update(
        {
            "receipts": receipts,
            "receipts_bytes": receipts_bytes,
        }
    )
    return result


def _repo_relative(path: Path, repo_root: Path, label: str) -> str:
    path = _absolute_path(Path(path))
    root = _absolute_path(Path(repo_root))
    if not path.is_relative_to(root) or path == root:
        raise BenchmarkValidationError(f"{label} must be inside the repository")
    relative = path.relative_to(root).as_posix()
    if not _safe_path(relative):
        raise BenchmarkValidationError(f"{label} repository path is unsafe")
    return relative


def _require_repo_path(
    path: Path, relative: str, repo_root: Path, label: str
) -> Path:
    expected = _absolute_path(Path(repo_root).joinpath(*relative.split("/")))
    if _absolute_path(Path(path)) != expected:
        raise BenchmarkValidationError(f"{label} path differs")
    return expected


def assemble_reviewed_source(
    capture_plan_path: Path,
    receipts_path: Path,
    capture_dir: Path,
    annotations_a_path: Path,
    annotations_b_path: Path,
    attribution_path: Path,
    rights_path: Path,
    license_paths: Sequence[Path],
    review_path: Path,
    source_spec_path: Path,
    repo_root: Path,
) -> dict:
    repo_root = _absolute_path(Path(repo_root))
    root_fd = _open_pinned_directory(repo_root, "repository")
    os.close(root_fd)
    expected_inputs = {
        "receipts": receipts_path,
        "annotations_a": annotations_a_path,
        "annotations_b": annotations_b_path,
        "attribution": attribution_path,
        "rights": rights_path,
    }
    for name, path in expected_inputs.items():
        _require_repo_path(path, EVIDENCE_PATHS[name], repo_root, name)
    review_path = _require_repo_path(
        review_path, EVIDENCE_PATHS["review"], repo_root, "review output"
    )
    source_spec_path = _require_repo_path(
        source_spec_path, _SOURCE_SPEC_PATH, repo_root, "source output"
    )
    if review_path == source_spec_path:
        raise BenchmarkValidationError("review and source outputs must differ")
    _validate_new_destination(review_path, "review output")
    _validate_new_destination(source_spec_path, "source output")

    if not isinstance(license_paths, Sequence) or isinstance(
        license_paths, (str, bytes)
    ):
        raise BenchmarkValidationError("license paths must be a sequence")
    expected_licenses = [
        _absolute_path(repo_root.joinpath(*relative.split("/")))
        for relative in sorted(LICENSE_PATHS)
    ]
    actual_licenses = [_absolute_path(Path(path)) for path in license_paths]
    if actual_licenses != expected_licenses:
        raise BenchmarkValidationError("license path inventory or order differs")

    capture_plan = _load_capture_plan(capture_plan_path)
    staging = verify_capture_staging(capture_plan, receipts_path, capture_dir)
    raw = {"receipts": staging["receipts_bytes"]}
    documents = {"receipts": staging["receipts"]}
    for name, path in (
        ("annotations_a", annotations_a_path),
        ("annotations_b", annotations_b_path),
        ("rights", rights_path),
    ):
        documents[name], raw[name], _identity = _read_canonical_json_file(
            path, name
        )
    raw["attribution"], _identity = _read_regular_file(
        attribution_path, "attribution"
    )
    if not raw["attribution"]:
        raise BenchmarkValidationError("attribution must not be empty")
    license_records = []
    for relative, path in zip(
        sorted(LICENSE_PATHS), expected_licenses, strict=True
    ):
        encoded, _identity = _read_regular_file(path, "license")
        license_records.append(
            {"path": relative, "sha256": hashlib.sha256(encoded).hexdigest()}
        )
    rights = _validate_rights(documents["rights"], license_records)
    bound = _receipt_bound_spec(capture_plan, documents["receipts"], rights)
    cases = sorted(bound["cases"], key=lambda case: case["id"])
    annotations_a = _validate_annotations(documents["annotations_a"], "A", cases)
    annotations_b = _validate_annotations(documents["annotations_b"], "B", cases)
    if not _same_json_value(annotations_a, annotations_b):
        raise BenchmarkValidationError("annotation A/B agreement differs")
    for annotation, case in zip(annotations_a, cases, strict=True):
        expected = {
            "id": case["id"],
            "board_present": case["board_present"],
            **(
                {"source_placement": case["source_placement"]}
                if case["board_present"]
                else {}
            ),
        }
        if not _same_json_value(annotation, expected):
            raise BenchmarkValidationError(
                f"annotation consensus differs from controlled input: {case['id']}"
            )
        case["board_present"] = annotation["board_present"]
        if case["board_present"]:
            case["source_placement"] = annotation["source_placement"]

    evidence = {
        name: {
            "path": EVIDENCE_PATHS[name],
            "sha256": hashlib.sha256(raw[name]).hexdigest(),
        }
        for name in ("receipts", "annotations_a", "annotations_b", "attribution", "rights")
    }
    evidence["licenses"] = license_records
    review = {
        "schema_version": 1,
        "benchmark_version": CAPTURED_VERSION,
        "annotation_sha256": {
            "a": evidence["annotations_a"]["sha256"],
            "b": evidence["annotations_b"]["sha256"],
        },
        "case_ids": [case["id"] for case in cases],
        "review_isolation": list(_REVIEW_ISOLATION),
        "a_b_agreement": True,
        "controlled_input_agreement": True,
        "model_output_used": False,
    }
    review_bytes = canonical_json_bytes(review)
    evidence["review"] = {
        "path": EVIDENCE_PATHS["review"],
        "sha256": hashlib.sha256(review_bytes).hexdigest(),
    }
    source_spec = copy.deepcopy(capture_plan)
    source_spec["evidence"] = evidence
    source_spec["cases"] = cases
    validate_captured_source_spec(
        source_spec, repo_root, verify_local_assets=False
    )
    _validate_review(review, source_spec, [case["id"] for case in cases])
    source_bytes = canonical_json_bytes(source_spec)

    review_identity = None
    source_identity = None
    try:
        review_identity = _write_new_file(review_path, review_bytes)
        source_identity = _write_new_file(source_spec_path, source_bytes)
        validate_captured_source_spec(
            source_spec, repo_root, verify_local_assets=True
        )
        return {"review": review, "source_spec": source_spec}
    except BaseException:
        _remove_owned_file(source_spec_path, source_identity)
        _remove_owned_file(review_path, review_identity)
        raise


def _commitment_paths(
    source_spec_path: Path,
    dataset_dir: Path,
    manifest_digest_path: Path,
    source_spec: dict,
    repo_root: Path,
) -> list[str]:
    dataset_relative = _repo_relative(dataset_dir, repo_root, "dataset")
    paths = [
        _repo_relative(source_spec_path, repo_root, "source specification"),
        _repo_relative(manifest_digest_path, repo_root, "manifest digest"),
        f"{dataset_relative}/manifest.json",
        *[
            f"{dataset_relative}/{case['path']}"
            for case in sorted(source_spec["cases"], key=lambda case: case["id"])
        ],
        *[source_spec["evidence"][name]["path"] for name in EVIDENCE_PATHS],
        *[
            record["path"]
            for record in source_spec["evidence"]["licenses"]
        ],
    ]
    if any(not _safe_path(path) for path in paths) or len(set(paths)) != len(paths):
        raise BenchmarkValidationError("commitment path inventory is unsafe or aliased")
    return sorted(paths)


def _read_commitment_files(
    repo_root: Path, paths: Sequence[str]
) -> tuple[dict[str, bytes], set[tuple[int, int]]]:
    repo_fd = _open_pinned_directory(repo_root, "commitment repository")
    encoded = {}
    identities = set()
    try:
        for relative in paths:
            if not _safe_path(relative):
                raise BenchmarkValidationError("unsafe commitment path")
            descriptor = _open_relative_regular_file(
                repo_fd, PurePosixPath(relative), "committed file"
            )
            try:
                opened = os.fstat(descriptor)
                identity = opened.st_dev, opened.st_ino
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or identity in identities
                ):
                    raise BenchmarkValidationError(
                        "committed file identities must be distinct regular files"
                    )
                blocks = []
                while block := os.read(descriptor, 1024 * 1024):
                    blocks.append(block)
                encoded[relative] = b"".join(blocks)
                identities.add(identity)
            except OSError as error:
                raise BenchmarkValidationError("could not read committed file") from error
            finally:
                os.close(descriptor)
        return encoded, identities
    finally:
        os.close(repo_fd)


def _commitment_bytes(
    source_spec_path: Path,
    dataset_dir: Path,
    manifest_digest_path: Path,
    source_spec: dict,
    repo_root: Path,
) -> bytes:
    paths = _commitment_paths(
        source_spec_path,
        dataset_dir,
        manifest_digest_path,
        source_spec,
        repo_root,
    )
    files, _identities = _read_commitment_files(repo_root, paths)
    return b"".join(
        f"{hashlib.sha256(files[path]).hexdigest()}  {path}\n".encode("ascii")
        for path in paths
    )


def freeze_captured_corpus(
    source_spec_path: Path,
    capture_dir: Path,
    destination: Path,
    manifest_digest_path: Path,
    freeze_digest_path: Path,
    repo_root: Path,
) -> dict:
    repo_root = _absolute_path(Path(repo_root))
    source_spec_path = _absolute_path(Path(source_spec_path))
    destination = _absolute_path(Path(destination))
    manifest_digest_path = _absolute_path(Path(manifest_digest_path))
    freeze_digest_path = _absolute_path(Path(freeze_digest_path))
    _repo_relative(source_spec_path, repo_root, "source specification")
    _repo_relative(destination, repo_root, "dataset")
    _repo_relative(manifest_digest_path, repo_root, "manifest digest")
    _repo_relative(freeze_digest_path, repo_root, "freeze digest")
    if len({destination, manifest_digest_path, freeze_digest_path}) != 3:
        raise BenchmarkValidationError("freeze outputs must be distinct")
    for path in (manifest_digest_path, freeze_digest_path):
        if path == destination or path.is_relative_to(destination):
            raise BenchmarkValidationError("freeze digests must be outside the dataset")
    _validate_new_destination(destination, "dataset")
    _validate_new_destination(manifest_digest_path, "manifest digest")
    _validate_new_destination(freeze_digest_path, "freeze digest")

    source_spec, _encoded, _identity = _read_canonical_json_file(
        source_spec_path, "source specification"
    )
    documents = validate_captured_evidence(source_spec, repo_root)
    snapshot = _snapshot_capture_buffers(
        source_spec, documents["receipts"], capture_dir, documents["rights"]["sites"]
    )
    plan = build_captured_case_plan(source_spec)
    materialized = []
    for case in plan:
        generated = copy.deepcopy(case)
        generated["file_sha256"] = snapshot["file_sha256"][case["id"]]
        generated["pixel_sha256"] = snapshot["pixel_sha256"][case["id"]]
        materialized.append(generated)
    manifest = {
        "schema_version": source_spec["schema_version"],
        "benchmark_version": source_spec["benchmark_version"],
        "claim": source_spec["claim"],
        "base_position_count": source_spec["base_position_count"],
        "limitations": source_spec["limitations"],
        "source_spec_sha256": source_spec_sha256(source_spec),
        "generation_runtime": runtime_fingerprint(),
        "expected_counts": source_spec["expected_counts"],
        "cases": materialized,
    }
    manifest_bytes = canonical_json_bytes(manifest)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    staging_identity = _owned_directory_identity(staging)
    destination_identity = None
    manifest_digest_identity = None
    freeze_digest_identity = None
    try:
        (staging / "images").mkdir()
        for case in plan:
            _write_staging_file(
                staging / case["path"], snapshot["buffers"][case["id"]]
            )
        _write_staging_file(staging / "manifest.json", manifest_bytes)
        verify_unanchored_manifest(staging, source_spec, verify_images=True)
        retained = load_verified_image_bytes(
            staging, materialized, require_minimal_png=True
        )
        if any(
            retained[identifier] != snapshot["buffers"][identifier]
            for identifier in retained
        ):
            raise BenchmarkValidationError("staged PNG bytes differ from capture")
        _rename_directory_noreplace(staging, destination)
        destination_identity = staging_identity
        if _owned_directory_identity(destination) != staging_identity:
            raise BenchmarkValidationError("published dataset identity differs")
        verify_unanchored_manifest(destination, source_spec, verify_images=True)
        published = load_verified_image_bytes(
            destination, materialized, require_minimal_png=True
        )
        if published != snapshot["buffers"]:
            raise BenchmarkValidationError("published PNG bytes differ from capture")

        manifest_digest_bytes = (
            f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json\n".encode(
                "ascii"
            )
        )
        manifest_digest_identity = _write_new_file(
            manifest_digest_path, manifest_digest_bytes
        )
        freeze_bytes = _commitment_bytes(
            source_spec_path,
            destination,
            manifest_digest_path,
            source_spec,
            repo_root,
        )
        freeze_digest_identity = _write_new_file(freeze_digest_path, freeze_bytes)
        freeze_sha256 = verify_freeze_commitment(
            source_spec_path,
            destination,
            manifest_digest_path,
            freeze_digest_path,
            repo_root,
        )
        return {
            "manifest": manifest,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "freeze_sha256": freeze_sha256,
        }
    except BaseException:
        if destination_identity is None:
            try:
                if _owned_directory_identity(destination) == staging_identity:
                    destination_identity = staging_identity
            except (FileNotFoundError, BenchmarkValidationError):
                pass
        _remove_owned_file(freeze_digest_path, freeze_digest_identity)
        _remove_owned_file(manifest_digest_path, manifest_digest_identity)
        _remove_owned_directory(destination, destination_identity)
        raise
    finally:
        _remove_owned_directory(staging, staging_identity)


def verify_freeze_commitment(
    source_spec_path: Path,
    dataset_dir: Path,
    manifest_digest_path: Path,
    freeze_digest_path: Path,
    repo_root: Path,
) -> str:
    repo_root = _absolute_path(Path(repo_root))
    source_spec_path = _absolute_path(Path(source_spec_path))
    dataset_dir = _absolute_path(Path(dataset_dir))
    manifest_digest_path = _absolute_path(Path(manifest_digest_path))
    freeze_digest_path = _absolute_path(Path(freeze_digest_path))
    source_spec, _encoded, _identity = _read_canonical_json_file(
        source_spec_path, "source specification"
    )
    validate_captured_source_spec(
        source_spec, repo_root, verify_local_assets=True
    )
    manifest = verify_manifest(
        dataset_dir, source_spec, manifest_digest_path, verify_images=False
    )
    buffers = load_verified_image_bytes(
        dataset_dir, manifest["cases"], require_minimal_png=True
    )
    if len(buffers) != EXPECTED_COUNTS["all"]:
        raise BenchmarkValidationError("verified image count differs")
    if len({case["file_sha256"] for case in manifest["cases"]}) != len(buffers):
        raise BenchmarkValidationError("duplicate encoded frozen PNG")
    if len({case["pixel_sha256"] for case in manifest["cases"]}) != len(buffers):
        raise BenchmarkValidationError("duplicate decoded frozen pixels")

    expected_paths = _commitment_paths(
        source_spec_path,
        dataset_dir,
        manifest_digest_path,
        source_spec,
        repo_root,
    )
    freeze_relative = _repo_relative(freeze_digest_path, repo_root, "freeze digest")
    if freeze_relative in expected_paths:
        raise BenchmarkValidationError("freeze commitment must not list itself")
    freeze_bytes, freeze_identity = _read_regular_file(
        freeze_digest_path, "freeze commitment"
    )
    try:
        lines = freeze_bytes.decode("ascii").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise BenchmarkValidationError("freeze commitment must be ASCII") from error
    parsed_paths = []
    parsed_hashes = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)\n", line)
        if match is None or not _safe_path(match.group(2)):
            raise BenchmarkValidationError("freeze commitment line is invalid")
        digest, relative = match.groups()
        if relative in parsed_hashes:
            raise BenchmarkValidationError("duplicate freeze commitment path")
        parsed_paths.append(relative)
        parsed_hashes[relative] = digest
    if not lines or parsed_paths != sorted(parsed_paths):
        raise BenchmarkValidationError("freeze commitment is not canonically sorted")
    if parsed_paths != expected_paths:
        raise BenchmarkValidationError("freeze commitment inventory differs")
    committed, identities = _read_commitment_files(repo_root, expected_paths)
    if freeze_identity in identities:
        raise BenchmarkValidationError("freeze commitment identity is aliased")
    for relative in expected_paths:
        if hashlib.sha256(committed[relative]).hexdigest() != parsed_hashes[relative]:
            raise BenchmarkValidationError(
                f"freeze commitment hash mismatch: {relative}"
            )
    return hashlib.sha256(freeze_bytes).hexdigest()
