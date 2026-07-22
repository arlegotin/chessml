from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import benchmarks.captured_board_recognition as captured
from benchmarks.board_recognition import canonical_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble, freeze, and verify the captured board benchmark."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write-plan", type=Path, metavar="PATH")
    modes.add_argument("--verify-captures", action="store_true")
    modes.add_argument("--assemble-source", action="store_true")
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--capture-plan", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--annotations-a", type=Path)
    parser.add_argument("--annotations-b", type=Path)
    parser.add_argument("--attribution", type=Path)
    parser.add_argument("--rights", type=Path)
    parser.add_argument("--licenses-dir", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--source-spec", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest-digest", type=Path)
    parser.add_argument("--freeze-digest", type=Path)
    return parser


def _mode(arguments: argparse.Namespace) -> tuple[str, set[str]]:
    if arguments.write_plan is not None:
        return "write", {"write_plan"}
    if arguments.verify_captures:
        return "captures", {"capture_plan", "receipts", "capture_dir"}
    if arguments.assemble_source:
        return "assemble", {
            "capture_plan",
            "receipts",
            "capture_dir",
            "annotations_a",
            "annotations_b",
            "attribution",
            "rights",
            "licenses_dir",
            "review_output",
            "source_output",
        }
    if arguments.freeze:
        return "freeze", {
            "source_spec",
            "capture_dir",
            "dataset",
            "manifest_digest",
            "freeze_digest",
        }
    return "verify", {
        "source_spec",
        "dataset",
        "manifest_digest",
        "freeze_digest",
    }


def _validate_arguments(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> str:
    mode, required = _mode(arguments)
    option_names = {
        "write_plan",
        "capture_plan",
        "receipts",
        "capture_dir",
        "annotations_a",
        "annotations_b",
        "attribution",
        "rights",
        "licenses_dir",
        "review_output",
        "source_output",
        "source_spec",
        "dataset",
        "manifest_digest",
        "freeze_digest",
    }
    provided = {
        name for name in option_names if getattr(arguments, name) is not None
    }
    missing = sorted(required - provided)
    irrelevant = sorted(provided - required)
    if missing or irrelevant:
        parser.error(
            f"{mode} mode argument mismatch: missing={missing}, irrelevant={irrelevant}"
        )
    if mode == "assemble":
        expected = REPO_ROOT / "benchmarks/licenses"
        try:
            actual_resolved = arguments.licenses_dir.resolve(strict=True)
            expected_resolved = expected.resolve(strict=True)
        except OSError as error:
            parser.error(f"licenses directory does not resolve: {error}")
        if actual_resolved != expected_resolved:
            parser.error("--licenses-dir must resolve to benchmarks/licenses")
    return mode


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    mode = _validate_arguments(parser, arguments)
    if mode == "write":
        result = captured.write_capture_plan(arguments.write_plan)
        summary = {"cases": len(result["cases"]), "output": str(arguments.write_plan)}
    elif mode == "captures":
        result = captured.verify_capture_staging(
            captured._load_capture_plan(arguments.capture_plan),
            arguments.receipts,
            arguments.capture_dir,
        )
        summary = {"cases": len(result["buffers"])}
    elif mode == "assemble":
        licenses = [
            REPO_ROOT.joinpath(*relative.split("/"))
            for relative in sorted(captured.LICENSE_PATHS)
        ]
        result = captured.assemble_reviewed_source(
            arguments.capture_plan,
            arguments.receipts,
            arguments.capture_dir,
            arguments.annotations_a,
            arguments.annotations_b,
            arguments.attribution,
            arguments.rights,
            licenses,
            arguments.review_output,
            arguments.source_output,
            REPO_ROOT,
        )
        summary = result["source_spec"]["expected_counts"]
    elif mode == "freeze":
        result = captured.freeze_captured_corpus(
            arguments.source_spec,
            arguments.capture_dir,
            arguments.dataset,
            arguments.manifest_digest,
            arguments.freeze_digest,
            REPO_ROOT,
        )
        summary = {
            **result["manifest"]["expected_counts"],
            "freeze_sha256": result["freeze_sha256"],
        }
    else:
        freeze_sha256 = captured.verify_freeze_commitment(
            arguments.source_spec,
            arguments.dataset,
            arguments.manifest_digest,
            arguments.freeze_digest,
            REPO_ROOT,
        )
        summary = {"freeze_sha256": freeze_sha256}
    sys.stdout.write(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
