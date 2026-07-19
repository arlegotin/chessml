from __future__ import annotations

import argparse
import os
import shutil
import stat
import tempfile
import textwrap
from pathlib import Path, PurePosixPath
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from benchmarks.board_recognition import (
    BenchmarkValidationError,
    PIECE_SYMBOLS,
    _reject_symlink_components,
    build_case_plan,
    canonical_json_bytes,
    file_sha256,
    load_source_spec,
    pixel_sha256,
    runtime_fingerprint,
    source_spec_sha256,
    validate_case_plan,
    validate_source_spec,
    verify_manifest,
    verify_unanchored_manifest,
)
from benchmarks.render_online_boards import render_case


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SPEC = REPO_ROOT / "benchmarks/board_recognition_v1.json"
CONTACT_FIELDS = (
    "base_position",
    "piece_set",
    "palette",
    "view",
    "layout",
    "quality",
    "channel_mode",
    "negative_template",
    "suite",
    "split",
)


def _save_png(image: Image.Image, destination: Path, renderer: dict) -> None:
    image.save(
        destination,
        format="PNG",
        optimize=renderer["png_optimize"],
        compress_level=renderer["png_compress_level"],
    )


def _owned_directory_identity(path: Path) -> tuple[int, int]:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise BenchmarkValidationError("owned path is not a real directory")
    return value.st_dev, value.st_ino


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        value = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISLNK(value.st_mode)
        and stat.S_ISDIR(value.st_mode)
        and (value.st_dev, value.st_ino) == identity
    ):
        shutil.rmtree(path)


def verify_rendered_corpus(
    dataset_dir: Path,
    spec: dict,
    repo_root: Path,
    *,
    render_case_fn=render_case,
) -> dict:
    validate_source_spec(spec, repo_root, verify_local_assets=True)
    expected_plan = build_case_plan(spec)
    manifest = verify_unanchored_manifest(dataset_dir, spec, verify_images=True)

    expected_files = {"manifest.json", *(case["path"] for case in expected_plan)}
    actual_files = {
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    actual_directories = {
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if actual_files != expected_files or actual_directories != {"images"}:
        raise BenchmarkValidationError("rendered corpus inventory differs")

    materialized_by_id = {case["id"]: case for case in manifest["cases"]}
    for expected in expected_plan:
        image = render_case_fn(expected, spec, repo_root)
        if not isinstance(image, Image.Image):
            raise BenchmarkValidationError("renderer did not return a Pillow image")
        image.load()
        materialized = materialized_by_id[expected["id"]]
        if image.mode != expected["image_mode"]:
            raise BenchmarkValidationError("re-rendered image mode differs")
        if image.size != tuple(expected["image_size"]):
            raise BenchmarkValidationError("re-rendered image size differs")
        if pixel_sha256(image) != materialized["pixel_sha256"]:
            raise BenchmarkValidationError("re-rendered case pixels differ")
    return manifest


def compare_corpora(
    first: Path,
    second: Path,
    spec: dict,
    repo_root: Path,
) -> dict:
    first_manifest = verify_rendered_corpus(first, spec, repo_root)
    verify_rendered_corpus(second, spec, repo_root)
    first_inventory = sorted(
        path.relative_to(first).as_posix()
        for path in first.rglob("*")
        if path.is_file()
    )
    second_inventory = sorted(
        path.relative_to(second).as_posix()
        for path in second.rglob("*")
        if path.is_file()
    )
    if first_inventory != second_inventory:
        raise BenchmarkValidationError("corpus inventories differ")
    for relative in first_inventory:
        if (first / relative).read_bytes() != (second / relative).read_bytes():
            raise BenchmarkValidationError(f"corpus file bytes differ: {relative}")
    return {"cases": len(first_manifest["cases"]), "files": len(first_inventory)}


def write_dataset(
    destination: Path,
    spec: dict,
    repo_root: Path,
    *,
    ordered_cases: Sequence[dict] | None = None,
    render_case_fn=render_case,
    image_writer=_save_png,
) -> dict:
    validate_source_spec(spec, repo_root, verify_local_assets=True)
    canonical_plan = build_case_plan(spec)
    cases = canonical_plan if ordered_cases is None else list(ordered_cases)
    try:
        validate_case_plan(sorted(cases, key=lambda case: case["id"]), spec)
    except (KeyError, TypeError) as error:
        raise BenchmarkValidationError("ordered cases differ from case plan") from error

    destination = Path(destination)
    parent = destination.parent
    _reject_symlink_components(parent, "destination parent")
    if parent.is_symlink() or not parent.is_dir():
        raise BenchmarkValidationError("destination parent must be a real directory")
    if os.path.lexists(destination):
        raise BenchmarkValidationError("destination already exists")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent)
    )
    staging_identity = _owned_directory_identity(staging)
    try:
        (staging / "images").mkdir()
        materialized_cases = []
        for case in cases:
            image = render_case_fn(case, spec, repo_root)
            if not isinstance(image, Image.Image):
                raise BenchmarkValidationError("renderer did not return a Pillow image")
            image.load()
            if image.mode != case["image_mode"]:
                raise BenchmarkValidationError("rendered image mode differs")
            if image.size != tuple(case["image_size"]):
                raise BenchmarkValidationError("rendered image size differs")

            path = staging / case["path"]
            image_writer(image, path, spec["renderer"])
            try:
                with Image.open(path) as saved:
                    saved.load()
                    if saved.format != "PNG":
                        raise BenchmarkValidationError("writer did not encode PNG")
                    if saved.mode != case["image_mode"]:
                        raise BenchmarkValidationError("written image mode differs")
                    if saved.size != tuple(case["image_size"]):
                        raise BenchmarkValidationError("written image size differs")
                    generated = dict(case)
                    generated["file_sha256"] = file_sha256(path)
                    generated["pixel_sha256"] = pixel_sha256(saved)
            except BenchmarkValidationError:
                raise
            except OSError as error:
                raise BenchmarkValidationError("could not reopen written PNG") from error
            materialized_cases.append(generated)

        manifest = {
            "schema_version": spec["schema_version"],
            "benchmark_version": spec["benchmark_version"],
            "claim": spec["claim"],
            "base_position_count": spec["base_position_count"],
            "limitations": spec["limitations"],
            "source_spec_sha256": source_spec_sha256(spec),
            "generation_runtime": runtime_fingerprint(),
            "expected_counts": spec["expected_counts"],
            "cases": sorted(materialized_cases, key=lambda case: case["id"]),
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        verify_rendered_corpus(
            staging, spec, repo_root, render_case_fn=render_case_fn
        )

        if os.path.lexists(destination):
            raise BenchmarkValidationError("destination appeared before publication")
        staging.rename(destination)
        try:
            if _owned_directory_identity(destination) != staging_identity:
                raise BenchmarkValidationError(
                    "published destination identity differs from staging"
                )
            return verify_rendered_corpus(
                destination, spec, repo_root, render_case_fn=render_case_fn
            )
        except BaseException:
            _remove_owned_directory(destination, staging_identity)
            raise
    except BaseException:
        _remove_owned_directory(staging, staging_identity)
        raise


def select_contact_sheet_cases(cases: Sequence[dict], spec: dict) -> list[dict]:
    ordered = sorted(cases, key=lambda case: case["id"])
    semantic = [
        {
            key: value
            for key, value in case.items()
            if key not in {"file_sha256", "pixel_sha256"}
        }
        for case in ordered
    ]
    validate_case_plan(semantic, spec)

    def coverage(case: dict) -> set[tuple[str, str]]:
        features = {
            (field, case[field]) for field in CONTACT_FIELDS if field in case
        }
        if case["board_present"]:
            features.update(
                ("piece_symbol", symbol)
                for symbol in set(case["canonical_placement"]) & PIECE_SYMBOLS
            )
        return features

    coverage_by_id = {case["id"]: coverage(case) for case in ordered}
    uncovered = set().union(*coverage_by_id.values())
    remaining = list(ordered)
    selected = []
    while uncovered:
        best = min(
            remaining,
            key=lambda case: (
                -len(coverage_by_id[case["id"]] & uncovered),
                case["id"],
            ),
        )
        gained = coverage_by_id[best["id"]] & uncovered
        if not gained:
            raise BenchmarkValidationError("contact-sheet coverage is incomplete")
        selected.append(best)
        uncovered -= gained
        remaining.remove(best)
    return selected


def write_contact_sheet(
    dataset_dir: Path, output: Path, cases: Sequence[dict]
) -> dict:
    if not cases:
        raise BenchmarkValidationError("contact sheet requires at least one case")
    dataset_dir = Path(dataset_dir)
    _reject_symlink_components(dataset_dir, "contact-sheet dataset")
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise BenchmarkValidationError("contact-sheet dataset must be a real directory")
    try:
        resolved_dataset = dataset_dir.resolve(strict=True)
    except OSError as error:
        raise BenchmarkValidationError(
            "contact-sheet dataset does not resolve"
        ) from error
    output = Path(output)
    _reject_symlink_components(output.parent, "contact-sheet parent")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise BenchmarkValidationError("contact-sheet parent must be a real directory")
    if os.path.lexists(output):
        raise BenchmarkValidationError("contact-sheet output already exists")
    try:
        resolved_output = output.parent.resolve(strict=True) / output.name
    except OSError as error:
        raise BenchmarkValidationError(
            "contact-sheet parent does not resolve"
        ) from error
    if resolved_output == resolved_dataset or resolved_output.is_relative_to(
        resolved_dataset
    ):
        raise BenchmarkValidationError(
            "contact-sheet output must be outside the dataset"
        )

    columns = 4
    cell_width, cell_height = 240, 170
    rows = (len(cases) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#171513")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, case in enumerate(cases):
        value = case.get("path") if isinstance(case, dict) else None
        relative = PurePosixPath(value) if isinstance(value, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != value
            or relative.suffix != ".png"
        ):
            raise BenchmarkValidationError("unsafe contact-sheet image path")
        path = dataset_dir.joinpath(*relative.parts)
        _reject_symlink_components(path, "contact-sheet image")
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as error:
            raise BenchmarkValidationError(
                "contact-sheet image does not resolve"
            ) from error
        if (
            not resolved_path.is_relative_to(resolved_dataset)
            or path.is_symlink()
            or not path.is_file()
        ):
            raise BenchmarkValidationError(
                "contact-sheet image must be a contained regular file"
            )
        try:
            with Image.open(path) as source:
                source.load()
                thumbnail = source.convert("RGB")
        except OSError as error:
            raise BenchmarkValidationError(
                f"could not decode contact-sheet image: {value}"
            ) from error
        thumbnail.thumbnail((224, 120), Image.Resampling.LANCZOS)
        column, row = index % columns, index // columns
        left = column * cell_width + (cell_width - thumbnail.width) // 2
        top = row * cell_height + 4 + (120 - thumbnail.height) // 2
        sheet.paste(thumbnail, (left, top))
        label = "\n".join(textwrap.wrap(case["id"], width=38))
        draw.multiline_text(
            (column * cell_width + 6, row * cell_height + 128),
            label,
            font=font,
            fill="#FFFFFF",
            spacing=1,
        )
    sheet.save(output, format="PNG", optimize=False, compress_level=9)
    return {"cases": len(cases), "size": list(sheet.size)}


def _validated_source() -> tuple[dict, list[dict]]:
    spec = load_source_spec(SOURCE_SPEC)
    validate_source_spec(spec, REPO_ROOT, verify_local_assets=True)
    plan = build_case_plan(spec)
    validate_case_plan(plan, spec)
    return spec, plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the frozen online-board benchmark"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--write", type=Path, metavar="DESTINATION")
    modes.add_argument("--verify", type=Path, metavar="DATASET")
    modes.add_argument(
        "--compare", nargs=2, type=Path, metavar=("FIRST", "SECOND")
    )
    modes.add_argument(
        "--contact-sheet", nargs=2, type=Path, metavar=("DATASET", "OUTPUT")
    )
    parser.add_argument("--case-order", choices=("normal", "reverse"))
    parser.add_argument("--digest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.write is None and arguments.case_order is not None:
        parser.error("--case-order requires --write")
    if arguments.verify is None and arguments.digest is not None:
        parser.error("--digest requires --verify")
    if arguments.verify is not None and arguments.digest is None:
        parser.error("--verify requires --digest")

    spec, plan = _validated_source()
    if not any(
        (
            arguments.write,
            arguments.verify,
            arguments.compare,
            arguments.contact_sheet,
        )
    ):
        counts = spec["expected_counts"]
        print(
            f"cases={counts['all']} positive={counts['positive']} "
            f"negative={counts['negative']}"
        )
        return 0

    if arguments.write is not None:
        ordered = plan if arguments.case_order != "reverse" else list(reversed(plan))
        manifest = write_dataset(
            arguments.write, spec, REPO_ROOT, ordered_cases=ordered
        )
        print(f"wrote cases={len(manifest['cases'])} destination={arguments.write}")
        return 0
    if arguments.verify is not None:
        verify_manifest(
            arguments.verify, spec, arguments.digest, verify_images=True
        )
        manifest = verify_rendered_corpus(arguments.verify, spec, REPO_ROOT)
        print(f"verified cases={len(manifest['cases'])} dataset={arguments.verify}")
        return 0
    if arguments.compare is not None:
        result = compare_corpora(
            arguments.compare[0], arguments.compare[1], spec, REPO_ROOT
        )
        print(f"compared cases={result['cases']} files={result['files']}")
        return 0

    dataset, output = arguments.contact_sheet
    manifest = verify_rendered_corpus(dataset, spec, REPO_ROOT)
    selected = select_contact_sheet_cases(manifest["cases"], spec)
    result = write_contact_sheet(dataset, output, selected)
    print(f"contact-sheet cases={result['cases']} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
