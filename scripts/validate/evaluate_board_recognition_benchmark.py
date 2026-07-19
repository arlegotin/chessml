from __future__ import annotations

import argparse
import os
import platform
import stat
import tempfile
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path

from benchmarks.board_recognition import (
    FAILURE_REASONS,
    BenchmarkValidationError,
    _reject_symlink_components,
    canonical_json_bytes,
    expand_placement,
    file_sha256,
    load_source_spec,
    score_predictions,
    verify_manifest,
)


class InvalidRecognitionResult(ValueError):
    pass


def _is_owned_regular_file(path: Path, device: int, inode: int) -> bool:
    try:
        current = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISREG(current.st_mode)
        and current.st_dev == device
        and current.st_ino == inode
    )


def _validate_report_destination(
    report_path: Path, dataset_dir: Path | None = None
) -> None:
    if os.path.lexists(report_path):
        raise FileExistsError(f"report already exists: {report_path}")
    parent = report_path.parent
    _reject_symlink_components(parent, "report parent")
    if parent.is_symlink() or not parent.is_dir():
        raise BenchmarkValidationError(
            "report parent must be a real existing non-symlink directory"
        )
    if dataset_dir is None:
        return
    try:
        resolved_dataset = dataset_dir.resolve(strict=True)
        resolved_report = parent.resolve(strict=True) / report_path.name
    except OSError as error:
        raise BenchmarkValidationError(
            "report dataset or parent does not resolve"
        ) from error
    if resolved_report == resolved_dataset or resolved_report.is_relative_to(
        resolved_dataset
    ):
        raise BenchmarkValidationError("report must be outside the dataset")


def run_predictions(
    dataset_dir: Path,
    cases: Sequence[dict],
    helper: object,
    *,
    picture_cls: type,
    success_cls: type,
    failure_cls: type,
) -> list[dict]:
    predictions = []
    for case in sorted(cases, key=lambda value: value["id"]):
        try:
            picture = picture_cls(dataset_dir / case["path"])
            result = helper.recognize(picture)
            if isinstance(result, success_cls):
                try:
                    placement = result.source_placement
                    expand_placement(placement)
                except Exception as error:
                    raise InvalidRecognitionResult(
                        "invalid recognition-success payload"
                    ) from error
                prediction = {
                    "id": case["id"],
                    "outcome": "success",
                    "source_placement": placement,
                }
            elif isinstance(result, failure_cls):
                try:
                    reason = result.reason.name
                except Exception as error:
                    raise InvalidRecognitionResult(
                        "invalid recognition-failure payload"
                    ) from error
                if not isinstance(reason, str) or reason not in FAILURE_REASONS:
                    raise InvalidRecognitionResult(
                        "invalid recognition-failure reason"
                    )
                prediction = {
                    "id": case["id"],
                    "outcome": "failure",
                    "reason": reason,
                }
            else:
                raise InvalidRecognitionResult("unexpected recognition result type")
        except Exception as error:
            prediction = {
                "id": case["id"],
                "outcome": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        predictions.append(prediction)
    return predictions


def write_report(
    report_path: Path,
    report: dict,
    *,
    write_fn: Callable[[Path, bytes], object] | None = None,
    rename_fn: Callable[[Path, Path], object] | None = None,
) -> None:
    _validate_report_destination(report_path)
    parent = report_path.parent

    encoded = canonical_json_bytes(report)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{report_path.name}.", suffix=".tmp", dir=parent
    )
    temp_path = Path(temp_name)
    owned = os.fstat(descriptor)
    rename_fn = rename_fn or os.rename
    try:
        if write_fn is None:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("could not write complete report")
                remaining = remaining[written:]
        else:
            write_fn(temp_path, encoded)
        if not _is_owned_regular_file(temp_path, owned.st_dev, owned.st_ino):
            raise BenchmarkValidationError(
                "report temp path is not the owned regular file"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        actual = b"".join(
            iter(lambda: os.read(descriptor, 1024 * 1024), b"")
        )
        if actual != encoded:
            raise BenchmarkValidationError(
                "report temp file differs from canonical content"
            )
        if os.path.lexists(report_path):
            raise FileExistsError(f"report already exists: {report_path}")
        if not _is_owned_regular_file(temp_path, owned.st_dev, owned.st_ino):
            raise BenchmarkValidationError(
                "report temp path is not the owned regular file"
            )
        rename_fn(temp_path, report_path)
    finally:
        try:
            if _is_owned_regular_file(temp_path, owned.st_dev, owned.st_ino):
                temp_path.unlink()
        finally:
            os.close(descriptor)


def inference_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        **{
            package: version(package)
            for package in (
                "torch",
                "torchvision",
                "timm",
                "lightning",
                "opencv-python",
                "numpy",
            )
        },
    }


def load_helper(
    board_detector_checkpoint: Path,
    square_classifier_checkpoint: Path,
    piece_classifier_checkpoint: Path,
    device: str,
) -> tuple[object, type, type, type]:
    from chessml.data.images.picture import Picture
    from chessml.models.lightning.board_detector_model import BoardDetector
    from chessml.models.lightning.piece_classifier_model import PieceClassifier
    from chessml.models.lightning.square_classifier_model import SquareClassifier
    from chessml.models.torch.vision_model_adapter import (
        MobileNetV3LargeClassifier,
        MobileNetV3SmallClassifier,
        MobileViTV2FPN,
    )
    from chessml.models.utils.board_recognition_helper import (
        BoardRecognitionHelper,
        RecognitionFailure,
        RecognitionSuccess,
    )

    board_detector = BoardDetector.load_from_checkpoint(
        board_detector_checkpoint,
        base_model_class=MobileViTV2FPN,
        base_model_kwargs={"pretrained": False},
        map_location=device,
        strict=True,
        weights_only=False,
    )
    square_classifier = SquareClassifier.load_from_checkpoint(
        square_classifier_checkpoint,
        base_model_class=MobileNetV3SmallClassifier,
        base_model_kwargs={"pretrained": False},
        map_location=device,
        strict=True,
        weights_only=False,
    )
    piece_classifier = PieceClassifier.load_from_checkpoint(
        piece_classifier_checkpoint,
        base_model_class=MobileNetV3LargeClassifier,
        base_model_kwargs={"pretrained": False},
        map_location=device,
        strict=True,
        weights_only=False,
    )
    for model in (board_detector, square_classifier, piece_classifier):
        model.eval()
    return (
        BoardRecognitionHelper(
            board_detector=board_detector,
            square_classifier=square_classifier,
            piece_classifier=piece_classifier,
        ),
        Picture,
        RecognitionSuccess,
        RecognitionFailure,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen board-recognition benchmark"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-spec", required=True, type=Path)
    parser.add_argument("--manifest-digest", required=True, type=Path)
    parser.add_argument("--board-detector-checkpoint", required=True, type=Path)
    parser.add_argument("--square-classifier-checkpoint", required=True, type=Path)
    parser.add_argument("--piece-classifier-checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source = load_source_spec(arguments.source_spec)
    manifest = verify_manifest(
        arguments.dataset, source, arguments.manifest_digest
    )
    _validate_report_destination(arguments.report, arguments.dataset)

    helper, picture_cls, success_cls, failure_cls = load_helper(
        arguments.board_detector_checkpoint,
        arguments.square_classifier_checkpoint,
        arguments.piece_classifier_checkpoint,
        arguments.device,
    )
    predictions = run_predictions(
        arguments.dataset,
        manifest["cases"],
        helper,
        picture_cls=picture_cls,
        success_cls=success_cls,
        failure_cls=failure_cls,
    )
    scores = score_predictions(manifest["cases"], predictions)
    report = {
        "benchmark_version": source["benchmark_version"],
        "claim": source["claim"],
        "base_position_count": source["base_position_count"],
        "limitations": source["limitations"],
        "manifest_sha256": arguments.manifest_digest.read_bytes()[:64].decode(
            "ascii"
        ),
        "checkpoints": {
            name: {"basename": path.name, "sha256": file_sha256(path)}
            for name, path in (
                ("board_detector", arguments.board_detector_checkpoint),
                ("square_classifier", arguments.square_classifier_checkpoint),
                ("piece_classifier", arguments.piece_classifier_checkpoint),
            )
        },
        "device": arguments.device,
        "inference_fingerprint": inference_fingerprint(),
        "predictions": predictions,
        "scores": scores,
    }
    write_report(arguments.report, report)
    return int(scores["overall"]["execution_error_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
