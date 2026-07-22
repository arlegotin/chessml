# Board Recognition Output Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make still-image board recognition return source-oriented piece placement or a typed per-image failure, and construct full FEN only from complete caller-supplied context.

**Architecture:** Keep the public types, placement helpers, and `build_fen()` in the existing `board_recognition_helper` module. Validate geometry in `BoardDetector` and translate only constrained-decoder errors in `PieceClassifier` so the helper can map expected image outcomes without swallowing model, device, configuration, or programming failures. Remove MetaPredictor from the core path, then migrate the only runtime callers and documentation.

**Tech Stack:** Python 3.14.6, chess 1.11.2 (python-chess), NumPy 2.4.6, OpenCV 4.13, OR-Tools 9.15, PyTorch/Lightning 2.12/2.6, pytest 9.1, uv 0.11.

**Specification:** `docs/superpowers/specs/2026-07-18-board-recognition-output-contract-design.md`

> **EXECUTION OVERRIDE — DO NOT COMMIT OR STAGE:** The user explicitly approved execution only in the working tree. Skip every `git add` and `git commit` step below. Implementation, tests, task reviews, and final verification remain mandatory.

## Global Constraints

- Core recognition returns only `RecognitionSuccess(source_placement, board_image)` or `RecognitionFailure(reason)`.
- `source_placement` is source-image row order: top to bottom, then left to right. Core recognition never rotates it.
- Remove `.board`, `.get_fen()`, `.flipped`, public square iteration, and the mandatory MetaPredictor dependency without a compatibility shim.
- Keep `NO_BOARD` in the public failure enum but do not emit it until P1-07 adds a real presence decision.
- Convert only named geometry and constrained-decoder domain errors to failure values. Model execution, device, checkpoint, configuration, caller-input, and programming failures remain exceptions.
- `build_fen()` requires orientation, turn, castling, en-passant, halfmove clock, and fullmove number as keyword-only arguments with no defaults.
- Return the exact validated caller-supplied FEN string; do not serialize it back through `Board.fen()`, which may normalize castling, fullmove, or en-passant fields.
- Keep the 12-label PieceClassifier, 13-plane `OnlyPieces`, 19-plane `FullPosition`, MetaPredictor model/training code, and checkpoints unchanged.
- Add no dependency, confidence schema, evaluator, benchmark, training repair, data generation, retraining, or checkpoint selection.
- The task-selected July 18 specification supersedes stale full-FEN/`flipped` checks in completed historical plans and specs. Do not rewrite those historical documents.
- Do not change or stage `BOARD_RECOGNITION_ARCHITECTURE_RESEARCH.tmp.md`, `BOARD_RECOGNITION_AUDIT.tmp.md`, or the approved specification. The plan may contain only this explicit execution override; otherwise preserve it.
- Do not stage or commit any implementation or documentation change. Treat every commit step below as skipped by direct user instruction.
- Do not publish a release or change the package version. A later release containing this pre-1.0 break requires a minor-version bump.
- Run commands from the repository root with `uv run --locked` unless a command does not execute Python.

## File Structure

- Modify `chessml/models/lightning/board_detector_model.py`: reject unusable detector quadrilaterals with one named geometry exception.
- Create `tests/test_board_detector_model.py`: cover each invalid geometry category and one valid warp.
- Modify `chessml/models/lightning/piece_classifier_model.py`: translate only constrained-assignment failures to one named decoder exception.
- Modify `tests/test_piece_classifier_model.py`: cover invalid solver status, decoder translation, and propagation of model execution errors.
- Modify `chessml/models/utils/board_recognition_helper.py`: define result types, source-order recognition, placement validation, failure mapping, orientation, and `build_fen()`.
- Modify `tests/test_board_recognition_helper.py`: replace authoritative-FEN tests with the complete output/failure/context contract.
- Modify `tests/test_board_recognition_integration.py`: make the local-checkpoint check a three-model typed-result smoke test.
- Modify `scripts/validate/validate_board_recognition.py`: recognize once per image and record source placement or failure without MetaPredictor or rendered FEN.
- Modify `README.md`: document observable placement, explicit failures, caller-owned FEN context, and MetaPredictor as an optional experimental prior.

---

### Task 1: Establish the geometry failure boundary

**Files:**

- Create: `tests/test_board_detector_model.py`
- Modify: `chessml/models/lightning/board_detector_model.py:12-155`

**Interfaces:**

- Consumes: `BoardDetector.predict_coords(Picture) -> np.ndarray` and the existing TL/TR/BR/BL point order.
- Produces: `InvalidBoardGeometryError(ValueError)` from `BoardDetector.extract_board_image()` only when four predicted points are unusable for a perspective warp.

- [ ] **Step 1: Add failing geometry-boundary tests**

Create `tests/test_board_detector_model.py`:

```python
import numpy as np
import pytest
import torch

from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import (
    BoardDetector,
    InvalidBoardGeometryError,
)


class ModelStub(torch.nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.output_features = output_features


def detector_with_coords(monkeypatch, coords):
    detector = BoardDetector(base_model_class=ModelStub)
    monkeypatch.setattr(
        detector,
        "predict_coords",
        lambda _image: np.asarray(coords, dtype=np.float32),
    )
    return detector


IMAGE = Picture(np.zeros((100, 100, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "coords",
    [
        [np.nan, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
        [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9],
        [0.1, 0.1, 0.9, 0.9, 0.1, 0.9, 0.7, 0.1],
        [0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9],
        [0.0, 0.0, 0.001, 0.0, 0.001, 0.001, 0.0, 0.001],
    ],
)
def test_extract_board_image_rejects_unusable_geometry(monkeypatch, coords):
    detector = detector_with_coords(monkeypatch, coords)

    with pytest.raises(InvalidBoardGeometryError):
        detector.extract_board_image(IMAGE)


def test_extract_board_image_accepts_a_valid_quadrilateral(monkeypatch):
    detector = detector_with_coords(
        monkeypatch,
        [0.1, 0.2, 0.9, 0.1, 0.8, 0.9, 0.2, 0.8],
    )

    extracted = detector.extract_board_image(IMAGE)

    assert isinstance(extracted, Picture)
    assert extracted.cv2.shape[0] > 0
    assert extracted.cv2.shape[1] > 0
```

The third invalid case is self-crossing with nonzero contour area, so it specifically exercises the convexity guard rather than passing through the zero-area guard.

- [ ] **Step 2: Run the geometry tests and verify the missing contract**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
```

Expected: collection fails because `InvalidBoardGeometryError` does not exist.

- [ ] **Step 3: Add the named exception and validate before OpenCV warping**

Add above `BoardDetector`:

```python
class InvalidBoardGeometryError(ValueError):
    pass
```

Replace `extract_board_image()` with:

```python
def extract_board_image(self, original_image: Picture) -> Picture:
    coords = self.predict_coords(original_image)
    width, height = original_image.pil.size

    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = coords
    source_points = np.float32(
        [
            [tl_x * width, tl_y * height],
            [tr_x * width, tr_y * height],
            [br_x * width, br_y * height],
            [bl_x * width, bl_y * height],
        ]
    )

    if (
        not np.isfinite(source_points).all()
        or len(np.unique(source_points, axis=0)) != 4
        or cv2.contourArea(source_points) <= 0
        or not cv2.isContourConvex(source_points)
    ):
        raise InvalidBoardGeometryError("Detector returned an unusable quadrilateral")

    width_a = np.linalg.norm(source_points[2] - source_points[3])
    width_b = np.linalg.norm(source_points[1] - source_points[0])
    target_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(source_points[1] - source_points[2])
    height_b = np.linalg.norm(source_points[0] - source_points[3])
    target_height = max(int(height_a), int(height_b))

    if target_width <= 0 or target_height <= 0:
        raise InvalidBoardGeometryError("Detector returned a zero-sized board")

    target_points = np.float32(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    extracted_image = cv2.warpPerspective(
        original_image.cv2,
        matrix,
        (target_width, target_height),
    )
    return Picture(extracted_image)
```

Do not catch shape-unpacking, image-decoding, or OpenCV errors here. They indicate model/output, caller-input, or programming faults rather than an expected four-point geometry rejection.

- [ ] **Step 4: Run the focused detector tests**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
```

Expected: all six tests pass.

- [ ] **Step 5: Commit the geometry boundary**

```bash
git add chessml/models/lightning/board_detector_model.py tests/test_board_detector_model.py
git commit -m "fix: reject invalid board geometry"
```

---

### Task 2: Establish the constrained-decoder failure boundary

**Files:**

- Modify: `chessml/models/lightning/piece_classifier_model.py:14-106,290-298`
- Modify: `tests/test_piece_classifier_model.py:1-130`

**Interfaces:**

- Consumes: known `constrained_argmax(logits)` solver outcomes: FEASIBLE/UNKNOWN as private `TimeoutError` subclasses and INFEASIBLE/MODEL_INVALID as private `RuntimeError` subclasses.
- Produces: `PieceDecodingError(RuntimeError)` from `PieceClassifier.classify_pieces()` only for those four known constrained-assignment outcomes. Arbitrary built-in decoder errors plus model preprocessing, inference, and tensor-conversion errors retain their original types and objects.

- [ ] **Step 1: Add failing decoder-boundary tests**

In `tests/test_piece_classifier_model.py`, add `torch` and the new public names to the imports:

```python
import torch

from chessml.data.images.picture import Picture
from chessml.models.lightning.piece_classifier_model import (
    PieceClassifier,
    PieceDecodingError,
    constrained_argmax,
)
```

Keep the existing `piece_classifier_model` module import and replace its one-line `constrained_argmax` import with the grouped import above. Append:

```python
class ClassifierModelStub(torch.nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.output_features = output_features

    def preprocess_image(self, _image):
        return torch.zeros((3, 8, 8), dtype=torch.float32)

    def forward(self, images):
        return torch.zeros(
            (len(images), self.output_features),
            dtype=torch.float32,
        )


class FailingClassifierModelStub(ClassifierModelStub):
    def forward(self, images):
        raise RuntimeError("device execution failed")


def test_constrained_argmax_reports_an_invalid_model(monkeypatch):
    class InvalidSolver:
        def __init__(self):
            self.parameters = SimpleNamespace()

        def Solve(self, model):
            return piece_classifier_model.cp_model.MODEL_INVALID

    monkeypatch.setattr(piece_classifier_model.cp_model, "CpSolver", InvalidSolver)

    with pytest.raises(RuntimeError, match="invalid"):
        constrained_argmax(logits_for(["k", "K"]))


@pytest.mark.parametrize(
    ("status", "cause_type"),
    [
        (piece_classifier_model.cp_model.FEASIBLE, TimeoutError),
        (piece_classifier_model.cp_model.UNKNOWN, TimeoutError),
        (piece_classifier_model.cp_model.INFEASIBLE, RuntimeError),
        (piece_classifier_model.cp_model.MODEL_INVALID, RuntimeError),
    ],
)
def test_classify_pieces_translates_known_solver_failures(
    monkeypatch,
    status,
    cause_type,
):
    class Solver:
        def __init__(self):
            self.parameters = SimpleNamespace()

        def Solve(self, model):
            return status

    monkeypatch.setattr(
        piece_classifier_model.cp_model,
        "CpSolver",
        Solver,
    )
    classifier = PieceClassifier(base_model_class=ClassifierModelStub)

    with pytest.raises(PieceDecodingError) as raised:
        classifier.classify_pieces(
            [Picture(np.zeros((8, 8, 3), dtype=np.uint8))]
        )

    assert isinstance(raised.value.__cause__, cause_type)


@pytest.mark.parametrize(
    "decoder_error",
    [
        RuntimeError("unexpected decoder failure"),
        TimeoutError("unexpected decoder timeout"),
    ],
)
def test_classify_pieces_does_not_translate_unexpected_decoder_errors(
    monkeypatch,
    decoder_error,
):
    classifier = PieceClassifier(base_model_class=ClassifierModelStub)

    def fail_decoding(_logits):
        raise decoder_error

    monkeypatch.setattr(
        piece_classifier_model,
        "constrained_argmax",
        fail_decoding,
    )

    with pytest.raises(type(decoder_error)) as raised:
        classifier.classify_pieces(
            [Picture(np.zeros((8, 8, 3), dtype=np.uint8))]
        )

    assert raised.value is decoder_error


def test_classify_pieces_does_not_translate_model_execution_errors():
    classifier = PieceClassifier(base_model_class=FailingClassifierModelStub)

    with pytest.raises(RuntimeError, match="device execution failed") as raised:
        classifier.classify_pieces(
            [Picture(np.zeros((8, 8, 3), dtype=np.uint8))]
        )

    assert type(raised.value) is RuntimeError
```

- [ ] **Step 2: Run the PieceClassifier tests and verify the missing exception**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py
```

Expected: collection fails because `PieceDecodingError` does not exist.

- [ ] **Step 3: Translate errors only around constrained assignment**

Add private marker subclasses for the known status branches, while preserving
their documented built-in categories, and the public boundary exception:

```python
class _ConstrainedArgmaxTimeoutError(TimeoutError):
    pass


class _ConstrainedArgmaxRuntimeError(RuntimeError):
    pass


class PieceDecodingError(RuntimeError):
    pass
```

Raise only those private markers for the four expected non-optimal statuses;
leave any unexpected status as an ordinary `RuntimeError`:

```python
if status == cp_model.FEASIBLE:
    raise _ConstrainedArgmaxTimeoutError(
        "CP-SAT stopped before proving the solution optimal"
    )
if status == cp_model.UNKNOWN:
    raise _ConstrainedArgmaxTimeoutError(
        "CP-SAT timed out before finding a solution"
    )
if status == cp_model.INFEASIBLE:
    raise _ConstrainedArgmaxRuntimeError(
        "CP-SAT constraint model is infeasible"
    )
if status == cp_model.MODEL_INVALID:
    raise _ConstrainedArgmaxRuntimeError(
        "CP-SAT constraint model is invalid"
    )
if status != cp_model.OPTIMAL:
    raise RuntimeError(f"CP-SAT returned unexpected status: {status}")
```

Convert the tensor before the exception boundary, then catch only the two
private status markers around the final constrained assignment:

```python
logits = logits.cpu().numpy()
try:
    return constrained_argmax(logits)
except (
    _ConstrainedArgmaxTimeoutError,
    _ConstrainedArgmaxRuntimeError,
) as error:
    raise PieceDecodingError(str(error)) from error
```

The `try` block must contain only `constrained_argmax()`. Tensor preprocessing,
`self(tensor_image)`, and tensor conversion stay above it. Generic
`RuntimeError`/`TimeoutError` must never be caught at this boundary.

- [ ] **Step 4: Run the focused decoder tests**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py
```

Expected: all existing and new PieceClassifier tests pass.

- [ ] **Step 5: Commit the decoder boundary**

```bash
git add chessml/models/lightning/piece_classifier_model.py tests/test_piece_classifier_model.py
git commit -m "fix: distinguish piece decoding failures"
```

---

### Task 3: Replace the authoritative-FEN recognition contract

**Files:**

- Modify: `chessml/models/utils/board_recognition_helper.py:1-175`
- Modify: `tests/test_board_recognition_helper.py:1-93`

**Interfaces:**

- Consumes: `InvalidBoardGeometryError`, `PieceDecodingError`, the three existing models, `Picture`, `chess.Board`, and source-row classifier output.
- Produces:
  - `BoardOrientation.WHITE_AT_BOTTOM | BLACK_AT_BOTTOM`
  - `RecognitionFailureReason.NO_BOARD | INVALID_GEOMETRY | DECODING_FAILED | INVALID_PLACEMENT`
  - frozen `RecognitionSuccess(source_placement: str, board_image: Picture)`
  - frozen `RecognitionFailure(reason: RecognitionFailureReason)`
  - `RecognitionResult = RecognitionSuccess | RecognitionFailure`
  - `build_fen(result, *, orientation, turn, castling, en_passant, halfmove_clock, fullmove_number) -> str`
  - `BoardRecognitionHelper(board_detector, square_classifier, piece_classifier).recognize(Picture) -> RecognitionResult`

- [ ] **Step 1: Replace old helper tests with failing contract tests**

Replace `tests/test_board_recognition_helper.py` with:

```python
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from chess import BLACK, WHITE, Board, square

from chessml.data.assets import PIECE_CLASSES
from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import (
    InvalidBoardGeometryError,
)
from chessml.models.lightning.piece_classifier_model import PieceDecodingError
from chessml.models.utils.board_recognition_helper import (
    BoardOrientation,
    BoardRecognitionHelper,
    RecognitionFailure,
    RecognitionFailureReason,
    RecognitionSuccess,
    build_fen,
)


class BoardDetectorStub:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)

    def extract_board_image(self, original_image):
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return original_image


class SquareClassifierStub:
    def __init__(self, occupied):
        self.occupied = occupied

    def classify_squares(self, images):
        assert len(images) == 64
        return self.occupied


class PieceClassifierStub:
    def __init__(self, classes, error=None):
        self.classes = classes
        self.error = error

    def classify_pieces(self, images):
        if self.error is not None:
            raise self.error
        assert len(images) == len(self.classes)
        return self.classes


def classifier_outputs(position):
    board = Board(f"{position} w - - 0 1")
    occupied = []
    classes = []

    for row in range(8):
        for column in range(8):
            piece = board.piece_at(square(column, 7 - row))
            occupied.append(int(piece is not None))
            if piece is not None:
                classes.append(PIECE_CLASSES[piece.symbol()])

    return occupied, classes


def helper_for(position, board_detector=None, piece_error=None):
    occupied, classes = classifier_outputs(position)
    return BoardRecognitionHelper(
        board_detector=board_detector or BoardDetectorStub(),
        square_classifier=SquareClassifierStub(occupied),
        piece_classifier=PieceClassifierStub(classes, error=piece_error),
    )


def recognize(position, board_detector=None, piece_error=None):
    return helper_for(
        position,
        board_detector=board_detector,
        piece_error=piece_error,
    ).recognize(IMAGE)


def rotate_position(position):
    return "/".join(row[::-1] for row in reversed(position.split("/")))


IMAGE = Picture(np.zeros((8, 8, 3), dtype=np.uint8))
ASYMMETRIC_POSITION = "r3k2r/8/8/3p4/8/2N5/8/R3K2R"
CONTEXT_POSITION = "r3k2r/8/8/3pP3/8/8/8/R3K2R"
INVALID_POSITION = "8/8/8/8/8/8/4k3/4K3"
EMPTY_POSITION = "8/8/8/8/8/8/8/8"


def test_recognize_returns_a_frozen_source_oriented_success():
    result = recognize(ASYMMETRIC_POSITION)

    assert isinstance(result, RecognitionSuccess)
    assert result.source_placement == ASYMMETRIC_POSITION
    assert result.board_image is IMAGE
    assert not hasattr(result, "board")
    assert not hasattr(result, "get_fen")
    assert not hasattr(result, "flipped")
    assert set(RecognitionFailureReason) == {
        RecognitionFailureReason.NO_BOARD,
        RecognitionFailureReason.INVALID_GEOMETRY,
        RecognitionFailureReason.DECODING_FAILED,
        RecognitionFailureReason.INVALID_PLACEMENT,
    }

    with pytest.raises(FrozenInstanceError):
        result.source_placement = "8/8/8/8/8/8/8/8"

    failure = RecognitionFailure(RecognitionFailureReason.NO_BOARD)
    with pytest.raises(FrozenInstanceError):
        failure.reason = RecognitionFailureReason.INVALID_GEOMETRY


def test_history_changes_only_when_the_caller_supplies_different_context():
    result = RecognitionSuccess(ASYMMETRIC_POSITION, IMAGE)

    first = build_fen(
        result,
        orientation=BoardOrientation.WHITE_AT_BOTTOM,
        turn=WHITE,
        castling="KQkq",
        en_passant="-",
        halfmove_clock=4,
        fullmove_number=12,
    )
    second = build_fen(
        result,
        orientation=BoardOrientation.WHITE_AT_BOTTOM,
        turn=BLACK,
        castling="-",
        en_passant="-",
        halfmove_clock=9,
        fullmove_number=33,
    )

    assert first == f"{ASYMMETRIC_POSITION} w KQkq - 4 12"
    assert second == f"{ASYMMETRIC_POSITION} b - - 9 33"
    assert result.source_placement == ASYMMETRIC_POSITION


@pytest.mark.parametrize(
    ("source_placement", "orientation"),
    [
        (CONTEXT_POSITION, BoardOrientation.WHITE_AT_BOTTOM),
        (
            rotate_position(CONTEXT_POSITION),
            BoardOrientation.BLACK_AT_BOTTOM,
        ),
    ],
)
def test_build_fen_uses_explicit_orientation_and_preserves_every_field(
    source_placement,
    orientation,
):
    result = RecognitionSuccess(source_placement, IMAGE)

    fen = build_fen(
        result,
        orientation=orientation,
        turn=WHITE,
        castling="KQkq",
        en_passant="d6",
        halfmove_clock=7,
        fullmove_number=23,
    )

    assert fen == f"{CONTEXT_POSITION} w KQkq d6 7 23"


def test_build_fen_requires_every_context_argument():
    result = RecognitionSuccess(CONTEXT_POSITION, IMAGE)

    with pytest.raises(TypeError):
        build_fen(
            result,
            orientation=BoardOrientation.WHITE_AT_BOTTOM,
            turn=WHITE,
            castling="KQkq",
            en_passant="d6",
            halfmove_clock=7,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("orientation", "white"),
        ("turn", "w"),
        ("castling", "QK"),
        ("castling", "KK"),
        ("en_passant", "a3"),
        ("halfmove_clock", -1),
        ("fullmove_number", 0),
    ],
)
def test_build_fen_rejects_invalid_or_normalized_context(field, value):
    result = RecognitionSuccess(CONTEXT_POSITION, IMAGE)
    context = {
        "orientation": BoardOrientation.WHITE_AT_BOTTOM,
        "turn": WHITE,
        "castling": "KQkq",
        "en_passant": "d6",
        "halfmove_clock": 7,
        "fullmove_number": 23,
    }
    context[field] = value

    with pytest.raises(ValueError):
        build_fen(result, **context)


def test_recognize_maps_invalid_geometry_to_a_failure():
    result = recognize(
        ASYMMETRIC_POSITION,
        board_detector=BoardDetectorStub(
            [InvalidBoardGeometryError("bad geometry")]
        ),
    )

    assert result == RecognitionFailure(
        RecognitionFailureReason.INVALID_GEOMETRY
    )


def test_recognize_maps_constrained_decoding_to_a_failure():
    result = recognize(
        ASYMMETRIC_POSITION,
        piece_error=PieceDecodingError("decoder failed"),
    )

    assert result == RecognitionFailure(
        RecognitionFailureReason.DECODING_FAILED
    )


@pytest.mark.parametrize("position", [INVALID_POSITION, EMPTY_POSITION])
def test_recognize_rejects_a_structurally_invalid_placement(position):
    result = recognize(position)

    assert result == RecognitionFailure(
        RecognitionFailureReason.INVALID_PLACEMENT
    )


def test_a_failed_frame_does_not_prevent_the_next_recognition():
    detector = BoardDetectorStub(
        [InvalidBoardGeometryError("bad geometry"), None]
    )
    helper = helper_for(ASYMMETRIC_POSITION, board_detector=detector)

    first = helper.recognize(IMAGE)
    second = helper.recognize(IMAGE)

    assert first == RecognitionFailure(
        RecognitionFailureReason.INVALID_GEOMETRY
    )
    assert isinstance(second, RecognitionSuccess)
    assert second.source_placement == ASYMMETRIC_POSITION


def test_recognize_does_not_hide_model_execution_errors():
    with pytest.raises(RuntimeError, match="device execution failed") as raised:
        recognize(
            ASYMMETRIC_POSITION,
            piece_error=RuntimeError("device execution failed"),
        )

    assert type(raised.value) is RuntimeError
```

- [ ] **Step 2: Run the helper tests and verify the old API fails the new contract**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_helper.py
```

Expected: collection fails because the new result, orientation, failure, and builder names do not exist.

- [ ] **Step 3: Replace the helper with the minimal approved implementation**

Replace `chessml/models/utils/board_recognition_helper.py` with:

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, TypeAlias

import cv2
from chess import BLACK, WHITE, Board, Color

from chessml.data.assets import BOARD_SIZE, INVERTED_PIECE_CLASSES
from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import (
    BoardDetector,
    InvalidBoardGeometryError,
)
from chessml.models.lightning.piece_classifier_model import (
    PieceClassifier,
    PieceDecodingError,
)
from chessml.models.lightning.square_classifier_model import SquareClassifier


class BoardOrientation(Enum):
    WHITE_AT_BOTTOM = auto()
    BLACK_AT_BOTTOM = auto()


class RecognitionFailureReason(Enum):
    NO_BOARD = auto()
    INVALID_GEOMETRY = auto()
    DECODING_FAILED = auto()
    INVALID_PLACEMENT = auto()


@dataclass(frozen=True)
class RecognitionSuccess:
    source_placement: str
    board_image: Picture


@dataclass(frozen=True)
class RecognitionFailure:
    reason: RecognitionFailureReason


RecognitionResult: TypeAlias = RecognitionSuccess | RecognitionFailure


def _rotate_placement(placement: str) -> str:
    return "/".join(row[::-1] for row in reversed(placement.split("/")))


def _is_valid_source_placement(placement: str) -> bool:
    candidates = (placement, _rotate_placement(placement))
    for candidate in candidates:
        for turn in ("w", "b"):
            try:
                board = Board(f"{candidate} {turn} - - 0 1")
            except ValueError:
                continue
            if board.is_valid():
                return True
    return False


def _iterate_squares(
    board_image: Picture,
    square_size: int,
) -> Iterator[Picture]:
    resized = cv2.resize(
        board_image.cv2,
        (BOARD_SIZE * square_size, BOARD_SIZE * square_size),
        interpolation=cv2.INTER_CUBIC,
    )
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            yield Picture(
                resized[
                    row * square_size : (row + 1) * square_size,
                    column * square_size : (column + 1) * square_size,
                ]
            )


def _source_placement(class_indexes: list[int]) -> str:
    rows = []
    for start in range(0, len(class_indexes), BOARD_SIZE):
        row = []
        empty_count = 0
        for class_index in class_indexes[start : start + BOARD_SIZE]:
            if class_index == -1:
                empty_count += 1
                continue
            if empty_count:
                row.append(str(empty_count))
                empty_count = 0
            row.append(INVERTED_PIECE_CLASSES[class_index])
        if empty_count:
            row.append(str(empty_count))
        rows.append("".join(row))
    return "/".join(rows)


def build_fen(
    result: RecognitionSuccess,
    *,
    orientation: BoardOrientation,
    turn: Color,
    castling: str,
    en_passant: str,
    halfmove_clock: int,
    fullmove_number: int,
) -> str:
    if not isinstance(result, RecognitionSuccess):
        raise TypeError("build_fen requires RecognitionSuccess")
    if not isinstance(orientation, BoardOrientation):
        raise ValueError("Invalid board orientation")
    if turn is WHITE:
        active_color = "w"
    elif turn is BLACK:
        active_color = "b"
    else:
        raise ValueError("Turn must be chess.WHITE or chess.BLACK")
    if (
        not isinstance(castling, str)
        or (
            castling != "-"
            and (
                not castling
                or castling
                != "".join(symbol for symbol in "KQkq" if symbol in castling)
            )
        )
    ):
        raise ValueError("Castling must be '-' or a canonical KQkq subset")
    if not isinstance(en_passant, str):
        raise ValueError("En-passant must be '-' or a target square")
    if type(halfmove_clock) is not int or halfmove_clock < 0:
        raise ValueError("Halfmove clock must be a non-negative integer")
    if type(fullmove_number) is not int or fullmove_number <= 0:
        raise ValueError("Fullmove number must be a positive integer")

    placement = result.source_placement
    if orientation is BoardOrientation.BLACK_AT_BOTTOM:
        placement = _rotate_placement(placement)
    fen = (
        f"{placement} {active_color} {castling} {en_passant} "
        f"{halfmove_clock} {fullmove_number}"
    )

    try:
        board = Board(fen)
    except ValueError as error:
        raise ValueError("Invalid FEN context") from error
    if not board.is_valid():
        raise ValueError("FEN context is inconsistent with the placement")
    return fen


class BoardRecognitionHelper:
    def __init__(
        self,
        board_detector: BoardDetector,
        square_classifier: SquareClassifier,
        piece_classifier: PieceClassifier,
    ):
        self.board_detector = board_detector
        self.square_classifier = square_classifier
        self.piece_classifier = piece_classifier

    def recognize(self, original_image: Picture) -> RecognitionResult:
        try:
            board_image = self.board_detector.extract_board_image(original_image)
        except InvalidBoardGeometryError:
            return RecognitionFailure(
                RecognitionFailureReason.INVALID_GEOMETRY
            )

        squares = list(_iterate_squares(board_image, square_size=64))
        square_classes = self.square_classifier.classify_squares(
            [square.bw for square in squares]
        )
        if len(square_classes) != len(squares):
            raise ValueError("SquareClassifier must return one label per square")

        occupied_indexes = [
            index
            for index, square_class in enumerate(square_classes)
            if square_class == 1
        ]
        pieces = [squares[index].bw for index in occupied_indexes]

        if pieces:
            try:
                piece_classes = self.piece_classifier.classify_pieces(pieces)
            except PieceDecodingError:
                return RecognitionFailure(
                    RecognitionFailureReason.DECODING_FAILED
                )
        else:
            piece_classes = []

        if len(piece_classes) != len(occupied_indexes):
            raise ValueError(
                "PieceClassifier must return one label per occupied square"
            )

        class_indexes = [-1] * len(squares)
        for index, piece_class in zip(occupied_indexes, piece_classes):
            class_indexes[index] = piece_class

        source_placement = _source_placement(class_indexes)
        if not _is_valid_source_placement(source_placement):
            return RecognitionFailure(
                RecognitionFailureReason.INVALID_PLACEMENT
            )
        return RecognitionSuccess(
            source_placement=source_placement,
            board_image=board_image,
        )
```

This deletes the stale debugging counter, unused imports, public square iteration, provisional `chess.Board` result, MetaPredictor call, metadata synthesis, and canonical rotation from core recognition.

- [ ] **Step 4: Run the complete focused contract**

Run:

```bash
uv run --locked python -m pytest -q \
  tests/test_board_detector_model.py \
  tests/test_piece_classifier_model.py \
  tests/test_board_recognition_helper.py
```

Expected: all geometry, decoder, result, failure, placement, context, and propagation tests pass.

- [ ] **Step 5: Verify MetaPredictor and old result behavior are absent from the core helper**

Run:

```bash
if rg -n 'MetaPredictor|OnlyPieces|get_fen|flipped|self\.board|self\.c' chessml/models/utils/board_recognition_helper.py; then
  exit 1
fi
```

Expected: exit zero with no matches.

- [ ] **Step 6: Commit the core contract**

```bash
git add chessml/models/utils/board_recognition_helper.py tests/test_board_recognition_helper.py
git commit -m "refactor: expose observable recognition results"
```

---

### Task 4: Migrate the integration smoke and validator

**Files:**

- Modify: `tests/test_board_recognition_integration.py:1-96`
- Modify: `scripts/validate/validate_board_recognition.py:1-149`

**Interfaces:**

- Consumes: the Task 3 helper constructor and result union.
- Produces: a three-checkpoint integration smoke and validator output containing one text record per input: a source placement on success or `failure:<REASON>` on failure.

- [ ] **Step 1: Replace the four-model/FEN integration test**

Replace `tests/test_board_recognition_integration.py` with:

```python
from pathlib import Path

import pytest

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


ROOT = Path(__file__).resolve().parents[1]
BOARD_CHECKPOINT = ROOT / "checkpoints/bd-MobileViTV2FPN-v1.ckpt"
SQUARE_CHECKPOINT = ROOT / "checkpoints/sc-9-bs=64-step=23296.ckpt"
PIECE_CHECKPOINT = ROOT / "checkpoints/pc-48-bs=128-step=18944.ckpt"
FRAME = ROOT / "test_data/input_frames/book_long/1.png"

REQUIRED_ASSETS = [
    BOARD_CHECKPOINT,
    SQUARE_CHECKPOINT,
    PIECE_CHECKPOINT,
    FRAME,
]
MISSING_ASSETS = [path for path in REQUIRED_ASSETS if not path.is_file()]

pytestmark = pytest.mark.skipif(
    bool(MISSING_ASSETS),
    reason="requires ignored local inference assets: "
    + ", ".join(str(path.relative_to(ROOT)) for path in MISSING_ASSETS),
)


def test_existing_checkpoints_run_board_recognition_on_cpu():
    board_detector = BoardDetector.load_from_checkpoint(
        BOARD_CHECKPOINT,
        base_model_class=MobileViTV2FPN,
        base_model_kwargs={"pretrained": False},
        map_location="cpu",
        strict=True,
        weights_only=False,
    )
    square_classifier = SquareClassifier.load_from_checkpoint(
        SQUARE_CHECKPOINT,
        base_model_class=MobileNetV3SmallClassifier,
        base_model_kwargs={"pretrained": False},
        map_location="cpu",
        strict=True,
        weights_only=False,
    )
    piece_classifier = PieceClassifier.load_from_checkpoint(
        PIECE_CHECKPOINT,
        base_model_class=MobileNetV3LargeClassifier,
        base_model_kwargs={"pretrained": False},
        map_location="cpu",
        strict=True,
        weights_only=False,
    )

    for model in (board_detector, square_classifier, piece_classifier):
        model.eval()

    result = BoardRecognitionHelper(
        board_detector=board_detector,
        square_classifier=square_classifier,
        piece_classifier=piece_classifier,
    ).recognize(Picture(FRAME))

    assert isinstance(result, (RecognitionSuccess, RecognitionFailure))
```

- [ ] **Step 2: Run the local-checkpoint smoke**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_integration.py
```

Expected in the provisioned workspace: one test passes. A typed failure is a valid smoke result; this test does not assert accuracy.

- [ ] **Step 3: Replace the validator's duplicate/FEN path**

Replace `scripts/validate/validate_board_recognition.py` with:

```python
from glob import glob
from pathlib import Path

from tqdm import tqdm

from chessml import config, script
from chessml.data.assets import BOARD_COLORS, PIECE_SETS
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.images.picture import Picture
from chessml.data.utils.file_lines_dataset import FileLinesDataset
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
)
from chessml.utils import reset_dir, write_lines_to_txt


script.add_argument(
    "-i",
    dest="input_dir",
    type=str,
    default="./test_data/input_frames/book_long",
)
script.add_argument("-ss", dest="square_size", type=int, default=32)
script.add_argument("-d", dest="device", type=str, default="mps")


@script
def main(args):
    board_detector = BoardDetector.load_from_checkpoint(
        "./checkpoints/bd-MobileViTV2FPN-v1.ckpt",
        base_model_class=MobileViTV2FPN,
        base_model_kwargs={"pretrained": False},
        map_location=args.device,
        strict=True,
        weights_only=False,
    )
    square_classifier = SquareClassifier.load_from_checkpoint(
        "./checkpoints/sc-9-bs=64-step=23296.ckpt",
        base_model_class=MobileNetV3SmallClassifier,
        base_model_kwargs={"pretrained": False},
        map_location=args.device,
        strict=True,
        weights_only=False,
    )
    piece_classifier = PieceClassifier.load_from_checkpoint(
        "./checkpoints/pc-48-bs=128-step=18944.ckpt",
        base_model_class=MobileNetV3LargeClassifier,
        base_model_kwargs={"pretrained": False},
        map_location=args.device,
        strict=True,
        weights_only=False,
    )

    for model in (board_detector, square_classifier, piece_classifier):
        model.eval()

    helper = BoardRecognitionHelper(
        board_detector=board_detector,
        square_classifier=square_classifier,
        piece_classifier=piece_classifier,
    )

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = reset_dir(
            input_dir.parent / f"{input_dir.stem}_recognized"
        )

        for image_name in tqdm(sorted(glob(f"{input_dir}/*.png"))):
            image_path = Path(image_name)
            result = helper.recognize(Picture(image_path))
            text_path = output_dir / f"{image_path.name}.txt"

            if isinstance(result, RecognitionFailure):
                write_lines_to_txt(
                    text_path,
                    [f"failure:{result.reason.name}"],
                )
                continue

            result.board_image.pil.resize(
                (args.square_size * 8, args.square_size * 8)
            ).save(output_dir / image_path.name)
            write_lines_to_txt(text_path, [result.source_placement])
        return

    dataset = BoardsImagesFromFENs(
        fens=FileLinesDataset(path=Path(config.dataset.path) / "unique_fens.txt"),
        piece_sets=PIECE_SETS,
        board_colors=BOARD_COLORS,
        square_size=64,
        shuffle_seed=10,
        limit=1024,
    )

    for picture, fen, flipped in dataset:
        expected = fen.split()[0]
        if flipped:
            expected = "/".join(
                row[::-1] for row in reversed(expected.split("/"))
            )
        result = helper.recognize(picture)
        print("----------")
        print("expected source placement:", expected)
        if isinstance(result, RecognitionFailure):
            print("recognition failure:", result.reason.name)
        else:
            print("recognized source placement:", result.source_placement)
```

This intentionally removes direct marking/extraction before `recognize()`, which currently runs detection twice and bypasses typed geometry failure. It also removes MetaPredictor, `OnlyPieces`, FEN rendering, and orientation output.

- [ ] **Step 4: Compile the migrated active callers**

Run:

```bash
uv run --locked python -m compileall -q \
  chessml/models/utils/board_recognition_helper.py \
  scripts/validate/validate_board_recognition.py \
  tests/test_board_recognition_integration.py
```

Expected: exit zero with no output.

- [ ] **Step 5: Run a one-frame validator smoke**

Run:

```bash
mkdir -p /private/tmp/chessml-recognition-contract-smoke
cp test_data/input_frames/book_long/1.png \
  /private/tmp/chessml-recognition-contract-smoke/1.png
uv run --locked python scripts/validate/validate_board_recognition.py \
  -d cpu \
  -i /private/tmp/chessml-recognition-contract-smoke
test -s \
  /private/tmp/chessml-recognition-contract-smoke_recognized/1.png.txt
```

Expected: the validator exits zero and writes either one source-placement line or one `failure:<REASON>` line. It never writes an invented full FEN.

- [ ] **Step 6: Commit the caller migration**

```bash
git add \
  tests/test_board_recognition_integration.py \
  scripts/validate/validate_board_recognition.py
git commit -m "refactor: migrate recognition callers"
```

---

### Task 5: Synchronize documentation and run the full gate

**Files:**

- Modify: `README.md:74-76,103-129,176-313`

**Interfaces:**

- Consumes: the Task 3 public types/builder and Task 4 three-checkpoint validator.
- Produces: user-facing documentation that distinguishes observed source placement, caller-supplied FEN context, typed failures, and optional MetaPredictor priors.

- [ ] **Step 1: Correct BoardDetector and MetaPredictor claims**

Change the BoardDetector extraction comment to:

```python
# Returns an unskewed board image. Unusable detector geometry raises
# InvalidBoardGeometryError; this model has no no-board decision.
extracted_board_image = model.extract_board_image(source)
```

Change the MetaPredictor table description to:

```markdown
| MetaPredictor (CNN) | Experimental prior for castling rights, side to move, and source viewpoint from piece placement. These fields are not observable facts from a still image and are not used by core recognition. | 5.7MB | [.ckpt](https://drive.google.com/file/d/1ovmG0ZRKD29SG25iARTNWbxOCZdMAv5m/view?usp=drive_link) |
```

Replace the opening MetaPredictor paragraph with:

```markdown
`MetaPredictor` is a standalone experimental prior. It predicts castling
rights, side to move, and viewpoint from piece placement, but identical
placements can have different history and source orientation is ambiguous.
Core board recognition therefore does not invoke this model or put its output
into FEN.
```

Keep the standalone load/predict example, but delete its `castling`, `turn`, and `fen` construction after `meta_predictor.predict(representation(board))`. End the example with:

```python
# These booleans are uncalibrated priors, not image-observed FEN fields.
# An application may inspect them separately, but core recognition never
# rotates placement or constructs FEN from them.
```

- [ ] **Step 2: Replace the full-FEN recognition example**

Rename `Retrieving FEN from image` to `Retrieving piece placement from image` and replace that section's prose/code through the maintainer-check paragraph with:

````markdown
### Retrieving piece placement from image
<a name="-retrieving-piece-placement"></a>

A still image can establish the source-oriented 8x8 piece grid, but it cannot
establish orientation or history-dependent FEN fields. `BoardRecognitionHelper`
therefore returns either `RecognitionSuccess` or `RecognitionFailure`.

```python
from chess import WHITE

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
    BoardOrientation,
    BoardRecognitionHelper,
    RecognitionFailure,
    build_fen,
)

board_detector = BoardDetector.load_from_checkpoint(
    "./checkpoints/bd-MobileViTV2FPN-v1.ckpt",
    base_model_class=MobileViTV2FPN,
    base_model_kwargs={"pretrained": False},
    map_location="cpu",
    strict=True,
    weights_only=False,
)
square_classifier = SquareClassifier.load_from_checkpoint(
    "./checkpoints/sc-9-bs=64-step=23296.ckpt",
    base_model_class=MobileNetV3SmallClassifier,
    base_model_kwargs={"pretrained": False},
    map_location="cpu",
    strict=True,
    weights_only=False,
)
piece_classifier = PieceClassifier.load_from_checkpoint(
    "./checkpoints/pc-48-bs=128-step=18944.ckpt",
    base_model_class=MobileNetV3LargeClassifier,
    base_model_kwargs={"pretrained": False},
    map_location="cpu",
    strict=True,
    weights_only=False,
)

for model in (board_detector, square_classifier, piece_classifier):
    model.eval()

helper = BoardRecognitionHelper(
    board_detector=board_detector,
    square_classifier=square_classifier,
    piece_classifier=piece_classifier,
)
result = helper.recognize(Picture("./image.jpeg"))

if isinstance(result, RecognitionFailure):
    print("Recognition failed:", result.reason.name)
else:
    print("Observed source placement:", result.source_placement)

    # Construct full FEN only when the application already knows every field.
    fen = build_fen(
        result,
        orientation=BoardOrientation.WHITE_AT_BOTTOM,
        turn=WHITE,
        castling="-",
        en_passant="-",
        halfmove_clock=0,
        fullmove_number=1,
    )
```

The complete core example and validator require these exact pre-provisioned,
trusted checkpoint paths:

- `./checkpoints/bd-MobileViTV2FPN-v1.ckpt`
- `./checkpoints/sc-9-bs=64-step=23296.ckpt`
- `./checkpoints/pc-48-bs=128-step=18944.ckpt`

The public download table does not publish this full three-file set. If any
checkpoint is missing, do not source these pickle-bearing files from
untrusted locations and do not use `weights_only=False` on an untrusted file.
With the trusted checkpoints and local frames already provisioned, run:

```bash
uv run python scripts/validate/validate_board_recognition.py -d mps
```

The validator records source placement or a typed failure per frame. It is a
runtime smoke tool, not the P2-01 labeled accuracy evaluator.
````

Preserve the rest of README unchanged.

- [ ] **Step 3: Scan active code and documentation for the removed contract**

Run:

```bash
if rg -n \
  'result\.get_fen|result\.flipped|meta_predictor=|returns None if no board|extract the final FEN|four-file set' \
  README.md chessml scripts tests; then
  exit 1
fi
```

Expected: exit zero with no matches. Standalone MetaPredictor model/training references remain, as required.

- [ ] **Step 4: Run the focused and complete automated gates**

Run:

```bash
uv run --locked python -m pytest -q \
  tests/test_board_detector_model.py \
  tests/test_piece_classifier_model.py \
  tests/test_board_recognition_helper.py \
  tests/test_board_recognition_integration.py
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
```

Expected: every focused and repository test passes, and compileall exits zero with no output. The pre-change baseline was `27 passed in 7.04s`; any failure is a regression, not an accepted baseline.

- [ ] **Step 5: Run the full provisioned validator smoke**

Run:

```bash
uv run --locked python scripts/validate/validate_board_recognition.py -d mps
rg --files -g '*.txt' test_data/input_frames/book_long_recognized | wc -l
```

Expected: tqdm processes all 167 frames without an uncaught exception, and the count is exactly `167`. Each text file contains either one source placement or one `failure:<REASON>` record. This proves runtime continuity only; it is not recognition-accuracy evidence.

- [ ] **Step 6: Review scope and whitespace**

Run:

```bash
git diff --check
git diff --check origin/main..HEAD
git status --short --branch
git log --oneline --decorate -5
```

Expected: no whitespace errors; task commits contain only the files listed in this plan. README is the only pending tracked change. The pre-existing untracked audit/research/spec/plan documents remain unstaged and unchanged.

- [ ] **Step 7: Commit the documentation migration**

```bash
git add README.md
git commit -m "docs: explain observable recognition output"
```

- [ ] **Step 8: Confirm the final repository state**

Run:

```bash
git status --short --branch
git log --oneline --decorate -5
```

Expected: the five task commits are the newest commits. Only the pre-existing untracked audit/research/spec/plan documents remain in the worktree.

Do not push, publish, release, retrain, generate data, or start P2-01 in this plan.
