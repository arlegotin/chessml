# MetaPredictor Board Representation Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the stable 13-plane board input contract so the published `mp-MetaPredictor-v1.ckpt` works unchanged, then re-enable MetaPredictor throughout recognition, validation, documentation, and training without changing the 12-class image-piece classifier contract.

**Architecture:** Keep `PIECE_CLASSES` as the shared 12-way piece-only vocabulary. `OnlyPieces` owns the model-facing compatibility boundary by reserving channel 0 for empty squares and shifting the 12 piece IDs to channels 1–12; `FullPosition` then appends its six metadata planes to produce 19 channels. `BoardRecognitionHelper` requires a MetaPredictor, predicts metadata from the raw recognized placement, and rotates a flipped placement exactly once before constructing the canonical FEN. The existing checkpoint remains byte-for-byte untouched.

**Tech Stack:** Python 3.11, NumPy, python-chess, PyTorch/Lightning, OpenCV, pytest, conda environment `chessml`.

## Global Constraints

- Do not change `PIECE_CLASSES`, `PIECE_CLASSES_NUMBER`, PieceClassifier output width, or SquareClassifier behavior. Those are the newer 12-piece and binary-occupancy contracts.
- Do not patch, rewrite, or retrain `checkpoints/mp-MetaPredictor-v1.ckpt`.
- Do not add a silent metadata fallback. Recognition must fail clearly at construction/load time if MetaPredictor is unavailable.
- Preserve the historical board tensor orientation: channels-first `(C, 8, 8)`, with A8 at `[channel, 0, 0]` and H1 at `[channel, 7, 7]`.
- Interpret `RecognitionResult.flipped` as the source image viewpoint. The returned FEN placement must always be canonical White-side orientation.
- Keep unrelated MetaPredictor loss-function, script-argument, and repository-version issues out of scope.
- Run every Python/test command after `conda activate chessml`; use CPU for the validation script because MPS is unavailable in the target environment.
- Preserve the known baseline failure in `tests/test_chessml.py` (`0.1.0` expected while the package is `0.1.1`); do not conceal or fix it as part of this change.

## File Structure

- Modify `chessml/data/boards/board_representation.py`: restore the 13/19-channel model input schemas.
- Create `tests/test_board_representation.py`: lock channel count, class offsets, orientation, dtype, one-hotness, and metadata-plane indices.
- Modify `chessml/models/utils/board_recognition_helper.py`: make MetaPredictor required and restore prediction/canonicalization.
- Create `tests/test_board_recognition_helper.py`: verify unflipped and 180-degree-flipped recognition with asymmetric positions.
- Modify `scripts/validate/validate_board_recognition.py`: load/pass/evaluate MetaPredictor and render from the detected viewpoint.
- Modify `scripts/train/train_meta_predictor.py`: use IterableDataset-safe loading and the metric actually logged by MetaPredictor.
- Modify `README.md`: synchronize direct inference and complete helper wiring examples.

---

### Task 1: Restore the historical board tensor schema

**Files:**

- Create: `tests/test_board_representation.py`
- Modify: `chessml/data/boards/board_representation.py:29-115`

- [ ] **Step 1: Add failing board-representation contract tests**

Create `tests/test_board_representation.py` with:

```python
import numpy as np
from chess import Board, Piece, square_file, square_rank

from chessml.data.assets import PIECE_CLASSES
from chessml.data.boards.board_representation import FullPosition, OnlyPieces


def test_only_pieces_uses_empty_zero_and_shifted_piece_channels():
    board = Board.empty()
    placements = []

    for square_index, (symbol, class_index) in enumerate(PIECE_CLASSES.items()):
        board.set_piece_at(square_index, Piece.from_symbol(symbol))
        placements.append((square_index, class_index))

    encoded = OnlyPieces()(board)

    assert encoded.shape == (13, 8, 8)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(
        encoded.sum(axis=0), np.ones((8, 8), dtype=np.float32)
    )
    assert encoded[0].sum() == 64 - len(PIECE_CLASSES)

    for square_index, class_index in placements:
        row = 7 - square_rank(square_index)
        column = square_file(square_index)
        assert encoded[class_index + 1, column, row] == 1


def test_full_position_appends_metadata_after_all_thirteen_piece_planes():
    board = Board()
    pieces = OnlyPieces()(board)
    encoded = FullPosition()(board)

    assert encoded.shape == (19, 8, 8)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[:13], pieces)
    assert pieces[0].sum() == 32
    assert pieces[PIECE_CLASSES["K"] + 1].sum() == 1
    np.testing.assert_array_equal(encoded[13], np.zeros((8, 8), dtype=np.float32))
    np.testing.assert_array_equal(
        encoded[14:18], np.ones((4, 8, 8), dtype=np.float32)
    )
    np.testing.assert_array_equal(encoded[18], np.ones((8, 8), dtype=np.float32))
```

- [ ] **Step 2: Run the tests and confirm the regression is exposed**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
pytest -q tests/test_board_representation.py
```

Expected: both tests fail because the branch currently returns `(12, 8, 8)` and `(18, 8, 8)`, and empty squares alias the last piece channel.

- [ ] **Step 3: Restore the compatibility boundary in `OnlyPieces`**

Replace the per-square class selection and one-hot width with:

```python
for square in SQUARES_180:
    piece = board.piece_at(square)
    piece_class = 0 if piece is None else PIECE_CLASSES[piece.symbol()] + 1
    pieces.append(piece_class)

reshaped = np.reshape(pieces, (BOARD_SIZE, BOARD_SIZE))

pieces = np.eye(PIECE_CLASSES_NUMBER + 1, dtype=np.float32)[reshaped]
pieces = np.swapaxes(pieces, 2, 0)

return pieces
```

This deliberately leaves `PIECE_CLASSES` at `0..11`; only the board representation adds the empty channel and offset.

Correct the `FullPosition` docstring from `13x8x8` to `19x8x8`. Do not reorder the appended planes: en passant remains channel 13, castling remains 14–17, and turn remains 18.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
pytest -q tests/test_board_representation.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the schema restoration**

```bash
git add chessml/data/boards/board_representation.py tests/test_board_representation.py
git commit -m "fix: restore board representation schema"
```

---

### Task 2: Restore MetaPredictor metadata and orientation handling

**Files:**

- Create: `tests/test_board_recognition_helper.py`
- Modify: `chessml/models/utils/board_recognition_helper.py:64-189`

- [ ] **Step 1: Add failing helper tests with an asymmetric board**

Create `tests/test_board_recognition_helper.py` with:

```python
import numpy as np
from chess import Board, square

from chessml.data.assets import PIECE_CLASSES
from chessml.data.images.picture import Picture
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper


class BoardDetectorStub:
    def extract_board_image(self, original_image):
        return original_image


class SquareClassifierStub:
    def __init__(self, occupied):
        self.occupied = occupied

    def classify_squares(self, images):
        assert len(images) == 64
        return self.occupied


class PieceClassifierStub:
    def __init__(self, classes):
        self.classes = classes

    def classify_pieces(self, images):
        assert len(images) == len(self.classes)
        return self.classes


class MetaPredictorStub:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, board):
        assert board.shape == (13, 8, 8)
        assert board.dtype == np.float32
        return self.prediction


def classifier_outputs(board):
    occupied = []
    classes = []

    for row in range(8):
        for column in range(8):
            piece = board.piece_at(square(column, 7 - row))
            occupied.append(int(piece is not None))
            if piece is not None:
                classes.append(PIECE_CLASSES[piece.symbol()])

    return occupied, classes


def recognize(position, prediction):
    source_board = Board(f"{position} w - - 0 1")
    occupied, classes = classifier_outputs(source_board)
    helper = BoardRecognitionHelper(
        board_detector=BoardDetectorStub(),
        square_classifier=SquareClassifierStub(occupied),
        piece_classifier=PieceClassifierStub(classes),
        meta_predictor=MetaPredictorStub(prediction),
    )

    return helper.recognize(Picture(np.zeros((8, 8, 3), dtype=np.uint8)))


def rotate_position(position):
    return "/".join(row[::-1] for row in reversed(position.split("/")))


CANONICAL_POSITION = "r3k2r/8/8/3p4/8/2N5/8/R3K2R"


def test_recognize_uses_meta_prediction_for_unflipped_position():
    result = recognize(
        CANONICAL_POSITION,
        (True, True, True, True, True, False),
    )

    assert result.get_fen() == f"{CANONICAL_POSITION} w KQkq - 0 1"
    assert result.flipped is False


def test_recognize_canonicalizes_a_flipped_position_exactly_once():
    result = recognize(
        rotate_position(CANONICAL_POSITION),
        (True, True, True, True, False, True),
    )

    assert result.get_fen() == f"{CANONICAL_POSITION} b KQkq - 0 1"
    assert result.flipped is True
```

The pawn and knight make the position non-symmetric, so an omitted or double 180-degree rotation cannot accidentally pass.

- [ ] **Step 2: Run the tests and confirm MetaPredictor is currently disconnected**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
pytest -q tests/test_board_recognition_helper.py
```

Expected: both tests fail because `BoardRecognitionHelper.__init__` does not accept `meta_predictor`.

- [ ] **Step 3: Make MetaPredictor a required helper dependency**

Change the constructor to accept and retain the existing model type:

```python
def __init__(
    self,
    board_detector: BoardDetector,
    square_classifier: SquareClassifier,
    piece_classifier: PieceClassifier,
    meta_predictor: MetaPredictor,
):
    self.board_detector = board_detector
    self.piece_classifier = piece_classifier
    self.square_classifier = square_classifier
    self.meta_predictor = meta_predictor
    self.c = 0
```

Do not make it optional and do not keep the six hard-coded booleans as a fallback.

- [ ] **Step 4: Restore prediction and canonicalize only the placement**

After constructing the provisional placement-only board, restore:

```python
(
    white_kingside_castling,
    white_queenside_castling,
    black_kingside_castling,
    black_queenside_castling,
    white_turn,
    flipped,
) = self.meta_predictor.predict(OnlyPieces()(result.board))
```

Delete the hard-coded metadata tuple. Before the final `set_fen`, restore:

```python
if flipped:
    fen_position = "/".join(row[::-1] for row in fen_rows[::-1])
```

Keep `result.flipped = flipped`. Do not rotate the board tensor before calling MetaPredictor; the model needs the source orientation to predict `flipped`.

- [ ] **Step 5: Run both focused suites**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
pytest -q tests/test_board_representation.py tests/test_board_recognition_helper.py
```

Expected: `4 passed`.

- [ ] **Step 6: Commit helper behavior**

```bash
git add chessml/models/utils/board_recognition_helper.py tests/test_board_recognition_helper.py
git commit -m "fix: restore MetaPredictor recognition metadata"
```

---

### Task 3: Re-enable downstream validation and repair upstream training wiring

**Files:**

- Modify: `scripts/validate/validate_board_recognition.py:71-116`
- Modify: `scripts/train/train_meta_predictor.py:65-72`
- Modify: `README.md:178-255`

- [ ] **Step 1: Re-enable the real checkpoint in the validation script**

Uncomment the MetaPredictor load/eval block:

```python
meta_predictor = MetaPredictor.load_from_checkpoint(
    "./checkpoints/mp-MetaPredictor-v1.ckpt",
    input_shape=OnlyPieces().shape,
    map_location=args.device,
)
meta_predictor.eval()
```

Pass `meta_predictor=meta_predictor` to `BoardRecognitionHelper`. Change the generated comparison board to:

```python
flipped=result.flipped,
```

This keeps the returned FEN canonical while rendering the comparison in the same viewpoint as the source image.

- [ ] **Step 2: Make MetaPredictor training compatible with its iterable dataset and metric**

Add these keyword arguments only to the `standard_training` call in `scripts/train/train_meta_predictor.py`:

```python
shuffle=False,
checkpoint_monitor="val_loss",
```

`BoardsFromFEN` is an `IterableDataset`, so PyTorch rejects `shuffle=True`. MetaPredictor logs `val_loss`, while the shared helper default is `val/loss`. Keep the shared defaults unchanged because other trainers intentionally use them.

- [ ] **Step 3: Synchronize the README examples**

In the direct MetaPredictor example, replace both stale `model` references:

```python
meta_predictor.eval()
```

and:

```python
) = meta_predictor.predict(representation(board))
```

Replace the `BoardRecognitionHelper` example with this complete, executable wiring:

```python
from chessml.data.boards.board_representation import OnlyPieces
from chessml.data.images.picture import Picture
from chessml.models.lightning.board_detector_model import BoardDetector
from chessml.models.lightning.meta_predictor_model import MetaPredictor
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.lightning.square_classifier_model import SquareClassifier
from chessml.models.torch.vision_model_adapter import (
    MobileNetV3LargeClassifier,
    MobileNetV3SmallClassifier,
    MobileViTV2FPN,
)
from chessml.models.utils.board_recognition_helper import BoardRecognitionHelper

board_detector = BoardDetector.load_from_checkpoint(
    "./checkpoints/bd-MobileViTV2FPN-v1.ckpt",
    base_model_class=MobileViTV2FPN,
    map_location="cpu",
)
board_detector.eval()

square_classifier = SquareClassifier.load_from_checkpoint(
    "./checkpoints/sc-9-bs=64-step=23296.ckpt",
    base_model_class=MobileNetV3SmallClassifier,
    map_location="cpu",
)
square_classifier.eval()

piece_classifier = PieceClassifier.load_from_checkpoint(
    "./checkpoints/pc-48-bs=128-step=18944.ckpt",
    base_model_class=MobileNetV3LargeClassifier,
    map_location="cpu",
)
piece_classifier.eval()

meta_predictor = MetaPredictor.load_from_checkpoint(
    "./checkpoints/mp-MetaPredictor-v1.ckpt",
    input_shape=OnlyPieces().shape,
    map_location="cpu",
)
meta_predictor.eval()

helper = BoardRecognitionHelper(
    board_detector=board_detector,
    square_classifier=square_classifier,
    piece_classifier=piece_classifier,
    meta_predictor=meta_predictor,
)

source = Picture("./image.jpeg")
result = helper.recognize(source)

fen = result.get_fen()
viewed_from_whites_perspective = not result.flipped
```

- [ ] **Step 4: Verify strict checkpoint compatibility without mutating it**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
python - <<'PY'
from chess import Board

from chessml.data.boards.board_representation import OnlyPieces
from chessml.models.lightning.meta_predictor_model import MetaPredictor

representation = OnlyPieces()
model = MetaPredictor.load_from_checkpoint(
    "checkpoints/mp-MetaPredictor-v1.ckpt",
    input_shape=representation.shape,
    map_location="cpu",
    strict=True,
).eval()
prediction = model.predict(representation(Board()))
assert representation.shape == (13, 8, 8)
assert len(prediction) == 6
print(representation.shape, prediction)
PY
```

Expected: strict loading succeeds, then prints `(13, 8, 8)` and a six-boolean prediction. No checkpoint file is written.

- [ ] **Step 5: Smoke-test the iterable training data path**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader

from chessml.data.boards.board_representation import OnlyPieces
from chessml.data.boards.boards_from_fen import BoardsFromFEN

with TemporaryDirectory() as directory:
    path = Path(directory) / "fens.txt"
    path.write_text("8/8/8/8/8/8/8/K6k w - - 0 1\n")
    dataset = BoardsFromFEN(path=path, transforms=[OnlyPieces()])
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    assert tuple(batch.shape) == (1, 13, 8, 8)
    print(tuple(batch.shape))
PY
python -m py_compile scripts/train/train_meta_predictor.py scripts/validate/validate_board_recognition.py
```

Expected: `(1, 13, 8, 8)` and no compile errors. Review the training diff to confirm the call site passes both `shuffle=False` and `checkpoint_monitor="val_loss"`.

- [ ] **Step 6: Commit pipeline and documentation wiring**

```bash
git add scripts/validate/validate_board_recognition.py scripts/train/train_meta_predictor.py README.md
git commit -m "fix: re-enable MetaPredictor pipeline"
```

---

### Task 4: Run regression and end-to-end verification

**Files:**

- Verify: all files changed in Tasks 1–3
- Generated/ignored: `test_data/input_frames/book_short_marked/`, `book_short_extracted/`, `book_short_boards/`, and `book_short_squares/`

- [ ] **Step 1: Run all new regression tests together**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
pytest -q tests/test_board_representation.py tests/test_board_recognition_helper.py
```

Expected: `4 passed`.

- [ ] **Step 2: Run the requested real recognition validation on CPU**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
python scripts/validate/validate_board_recognition.py -d cpu
```

Expected: exit code 0 after processing all eight PNG inputs. MetaPredictor must be visibly active in the script—no commented load/pass lines and no hard-coded helper fallback. Matplotlib font-cache and Lightning checkpoint-upgrade notices are benign.

- [ ] **Step 3: Validate every generated FEN**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
python - <<'PY'
from pathlib import Path

from chess import Board

paths = sorted(Path("test_data/input_frames/book_short_boards").glob("*.png.txt"))
assert len(paths) == 8, paths
for path in paths:
    Board(path.read_text().strip())
print("validated 8 recognized FENs")
PY
```

Expected: `validated 8 recognized FENs`.

- [ ] **Step 4: Run the full test suite and classify the known unrelated failure**

Run:

```bash
source /Users/artemlegotin/anaconda3/etc/profile.d/conda.sh
conda activate chessml
pytest -q
```

Expected repository baseline: the four new tests pass; `tests/test_chessml.py::test_version` still fails because it expects `0.1.0` while `chessml.__version__` is `0.1.1`. Any additional failure is a regression and must be fixed before continuing.

- [ ] **Step 5: Review scope, whitespace, and repository state**

Run:

```bash
git diff --check HEAD~3..HEAD
git diff --stat HEAD~3..HEAD
git diff HEAD~3..HEAD -- \
  chessml/data/boards/board_representation.py \
  chessml/models/utils/board_recognition_helper.py \
  scripts/validate/validate_board_recognition.py \
  scripts/train/train_meta_predictor.py \
  tests/test_board_representation.py \
  tests/test_board_recognition_helper.py \
  README.md
git status --short
```

Expected: no whitespace errors; only the seven planned source/test/doc paths appear in the implementation diff; working tree is clean after the three task commits. Confirm `git status --ignored --short checkpoints/mp-MetaPredictor-v1.ckpt` still reports the checkpoint as ignored rather than modified/tracked.

- [ ] **Step 6: Perform the completion review**

Before claiming completion, invoke `superpowers:verification-before-completion`, report the exact test/validation outcomes, explicitly distinguish the pre-existing version-test failure, and confirm the checkpoint was neither patched nor retrained.
