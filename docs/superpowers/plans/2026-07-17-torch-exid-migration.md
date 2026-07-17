# `torch_exid` Compatibility Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `torch_exid.ExtendedIterableDataset` with one local, behavior-compatible PyTorch iterable dataset while preserving board-recognition training imports, data behavior, Lightning checkpoints, and the exact MPS validator command.

**Architecture:** Add one internal base class under the existing `chessml.data` package, implemented with `torch.utils.data.IterableDataset` and `random.Random`. Characterization tests lock the upstream behavior before the eight consumers switch imports; no model, trainer, checkpoint, or unrelated dependency changes. Verification exercises the affected dataset seams, protected dependency versions, Lightning entry points, real checkpoints, and the complete MPS validator.

**Tech Stack:** Python 3.11.12, PyTorch 2.7.0, torchvision 0.22.0, Lightning 2.5.1.post0, timm 1.0.15, python-chess, pytest, Conda environment `chessml`, macOS MPS.

## Global Constraints

- Use the approved design in `docs/superpowers/specs/2026-07-17-torch-exid-migration-design.md` as the behavior contract.
- Do not upgrade Python, PyTorch, torchvision, Lightning, timm, OpenCV, torchmetrics, OR-Tools, or any other dependency in this change.
- Do not add `torchdata` or another replacement dependency; use the standard library and the already-installed PyTorch base class.
- Preserve transform, `skip_next`, offset, limit, fixed-chunk shuffle, generated-seed, natural-reset, and single-worker behavior from `torch-exid` 0.1.5.
- Preserve the existing dataset-instance state limitation for early-abandoned/concurrent iterators and the existing duplicate-stream behavior with multiple DataLoader workers. Do not add sharding in this migration.
- Keep all models, tensor layouts, timm identifiers, preprocessing, Lightning hooks, Trainer configuration, and checkpoint formats unchanged.
- Do not fix the known dynamic BoardDetector offset/limit, iterable shuffle, checkpoint-monitor, validator mutation, or package-version defects.
- Never import a `scripts/train/*.py` module directly: the module-level `@script` decorator parses arguments and can start training. Use `--help` for side-effect-safe import smoke checks.
- Run Python commands from the repository root after `conda activate chessml` unless a step explicitly performs activation.
- The final acceptance gate is exactly `python scripts/validate/validate_board_recognition.py -d mps` in the activated `chessml` environment.
- Preserve the known baseline failure in `tests/test_chessml.py` (`0.1.0` expected while the package is `0.1.1`); record it without skipping or changing it.

## File Structure

- Create `chessml/data/iterable_dataset.py`: internal `ExtendedIterableDataset` compatibility implementation and the only new production module.
- Create `tests/test_iterable_dataset.py`: upstream behavior characterization plus `FileLinesDataset` and `BoardsFromFEN` integration coverage.
- Modify `chessml/data/utils/file_lines_dataset.py`: import the local base.
- Modify `chessml/data/games/games_from_pgn.py`: import the local base.
- Modify `chessml/data/boards/boards_from_games.py`: import the local base.
- Modify `chessml/data/policy/policies_from_games.py`: import the local base.
- Modify `chessml/data/images/boards_images_from_fens.py`: import the local base.
- Modify `chessml/data/images/augmented_boards_images.py`: import the local base.
- Modify `chessml/data/images/pieces_images.py`: import the local base.
- Modify `scripts/train/train_value_model.py`: import the local base while retaining `BoardsAndValues.skip_next()` behavior.
- Modify `requirements.txt`: remove only `torch-exid>=0.1.5`.

---

### Task 1: Implement the local compatibility class from behavior tests

**Files:**

- Create: `tests/test_iterable_dataset.py`
- Create: `chessml/data/iterable_dataset.py`

**Interfaces:**

- Consumes: `torch.utils.data.IterableDataset`; Python `random.Random`; subclass implementations of `generator() -> Iterator[Any]`.
- Produces: `ExtendedIterableDataset(shuffle_buffer=1, shuffle_seed=None, offset=0, limit=-1, transforms=None, transforms_required=False, *args, **kwargs)`; `skip_next() -> None`; `generator() -> Iterator[Any]`; `__iter__() -> Iterator[Any]`.

- [ ] **Step 1: Create the failing upstream-behavior tests**

Create `tests/test_iterable_dataset.py` with:

```python
import pytest
from torch.utils.data import DataLoader, IterableDataset

from chessml.data.iterable_dataset import ExtendedIterableDataset


class IntegersDataset(ExtendedIterableDataset):
    def __init__(self, stop=8, skip_odd=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop
        self.skip_odd = skip_odd

    def generator(self):
        for value in range(self.stop):
            if self.skip_odd and value % 2:
                self.skip_next()
            yield value


class MutableDataset(ExtendedIterableDataset):
    def generator(self):
        state = {"value": -1}

        for value in range(3):
            state["value"] = value
            yield state
            state["value"] = -1


def test_default_order_and_zero_finite_and_unlimited_limits():
    assert list(IntegersDataset(stop=4)) == [0, 1, 2, 3]
    assert list(IntegersDataset(stop=6, offset=2, limit=3)) == [2, 3, 4]
    assert list(IntegersDataset(stop=6, limit=0)) == []


def test_applies_transforms_left_to_right_before_offset_and_limit():
    seen = []

    def record_and_add_ten(value):
        seen.append(value)
        return value + 10

    dataset = IntegersDataset(
        stop=5,
        offset=2,
        limit=2,
        transforms=[record_and_add_ten, lambda value: value * 2],
    )

    assert list(dataset) == [24, 26]
    assert seen == [0, 1, 2, 3]


@pytest.mark.parametrize("transforms", [None, []])
def test_requires_nonempty_transforms_when_requested(transforms):
    with pytest.raises(ValueError, match="requires transforms"):
        IntegersDataset(transforms_required=True, transforms=transforms)


def test_skip_next_is_transformed_and_does_not_consume_offset_or_limit():
    seen = []

    def record(value):
        seen.append(value)
        return value

    dataset = IntegersDataset(
        stop=10,
        skip_odd=True,
        offset=1,
        limit=3,
        transforms=[record],
    )

    assert list(dataset) == [2, 4, 6]
    assert seen == list(range(7))


def test_seeded_shuffle_repeats_the_same_permutation_for_every_chunk():
    dataset = IntegersDataset(
        stop=8,
        shuffle_buffer=3,
        shuffle_seed=42,
    )
    expected = [1, 0, 2, 4, 3, 5, 7, 6]

    assert list(dataset) == expected
    assert list(dataset) == expected


def test_generated_seed_is_stable_for_full_reiteration():
    dataset = IntegersDataset(stop=8, shuffle_buffer=3)
    first = list(dataset)

    assert list(dataset) == first
    assert sorted(first) == list(range(8))


def test_transforms_snapshot_mutable_items_before_the_generator_resumes():
    dataset = MutableDataset(transforms=[lambda state: state["value"]])

    assert list(dataset) == [0, 1, 2]


def test_default_generator_reports_missing_subclass_implementation():
    with pytest.raises(NotImplementedError, match="implement generator"):
        list(ExtendedIterableDataset())


def test_remains_a_single_worker_iterable_dataset_and_leaves_batching_to_torch():
    dataset = IntegersDataset(stop=5)
    loader = DataLoader(dataset, batch_size=2, num_workers=0)

    assert isinstance(dataset, IterableDataset)
    assert [batch.tolist() for batch in loader] == [[0, 1], [2, 3], [4]]
```

- [ ] **Step 2: Run the test to prove the local module is absent**

Run:

```bash
python -m pytest -q tests/test_iterable_dataset.py
```

Expected: test collection fails with `ModuleNotFoundError: No module named 'chessml.data.iterable_dataset'`.

- [ ] **Step 3: Add the minimum compatibility implementation**

Create `chessml/data/iterable_dataset.py` with:

```python
from random import Random
from typing import Any, Callable, Iterator

from torch.utils.data import IterableDataset


class ExtendedIterableDataset(IterableDataset):
    def __init__(
        self,
        shuffle_buffer: int = 1,
        shuffle_seed: int | None = None,
        offset: int = 0,
        limit: int = -1,
        transforms: list[Callable[[Any], Any]] | None = None,
        transforms_required: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if transforms_required and not transforms:
            raise ValueError("ExtendedIterableDataset requires transforms")

        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = (
            shuffle_seed
            if shuffle_seed is not None
            else Random().randint(0, 999331)
        )
        self.buffer: list[Any] = []
        self.offset = offset
        self.limit = limit
        self.counter = 0
        self._skip_next = False
        self.transforms = [] if transforms is None else transforms

    def skip_next(self) -> None:
        self._skip_next = True

    def generator(self) -> Iterator[Any]:
        raise NotImplementedError("please implement generator method")

    @property
    def _limit_allows_one_more(self) -> bool:
        return self.limit < 0 or self.counter < self.limit + self.offset

    def _items_with_conditions(self) -> Iterator[Any]:
        source = self.generator()

        while True:
            try:
                if not self._limit_allows_one_more:
                    raise StopIteration

                item = next(source)

                for transform in self.transforms:
                    item = transform(item)

                if self._skip_next:
                    self._skip_next = False
                    continue

                self.counter += 1

                if self.counter - 1 < self.offset:
                    continue

                yield item
            except StopIteration:
                break

    def _flush_buffer(self) -> Iterator[Any]:
        Random(self.shuffle_seed).shuffle(self.buffer)

        for item in self.buffer:
            yield item

        self.buffer = []

    def _iterate(self) -> Iterator[Any]:
        if self.shuffle_buffer > 1:
            for item in self._items_with_conditions():
                self.buffer.append(item)

                if len(self.buffer) >= self.shuffle_buffer:
                    yield from self._flush_buffer()

            if self.buffer:
                yield from self._flush_buffer()
        else:
            yield from self._items_with_conditions()

        self.counter = 0
        self.buffer = []
        self._skip_next = False

    def __iter__(self) -> Iterator[Any]:
        return self._iterate()
```

Do not replace the dataset fields with iterator-local state or a `finally`
reset. The approved migration deliberately retains upstream early-abandon and
concurrent-iterator behavior rather than mixing a bug fix into dependency
removal.

- [ ] **Step 4: Run the focused behavior tests**

Run:

```bash
python -m pytest -q tests/test_iterable_dataset.py
```

Expected: all tests pass.

- [ ] **Step 5: Review and commit the local behavior boundary**

Run:

```bash
git diff --check
git diff -- chessml/data/iterable_dataset.py tests/test_iterable_dataset.py
git add chessml/data/iterable_dataset.py tests/test_iterable_dataset.py
git commit -m "refactor: add local iterable dataset"
```

Expected: the diff contains only the local class and its behavior tests; the commit succeeds.

---

### Task 2: Switch every consumer and remove the external dependency

**Files:**

- Modify: `tests/test_iterable_dataset.py`
- Modify: `chessml/data/utils/file_lines_dataset.py:4`
- Modify: `chessml/data/games/games_from_pgn.py:7`
- Modify: `chessml/data/boards/boards_from_games.py:2`
- Modify: `chessml/data/policy/policies_from_games.py:2`
- Modify: `chessml/data/images/boards_images_from_fens.py:2`
- Modify: `chessml/data/images/augmented_boards_images.py:1`
- Modify: `chessml/data/images/pieces_images.py:4`
- Modify: `scripts/train/train_value_model.py:12`
- Modify: `requirements.txt:15`

**Interfaces:**

- Consumes: `chessml.data.iterable_dataset.ExtendedIterableDataset` from Task 1.
- Produces: all existing ChessML iterable dataset classes inherit the local base; no production source or manifest imports/requires `torch_exid`.

- [ ] **Step 1: Add failing project-integration tests**

Add these imports near the top of `tests/test_iterable_dataset.py`:

```python
from chess import Board

from chessml.data.boards.boards_from_fen import BoardsFromFEN
from chessml.data.utils.file_lines_dataset import FileLinesDataset
```

Append these tests:

```python
def test_file_lines_dataset_uses_the_local_pipeline(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("zero\none\ntwo\nthree\n")
    dataset = FileLinesDataset(
        path=path,
        transforms=[str.upper],
        offset=1,
        limit=2,
    )

    assert isinstance(dataset, ExtendedIterableDataset)
    assert list(dataset) == ["ONE", "TWO"]


def test_boards_from_fen_snapshots_reused_boards_in_a_dataloader(tmp_path):
    first = Board()
    second = first.copy()
    second.push_san("e4")
    path = tmp_path / "fens.txt"
    path.write_text(f"{first.fen()}\n{second.fen()}\n")
    dataset = BoardsFromFEN(path=path, transforms=[lambda board: board.fen()])
    loader = DataLoader(dataset, batch_size=2, num_workers=0)

    assert isinstance(dataset, ExtendedIterableDataset)
    batch = next(iter(loader))
    assert batch == [first.fen(), second.fen()]
```

- [ ] **Step 2: Run only the new integration tests and prove imports still use the external class**

Run:

```bash
python -m pytest -q \
  tests/test_iterable_dataset.py::test_file_lines_dataset_uses_the_local_pipeline \
  tests/test_iterable_dataset.py::test_boards_from_fen_snapshots_reused_boards_in_a_dataloader
```

Expected: both tests fail at the `isinstance(..., ExtendedIterableDataset)` assertions because `FileLinesDataset` still inherits the external class.

- [ ] **Step 3: Replace all eight imports**

In each of these files:

```text
chessml/data/utils/file_lines_dataset.py
chessml/data/games/games_from_pgn.py
chessml/data/boards/boards_from_games.py
chessml/data/policy/policies_from_games.py
chessml/data/images/boards_images_from_fens.py
chessml/data/images/augmented_boards_images.py
chessml/data/images/pieces_images.py
scripts/train/train_value_model.py
```

replace the exact import:

```python
from torch_exid import ExtendedIterableDataset
```

with:

```python
from chessml.data.iterable_dataset import ExtendedIterableDataset
```

Do not change any constructor arguments, generator bodies, transforms,
shuffle settings, offsets, limits, or `skip_next()` calls.

- [ ] **Step 4: Remove only the obsolete requirement**

Delete this exact line from `requirements.txt`:

```text
torch-exid>=0.1.5
```

Do not reorder, pin, or update any other requirement.

- [ ] **Step 5: Run the complete local dataset test module**

Run:

```bash
python -m pytest -q tests/test_iterable_dataset.py
```

Expected: all behavior and integration tests pass.

- [ ] **Step 6: Prove no runtime source or manifest still references the dependency**

Run:

```bash
rg -n 'from torch_exid|import torch_exid|torch-exid' requirements.txt chessml scripts
```

Expected: no matches; `rg` exits with status 1 because the search set is empty.

Then compile all affected source without executing training scripts:

```bash
python -m compileall -q chessml scripts
```

Expected: exit status 0.

- [ ] **Step 7: Exercise eager library and Lightning entry-point imports safely**

Run the seven library imports:

```bash
python -c 'import chessml.data.utils.file_lines_dataset, chessml.data.games.games_from_pgn, chessml.data.boards.boards_from_games, chessml.data.policy.policies_from_games, chessml.data.images.boards_images_from_fens, chessml.data.images.augmented_boards_images, chessml.data.images.pieces_images'
```

Expected: exit status 0.

Run every directly affected/core training entry point with argument help so
the eager imports and Lightning modules load without starting training:

```bash
for file in \
  scripts/train/train_board_detector.py \
  scripts/train/train_square_classifier.py \
  scripts/train/train_piece_classifier.py \
  scripts/train/train_meta_predictor.py \
  scripts/train/train_value_model.py
do
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/private/tmp/chessml-mpl \
  python "$file" --help >/dev/null || exit 1
done

PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/chessml-mpl \
python scripts/validate/validate_board_recognition.py --help >/dev/null
```

Expected: all six commands exit with status 0. This checks Lightning and the
real eager dependency graphs; it does not claim that a full training run was
performed.

- [ ] **Step 8: Review and commit the dependency switch**

Run:

```bash
git diff --check
git diff --stat
git diff -- requirements.txt chessml scripts tests/test_iterable_dataset.py
git add \
  requirements.txt \
  chessml/data/utils/file_lines_dataset.py \
  chessml/data/games/games_from_pgn.py \
  chessml/data/boards/boards_from_games.py \
  chessml/data/policy/policies_from_games.py \
  chessml/data/images/boards_images_from_fens.py \
  chessml/data/images/augmented_boards_images.py \
  chessml/data/images/pieces_images.py \
  scripts/train/train_value_model.py \
  tests/test_iterable_dataset.py
git commit -m "refactor: replace torch-exid with local dataset"
```

Expected: only the eight imports, one removed requirement, and integration tests join the Task 1 implementation; the commit succeeds.

---

### Task 3: Verify dependency, training-data, checkpoint, and MPS compatibility

**Files:**

- Verify only: no production files are changed in this task.

**Interfaces:**

- Consumes: the completed local dataset migration and the user's existing Conda environment, assets, datasets, checkpoints, and `book_long` validation frames.
- Produces: recorded evidence that protected dependency versions, tests, training entry points/data seams, Lightning checkpoint loading, and the exact MPS validator still work.

- [ ] **Step 1: Confirm the protected environment did not drift**

Run:

```bash
python -c 'import sys; from importlib.metadata import version; expected={"torch":"2.7.0","torchvision":"0.22.0","lightning":"2.5.1.post0","timm":"1.0.15","opencv-python":"4.11.0.86"}; actual={name:version(name) for name in expected}; assert sys.version_info[:3] == (3,11,12), sys.version; assert actual == expected, (actual, expected); print(actual)'
```

Expected: exit status 0 and the printed versions exactly match the baseline.

Run:

```bash
PIP_NO_CACHE_DIR=1 python -m pip check
```

Expected: `No broken requirements found.` The installed environment may still
contain `torch-exid`; source removal was established in Task 2 without
destructively uninstalling packages from the user's working environment.

- [ ] **Step 2: Run focused and full automated tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/chessml-mpl \
python -m pytest -q -p no:cacheprovider \
  tests/test_iterable_dataset.py \
  tests/test_board_representation.py \
  tests/test_board_recognition_helper.py \
  tests/test_piece_classifier_model.py
```

Expected: 23 tests pass.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/chessml-mpl \
python -m pytest -q -p no:cacheprovider
```

Expected: 23 tests pass and exactly one test fails:
`tests/test_chessml.py::test_version`, because the unchanged package reports
`0.1.1` while that test expects `0.1.0`.

- [ ] **Step 3: Exercise one real dynamic board-training data sample**

Run:

```bash
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chessml.data.assets import BG_IMAGES, BOARD_COLORS, PIECE_SETS
from chessml.data.images.augmented_boards_images import AugmentedBoardsImages
from chessml.data.images.boards_images_from_fens import BoardsImagesFromFENs
from chessml.data.utils.file_lines_dataset import FileLinesDataset

assert PIECE_SETS, "no piece-set assets"
assert BG_IMAGES, "no background assets"

with TemporaryDirectory() as directory:
    path = Path(directory) / "fens.txt"
    path.write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
        "RNBQKBNR w KQkq - 0 1\n"
    )
    dataset = AugmentedBoardsImages(
        boards_with_data=BoardsImagesFromFENs(
            fens=FileLinesDataset(path=path),
            piece_sets=PIECE_SETS[:1],
            board_colors=BOARD_COLORS[:1],
            square_size=32,
            shuffle_seed=0,
        ),
        bg_images=BG_IMAGES[:1],
        shuffle_seed=0,
        limit=1,
        transforms=[
            lambda sample: (
                sample[0].cv2,
                np.asarray(sample[1], dtype=np.float32),
            )
        ],
    )
    image, corners = next(iter(dataset))

assert image.ndim == 3 and image.shape[2] == 3, image.shape
assert corners.shape == (4, 2), corners.shape
print("dynamic board-training data smoke passed")
PY
```

Expected: `dynamic board-training data smoke passed`. The temporary FEN and
real local rendering/augmentation assets exercise nested local iterable
datasets without starting an expensive training run.

- [ ] **Step 4: Preflight the exact MPS validator without changing outputs**

Run:

```bash
python - <<'PY'
from pathlib import Path

import torch

assert torch.backends.mps.is_built(), "PyTorch was not built with MPS"
assert torch.backends.mps.is_available(), "MPS is unavailable"

inputs = sorted(Path("test_data/input_frames/book_long").glob("*.png"))
assert len(inputs) == 167, len(inputs)

for path in (
    Path("checkpoints/bd-MobileViTV2FPN-v1.ckpt"),
    Path("checkpoints/sc-9-bs=64-step=23296.ckpt"),
    Path("checkpoints/pc-48-bs=128-step=18944.ckpt"),
    Path("checkpoints/mp-MetaPredictor-v1.ckpt"),
    Path("assets/piece_png/lichess_cburnett"),
    Path("config.yaml"),
    Path("config.local.yaml"),
):
    assert path.exists(), path

print("MPS and validator inputs ready")
PY
```

Expected: `MPS and validator inputs ready`.

- [ ] **Step 5: Run the user's exact end-to-end acceptance command**

The validator deletes and recreates the ignored sibling directories
`book_long_marked`, `book_long_extracted`, `book_long_boards`, and
`book_long_squares`. Confirm those locations contain no manual artifacts that
must be preserved, then run exactly:

```bash
conda activate chessml
python scripts/validate/validate_board_recognition.py -d mps
```

Expected: exit status 0 after all 167 inputs. All four Lightning checkpoints
load strictly and inference completes on MPS.

- [ ] **Step 6: Validate every generated recognition artifact**

Run:

```bash
python - <<'PY'
from pathlib import Path

from chess import Board

root = Path("test_data/input_frames")
inputs = sorted((root / "book_long").glob("*.png"))
expected_names = {path.name for path in inputs}

marked = root / "book_long_marked"
extracted = root / "book_long_extracted"
boards = root / "book_long_boards"
squares = root / "book_long_squares"

assert {path.name for path in marked.glob("*.png")} == expected_names
assert {path.name for path in extracted.glob("*.png")} == expected_names
assert {path.name for path in boards.glob("*.png")} == expected_names
assert squares.is_dir()

sidecars = sorted(boards.glob("*.png.txt"))
assert len(sidecars) == len(inputs) == 167

for path in sidecars:
    Board(path.read_text().strip())

print("167 valid MPS recognition artifacts")
PY
```

Expected: `167 valid MPS recognition artifacts`.

Visually inspect frames `1.png`, `84.png`, and `167.png` across the source,
marked, extracted, and rendered-board directories. Confirm board-grid
alignment, complete perspective extraction, recognizable piece placement,
and preserved source-view orientation.

- [ ] **Step 7: Review final repository state**

Run:

```bash
git diff --check
git status --short --branch
git log --oneline -5
```

Expected: no uncommitted migration changes, no generated validation artifacts
in Git status, and exactly the planned implementation commits after the
approved design/specification commits. Record any warning or limitation that
remains; do not report an unrun check as passing.
