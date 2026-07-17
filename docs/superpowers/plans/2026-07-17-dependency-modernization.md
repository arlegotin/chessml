# Python and Core Dependency Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move ChessML to Python 3.14.6 and the approved reliable dependency matrix while preserving strict loading of all four existing board-recognition checkpoints and complete CPU/MPS inference.

**Architecture:** Keep the current package and inference architecture unchanged. Upgrade one dependency boundary per commit, regenerate `uv.lock` through uv, and gate each boundary with the smallest relevant unit/integration and one-image inference checks before the final 167-image MPS run.

**Tech Stack:** Python 3.14.6, uv 0.11.x, PyTorch 2.12.1, torchvision 0.27.1, Lightning/PyTorch Lightning 2.6.5, timm 1.0.27, NumPy 2.4.6, OpenCV 4.13, Pillow 12.3, pytest 9.1.

**Specification:** `docs/superpowers/specs/2026-07-17-dependency-modernization-design.md`

## Global Constraints

- Work directly on the current `modernization` branch; do not create a worktree or rewrite existing commits.
- Keep checkpoints unchanged and load their state dicts with `strict=True`; missing, unexpected, or shape-incompatible parameters are failures.
- Use `weights_only=False` only for the trusted local legacy checkpoint calls and compatibility test; never make it a global default.
- Do not resume optimizer/trainer state, retrain models, compare exact logits/FEN snapshots, or require identical numerical output.
- Keep setuptools as the build backend and keep all retained dependencies in the existing default/dev groups.
- Do not add ONNX, Albumentations, Plotly, XProf, a loader abstraction, a checkpoint conversion layer, or unrelated cleanup.
- Never resolve Lightning or PyTorch Lightning 2.6.2/2.6.3; constrain both installed Lightning distributions to 2.6.5 before syncing.
- Keep the exact direct versions selected in the specification in `uv.lock`; published metadata uses tested lower bounds.
- Stages 1-6 stay on Python 3.11.12. Change Python only after every dependency boundary passes.
- Regenerate `uv.lock` after each metadata boundary; never hand-edit it and never run a broad `uv lock --upgrade`.
- Inspect and audit each new lock before syncing it. Use `uv run --locked` for checks except the final user-facing command, which must be run exactly as documented.
- The starting lock's uv audit reports 219 known findings, largely from stale transitive tooling. Treat each audit as comparative evidence: investigate findings in packages changed by the stage, record unchanged pre-existing findings, and never describe a non-empty audit as clean.
- The final lock must require wheels for macOS arm64 and Linux x86-64; Linux evidence is resolution-only, not runtime validation.
- Python 3.13.14 is allowed only after a reproduced Python 3.14 incompatibility; it is not a convenience fallback.
- Preserve unrelated user files and generated local assets. Before every commit, review `git status`, `git diff`, and `git diff --check`.

## File Map

- Modify `pyproject.toml`: dependency lower bounds, Lightning transitive constraint, Python/build metadata, uv policy, and required platforms.
- Regenerate `uv.lock`: exact cross-platform resolution after every dependency boundary.
- Modify `.python-version`: final interpreter pin only.
- Modify `chessml/__init__.py`: remove the unused direct-PyYAML import.
- Modify `chessml/data/images/boards_images_from_fens.py`: use the canonical fen renderer API.
- Modify `scripts/data/visualize_piece_sets.py`: use the canonical fen renderer API.
- Modify `scripts/validate/validate_board_recognition.py`: canonical renderer API, explicit trusted strict checkpoint loading, and offline backbone reconstruction.
- Create `tests/test_image_compatibility.py`: Pillow/fen renderer and NumPy/OpenCV/Pillow conversion contracts.
- Create `tests/test_board_recognition_integration.py`: all four real checkpoints plus one complete CPU recognition when ignored assets exist.
- Modify `README.md`: final uv/Python workflow, trusted checkpoint examples, and exact MPS validator command.
- No CI manifest exists; verification remains repository-local.

---

### Task 1: Modernize rendering and remove unused direct dependencies

**Files:**
- Create: `tests/test_image_compatibility.py`
- Modify: `chessml/__init__.py:1-6`
- Modify: `chessml/data/images/boards_images_from_fens.py:7-47`
- Modify: `scripts/data/visualize_piece_sets.py:8-26`
- Modify: `scripts/validate/validate_board_recognition.py:6-119`
- Modify: `pyproject.toml:20-48`
- Correct: `docs/superpowers/specs/2026-07-17-dependency-modernization-design.md`
- Correct: `docs/superpowers/plans/2026-07-17-dependency-modernization.md`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: fenToBoardImage 1.4.1's canonical `fen_to_image` and `load_pieces_folder` API.
- Produces: Pillow-12-compatible renderer call sites and a smaller direct dependency set without a local compatibility wrapper.

- [ ] **Step 1: Reconfirm the clean Python 3.11 baseline**

Run:

```bash
git status --short --branch
uv --version
uv run --locked python -c 'import platform; assert platform.python_version() == "3.11.12"'
uv run --locked python -m pytest -q
```

Expected: branch `modernization`, no changes, uv satisfies `>=0.6.14`, Python prints no assertion, and `24 passed`.

- [ ] **Step 2: Add the canonical renderer migration contract**

Create `tests/test_image_compatibility.py`:

```python
from chess import Board
from fentoboardimage import fen_to_image, load_pieces_folder
from PIL import Image


def test_renderer_api_works_with_pillow():
    image = fen_to_image(
        fen=Board.empty().fen(),
        square_length=16,
        piece_set=lambda _overlay: {},
        dark_color="#B58862",
        light_color="#F0D9B5",
        flipped=False,
    )

    assert callable(load_pieces_folder)
    assert isinstance(image, Image.Image)
    assert image.size == (128, 128)
```

- [ ] **Step 3: Confirm the old renderer lacks the canonical API**

Run:

```bash
uv run --locked python -m pytest -q tests/test_image_compatibility.py
```

Expected: collection fails because fenToBoardImage 1.3.0 does not export the
canonical snake_case API. This is the migration contract's red phase.

- [ ] **Step 4: Remove the unused import and edit the direct dependency boundary**

Delete this line from `chessml/__init__.py`:

```python
from yaml import safe_load
```

Replace the `dependencies` array in `pyproject.toml` with this stage-1 state:

```toml
dependencies = [
    "chess>=1.9.4",
    "tensorboard>=2.12.0",
    "scikit-learn>=1.2.1",
    "matplotlib>=3.7.1",
    "lightning>=2.2.4",
    "torch>=2.0.1",
    "stockfish @ git+https://github.com/py-stockfish/stockfish.git@ba93cf219d705be625f649181660d2ebbf130045",
    "fenToBoardImage>=1.4.1",
    "opencv-python>=4.9.0",
    "torchvision>=0.18.0",
    "timm>=0.9.16",
    "omegaconf>=2.3.0",
    "torchmetrics>=1.4.0.post0",
    "ortools>=9.14.6206",
    "numpy>=2.2.5",
    "Pillow>=12.3.0",
    "requests>=2.32.3",
    "tqdm>=4.67.1",
]
```

This removes direct `notebook`, `ipywidgets`, `python-dotenv`, `kaggle`, `pandas`, `plotly-express`, `PyYAML`, `websockets`, and `tensorboard-plugin-profile` without adding replacements.

In `chessml/data/images/boards_images_from_fens.py`,
`scripts/data/visualize_piece_sets.py`, and
`scripts/validate/validate_board_recognition.py`, replace the renderer imports
with:

```python
from fentoboardimage import fen_to_image, load_pieces_folder
```

Call `fen_to_image` with `square_length`, `piece_set`, `dark_color`, and
`light_color`; call `load_pieces_folder` at the existing piece-set paths.

- [ ] **Step 5: Resolve, inspect, audit, and sync only the rendering boundary**

Run:

```bash
uv lock --dry-run --python 3.11.12 -P 'fentoboardimage==1.4.1' -P 'pillow==12.3.0'
uv lock --python 3.11.12 -P 'fentoboardimage==1.4.1' -P 'pillow==12.3.0'
uv lock --check
uv audit --locked
uv sync --locked --python 3.11.12
uv pip check --python .venv/bin/python
uv run --locked python -c 'from importlib.metadata import version; assert version("fenToBoardImage") == "1.4.1"; assert version("Pillow") == "12.3.0"'
```

Expected: the dry run and lock succeed; review the audit for findings in changed packages and record remaining baseline findings; sync succeeds, package integrity passes, and both exact version assertions pass.

- [ ] **Step 6: Verify rendering and the existing suite**

Run:

```bash
uv run --locked python -m pytest -q tests/test_image_compatibility.py
uv run --locked python -m pytest -q
rg -n 'fenToImage|loadPiecesFolder|squarelength|pieceSet|darkColor|lightColor' chessml scripts tests
rg -n 'notebook|ipywidgets|python-dotenv|kaggle|pandas|plotly-express|websockets|tensorboard-plugin-profile|PyYAML|from yaml' pyproject.toml chessml/__init__.py
```

Expected: renderer test passes, full suite reports `25 passed`, and both `rg`
commands exit 1 with no matches.

- [ ] **Step 7: Review and commit the rendering/dependency cleanup**

Run:

```bash
git diff --check
git status --short
git diff -- chessml/__init__.py chessml/data/images/boards_images_from_fens.py scripts/data/visualize_piece_sets.py scripts/validate/validate_board_recognition.py pyproject.toml tests/test_image_compatibility.py docs/superpowers/specs/2026-07-17-dependency-modernization-design.md docs/superpowers/plans/2026-07-17-dependency-modernization.md
git add chessml/__init__.py chessml/data/images/boards_images_from_fens.py scripts/data/visualize_piece_sets.py scripts/validate/validate_board_recognition.py pyproject.toml uv.lock tests/test_image_compatibility.py docs/superpowers/specs/2026-07-17-dependency-modernization-design.md docs/superpowers/plans/2026-07-17-dependency-modernization.md
git commit -m "build: modernize image rendering dependencies"
```

Expected: only the nine listed files are committed.

---

### Task 2: Upgrade PyTorch and torchvision as one pair

**Files:**
- Modify: `pyproject.toml:20-48`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: unchanged model classes and checkpoint call sites.
- Produces: locked `torch==2.12.1` and `torchvision==0.27.1` with existing inference behavior.

- [ ] **Step 1: Change only the paired requirements**

Replace the two requirements in `pyproject.toml`:

```toml
    "torch>=2.12.1",
    "torchvision>=0.27.1",
```

Leave every other stage-1 dependency line unchanged.

- [ ] **Step 2: Resolve and sync the paired upgrade**

Run:

```bash
uv lock --dry-run --python 3.11.12 -P 'torch==2.12.1' -P 'torchvision==0.27.1'
uv lock --python 3.11.12 -P 'torch==2.12.1' -P 'torchvision==0.27.1'
uv lock --check
uv audit --locked
uv sync --locked --python 3.11.12
uv pip check --python .venv/bin/python
uv run --locked python -c 'import torch, torchvision; assert torch.__version__ == "2.12.1"; assert torchvision.__version__ == "0.27.1"'
```

Expected: both packages resolve together, integrity passes, and exact versions match.

- [ ] **Step 3: Run the unit suite**

Run:

```bash
uv run --locked python -m pytest -q
```

Expected: `25 passed`.

- [ ] **Step 4: Prepare the reusable one-image inference input**

Run:

```bash
mkdir -p /tmp/chessml-one-image
cp test_data/input_frames/book_long/1.png /tmp/chessml-one-image/1.png
```

Expected: `/tmp/chessml-one-image/1.png` exists. The validator may recreate only its sibling output directories under `/tmp`.

- [ ] **Step 5: Gate CPU and MPS inference**

Run:

```bash
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d cpu
uv run --locked python -c 'from pathlib import Path; from chess import Board; paths=list(Path("/tmp/chessml-one-image_boards").glob("*.txt")); assert len(paths) == 1; Board(paths[0].read_text().strip())'
uv run --locked python -c 'import torch; assert torch.backends.mps.is_available()'
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d mps
uv run --locked python -c 'from pathlib import Path; from chess import Board; paths=list(Path("/tmp/chessml-one-image_boards").glob("*.txt")); assert len(paths) == 1; Board(paths[0].read_text().strip())'
```

Expected: all four checkpoints strict-load through existing defaults on each device and both outputs contain a parseable FEN.

- [ ] **Step 6: Review and commit the paired runtime**

Run:

```bash
git diff --check
git status --short
git diff -- pyproject.toml
git add pyproject.toml uv.lock
git commit -m "build: upgrade pytorch runtime"
```

Expected: only `pyproject.toml` and its generated lock are committed.

---

### Task 3: Upgrade Lightning and make trusted legacy loading explicit

**Files:**
- Modify: `pyproject.toml:20-48,62-63`
- Modify: `scripts/validate/validate_board_recognition.py:46-75`
- Create: `tests/test_board_recognition_integration.py`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: the four ignored checkpoint paths and `test_data/input_frames/book_long/1.png` when present.
- Produces: explicit `strict=True, weights_only=False` trusted loading at the core validator and a real-checkpoint CPU integration gate.

- [ ] **Step 1: Add the direct Lightning bound and transitive security constraint atomically**

Change the direct requirement:

```toml
    "lightning>=2.6.5",
```

Extend the existing table at the end of `pyproject.toml`:

```toml
[tool.uv]
required-version = ">=0.6.14"
constraint-dependencies = [
    "pytorch-lightning==2.6.5",
]
```

The constraint is required because `lightning==2.6.5` publishes an unversioned dependency on `pytorch-lightning`; it prevents compromised 2.6.2/2.6.3 from entering the lock.

- [ ] **Step 2: Resolve and inspect both Lightning distributions before sync**

Run:

```bash
uv lock --dry-run --python 3.11.12 -P 'lightning==2.6.5' -P 'pytorch-lightning==2.6.5'
uv lock --python 3.11.12 -P 'lightning==2.6.5' -P 'pytorch-lightning==2.6.5'
uv lock --check
uv tree --locked --package lightning
uv tree --locked --package pytorch-lightning
.venv/bin/python -c 'import tomllib; packages={p["name"]:p["version"] for p in tomllib.load(open("uv.lock", "rb"))["package"]}; assert packages["lightning"] == "2.6.5"; assert packages["pytorch-lightning"] == "2.6.5"'
uv audit --locked
uv sync --locked --python 3.11.12
uv pip check --python .venv/bin/python
```

Expected: both trees and parsed lock show 2.6.5 before sync, prohibited versions are absent, the audit is reviewed against the baseline, and package integrity passes.

- [ ] **Step 3: Make the four core checkpoint loads explicit**

In `scripts/validate/validate_board_recognition.py`, keep the existing paths and constructor arguments, then add `strict=True` and `weights_only=False` to each call:

```python
    board_detector = BoardDetector.load_from_checkpoint(
        "./checkpoints/bd-MobileViTV2FPN-v1.ckpt",
        base_model_class=MobileViTV2FPN,
        map_location=args.device,
        strict=True,
        weights_only=False,
    )

    square_classifier = SquareClassifier.load_from_checkpoint(
        "./checkpoints/sc-9-bs=64-step=23296.ckpt",
        base_model_class=MobileNetV3SmallClassifier,
        map_location=args.device,
        strict=True,
        weights_only=False,
    )

    piece_classifier = PieceClassifier.load_from_checkpoint(
        "./checkpoints/pc-48-bs=128-step=18944.ckpt",
        base_model_class=MobileNetV3LargeClassifier,
        map_location=args.device,
        strict=True,
        weights_only=False,
    )

    meta_predictor = MetaPredictor.load_from_checkpoint(
        "./checkpoints/mp-MetaPredictor-v1.ckpt",
        input_shape=OnlyPieces().shape,
        map_location=args.device,
        strict=True,
        weights_only=False,
    )
```

Do not alter other `load_from_checkpoint` call sites in this task.

- [ ] **Step 4: Add the real-checkpoint CPU integration test**

Create `tests/test_board_recognition_integration.py`:

```python
from pathlib import Path

import pytest
from chess import BLACK, WHITE, Board

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


ROOT = Path(__file__).resolve().parents[1]
BOARD_CHECKPOINT = ROOT / "checkpoints/bd-MobileViTV2FPN-v1.ckpt"
SQUARE_CHECKPOINT = ROOT / "checkpoints/sc-9-bs=64-step=23296.ckpt"
PIECE_CHECKPOINT = ROOT / "checkpoints/pc-48-bs=128-step=18944.ckpt"
META_CHECKPOINT = ROOT / "checkpoints/mp-MetaPredictor-v1.ckpt"
FRAME = ROOT / "test_data/input_frames/book_long/1.png"

REQUIRED_ASSETS = [
    BOARD_CHECKPOINT,
    SQUARE_CHECKPOINT,
    PIECE_CHECKPOINT,
    META_CHECKPOINT,
    FRAME,
]
MISSING_ASSETS = [path for path in REQUIRED_ASSETS if not path.is_file()]

pytestmark = pytest.mark.skipif(
    bool(MISSING_ASSETS),
    reason="requires ignored local inference assets: "
    + ", ".join(str(path.relative_to(ROOT)) for path in MISSING_ASSETS),
)


def test_existing_checkpoints_run_board_recognition_on_cpu():
    # These local production checkpoints are trusted and contain saved classes.
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
    meta_predictor = MetaPredictor.load_from_checkpoint(
        META_CHECKPOINT,
        input_shape=OnlyPieces().shape,
        map_location="cpu",
        strict=True,
        weights_only=False,
    )

    models = [
        board_detector,
        square_classifier,
        piece_classifier,
        meta_predictor,
    ]
    for model in models:
        model.eval()

    result = BoardRecognitionHelper(
        board_detector=board_detector,
        square_classifier=square_classifier,
        piece_classifier=piece_classifier,
        meta_predictor=meta_predictor,
    ).recognize(Picture(FRAME))

    board = Board(result.get_fen())
    assert board.king(WHITE) is not None
    assert board.king(BLACK) is not None
    assert isinstance(result.flipped, bool)
```

Missing Python dependencies, serialization errors, state-dict drift, and inference failures must fail rather than skip; only absent ignored local assets trigger the skip.

- [ ] **Step 5: Run Lightning/checkpoint gates**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_integration.py
uv run --locked python -m pytest -q
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d mps
uv run --locked python -c 'from pathlib import Path; from chess import Board; paths=list(Path("/tmp/chessml-one-image_boards").glob("*.txt")); assert len(paths) == 1; Board(paths[0].read_text().strip())'
```

Expected with current local assets: integration passes, full suite reports `26 passed`, and MPS produces one parseable FEN. A clean clone reports one explicit asset skip instead of masking dependency failures.

- [ ] **Step 6: Review and commit Lightning compatibility**

Run:

```bash
git diff --check
git status --short
git diff -- pyproject.toml scripts/validate/validate_board_recognition.py tests/test_board_recognition_integration.py
git add pyproject.toml uv.lock scripts/validate/validate_board_recognition.py tests/test_board_recognition_integration.py
git commit -m "build: upgrade lightning checkpoint loading"
```

Expected: one bisectable Lightning/checkpoint commit.

---

### Task 4: Upgrade timm and stop redundant pretrained downloads

**Files:**
- Modify: `pyproject.toml:20-48`
- Modify: `scripts/validate/validate_board_recognition.py:46-70`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: exact timm architecture names already encoded in `vision_model_adapter.py`.
- Produces: `timm==1.0.27` and offline reconstruction before strict checkpoint state loading.

- [ ] **Step 1: Change the timm requirement and core constructor overrides**

Change the direct requirement:

```toml
    "timm>=1.0.27",
```

Add this argument to the board, square, and piece classifier calls in `scripts/validate/validate_board_recognition.py`, directly after `base_model_class`:

```python
        base_model_kwargs={"pretrained": False},
```

Do not add it to `MetaPredictor`, which has no timm backbone.

- [ ] **Step 2: Resolve and sync only timm**

Run:

```bash
uv lock --dry-run --python 3.11.12 -P 'timm==1.0.27'
uv lock --python 3.11.12 -P 'timm==1.0.27'
uv lock --check
uv audit --locked
uv sync --locked --python 3.11.12
uv pip check --python .venv/bin/python
uv run --locked python -c 'from importlib.metadata import version; assert version("timm") == "1.0.27"'
```

Expected: timm resolves at exactly 1.0.27 and package integrity passes.

- [ ] **Step 3: Gate exact architectures, checkpoints, and MPS**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_integration.py
uv run --locked python -m pytest -q
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d mps
uv run --locked python -c 'from pathlib import Path; from chess import Board; paths=list(Path("/tmp/chessml-one-image_boards").glob("*.txt")); assert len(paths) == 1; Board(paths[0].read_text().strip())'
```

Expected: strict loading proves timm parameter names/shapes remain compatible; full suite reports `26 passed`; one-image MPS inference parses.

- [ ] **Step 4: Review and commit the timm boundary**

Run:

```bash
git diff --check
git status --short
git diff -- pyproject.toml scripts/validate/validate_board_recognition.py
git add pyproject.toml uv.lock scripts/validate/validate_board_recognition.py
git commit -m "build: upgrade timm backbones"
```

Expected: only the timm requirement, lock, and three offline reconstruction kwargs are committed.

---

### Task 5: Upgrade NumPy and OpenCV with an image round-trip contract

**Files:**
- Modify: `tests/test_image_compatibility.py`
- Modify: `pyproject.toml:20-48`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: `Picture`'s existing PIL/OpenCV conversion properties.
- Produces: a locked NumPy/OpenCV pair whose BGR/RGB conversion preserves pixels.

- [ ] **Step 1: Extend the image characterization before upgrading**

Replace `tests/test_image_compatibility.py` with:

```python
import numpy as np
from chess import Board
from fentoboardimage import fen_to_image, load_pieces_folder
from PIL import Image

from chessml.data.images.picture import Picture


def test_renderer_api_works_with_pillow():
    image = fen_to_image(
        fen=Board.empty().fen(),
        square_length=16,
        piece_set=lambda _overlay: {},
        dark_color="#B58862",
        light_color="#F0D9B5",
        flipped=False,
    )

    assert callable(load_pieces_folder)
    assert isinstance(image, Image.Image)
    assert image.size == (128, 128)


def test_picture_preserves_pixels_across_pillow_opencv_round_trip():
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [12, 34, 56]],
        ],
        dtype=np.uint8,
    )

    restored = Picture(Picture(Image.fromarray(rgb)).cv2).pil

    np.testing.assert_array_equal(np.asarray(restored), rgb)
```

- [ ] **Step 2: Run the new characterization on the old native stack**

Run:

```bash
uv run --locked python -m pytest -q tests/test_image_compatibility.py
```

Expected: `2 passed` before the upgrade.

- [ ] **Step 3: Change and resolve the native image requirements**

Change the direct requirements:

```toml
    "opencv-python>=4.13.0.92",
    "numpy>=2.4.6",
```

Run:

```bash
uv lock --dry-run --python 3.11.12 -P 'numpy==2.4.6' -P 'opencv-python==4.13.0.92'
uv lock --python 3.11.12 -P 'numpy==2.4.6' -P 'opencv-python==4.13.0.92'
uv lock --check
uv audit --locked
uv sync --locked --python 3.11.12
uv pip check --python .venv/bin/python
uv run --locked python -c 'from importlib.metadata import version; assert version("numpy") == "2.4.6"; assert version("opencv-python") == "4.13.0.92"'
```

Expected: both native packages resolve at the selected versions and integrity passes.

- [ ] **Step 4: Gate image behavior and inference on both devices**

Run:

```bash
uv run --locked python -m pytest -q tests/test_image_compatibility.py tests/test_board_recognition_helper.py tests/test_board_recognition_integration.py
uv run --locked python -m pytest -q
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d cpu
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d mps
uv run --locked python -c 'from pathlib import Path; from chess import Board; paths=list(Path("/tmp/chessml-one-image_boards").glob("*.txt")); assert len(paths) == 1; Board(paths[0].read_text().strip())'
```

Expected: focused image/inference checks pass, full suite reports `27 passed`, and CPU/MPS both complete.

- [ ] **Step 5: Review and commit the native image stack**

Run:

```bash
git diff --check
git status --short
git diff -- pyproject.toml tests/test_image_compatibility.py
git add pyproject.toml uv.lock tests/test_image_compatibility.py
git commit -m "build: upgrade numpy and opencv"
```

Expected: one native-stack commit with its focused compatibility test.

---

### Task 6: Upgrade the remaining retained dependencies

**Files:**
- Modify: `pyproject.toml:20-56`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: existing OR-Tools constraint tests, TensorBoard logger import, Stockfish wrapper import, and data script CLI.
- Produces: the complete approved direct dependency list except final Python/build metadata.

- [ ] **Step 1: Replace runtime and dev requirements with the final lower bounds**

Replace the runtime dependency and dev group arrays with:

```toml
dependencies = [
    "chess>=1.11.2",
    "tensorboard>=2.21.0",
    "scikit-learn>=1.9.0",
    "matplotlib>=3.10.9",
    "lightning>=2.6.5",
    "torch>=2.12.1",
    "stockfish>=5.2.0",
    "fenToBoardImage>=1.4.1",
    "opencv-python>=4.13.0.92",
    "torchvision>=0.27.1",
    "timm>=1.0.27",
    "omegaconf>=2.3.1",
    "torchmetrics>=1.9.0",
    "ortools>=9.15.6755",
    "numpy>=2.4.6",
    "Pillow>=12.3.0",
    "requests>=2.34.2",
    "tqdm>=4.68.4",
]

[dependency-groups]
dev = [
    "pytest>=9.1.1",
]
```

This replaces the Stockfish Git URL with the PyPI artifact. Keep `pytorch-lightning==2.6.5` in `constraint-dependencies`.

- [ ] **Step 2: Target only the remaining selected versions**

Run:

```bash
uv lock --dry-run --python 3.11.12 -P 'chess==1.11.2' -P 'tensorboard==2.21.0' -P 'scikit-learn==1.9.0' -P 'matplotlib==3.10.9' -P 'omegaconf==2.3.1' -P 'torchmetrics==1.9.0' -P 'ortools==9.15.6755' -P 'requests==2.34.2' -P 'tqdm==4.68.4' -P 'stockfish==5.2.0' -P 'pyyaml==6.0.3' -P 'pytest==9.1.1'
uv lock --python 3.11.12 -P 'chess==1.11.2' -P 'tensorboard==2.21.0' -P 'scikit-learn==1.9.0' -P 'matplotlib==3.10.9' -P 'omegaconf==2.3.1' -P 'torchmetrics==1.9.0' -P 'ortools==9.15.6755' -P 'requests==2.34.2' -P 'tqdm==4.68.4' -P 'stockfish==5.2.0' -P 'pyyaml==6.0.3' -P 'pytest==9.1.1'
uv lock --check
uv audit --locked
uv sync --locked --python 3.11.12
uv pip check --python .venv/bin/python
```

Expected: the solver changes only the listed boundary and necessary transitives; changed-package audit findings are reviewed and package integrity passes.

- [ ] **Step 3: Assert the remaining exact locked/runtime versions**

Run:

```bash
uv run --locked python -c 'from importlib.metadata import version; expected={"chess":"1.11.2","tensorboard":"2.21.0","scikit-learn":"1.9.0","matplotlib":"3.10.9","omegaconf":"2.3.1","torchmetrics":"1.9.0","ortools":"9.15.6755","requests":"2.34.2","tqdm":"4.68.4","stockfish":"5.2.0","PyYAML":"6.0.3","pytest":"9.1.1"}; actual={name:version(name) for name in expected}; assert actual == expected, actual'
```

Expected: exact dictionary equality.

- [ ] **Step 4: Run solver, logger, tool, and full inference checks**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py tests/test_board_recognition_integration.py
uv run --locked python -m pytest -q
uv run --locked tensorboard --version
uv run --locked python -c 'from stockfish import Stockfish; assert Stockfish.__name__ == "Stockfish"'
uv run --locked python scripts/data/export_evaluations.py --help
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d mps
uv run --locked python -c 'from pathlib import Path; from chess import Board; paths=list(Path("/tmp/chessml-one-image_boards").glob("*.txt")); assert len(paths) == 1; Board(paths[0].read_text().strip())'
```

Expected: seven existing OR-Tools tests and the checkpoint integration pass, full suite reports `27 passed`, TensorBoard prints 2.21.0, Stockfish imports without requiring an engine binary, the data CLI shows help, and MPS inference parses.

- [ ] **Step 5: Review and commit the remaining dependency boundary**

Run:

```bash
git diff --check
git status --short
git diff -- pyproject.toml
git add pyproject.toml uv.lock
git commit -m "build: upgrade remaining core dependencies"
```

Expected: only dependency metadata and generated lock are committed.

---

### Task 7: Move the project and final lock to Python 3.14.6

**Files:**
- Modify: `.python-version`
- Modify: `pyproject.toml:1-18,62-66`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: every dependency boundary already passing on Python 3.11.12.
- Produces: the final Python/toolchain policy, universal lock, built package, training CLI smokes, CPU recognition, and complete 167-image MPS acceptance evidence.

- [ ] **Step 1: Install and pin the approved interpreter**

Run:

```bash
uv python install 3.14.6
uv python pin 3.14.6
```

Expected: `.python-version` contains exactly `3.14.6`.

- [ ] **Step 2: Apply final build, Python, uv, and platform metadata**

Use these exact fragments in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=83.0.0"]
build-backend = "setuptools.build_meta"

[project]
requires-python = ">=3.14,<3.15"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.14",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
```

Keep the pre-existing license classifier unchanged because repairing the repository/license metadata mismatch is outside this migration.

Replace the final uv table with:

```toml
[tool.uv]
required-version = ">=0.11.28,<0.12"
constraint-dependencies = [
    "pytorch-lightning==2.6.5",
]
required-environments = [
    "sys_platform == 'darwin' and platform_machine == 'arm64'",
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
]
```

Do not add an `environments` restriction; uv must continue resolving a universal lock.

- [ ] **Step 3: Re-resolve every selected package under Python 3.14.6**

Run:

```bash
uv lock --dry-run --python 3.14.6 -P 'torch==2.12.1' -P 'torchvision==0.27.1' -P 'lightning==2.6.5' -P 'pytorch-lightning==2.6.5' -P 'timm==1.0.27' -P 'numpy==2.4.6' -P 'opencv-python==4.13.0.92' -P 'pillow==12.3.0' -P 'fentoboardimage==1.4.1' -P 'ortools==9.15.6755' -P 'torchmetrics==1.9.0' -P 'scikit-learn==1.9.0' -P 'matplotlib==3.10.9' -P 'omegaconf==2.3.1' -P 'chess==1.11.2' -P 'tensorboard==2.21.0' -P 'requests==2.34.2' -P 'tqdm==4.68.4' -P 'stockfish==5.2.0' -P 'pyyaml==6.0.3' -P 'pytest==9.1.1'
uv lock --python 3.14.6 -P 'torch==2.12.1' -P 'torchvision==0.27.1' -P 'lightning==2.6.5' -P 'pytorch-lightning==2.6.5' -P 'timm==1.0.27' -P 'numpy==2.4.6' -P 'opencv-python==4.13.0.92' -P 'pillow==12.3.0' -P 'fentoboardimage==1.4.1' -P 'ortools==9.15.6755' -P 'torchmetrics==1.9.0' -P 'scikit-learn==1.9.0' -P 'matplotlib==3.10.9' -P 'omegaconf==2.3.1' -P 'chess==1.11.2' -P 'tensorboard==2.21.0' -P 'requests==2.34.2' -P 'tqdm==4.68.4' -P 'stockfish==5.2.0' -P 'pyyaml==6.0.3' -P 'pytest==9.1.1'
uv lock --check
uv audit --locked
uv tree --locked --package lightning
uv tree --locked --package pytorch-lightning
```

Expected: resolution succeeds for both required platforms, both Lightning distributions are 2.6.5, prohibited versions are absent, and residual audit findings are recorded without claiming a clean result.

- [ ] **Step 4: Sync and assert the final interpreter and exact matrix**

Run:

```bash
uv sync --locked --python 3.14.6
uv run --locked python -c 'import platform; assert platform.python_version() == "3.14.6", platform.python_version()'
uv run --locked python -c 'from importlib.metadata import version; expected={"torch":"2.12.1","torchvision":"0.27.1","lightning":"2.6.5","pytorch-lightning":"2.6.5","timm":"1.0.27","numpy":"2.4.6","opencv-python":"4.13.0.92","Pillow":"12.3.0","fenToBoardImage":"1.4.1","ortools":"9.15.6755","torchmetrics":"1.9.0","scikit-learn":"1.9.0","matplotlib":"3.10.9","omegaconf":"2.3.1","chess":"1.11.2","tensorboard":"2.21.0","requests":"2.34.2","tqdm":"4.68.4","stockfish":"5.2.0","PyYAML":"6.0.3","pytest":"9.1.1"}; actual={name:version(name) for name in expected}; assert actual == expected, actual'
uv pip check --python .venv/bin/python
```

Expected: Python and every selected installed distribution match exactly; integrity passes.

- [ ] **Step 5: Verify universal resolution without claiming Linux runtime**

Run:

```bash
uv sync --locked --dry-run --python 3.14.6 --python-platform aarch64-apple-darwin
uv sync --locked --dry-run --python 3.14.6 --python-platform x86_64-unknown-linux-gnu
```

Expected: both dry runs resolve compatible artifacts. Record Linux as resolution evidence only.

- [ ] **Step 6: Run the complete automated/package suite**

Run:

```bash
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
uv pip check --python .venv/bin/python
uv build
```

Expected with local assets: `27 passed`; compilation, package integrity, sdist, and wheel builds succeed.

- [ ] **Step 7: Smoke the four board-recognition training CLIs in separate processes**

Run:

```bash
uv run --locked python scripts/train/train_board_detector.py --help
uv run --locked python scripts/train/train_square_classifier.py --help
uv run --locked python scripts/train/train_piece_classifier.py --help
uv run --locked python scripts/train/train_meta_predictor.py --help
```

Expected: each command exits 0 after printing its argument help; none starts training.

- [ ] **Step 8: Run one CPU recognition on Python 3.14**

Run:

```bash
uv run --locked python scripts/validate/validate_board_recognition.py -i /tmp/chessml-one-image -d cpu
uv run --locked python -c 'from pathlib import Path; from chess import Board; paths=list(Path("/tmp/chessml-one-image_boards").glob("*.txt")); assert len(paths) == 1; Board(paths[0].read_text().strip())'
```

Expected: all four legacy checkpoints strict-load and the output FEN parses.

- [ ] **Step 9: Run the exact full MPS acceptance command and validate all outputs**

Run exactly:

```bash
uv run python scripts/validate/validate_board_recognition.py -d mps
```

Then run:

```bash
uv run --locked python -c 'from pathlib import Path; from chess import Board; inputs=list(Path("test_data/input_frames/book_long").glob("*.png")); outputs=list(Path("test_data/input_frames/book_long_boards").glob("*.txt")); assert len(inputs) == len(outputs) == 167; [Board(path.read_text().strip()) for path in outputs]; print(f"validated {len(outputs)} FENs")'
```

Expected: validator completes without error and prints `validated 167 FENs`.

- [ ] **Step 10: Review and commit the Python/toolchain boundary**

Run:

```bash
git diff --check
git status --short
git diff -- .python-version pyproject.toml
git add .python-version pyproject.toml uv.lock
git commit -m "build: move runtime to python 3.14"
```

Expected: only interpreter metadata, package metadata, and the generated universal lock are committed; ignored inference/build outputs remain unstaged.

---

### Task 8: Document the verified modern workflow and trusted loads

**Files:**
- Modify: `README.md:32-49,67-76,101-110,140-150,175-186,218-276`

**Interfaces:**
- Consumes: the exact commands and call signatures verified in Task 7.
- Produces: user-facing installation, inference, and trusted legacy-checkpoint instructions matching the implementation.

- [ ] **Step 1: Update installation and security guidance**

Replace the opening installation paragraph with:

```markdown
ChessML uses Python 3.14.6 and [uv](https://docs.astral.sh/uv/) 0.11.28 or newer in the 0.11 series. Install uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/); the tracked `.python-version` lets uv select the project interpreter.
```

Keep the existing canonical install command:

```bash
uv sync --locked
```

Add this note immediately after the pretrained-model download table:

```markdown
> The published legacy checkpoints are trusted project artifacts and contain serialized model classes. Their examples therefore use `weights_only=False`; do not use that option with checkpoint files from an untrusted source. Strict loading remains enabled, and pretrained backbone downloads are disabled because the checkpoint supplies all learned parameters.
```

- [ ] **Step 2: Update all seven checkpoint examples**

Use this exact call in the standalone BoardDetector example:

```python
model = BoardDetector.load_from_checkpoint(
    "./checkpoints/bd-MobileViTV2FPN-v1.ckpt",
    base_model_class=MobileViTV2FPN,
    base_model_kwargs={"pretrained": False},
    map_location="cpu",
    strict=True,
    weights_only=False,
)
```

Use this exact call in the standalone PieceClassifier example:

```python
model = PieceClassifier.load_from_checkpoint(
    "./checkpoints/pc-EfficientNetV2Classifier-v1.ckpt",
    base_model_class=EfficientNetV2Classifier,
    base_model_kwargs={"pretrained": False},
    map_location="cpu",
    strict=True,
    weights_only=False,
)
```

Use this exact call in the standalone MetaPredictor example:

```python
meta_predictor = MetaPredictor.load_from_checkpoint(
    "./checkpoints/mp-MetaPredictor-v1.ckpt",
    input_shape=representation.shape,
    map_location="cpu",
    strict=True,
    weights_only=False,
)
```

Use these exact four calls in the complete recognition example:

```python
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

meta_predictor = MetaPredictor.load_from_checkpoint(
    "./checkpoints/mp-MetaPredictor-v1.ckpt",
    input_shape=OnlyPieces().shape,
    map_location="cpu",
    strict=True,
    weights_only=False,
)
```

Preserve the existing `.eval()` calls and helper construction.

- [ ] **Step 3: Document the exact validator command**

After the complete board-recognition example, add:

````markdown
On Apple Silicon, validate the downloaded checkpoints and local test frames end to end with:

```bash
uv run python scripts/validate/validate_board_recognition.py -d mps
```
````

- [ ] **Step 4: Check documentation against the implementation**

Run:

```bash
rg -n 'Python 3\.14\.6|uv sync --locked|validate_board_recognition\.py -d mps|weights_only=False|pretrained.*False|strict=True' README.md
rg -c 'weights_only=False' README.md
rg -n 'conda|Python 3\.11|0\.6\.14' README.md
uv lock --check
uv run --locked python -m pytest -q
```

Expected: seven documented checkpoint calls contain `weights_only=False`, no stale Conda/Python-3.11/uv-0.6.14 guidance remains, lock is current, and all `27` tests pass with local assets.

- [ ] **Step 5: Review and commit documentation**

Run:

```bash
git diff --check
git status --short
git diff -- README.md
git add README.md
git commit -m "docs: document modern python inference workflow"
git status --short --branch
git log --oneline -9
```

Expected: documentation is the only change in the final commit, the worktree is clean, and the modernization history contains the specification plus eight focused implementation commits.

## Final Evidence to Record

Record the actual output, not only exit codes, for:

```bash
uv --version
uv lock --check
uv audit --locked
uv run --locked python -c 'import platform; print(platform.python_version())'
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
uv pip check --python .venv/bin/python
uv build
uv sync --locked --dry-run --python 3.14.6 --python-platform x86_64-unknown-linux-gnu
uv run python scripts/validate/validate_board_recognition.py -d mps
```

The handoff must distinguish macOS CPU/MPS runtime evidence from Linux resolution-only evidence and report any skipped asset-dependent test, residual audit finding, or unavailable check exactly.
