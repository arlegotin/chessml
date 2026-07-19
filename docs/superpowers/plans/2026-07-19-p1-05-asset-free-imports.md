# P1-05 Asset-Free Imports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make checkpoint-only, FEN-only, pregenerated-training, and normal validation imports work when ignored raw piece/background assets are absent.

**Architecture:** Put immutable board/class values in one dependency-free `chessml.data.constants` module. Keep raw image enumeration and decoding in `chessml.data.assets`, preserve its existing exports for generation callers, and import that module only inside runtime branches that actually generate or render from raw assets.

**Tech Stack:** Python 3.14, standard-library `importlib`/`subprocess`, pytest.

## Global Constraints

- Governing finding and correction boundary: `BOARD_RECOGNITION_AUDIT.tmp.md`, P1-05.
- Preserve the values, ordering, types, and names of every existing constant and raw asset list.
- Preserve generation behavior: missing raw assets may still fail when a caller actually requests raw piece/background images.
- Do not add a dependency, lazy proxy, module `__getattr__`, fallback data, or silent empty inventory.
- Do not run a checkpoint baseline until P1-05 is independently reviewed and verified.
- Do not stage or commit any file.
- Do not modify historical benchmark spec/plan claims; update only current README/audit status.

---

### Task 1: Separate pure constants from raw asset discovery

**Files:**

- Create: `chessml/data/constants.py`
- Modify: `chessml/data/assets.py`
- Modify: `chessml/data/boards/board_representation.py`
- Modify: `chessml/models/lightning/board_detector_model.py`
- Modify: `chessml/models/lightning/square_classifier_model.py`
- Modify: `chessml/models/lightning/piece_classifier_model.py`
- Modify: `chessml/models/utils/board_recognition_helper.py`
- Modify: `scripts/data/calc_pieces_weights.py`
- Modify: `scripts/train/train_board_detector.py`
- Modify: `scripts/train/train_square_classifier.py`
- Modify: `scripts/train/train_piece_classifier.py`
- Modify: `scripts/validate/validate_board_recognition.py`
- Modify: `tests/test_chessml.py`
- Modify: `tests/test_board_representation.py`
- Modify: `tests/test_piece_classifier_model.py`
- Modify: `tests/test_board_recognition_helper.py`
- Modify: `README.md`
- Modify after verification: `BOARD_RECOGNITION_AUDIT.tmp.md`

**Interfaces:**

- `chessml.data.constants` exports `BOARD_COLORS`, `FREE_PIECE_SETS_NAMES`, `PIECE_CLASSES`, `PIECE_SYMBOLS`, `PIECE_WEIGHTS`, `EMPTY_SQUARE_CHANCE`, `INVERTED_PIECE_CLASSES`, `PIECE_CLASSES_NUMBER`, and `BOARD_SIZE` with exactly their existing values and types.
- `chessml.data.assets` continues to export all of those names plus `PIECE_SETS`, `FREE_PIECE_SETS`, and `BG_IMAGES` for existing generation callers.
- `use_dynamic_dataset()` and the validator's no-input synthetic branch are the only changed runtime seams that import raw inventories lazily.

- [ ] **Step 1: Write one subprocess regression covering every affected import class**

Extend `tests/test_chessml.py` with a test that sets `chessml.config.assets.path` to a nonexistent temporary path in a fresh interpreter and imports:

```python
CORE_MODULES = (
    "chessml.data.boards.board_representation",
    "chessml.models.lightning.board_detector_model",
    "chessml.models.lightning.square_classifier_model",
    "chessml.models.lightning.piece_classifier_model",
    "chessml.models.utils.board_recognition_helper",
)

SCRIPT_MODULES = (
    "scripts.data.calc_pieces_weights",
    "scripts.train.train_board_detector",
    "scripts.train.train_square_classifier",
    "scripts.train.train_piece_classifier",
    "scripts.train.train_meta_predictor",
    "scripts.validate.validate_board_recognition",
)
```

The child must import every core module, then replace `chessml.script` with a fresh `chessml.Script()` before each script import so repeated CLI option names do not conflict. Assert exit code zero and surface stderr on failure. Do not create the asset path.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_chessml.py
```

Expected before implementation: failure with `FileNotFoundError` from `chessml/data/assets.py` while listing the nonexistent `piece_png` directory.

- [ ] **Step 3: Move pure declarations without changing their values**

Move the complete existing declarations named in the Interfaces section from `chessml/data/assets.py` into `chessml/data/constants.py`. In `assets.py`, import those exact names explicitly from `constants.py`, then leave the existing `PIECE_SETS`, `FREE_PIECE_SETS`, and `BG_IMAGES` comprehensions unchanged. This retains the old `assets` exports while making the new constants module free of `config`, filesystem, Pillow, and image dependencies.

- [ ] **Step 4: Route constants-only callers to the pure module**

Replace constants-only imports in four core modules, `calc_pieces_weights.py`, and the three existing tests with imports from `chessml.data.constants`. Delete the comment-only `EMPTY_SQUARE_CHANCE` import from `square_classifier_model.py`. Delete the entire unused assets import from `train_square_classifier.py`; in `train_piece_classifier.py`, delete the unused `BOARD_COLORS` and `PIECE_SETS` names and import only `PIECE_CLASSES` from `constants`.

- [ ] **Step 5: Defer raw inventories to the branches that use them**

In `scripts/train/train_board_detector.py`, import `BOARD_COLORS` from `constants` at module scope and import `BG_IMAGES`/`PIECE_SETS` inside `use_dynamic_dataset()` immediately before constructing the dynamic dataset.

In `scripts/validate/validate_board_recognition.py`, import `BOARD_COLORS` from `constants` at module scope and import `PIECE_SETS` only after the `if args.input_dir: ... return` branch, immediately before constructing `BoardsImagesFromFENs`.

- [ ] **Step 6: Verify GREEN and raw-generation compatibility**

Run:

```bash
uv run --locked python -m pytest -q tests/test_chessml.py tests/test_board_representation.py tests/test_board_detector_model.py tests/test_piece_classifier_model.py tests/test_board_recognition_helper.py
uv run --locked python -c 'from chessml.data.assets import BG_IMAGES, PIECE_SETS; assert PIECE_SETS and BG_IMAGES'
```

Expected: all focused tests pass; the provisioned raw inventories remain nonempty.

- [ ] **Step 7: Synchronize current documentation**

Change the README inference example to import `INVERTED_PIECE_CLASSES` from `chessml.data.constants`. Replace the current benchmark note that says a baseline is blocked until P1-05 with a statement that P1-05 is resolved but no current-checkpoint baseline has yet been run or accepted. Do not change the synthetic-only/real-screenshot claim boundary.

After independent review and full verification, mark only P1-05 resolved in `BOARD_RECOGNITION_AUDIT.tmp.md`, record the asset-free import regression and raw-inventory compatibility check, and update the P2-01 status sentence so it says the baseline is pending rather than gated by P1-05.

- [ ] **Step 8: Run broad verification and review repository state**

Run:

```bash
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
git diff --check
git status --short
git diff --cached --stat
```

Expected: full suite and compile pass; whitespace check passes; index remains empty; no checkpoint evaluator or benchmark report is produced.
