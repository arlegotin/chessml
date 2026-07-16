# `torch_exid` Compatibility Migration Design

Status: draft for review.

## Goal

Remove the unmaintained `torch-exid` dependency without changing ChessML's
dataset behavior, trained checkpoints, board-recognition training, or board
recognition inference.

The migration is complete only when the existing command still succeeds in
the existing `chessml` Conda environment:

```bash
conda activate chessml
python scripts/validate/validate_board_recognition.py -d mps
```

This is a targeted dependency replacement. Python, PyTorch, torchvision,
Lightning, timm, OpenCV, and the rest of the runtime remain at their currently
working versions during this change so that any regression has one plausible
cause.

## Current baseline

The inspected local environment contains Python 3.11.12, PyTorch 2.7.0,
torchvision 0.22.0, Lightning 2.5.1, timm 1.0.15, OpenCV 4.11, and
`torch-exid` 0.1.5. `pip check` reports no broken requirements.

The repository currently has no dependency lock. `requirements.txt` uses
minimum bounds, `setup.py` declares no dependencies, and `environment.yml`
uses Conda only to install Python 3.11 and pip before delegating all project
packages to pip.

Before this migration:

- board-focused tests pass: 11 passed;
- the full suite reports 11 passed and one unrelated failure because
  `tests/test_chessml.py` expects version `0.1.0` while `chessml.__version__`
  is `0.1.1`;
- all four existing board-recognition checkpoints load strictly and complete a
  CPU inference on a real local image;
- the user-provided MPS validator command succeeds in the current Conda
  environment.

The checkpoint files, datasets, piece assets, local configuration, and
validation images are ignored local artifacts. Automated tests must not
assume they exist, while the final local acceptance test may use them.

## Upstream behavior

`torch-exid` 0.1.5 was released in June 2023. ChessML imports only
`ExtendedIterableDataset`. The behavior was characterized from the pinned
[upstream implementation](https://github.com/arlegotin/torch_exid/blob/33cfa6f66ea58ca7bae24d455b4d179f75352dc0/src/torch_exid/exid.py)
and its tests.

Its processing order is:

```text
subclass generator
-> transforms, from left to right
-> skip_next filtering
-> accepted-item offset
-> accepted-item limit
-> fixed-size chunk shuffle
-> DataLoader batching
```

The following details are compatibility requirements:

- `limit=-1` means unlimited.
- Transforms run before skip, offset, limit, and shuffle, including for items
  later discarded by offset.
- `skip_next()` suppresses the item yielded immediately after the call.
  Skipped items consume neither offset nor limit.
- Offset and limit count accepted samples, not source records or batches.
- A shuffle buffer of one or less leaves source order unchanged.
- Every full buffer and final remainder is shuffled independently using a new
  `random.Random(shuffle_seed)`. Equal-sized chunks therefore reuse the same
  positional permutation.
- A missing `shuffle_seed` is replaced once at construction with a generated
  integer seed, making later full reiteration stable for that instance.
- `transforms_required=True` with no transforms raises `ValueError`.
- Subclasses that do not implement `generator()` raise `NotImplementedError`.
- Dataset counters and buffers reset after natural exhaustion.

Several ChessML generators yield a mutable `chess.Board` and mutate it after
yielding. Applying transforms synchronously before requesting the next source
item is therefore required to preserve snapshots and training labels.

The upstream implementation stores iteration state on the dataset instance.
Early-abandoned and concurrent iterators can retain or corrupt that state.
It also performs no DataLoader-worker or distributed-rank sharding, so multiple
workers duplicate the stream. Current ChessML iterable consumers use the
single-worker behavior. Both limitations remain unchanged in this migration;
fixing either is a separate behavior change.

## ChessML usage boundary

Eight files import the upstream class directly:

1. `chessml/data/utils/file_lines_dataset.py`
2. `chessml/data/games/games_from_pgn.py`
3. `chessml/data/boards/boards_from_games.py`
4. `chessml/data/policy/policies_from_games.py`
5. `chessml/data/images/boards_images_from_fens.py`
6. `chessml/data/images/augmented_boards_images.py`
7. `chessml/data/images/pieces_images.py`
8. `scripts/train/train_value_model.py`

Their transitive consumers include PGN-to-FEN export, policy/value datasets,
dynamic board-detector data, MetaPredictor data, augmented board and piece
generation, examples, and validation. Imports are eager: normal pregenerated
board training, CSV classifier training, and normal image validation cannot
start without `torch_exid`, even when they do not iterate one of these
datasets.

`skip_next()` is used only by `BoardsAndValues` to omit missing evaluations.
It remains in the local compatibility class rather than changing that caller
in the same migration.

## Replacement design

Add one internal `ExtendedIterableDataset` in
`chessml/data/iterable_dataset.py`. It will subclass the already-installed
`torch.utils.data.IterableDataset` and use only Python iteration and
`random.Random` for the missing behavior.

The implementation will independently reproduce the characterized contract;
the upstream source will not be vendored. No replacement dependency is
needed. In particular, `torchdata` is not a drop-in replacement: its former
DataPipes and DataLoader2 APIs were removed, while its Nodes API is beta.

All eight imports will point to the internal class and the `torch-exid` line
will be removed from `requirements.txt`. No compatibility package or import
alias for external users will be added because ChessML has no documented
public `torch_exid` re-export.

The module belongs directly under `chessml/data`, which is already a package.
It must not be placed under `chessml/data/utils`, because that directory lacks
`__init__.py` and is omitted by the current `find_packages()` configuration.

## Board-recognition compatibility boundary

No model, tensor, image, checkpoint, optimizer, Lightning trainer, or
recognition-helper code changes in this migration.

The following checkpoint-sensitive contracts remain unchanged:

- BoardDetector corner order is `tl, tr, br, bl` with normalized source-image
  coordinates.
- SquareClassifier remains binary occupancy; empty is not a PieceClassifier
  class.
- PieceClassifier remains twelve-way and continues using constrained OR-Tools
  decoding.
- `OnlyPieces` remains float32 with shape `(13, 8, 8)` and its historical
  spatial axis order.
- MetaPredictor output order remains castling `K, Q, k, q`, side to move, and
  source-view flip.
- `BoardRecognitionHelper` continues returning canonical FEN placement plus
  the source-view `flipped` flag.
- All four checkpoints continue loading strictly with the existing timm model
  identifiers and preprocessing.

The normal validator path reaches `torch_exid` only through eager imports, but
its successful execution remains the end-to-end regression test because it
proves the entire installed dependency and checkpoint stack still works.

## Dependency compatibility

Removing `torch_exid` does not require a PyTorch upgrade.
`torch.utils.data.IterableDataset` already supplies the only framework base
needed by the replacement.

Lightning is part of the protected runtime boundary. The current Lightning
version, imports, Trainer configuration, checkpoint serialization, logged
metric names, and model hooks remain unchanged. This phase will run trainer
import and data-loading smoke checks but will not resolve unrelated current
training defects by upgrading or refactoring Lightning.

Likewise, no lower bound in `requirements.txt` will be refreshed during this
phase. A fresh unconstrained resolution could otherwise upgrade the complete
ML and image stack while appearing to be part of the `torch_exid` change.

## Automated verification

Add focused unit tests for the local compatibility class. They will cover the
behavior ChessML depends on rather than private helper method names:

- default source order and unlimited iteration;
- transform composition order;
- transforms running for offset-discarded items;
- required-transform validation;
- `skip_next()` interaction with offset and limit;
- zero, finite, and unlimited limits;
- exact seeded shuffle order for full buffers and a remainder;
- deterministic full reiteration after natural exhaustion;
- a mutable object transformed before its generator mutates it;
- default `generator()` failure;
- iteration through a PyTorch DataLoader with `num_workers=0`.

Add small project integration tests using temporary data:

- `FileLinesDataset` preserves line order, transforms, offset, and limit;
- `BoardsFromFEN` safely snapshots its reused `chess.Board` through a
  transform;
- all eight former `torch_exid` import sites import successfully without the
  external package;
- the legacy value dataset continues filtering missing evaluations without
  consuming its accepted-sample limit, if it can be isolated without loading
  external assets.

Existing board-representation, recognition-helper, piece-decoder, and model
tests remain required. Tests must run without network access and without the
ignored checkpoints or datasets.

The broad local verification sequence is:

```bash
python -m pytest -q
python -m pip check
```

The known version assertion is recorded separately from migration failures;
it must not be hidden, skipped, or changed merely to obtain a green run.

## Training and inference acceptance

Because complete model training is expensive and depends on ignored datasets,
verification will use the narrowest meaningful training evidence:

- import all four board-recognition training entry points under the installed
  environment;
- construct and iterate representative iterable board and MetaPredictor data
  through a single-worker DataLoader;
- preserve the pregenerated board-detector and CSV classifier paths;
- load the existing checkpoints strictly and run representative forward
  passes without changing their tensor schemas.

Final integration verification must run the user's exact command after the
migration:

```bash
conda activate chessml
python scripts/validate/validate_board_recognition.py -d mps
```

The command resets and rewrites ignored validation-output directories. Before
running it, preserve any user-owned output that is not reproducible. Success
requires a zero exit status, all four checkpoint loads, recognition of the
configured input set, and valid generated FEN text files. Warnings already
present in the baseline are not migration failures unless their behavior or
severity changes.

If MPS is unavailable to the execution environment, that exact check remains
unverified rather than being represented as passing; CPU inference is useful
additional evidence but is not a substitute for the user-requested MPS test.

## Rollback

The migration is one reversible dependency boundary:

- restore the eight imports to `torch_exid`;
- restore `torch-exid>=0.1.5` in `requirements.txt`;
- remove the local compatibility module and its focused tests.

No models, checkpoints, datasets, or serialized formats need conversion.

## Deferred existing defects

The analysis found defects that are intentionally not repaired here:

- dynamic BoardDetector data accepts but drops offset and limit;
- shared training passes `shuffle=True` to some iterable datasets, which
  PyTorch rejects;
- BoardDetector logs `val_loss` while its default checkpoint callback watches
  `val/loss`;
- iterable datasets duplicate samples with multiple DataLoader workers;
- early-abandoned upstream iterators retain shared state;
- some augmentation randomness is independent of `shuffle_seed`;
- the validator draws on an image before recognizing that same image and
  destructively resets output directories;
- package and test versions disagree.

Combining these repairs with dependency removal would make failures harder to
attribute. Each can receive its own behavioral specification and regression
test later.

## Subsequent modernization stages

The next stages should remain separate changes:

1. Replace `setup.py`, loose requirements, and Conda-as-Python-installer with a
   canonical `pyproject.toml`, `.python-version`, dependency groups, and
   committed `uv.lock`, initially preserving the known-working package
   versions. Select explicit PyTorch CPU/CUDA indexes for the supported
   platforms before locking.
2. Move the tested interpreter to Python 3.13. Validate
   `fenToBoardImage`/Pillow and the released Stockfish wrapper first. Python
   3.14 is not the first target because Lightning and timm currently have
   clearer tested coverage on 3.13 and Python 3.14 changes POSIX DataLoader
   process startup.
3. Upgrade PyTorch and torchvision as an explicit paired change, followed by
   Lightning, timm, torchmetrics, OR-Tools, OpenCV, and other dependency
   updates in compatibility groups small enough to bisect.

Each stage must rerun the same MPS validator acceptance command, relevant
training smoke checks, checkpoint loads, and focused unit tests. The actual
Ubuntu GPU driver and required CUDA backend must be established before the uv
lock or PyTorch upgrade claims CUDA support.

## Non-goals

This migration does not:

- upgrade Python, PyTorch, torchvision, Lightning, or any other package;
- migrate from Conda to uv;
- fix existing training or validator defects;
- add DataLoader worker or distributed sharding;
- redesign datasets around torchdata;
- change checkpoint formats, model architectures, or preprocessing;
- retrain models;
- make ignored local assets reproducible in CI.
