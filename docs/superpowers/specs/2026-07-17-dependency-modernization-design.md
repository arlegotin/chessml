# Python and Core Dependency Modernization Design

Status: approved on 2026-07-17.

## Goal

Modernize ChessML to a current, reliable Python and dependency baseline while
preserving the existing pretrained board-recognition inference path.

The target developer workflow remains:

```bash
uv sync --locked
uv run python scripts/validate/validate_board_recognition.py -d mps
```

The migration must keep all four existing board-recognition checkpoints
loadable and must complete inference over the 167 local validation images on
macOS MPS. Exact logits, exact floating-point values, and identical recognized
FENs are not required. Every produced FEN must still be valid.

Continuing training from optimizer, scheduler, callback, or trainer-loop state
is not required. Training entry points must continue to import and expose their
command-line help, but no expensive training run is part of acceptance.

## Non-goals

This migration will not:

- retrain, fine-tune, convert, or rewrite any checkpoint;
- adopt PyTorch nightlies or newly released dependency lines without a useful
  public soak period;
- add ONNX, ONNX Runtime, Albumentations, a new build backend, a new model
  format, or a custom dependency abstraction;
- guarantee exact numerical output across PyTorch, MPS, NumPy, OpenCV, or
  Pillow versions;
- claim Linux/CUDA runtime validation from the macOS host;
- resume training from a full Lightning trainer state;
- reorganize training and data tools into optional dependency groups;
- repair unrelated application bugs or stale scripts;
- resolve the pre-existing disagreement between the repository license file
  and the MIT package classifier.

## Current evidence

The starting branch is `modernization` at `bf31919`, with a clean worktree.
The uv migration established one canonical `pyproject.toml`, Python 3.11.12,
and a 205-package lock.

The current core versions are:

| Component | Current |
| --- | ---: |
| Python | 3.11.12 |
| PyTorch | 2.7.0 |
| torchvision | 0.22.0 |
| Lightning | 2.5.1.post0 |
| timm | 1.0.15 |
| torchmetrics | 1.7.1 |
| NumPy | 2.2.5 |
| OpenCV | 4.11.0.86 |
| Pillow | 9.5.0 |
| OR-Tools | 9.14.6206 |

The existing test baseline is 24 passing tests. All four local checkpoints
strict-load on CPU in the current environment, and one complete CPU
recognition produces a parseable FEN. The previous uv migration also ran the
exact MPS validator successfully over all 167 local images.

The checkpoints are full Lightning checkpoints:

| Checkpoint | Lightning writer | State entries | Inference constructor |
| --- | ---: | ---: | --- |
| `bd-MobileViTV2FPN-v1.ckpt` | 2.2.4 | 303 | `MobileViTV2FPN` |
| `sc-9-bs=64-step=23296.ckpt` | 2.5.1.post0 | 249 | `MobileNetV3SmallClassifier` |
| `pc-48-bs=128-step=18944.ckpt` | 2.5.1.post0 | 319 | `MobileNetV3LargeClassifier` |
| `mp-MetaPredictor-v1.ckpt` | 2.2.4 | 55 | `MetaPredictor` |

The square- and piece-classifier checkpoints contain pickled Python class
objects in their saved hyperparameters. They therefore require trusted
full-checkpoint loading rather than PyTorch's restricted weights-only loader.

## Runtime boundary

`scripts/validate/validate_board_recognition.py` is the end-to-end acceptance
path. It performs these steps:

1. reconstruct four Lightning modules and strictly load their state dicts;
2. create timm backbones and torchvision feature/loss components;
3. detect and unskew the board with PyTorch, NumPy, OpenCV, and Pillow;
4. classify occupancy and pieces on MPS;
5. solve chess-piece constraints with OR-Tools on CPU;
6. encode the board and predict metadata;
7. construct a python-chess board and final FEN;
8. render validation output through fenToBoardImage and Pillow.

Because model modules import validation and plotting libraries eagerly,
scikit-learn, Matplotlib, and torchmetrics remain inference-time dependencies
for this migration even though their primary purpose is training validation.
Moving those imports or splitting optional features is a later cleanup.

## Version selection policy

Choose the newest stable release that is both officially compatible and has a
meaningful patch or ecosystem soak period. Do not equate "latest" with
"reliable."

The following newly released lines are deliberately excluded:

- PyTorch 2.13.0 and torchvision 0.28.0, released nine days before this design;
- NumPy 2.5.1, released thirteen days before this design;
- timm 1.0.28, released six days before this design;
- OpenCV 5.0, a new major release;
- Matplotlib 3.11.0, a new minor line with substantially less field time.

Lightning 2.6.2 and 2.6.3 are prohibited because Lightning's official
`GHSA-w37p-236h-pfx3` advisory identifies them as compromised releases.
Lightning 2.6.5 is a later legitimate release.

The project lock records exact versions. Published project metadata continues
to use tested lower bounds rather than duplicating every lock entry as an
equality pin. `uv sync --locked` is the supported reproducible installation
path.

## Target baseline

| Component | Target | Selection reason |
| --- | ---: | --- |
| Python | 3.14.6 | Stable for nine months and six patch releases |
| PyTorch | 2.12.1 | Patched preceding release line; skip new 2.13 |
| torchvision | 0.27.1 | Official PyTorch 2.12 pair |
| Lightning | 2.6.5 | Current patched, uncompromised 2.6 release |
| torchmetrics | 1.9.0 | Current stable release |
| timm | 1.0.27 | Mature patch before six-day-old 1.0.28 |
| NumPy | 2.4.6 | Mature patched line before new 2.5 |
| OpenCV | 4.13.0.92 | Current 4.x line; avoid new major 5 |
| Pillow | 12.3.0 | Current maintained release |
| fenToBoardImage | 1.4.1 | Current release with canonical snake_case API |
| OR-Tools | 9.15.6755 | Current release with CPython 3.14 arm64 wheels |
| scikit-learn | 1.9.0 | Current release with CPython 3.14 arm64 wheels |
| Matplotlib | 3.10.9 | Mature patch line before 3.11 |
| OmegaConf | 2.3.1 | Current stable patch |
| python-chess | 1.11.2 | Already current |
| TensorBoard | 2.21.0 | Current training logger/CLI release |
| Requests | 2.34.2 | Current patched release |
| tqdm | 4.68.4 | Current patched release |
| Stockfish wrapper | 5.2.0 | Maintained PyPI release with provenance |
| PyYAML | 6.0.3 transitive | Supplied through retained dependencies |
| pytest | 9.1.1 | Current stable test runner |
| setuptools | 83.0.0 | Retain the existing build backend |
| uv | 0.11.28 | Current stable project manager |

If a verified dependency incompatibility prevents Python 3.14, Python 3.13.14
is the only approved fallback. The fallback must be justified by a reproduced
failure; it is not a convenience option.

Primary release and compatibility references:

- [Python support status](https://devguide.python.org/versions/)
- [PyTorch releases](https://github.com/pytorch/pytorch/releases)
- [torchvision compatibility matrix](https://pypi.org/project/torchvision/)
- [Lightning releases](https://github.com/Lightning-AI/pytorch-lightning/releases)
- [Lightning security advisory](https://github.com/Lightning-AI/pytorch-lightning/security/advisories/GHSA-w37p-236h-pfx3)
- [NumPy releases](https://pypi.org/project/numpy/)
- [fenToBoardImage 1.4.1](https://pypi.org/project/fenToBoardImage/)

## Dependency ownership and cleanup

Keep and modernize dependencies used by tracked package code, scripts, or the
documented workflow:

- core/runtime: python-chess, NumPy, PyTorch, torchvision, Lightning,
  OmegaConf, Pillow, OpenCV, timm, OR-Tools, scikit-learn, Matplotlib, and
  torchmetrics;
- training: TensorBoard;
- data and validation tools: fenToBoardImage, Requests, tqdm, and Stockfish;
- development: pytest.

Remove these direct dependencies because no tracked live code or documented
workflow uses them:

- `notebook` and `ipywidgets` — no notebooks are tracked;
- `python-dotenv` — no imports or configuration use;
- `kaggle` — no imports or documented Kaggle workflow;
- `pandas` — no direct import; OR-Tools may retain it transitively;
- `plotly-express` — only commented-out code and no release since 2019;
- `websockets` — no imports;
- `tensorboard-plugin-profile` — no profiling workflow, and XProf supersedes
  the plugin;
- direct `PyYAML` — its only source import is unused; OmegaConf and Lightning
  still provide PyYAML transitively.

Delete the unused `safe_load` import from `chessml/__init__.py` when removing
the direct PyYAML declaration. Do not add Plotly, XProf, or another replacement
for an unused feature.

fenToBoardImage 1.4.1 keeps deprecated camel-case function aliases but does
not translate the 1.3 keyword names. Migrate all three live renderer call sites
to `fen_to_image`, `load_pieces_folder`, and snake_case keyword arguments. Do
not add a local compatibility wrapper.

Replace the pinned Stockfish Git dependency with the maintained `stockfish`
5.2.0 PyPI artifact. This changes only a data-evaluation tool; the wrapper
still requires a separately installed Stockfish engine executable.

Do not introduce optional groups in this phase. Retained training and data
tools remain available after the existing `uv sync --locked` command.

## Python and packaging policy

Change `.python-version` to `3.14.6` and narrow project metadata to:

```toml
requires-python = ">=3.14,<3.15"
```

Replace the Python 3.11 classifier with Python 3.14. Keep setuptools as the
build backend and raise its build requirement to the selected modern release.
Changing build backends while changing the entire runtime would provide no
compatibility benefit.

Require the tested uv series:

```toml
[tool.uv]
required-version = ">=0.11.28,<0.12"
```

Regenerate and commit `uv.lock` after every intentional dependency boundary.
Do not hand-edit the generated lock.

## Checkpoint compatibility changes

Use Lightning's normal `load_from_checkpoint` path so its documented
checkpoint-schema migration handles the older 2.2.4 and 2.5.1 checkpoint
metadata in memory.

At trusted repository checkpoint call sites used by board-recognition
inference, pass:

```python
weights_only=False
```

This is required because two existing checkpoints contain class objects in
saved hyperparameters. It must not become a global loader default and must not
be used for untrusted checkpoint files.

For the board detector, square classifier, and piece classifier, also override
the reconstructed backbone with:

```python
base_model_kwargs={"pretrained": False}
```

The checkpoint strictly supplies every learned parameter, so downloading
ImageNet weights immediately before overwriting them is unnecessary. Strict
state loading remains enabled; missing or unexpected keys are migration
failures, not warnings to suppress.

Update the corresponding README examples to show the verified loading mode.
Do not create a wrapper or checkpoint migration abstraction for a handful of
explicit call sites.

## Staged implementation

Keep failure attribution and rollback simple by using these commit boundaries:

1. Upgrade fenToBoardImage and Pillow, remove obsolete direct dependencies,
   and validate rendering on Python 3.11.
2. Upgrade PyTorch and torchvision together and run strict checkpoint plus
   CPU/MPS inference gates.
3. Upgrade Lightning, add explicit trusted-checkpoint loading, and rerun all
   checkpoint gates.
4. Upgrade timm, disable redundant pretrained initialization, and rerun all
   checkpoint gates.
5. Upgrade the NumPy/OpenCV native image stack and rerun image/inference gates.
6. Upgrade OR-Tools, torchmetrics, scikit-learn, Matplotlib, TensorBoard, and
   retained data tools with focused tests and smokes.
7. Move to Python 3.14.6, update uv/setuptools metadata, regenerate the final
   lock, and run the complete acceptance suite.
8. Update user documentation with the final tested versions and commands.

A stage that fails its compatibility gate is fixed or reverted before the next
stage. Do not stack speculative fixes across dependency boundaries.

## Automated coverage

Keep the existing 24 unit tests. Add one focused local-asset checkpoint
compatibility test that:

- skips only when the ignored checkpoint assets are absent;
- reconstructs all four production models on CPU;
- disables pretrained backbone downloads;
- explicitly performs trusted full-checkpoint loading;
- relies on Lightning/PyTorch strict state loading to reject missing,
  unexpected, or shape-incompatible parameters.

The checkpoint test is an integration test, not a replacement for the unit
tests. A clean clone cannot exercise it because the 274 MB of checkpoints are
intentionally not stored in Git.

No test will hard-code model logits or FEN snapshots across library versions.
Existing unit tests continue to cover board representation, FEN orientation,
dataset behavior, and OR-Tools constraint invariants.

## Verification

Run focused tests after each stage and the following complete suite before
completion:

```bash
uv lock --check
uv sync --locked
uv run python -m pytest -q
uv run python -m compileall -q chessml scripts tests
uv pip check --python .venv/bin/python
uv build
```

Strict-load all four checkpoints on CPU and run one real validation image
through the complete recognition helper. Require a syntactically valid
python-chess FEN; exact prediction equality with the old environment is not
required.

Run `--help` in separate processes for the four board-recognition training
entry points:

- `scripts/train/train_board_detector.py`;
- `scripts/train/train_square_classifier.py`;
- `scripts/train/train_piece_classifier.py`;
- `scripts/train/train_meta_predictor.py`.

Do not import training scripts as modules because their script decorator can
start training.

The final host-only inference command is exactly:

```bash
uv run python scripts/validate/validate_board_recognition.py -d mps
```

It must process all 167 local PNG inputs. Verify that 167 FEN text files are
created and that python-chess parses each one. Generated output directories
are ignored local artifacts and may be recreated by the existing validator.

The final macOS arm64 resolution target is macOS 14+ because the approved
PyTorch 2.12.1 Python 3.14 wheel has that minimum. The universal lock must also
resolve compatible artifacts for Linux x86-64. Linux/CUDA evidence is limited
to dependency resolution and imports because the current host has no CUDA
device. Do not report Linux/CUDA runtime success.

## Completion criteria

The modernization is complete only when:

- Python 3.14.6 and every selected direct version are present in the final
  locked environment;
- prohibited or removed dependencies are absent as direct requirements and
  compromised Lightning versions are absent from the lock;
- the full unit and checkpoint compatibility suites pass;
- all four checkpoints strict-load without rewriting files or weakening
  strictness;
- CPU recognition yields a valid FEN;
- the exact MPS validator processes and validates all 167 images;
- package integrity, compilation, build, and training-help smokes pass;
- README commands and version claims match the verified implementation;
- the worktree contains no accidental generated outputs or unrelated edits.
