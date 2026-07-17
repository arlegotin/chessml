# Conda-to-uv Migration Design

Status: approved on 2026-07-17.

## Goal

Replace ChessML's Conda/pip bootstrap with a native uv project without
changing Python or library versions, model behavior, checkpoints, training
semantics, or inference results.

For an existing configured checkout, the new developer workflow is:

```bash
uv sync --locked
uv run python scripts/validate/validate_board_recognition.py -d mps
```

Project commands continue to require the ignored `config.local.yaml` that
`chessml/__init__.py` loads today. README will state that a developer without
local overrides must create this file with an empty YAML mapping (`{}`). Making
the override optional in application code is a separate behavior fix.

This is a package-manager migration only. Python, PyTorch, torchvision,
Lightning, timm, OpenCV, OR-Tools, and all other retained dependencies will be
upgraded later as separate, targeted changes.

## Current baseline

The repository currently spreads environment metadata across three files:

- `environment.yml` creates a Python 3.11 Conda environment and installs pip;
- `requirements.txt` supplies all project dependencies to pip using minimum
  bounds and one unpinned Git dependency;
- `setup.py` supplies incomplete package metadata but declares no
  dependencies.

Conda does not install a project library directly. Its configured channels are
therefore irrelevant to the current resolved package set; pip and PyPI already
provide Python, ML, and image-library packages after Conda creates the
interpreter.

The working local environment is macOS arm64 with Python 3.11.12. Its core
versions are PyTorch 2.7.0, torchvision 0.22.0, Lightning 2.5.1.post0, timm
1.0.15, OpenCV 4.11.0.86, NumPy 2.2.5, Pillow 9.5.0, torchmetrics 1.7.1, and
ORTools 9.14.6206. The installed Stockfish wrapper came from commit
`ba93cf219d705be625f649181660d2ebbf130045` of
`https://github.com/py-stockfish/stockfish.git`.

The previous `torch_exid` migration established this baseline:

- the focused dataset and board tests pass;
- the full suite has one known failure because package code reports version
  `0.1.1` while `setup.py` and `tests/test_chessml.py` expect `0.1.0`;
- the existing Conda environment passes `pip check`;
- all four board-recognition checkpoints load and the complete MPS validator
  succeeds on all 167 local validation images.

The ignored checkpoints, validation images, generated outputs, local config,
datasets, and piece assets remain local acceptance-test inputs. A fresh clone
cannot run that end-to-end validation without them.

## Chosen approach

Use uv's [native project workflow](https://docs.astral.sh/uv/guides/projects/)
with one canonical `pyproject.toml` and one committed
[universal `uv.lock`](https://docs.astral.sh/uv/concepts/projects/layout/).

The rejected alternatives are:

- using only `uv pip` with the existing files, because it would retain three
  overlapping sources of dependency and packaging truth;
- pruning, regrouping, or upgrading libraries during the migration, because a
  regression would no longer be attributable to the package-manager change;
- adding platform-selectable PyTorch extras or custom CUDA indexes, because
  the current repository does not select such an index and the Ubuntu CUDA
  backend has not been characterized on its actual host.

## Canonical project files

Add and commit:

- `pyproject.toml` for package metadata, build configuration, runtime
  dependencies, and the development dependency group;
- `.python-version` containing `3.11.12` for uv's managed interpreter choice;
- `uv.lock` containing exact direct and transitive resolutions.

Update `.gitignore` to stop ignoring `.python-version`; `.venv` remains
ignored.

Remove:

- `environment.yml`;
- `requirements.txt`;
- `setup.py`.

No generated `requirements.txt`, compatibility wrapper, Make target, shell
activation script, or second lockfile will be retained. `uv sync` creates and
maintains the repository-local `.venv`, which is already ignored.

`pyproject.toml` will preserve `requires-python = ">=3.11"` from `setup.py`.
The committed `.python-version` selects the validated 3.11.12 development
interpreter, and no other Python version will be claimed as tested. Narrowing
the public compatibility range would be a separate metadata-policy change.

The repository will not pin or install uv itself. The existing uv 0.6.4 meets
the [upstream minimum](https://docs.astral.sh/uv/guides/integration/pytorch/)
for the project and PyTorch features used here. A normal uv installation plus
`uv sync --locked` is the only bootstrap requirement.

## Packaging metadata

Keep setuptools as the build backend. Changing package manager and build
backend together provides no compatibility benefit. Configure
[namespace discovery](https://setuptools.pypa.io/en/stable/userguide/package_discovery.html)
in `pyproject.toml` rather than adding empty package files solely for the
build.

Package discovery will include only `chessml*` and enable namespace-package
discovery. The current `find_packages()` configuration incorrectly omits
imported directories without `__init__.py`, including:

- `chessml.data.policy`, `chessml.data.utils`, and `chessml.data.values`;
- `chessml.models` and its `lightning`, `torch`, and `utils` children;
- `chessml.play` and `chessml.train`.

It also discovers top-level `scripts` and `tests`, which are not distributable
ChessML packages. A built-wheel content check will prove that the `chessml`
namespaces are included and `scripts`/`tests` are excluded.

Use the existing runtime version `0.1.1` as the canonical package version and
align the version test with it. Distribution metadata therefore changes from
its stale `0.1.0` to the already exposed runtime value; this migration does not
publish a release.

All other name, description, author, URL, classifier, and license metadata
will be migrated from `setup.py` without opportunistic edits. The apparent
license-classifier disagreement is recorded for later review rather than
silently changed here.

## Dependency policy

Preserve the existing direct dependency specifiers in published project
metadata instead of replacing every lower bound with an equality pin. The
committed lock, not restrictive wheel metadata, records exact environment
versions.

Seed the initial resolution reproducibly from
`python -m pip freeze --all` in the working Conda environment. Convert that
snapshot to temporary uv `constraint-dependencies`, excluding the local
ChessML distribution, and create the first lock. Then remove the temporary
constraints and relock without `--upgrade`; uv uses the existing lock as its
version preference. This follows uv's
[pip-to-project migration guidance](https://docs.astral.sh/uv/guides/migration/pip-to-project/).
The temporary snapshot is migration evidence, not a second committed
dependency manifest.

Compare the resulting uv environment with the Conda snapshot. Every retained
runtime distribution common to the current platform must keep its version; a
mismatch stops this migration and becomes a separately approved dependency
change. Platform-only packages and packaging tools are compared separately
rather than forced into runtime dependencies.

This does not mean copying every installed distribution into
`pyproject.toml`. In particular, the removed `torch-exid`, unused `torchdata`,
editable project metadata, and incidental transitive packages will not become
direct dependencies.

Preserve every dependency currently declared in `requirements.txt`, even when
repository search finds no live import. Removing notebook, Kaggle, plotting,
or other historical tooling is a separate dependency-cleanup change. Move
only pytest to the standard `dev` dependency group.

Add the following already-installed packages as direct dependencies because
tracked code imports them directly:

- NumPy 2.2.5;
- Pillow 9.5.0;
- requests 2.32.3;
- tqdm 4.67.1.

This corrects ownership in package metadata without changing the installed
environment. Pillow must remain below 10 because `fenToBoardImage` 1.3.0
requires `Pillow>=9,<10`.

Use lower bounds matching the validated versions for NumPy, requests, and
tqdm, and use `Pillow>=9,<10`. The initial lock retains the exact versions
listed above.

Keep the Stockfish wrapper as a Git source and pin its currently installed
full commit in `pyproject.toml`. This preserves the existing fork and code
while preventing a later lock regeneration from following a moving default
branch.

The resulting locked direct dependency baseline is the installed version of
each of the following:

```text
chess 1.11.2                    notebook 7.4.2
ipywidgets 8.1.7                python-dotenv 1.1.0
tensorboard 2.19.0              kaggle 1.7.4.5
scikit-learn 1.6.1             pandas 2.2.3
plotly-express 0.4.1            matplotlib 3.10.3
PyYAML 6.0.2                    lightning 2.5.1.post0
websockets 15.0.1              torch 2.7.0
fenToBoardImage 1.3.0           opencv-python 4.11.0.86
torchvision 0.22.0             timm 1.0.15
omegaconf 2.3.0                torchmetrics 1.7.1
tensorboard-plugin-profile 2.19.9
ortools 9.14.6206               numpy 2.2.5
Pillow 9.5.0                   requests 2.32.3
tqdm 4.67.1                    stockfish Git commit ba93cf2...
```

The `dev` group contains pytest 8.3.5. No dependency groups for notebooks,
training, validation, CUDA, or optional features will be introduced in this
phase; those groups would change default availability or add unused choices.

For this research application, `project.dependencies` represents the complete
repository environment, including script and notebook tooling, rather than a
minimal reusable-library wheel. This intentionally preserves today's default
availability. Separating publishable runtime dependencies from repository
tools belongs to the later dependency-cleanup track.

## PyTorch platform policy

Lock torch at 2.7.0 and torchvision at 0.22.0, which are the pair installed by
the working environment and required by torchvision metadata. Preserve their
existing lower-bound project specifiers; the lock carries the exact versions.

Resolve them from PyPI, exactly as the current unqualified requirements do.
[PyPI publishes](https://pypi.org/simple/torch/) the required Python 3.11
macOS arm64 and Linux x86-64 wheels. The macOS wheel supplies the existing MPS
backend; there is no MPS-specific package index.

Do not add PyTorch CPU/CUDA indexes or uv source markers. Doing so would select
a CUDA variant that the repository did not previously specify. The universal
lock must resolve for macOS arm64 and Linux x86-64, but only the local macOS
MPS runtime can be functionally validated in this phase.

This preserves repository-declared behavior, not an undocumented machine
override. If the Ubuntu training host currently installs from a custom
`cu118`, `cu126`, or `cu128` index, that backend must be captured and tested as
a subsequent platform-specific dependency change using its `nvidia-smi` and
`torch.version.cuda` evidence.

## Developer workflow and documentation

README installation becomes:

```bash
uv sync --locked
uv run python scripts/sanity_check.py
```

Before the sanity check, a checkout without local overrides must create the
ignored `config.local.yaml` containing `{}`. Existing configured checkouts
retain their current file.

Current user-facing README commands and the live `process_video.sh` command
example will use `uv run`, including training, data-export, validation, and
TensorBoard commands. Historical design and implementation documents remain
historical. Shell activation is neither required nor documented. A developer
may activate `.venv` manually, but the project contract is `uv run`.

Routine dependency changes will use uv's project commands and commit both
`pyproject.toml` and `uv.lock`. Later upgrade tracks will intentionally change
one compatible dependency set at a time and rerun the same acceptance checks.

## Verification

Package-manager behavior is best verified at the environment and artifact
boundaries; no new application abstraction or unit-test module is needed.

Run the migration first against a fresh uv-managed `.venv` in the configured
working checkout, not the active Conda environment. Required checks are:

```bash
uv lock --check
uv sync --locked
uv run python -m pytest -q
uv pip check
uv build
```

Inspect the wheel to confirm:

- every tracked importable `chessml*` namespace is present;
- `scripts` and `tests` are absent;
- the distribution version and `chessml.__version__` are both `0.1.1`;
- dependency metadata matches `pyproject.toml`;
- `torch_exid` is absent from both declared and installed dependencies.

Install the wheel into an isolated temporary environment and run imports from
outside the repository so the source tree cannot mask missing packages. Give
that temporary working directory `config.yaml` plus an empty
`config.local.yaml`, then import each discovered `chessml*` namespace.

Compare the Conda and uv package snapshots and fail the migration on any
unexplained retained-version change. Inspect the lock's PyTorch and torchvision
artifacts for both CPython 3.11 macOS arm64 and Linux x86-64. This is static
Linux resolution evidence only, not a CUDA runtime test.

Run `--help` import smoke checks for these four board-recognition training
scripts without starting expensive training:

- `scripts/train/train_board_detector.py`;
- `scripts/train/train_square_classifier.py`;
- `scripts/train/train_piece_classifier.py`;
- `scripts/train/train_meta_predictor.py`.

Run the existing focused dataset, board-representation, checkpoint-loading,
recognition-helper, and decoder tests as part of the full suite.

The end-to-end acceptance command is:

```bash
uv run python scripts/validate/validate_board_recognition.py -d mps
```

It must exit successfully, load all four existing checkpoints, process the
configured 167 input images, and produce syntactically valid FEN output for
every image. Existing warnings are not migration failures unless their
behavior or severity changes.

The validator destructively recreates ignored output directories. Inspect
them before execution and preserve any non-reproducible user data. The command
must run from the repository root because current configuration loading and
asset paths are relative to that directory.

Static universal resolution is necessary but does not prove CUDA execution.
If the Ubuntu host is unavailable, report Linux/CUDA runtime validation as not
run rather than treating macOS MPS success as proof of it.

## Rollback

The migration is one reversible packaging boundary:

- restore `environment.yml`, `requirements.txt`, and `setup.py`;
- remove `pyproject.toml`, `.python-version`, and `uv.lock`;
- restore the `.python-version` ignore rule;
- restore Conda commands in README.

No code, model, checkpoint, data, or serialized format requires conversion.

## Deferred work

The following remain separate modernization tracks:

- Python upgrade;
- PyTorch/torchvision, Lightning, timm, torchmetrics, OpenCV, OR-Tools, and
  other library upgrades;
- dependency pruning and finer dependency groups;
- an explicit, host-tested Linux CUDA index;
- CI coverage on macOS and Linux;
- package-license metadata reconciliation;
- configuration and assets that currently assume the repository working
  directory;
- optional handling or a tracked template for `config.local.yaml`;
- existing training, validator, and iterable-dataset defects already recorded
  in the `torch_exid` migration design.

## Non-goals

This migration does not:

- upgrade or downgrade Python or any retained library;
- retrain models or rewrite checkpoints;
- change tensor shapes, model architectures, preprocessing, training loops,
  datasets, inference, or FEN output;
- remove historical dependencies merely because current source search finds
  no import;
- add Docker, CI, pre-commit, environment activation helpers, alternate
  lockfiles, or dependency automation;
- publish the ChessML package;
- claim Linux CUDA runtime compatibility without running on that host.
