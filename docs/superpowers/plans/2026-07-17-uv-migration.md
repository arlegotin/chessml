# Conda-to-uv Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Conda/pip project bootstrap with a native, locked uv project while preserving Python 3.11.12, every retained runtime-library version, board-recognition training imports, checkpoints, and MPS inference.

**Architecture:** Consolidate package metadata and dependencies in one PEP 621 `pyproject.toml`, select Python with `.python-version`, and commit uv's universal lock. Seed the first lock from the working Conda environment through temporary constraints, remove those constraints, then compare installed distributions before documentation and end-to-end validation.

**Tech Stack:** uv 0.6.4 or newer, Python 3.11.12, setuptools, PyTorch 2.7.0, torchvision 0.22.0, Lightning 2.5.1.post0, pytest, macOS MPS.

## Global Constraints

- Use `docs/superpowers/specs/2026-07-17-uv-migration-design.md` as the approved contract.
- Work on the existing `modernization` branch; do not create a branch or worktree.
- This phase changes package management only. Do not upgrade or downgrade Python or any retained runtime library.
- Preserve the dependency specifiers from `requirements.txt`; exact versions belong in `uv.lock`.
- Preserve the Stockfish fork at commit `ba93cf219d705be625f649181660d2ebbf130045`.
- Do not reintroduce `torch-exid`, promote `torchdata`, prune historical dependencies, or add new CUDA indexes.
- Keep PyTorch and torchvision on PyPI, matching the current repository source policy.
- Keep setuptools as the build backend and discover only `chessml*`, including namespace packages.
- Keep the public Python requirement `>=3.11`; `.python-version` selects the tested 3.11.12 interpreter.
- `config.local.yaml` remains ignored and required. Do not change application config loading in this migration.
- Do not modify model code, tensor layouts, preprocessing, trainers, datasets, checkpoints, or validator behavior.
- Never import a `scripts/train/*.py` module directly; its decorator can start training. Use a separate `--help` process.
- Treat any retained runtime-version mismatch between Conda and uv as a stop condition, not an acceptable migration difference.
- A universal lock and Linux wheel entries are static evidence only; do not claim Linux CUDA execution without the host.
- The final inference gate is exactly `uv run python scripts/validate/validate_board_recognition.py -d mps` from the repository root.

## File Structure

- Create `pyproject.toml`: canonical PEP 621 metadata, build configuration, runtime dependencies, and pytest development group.
- Create `.python-version`: select Python 3.11.12 for uv.
- Create `uv.lock`: generated universal dependency lock seeded from the current Conda environment.
- Modify `.gitignore`: allow `.python-version` to be committed while continuing to ignore `.venv`.
- Modify `tests/test_chessml.py`: align the stale distribution-version assertion with the existing runtime version 0.1.1.
- Delete `environment.yml`: remove the Conda bootstrap.
- Delete `requirements.txt`: remove the duplicate pip dependency manifest.
- Delete `setup.py`: remove the duplicate legacy package metadata.
- Modify `README.md`: document uv installation, sync, configuration prerequisite, and `uv run` commands.
- Modify `scripts/validate/process_video.sh`: update its live validation command example to `uv run`.

---

### Task 1: Create the canonical uv project and preserve the environment

**Files:**

- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `uv.lock`
- Modify: `.gitignore:8`
- Modify: `tests/test_chessml.py:4`
- Delete: `environment.yml`
- Delete: `requirements.txt`
- Delete: `setup.py`
- Temporary: `/tmp/chessml-lock-project/pyproject.toml`
- Temporary: `/tmp/chessml-lock-project/README.md`
- Temporary: `/tmp/chessml-lock-project/render_constraints.py`
- Temporary: `/tmp/chessml_compare_envs.py`
- Temporary: `/tmp/chessml_verify_wheel.py`

**Interfaces:**

- Consumes: the installed distributions visible to `/Users/artemlegotin/anaconda3/envs/chessml/bin/python`; the dependency specifiers in the legacy `requirements.txt`; package metadata from `setup.py`.
- Produces: `uv sync --locked` and `uv run python scripts/sanity_check.py` backed by `.venv`; distribution `chessml==0.1.1`; a wheel containing all `chessml*` namespaces and excluding `scripts` and `tests`.

- [ ] **Step 1: Record the two failing packaging behaviors before editing**

Run the existing version test with the current environment:

```bash
/Users/artemlegotin/anaconda3/envs/chessml/bin/python -m pytest -q tests/test_chessml.py
```

Expected: one failure because the test expects `0.1.0` while `chessml.__version__` is `0.1.1`.

Run the current package-discovery assertion:

```bash
/Users/artemlegotin/anaconda3/envs/chessml/bin/python -c 'from setuptools import find_packages; packages = set(find_packages()); assert "chessml.models.lightning" in packages, sorted(packages)'
```

Expected: `AssertionError`; the printed package list omits `chessml.models.lightning` and the other namespace directories.

Confirm that the starting environment itself is internally consistent:

```bash
/Users/artemlegotin/anaconda3/envs/chessml/bin/python -m pip check
```

Expected: `No broken requirements found.`

- [ ] **Step 2: Replace the three legacy manifests with canonical project metadata**

Create `pyproject.toml` with exactly:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "chessml"
version = "0.1.1"
authors = [
    { name = "Artem Legotin", email = "arlegotin@gmail.com" },
]
description = "A Python package for advanced chess analysis"
readme = "README.md"
requires-python = ">=3.11"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "chess>=1.9.4",
    "notebook>=6.5.3",
    "ipywidgets>=8.0.4",
    "python-dotenv>=1.0.0",
    "tensorboard>=2.12.0",
    "kaggle>=1.5.13",
    "scikit-learn>=1.2.1",
    "pandas>=1.5.3",
    "plotly-express>=0.4.1",
    "matplotlib>=3.7.1",
    "PyYAML>=6.0",
    "lightning>=2.2.4",
    "websockets>=11.0.3",
    "torch>=2.0.1",
    "stockfish @ git+https://github.com/py-stockfish/stockfish.git@ba93cf219d705be625f649181660d2ebbf130045",
    "fenToBoardImage>=1.3.0",
    "opencv-python>=4.9.0",
    "torchvision>=0.18.0",
    "timm>=0.9.16",
    "omegaconf>=2.3.0",
    "torchmetrics>=1.4.0.post0",
    "tensorboard-plugin-profile>=2.15.1",
    "ortools>=9.14.6206",
    "numpy>=2.2.5",
    "Pillow>=9,<10",
    "requests>=2.32.3",
    "tqdm>=4.67.1",
]

[project.urls]
Homepage = "https://github.com/arlegotin/chessml"

[dependency-groups]
dev = [
    "pytest>=5.2",
]

[tool.setuptools.packages.find]
include = ["chessml*"]
namespaces = true
```

Create `.python-version` with:

```text
3.11.12
```

Use `apply_patch` to remove only the `.python-version` line from `.gitignore`, change `tests/test_chessml.py` to:

```python
from chessml import __version__


def test_version():
    assert __version__ == "0.1.1"
```

Use `apply_patch` to delete `environment.yml`, `requirements.txt`, and `setup.py` after their content has been transferred. Do not edit application modules.

- [ ] **Step 3: Generate a temporary constraint block from the working environment**

Create the temporary directory and copy only the canonical project inputs:

```bash
test ! -e /tmp/chessml-lock-project
mkdir -p /tmp/chessml-lock-project
cp pyproject.toml /tmp/chessml-lock-project/pyproject.toml
cp README.md /tmp/chessml-lock-project/README.md
```

Expected: the absence check proves no stale lock can influence resolution, then the new directory contains only the two copied files.

Create `/tmp/chessml-lock-project/render_constraints.py` with:

```python
from importlib.metadata import distributions
from re import sub
import subprocess
import sys

from packaging.requirements import Requirement


def normalize(name: str) -> str:
    return sub(r"[-_.]+", "-", name).lower()


excluded = {
    "chessml",
    "pip",
    "setuptools",
    "torch-exid",
    "torchdata",
    "wheel",
}
installed = {
    normalize(distribution.metadata["Name"]): distribution.version
    for distribution in distributions()
}
versions = {}

freeze = subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze", "--all"],
    text=True,
)

for line in freeze.splitlines():
    if not line or line.startswith(("#", "-e ")):
        continue

    requirement = Requirement(line)
    canonical_name = normalize(requirement.name)
    if canonical_name not in excluded:
        versions[canonical_name] = installed[canonical_name]

print("[tool.uv]")
print("constraint-dependencies = [")
for name, version in sorted(versions.items()):
    print(f'    "{name}=={version}",')
print("]")
```

Run:

```bash
/Users/artemlegotin/anaconda3/envs/chessml/bin/python /tmp/chessml-lock-project/render_constraints.py
```

Expected: the script consumes `pip freeze --all` and prints a complete `[tool.uv]` TOML block whose retained versions include `torch==2.7.0`, `torchvision==0.22.0`, `lightning==2.5.1.post0`, `pillow==9.5.0`, and `pytest==8.3.5`, and which omits ChessML, packaging tools, `torch-exid`, and `torchdata`.

Append the command's complete stdout to `/tmp/chessml-lock-project/pyproject.toml` with `apply_patch`. Do not append it to the repository's canonical `pyproject.toml`.

- [ ] **Step 4: Generate the constrained lock, then remove the temporary constraints**

Create the first universal lock from the temporary project:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv lock --project /tmp/chessml-lock-project
```

Expected: resolution succeeds and creates `/tmp/chessml-lock-project/uv.lock`.

Copy the generated artifact into the repository, then resolve once without the temporary constraints. uv must use the copied lock as its version preference:

```bash
cp /tmp/chessml-lock-project/uv.lock uv.lock
UV_CACHE_DIR=/tmp/chessml-uv-cache uv lock
UV_CACHE_DIR=/tmp/chessml-uv-cache uv lock --check
```

Expected: both repository lock commands exit zero. `pyproject.toml` contains no `constraint-dependencies`; `uv.lock` retains the working direct versions, including torch 2.7.0, torchvision 0.22.0, Lightning 2.5.1.post0, Pillow 9.5.0, and pytest 8.3.5.

Verify the source and exclusions:

```bash
rg -n 'ba93cf219d705be625f649181660d2ebbf130045|torch-exid|torchdata|constraint-dependencies' pyproject.toml uv.lock
```

Expected: the Stockfish commit appears; `torch-exid`, `torchdata`, and `constraint-dependencies` do not appear.

Verify that the universal lock records both supported artifact families:

```bash
/Users/artemlegotin/anaconda3/envs/chessml/bin/python -c 'from pathlib import Path; text = Path("uv.lock").read_text(); artifacts = ["torch-2.7.0-cp311-none-macosx_11_0_arm64.whl", "torch-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl", "torchvision-0.22.0-cp311-cp311-macosx_11_0_arm64.whl", "torchvision-0.22.0-cp311-cp311-manylinux_2_28_x86_64.whl"]; missing = [artifact for artifact in artifacts if artifact not in text]; assert not missing, missing; print("macOS and Linux PyTorch artifacts locked")'
```

Expected: `macOS and Linux PyTorch artifacts locked`. This checks resolution metadata, not Linux CUDA execution.

- [ ] **Step 5: Create a fresh uv-managed Python environment**

Confirm that no pre-existing `.venv` will be overwritten:

```bash
test ! -e .venv
```

Expected: exit zero. If `.venv` exists, inspect and preserve it before continuing.

Install the selected interpreter and sync the locked environment:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv python install 3.11.12
UV_CACHE_DIR=/tmp/chessml-uv-cache uv sync --locked
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python -c 'import platform, sys; assert sys.version_info[:3] == (3, 11, 12); print(platform.platform(), sys.version)'
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python -c 'from importlib.util import find_spec; assert find_spec("torch_exid") is None; assert find_spec("torchdata") is None; print("removed packages absent")'
UV_CACHE_DIR=/tmp/chessml-uv-cache uv pip check
```

Expected: uv creates `.venv`, the interpreter assertion passes, removed packages are absent, and `uv pip check` exits zero with no incompatible packages.

- [ ] **Step 6: Prove that retained environment versions did not drift**

Create `/tmp/chessml_compare_envs.py` with:

```python
import json
import subprocess
from importlib.metadata import distributions
from re import sub


BASELINE_PYTHON = "/Users/artemlegotin/anaconda3/envs/chessml/bin/python"
PACKAGING_TOOLS = {
    "pip",
    "setuptools",
    "wheel",
}
PROJECT_ONLY = {
    "chessml",
}
EXPLICIT_REMOVALS = {
    "torch-exid",
    "torchdata",
}


def normalize(name: str) -> str:
    return sub(r"[-_.]+", "-", name).lower()


def versions():
    return {
        normalize(distribution.metadata["Name"]): distribution.version
        for distribution in distributions()
    }


baseline_code = r'''
import json
from importlib.metadata import distributions
from packaging.markers import default_environment
from packaging.requirements import Requirement
from re import sub

def normalize(name):
    return sub(r"[-_.]+", "-", name).lower()

roots = {
    "chess", "fentoboardimage", "ipywidgets", "kaggle", "lightning",
    "matplotlib", "notebook", "numpy", "omegaconf", "opencv-python",
    "ortools", "pandas", "pillow", "plotly-express", "pytest",
    "python-dotenv", "pyyaml", "requests", "scikit-learn", "stockfish",
    "tensorboard", "tensorboard-plugin-profile", "timm", "torch",
    "torchmetrics", "torchvision", "tqdm", "websockets",
}
installed = {
    normalize(distribution.metadata["Name"]): distribution
    for distribution in distributions()
}
environment = default_environment()
environment["extra"] = ""
graph = {}

for name, distribution in installed.items():
    dependencies = []
    for requirement_text in distribution.requires or []:
        requirement = Requirement(requirement_text)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            dependencies.append(normalize(requirement.name))
    graph[name] = dependencies

retained = set()
pending = list(roots)
while pending:
    name = pending.pop()
    if name in retained:
        continue
    if name not in installed:
        raise SystemExit(f"missing baseline dependency: {name}")
    retained.add(name)
    pending.extend(graph[name])

print(json.dumps({
    "all_versions": {
        name: distribution.version
        for name, distribution in installed.items()
    },
    "retained": sorted(retained),
    "versions": {
        name: installed[name].version
        for name in retained
    },
}))
'''

baseline = json.loads(
    subprocess.check_output([BASELINE_PYTHON, "-c", baseline_code], text=True)
)
current = versions()
retained = set(baseline["retained"]) - PACKAGING_TOOLS
baseline_only = (
    baseline["all_versions"].keys()
    - current.keys()
    - PACKAGING_TOOLS
    - PROJECT_ONLY
    - EXPLICIT_REMOVALS
)
missing = {
    name: baseline["versions"][name]
    for name in sorted(retained - current.keys())
}
mismatches = {
    name: {"conda": baseline["versions"][name], "uv": current[name]}
    for name in sorted(retained & current.keys())
    if baseline["versions"][name] != current[name]
}
unexpected = {
    name: current[name]
    for name in sorted(current.keys() - retained - PACKAGING_TOOLS - PROJECT_ONLY)
}
packaging_tools = {
    name: {
        "conda": baseline["versions"].get(name),
        "uv": current.get(name),
    }
    for name in sorted(PACKAGING_TOOLS)
}
baseline_only_unretained = {
    name: baseline["all_versions"][name]
    for name in sorted(baseline_only - retained)
}

if missing or mismatches or unexpected:
    print(json.dumps({
        "missing": missing,
        "mismatches": mismatches,
        "unexpected": unexpected,
        "baseline_only_unretained": baseline_only_unretained,
        "packaging_tools": packaging_tools,
    }, indent=2))
    raise SystemExit(1)

print(json.dumps({
    "retained_versions": len(retained),
    "baseline_only_unretained": baseline_only_unretained,
    "packaging_tools": packaging_tools,
}, indent=2))
```

Run it inside the uv environment:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python /tmp/chessml_compare_envs.py
```

Expected: exit zero, a positive `retained_versions` count, a classified `baseline_only_unretained` map, and a separately reported packaging-tool comparison. Review every baseline-only entry to confirm it is outside the declared dependency closure. Any missing retained package, version mismatch, or unexpected runtime package stops this migration.

- [ ] **Step 7: Run tests and verify the built distribution outside the source tree**

Run the complete test suite:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python -m pytest -q
```

Expected: 24 tests pass, including the corrected version assertion and the existing iterable-dataset coverage.

Build the source distribution and wheel:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv build
```

Expected: `dist/chessml-0.1.1.tar.gz` and `dist/chessml-0.1.1-py3-none-any.whl` are created.

Create `/tmp/chessml_verify_wheel.py` with:

```python
from email.parser import Parser
from pathlib import Path
import tomllib
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools import find_namespace_packages


wheel = Path("dist/chessml-0.1.1-py3-none-any.whl")
source_packages = set(find_namespace_packages(include=["chessml*"]))


def normalized_requirement(value: str):
    requirement = Requirement(value)
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(requirement.extras)),
        str(requirement.specifier),
        requirement.url,
        str(requirement.marker) if requirement.marker is not None else None,
    )

with ZipFile(wheel) as archive:
    names = set(archive.namelist())
    wheel_packages = {
        package
        for package in source_packages
        if any(name.startswith(f"{package.replace('.', '/')}/") for name in names)
    }
    assert wheel_packages == source_packages, sorted(source_packages - wheel_packages)
    assert not any(name.startswith("scripts/") for name in names)
    assert not any(name.startswith("tests/") for name in names)

    metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
    metadata = Parser().parsestr(archive.read(metadata_name).decode())

assert metadata["Version"] == "0.1.1"
project = tomllib.loads(Path("pyproject.toml").read_text())
expected_requirements = {
    normalized_requirement(value)
    for value in project["project"]["dependencies"]
}
wheel_requirements = {
    normalized_requirement(value)
    for value in metadata.get_all("Requires-Dist", [])
}
assert wheel_requirements == expected_requirements, {
    "missing": sorted(expected_requirements - wheel_requirements, key=repr),
    "unexpected": sorted(wheel_requirements - expected_requirements, key=repr),
}
print("wheel contents and metadata verified")
```

Run:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python /tmp/chessml_verify_wheel.py
```

Expected: `wheel contents and metadata verified`.

Create an isolated wheel environment and working directory:

```bash
test ! -e /tmp/chessml-wheel-cwd
test ! -e /tmp/chessml-wheel-venv
mkdir -p /tmp/chessml-wheel-cwd
cp config.yaml /tmp/chessml-wheel-cwd/config.yaml
UV_CACHE_DIR=/tmp/chessml-uv-cache uv venv /tmp/chessml-wheel-venv --python 3.11.12
UV_CACHE_DIR=/tmp/chessml-uv-cache uv pip install --python /tmp/chessml-wheel-venv/bin/python --no-deps dist/chessml-0.1.1-py3-none-any.whl
UV_CACHE_DIR=/tmp/chessml-uv-cache uv pip install --python /tmp/chessml-wheel-venv/bin/python 'PyYAML==6.0.2' 'omegaconf==2.3.0'
```

Use `apply_patch` to create `/tmp/chessml-wheel-cwd/config.local.yaml` with:

```yaml
{}
```

From `/tmp/chessml-wheel-cwd`, run:

```bash
/tmp/chessml-wheel-venv/bin/python -c 'import importlib; from importlib.metadata import version; names = ["chessml", "chessml.data", "chessml.data.boards", "chessml.data.games", "chessml.data.images", "chessml.data.policy", "chessml.data.utils", "chessml.data.values", "chessml.models", "chessml.models.lightning", "chessml.models.torch", "chessml.models.utils", "chessml.play", "chessml.train"]; [importlib.import_module(name) for name in names]; assert version("chessml") == "0.1.1"; print("isolated wheel imports verified")'
```

Expected: `isolated wheel imports verified`. Running outside the repository ensures source files cannot mask wheel omissions.

- [ ] **Step 8: Review and commit the package-manager boundary**

Run:

```bash
git diff --check
git status --short
git diff -- .gitignore .python-version pyproject.toml tests/test_chessml.py environment.yml requirements.txt setup.py
```

Expected: only the planned packaging files, lock, and version assertion changed; no application module changed.

Stage the explicit file set and review the actual commit contents:

```bash
git add .gitignore .python-version pyproject.toml uv.lock tests/test_chessml.py environment.yml requirements.txt setup.py
git diff --cached --check
git diff --cached --stat
git diff --cached -- .gitignore .python-version pyproject.toml tests/test_chessml.py environment.yml requirements.txt setup.py
```

Expected: the staged diff contains only the planned package-manager boundary and has no whitespace errors.

Commit:

```bash
git commit -m "build: replace conda environment with uv"
```

Expected: one coherent packaging commit succeeds.

---

### Task 2: Document the uv workflow and smoke-test training imports

**Files:**

- Modify: `README.md:31-61,86,96-98,135-137,174-176,304-311`
- Modify: `scripts/validate/process_video.sh:23`

**Interfaces:**

- Consumes: `.python-version`, `pyproject.toml`, and `uv.lock` from Task 1.
- Produces: one activation-free user workflow using `uv sync --locked` and `uv run`; no historical design document changes.

- [ ] **Step 1: Confirm that live documentation still names the legacy workflow**

Run:

```bash
rg -n -i '\bconda\b|environment\.yml|(^|`)python scripts/|`tensorboard --' README.md scripts/validate/process_video.sh
```

Expected: README reports the Conda installation and unwrapped commands; `process_video.sh` contains one unwrapped validation example.

- [ ] **Step 2: Replace only current user-facing commands**

Replace the README installation section with:

````markdown
ChessML uses [uv](https://docs.astral.sh/uv/) to manage Python and project dependencies. Install uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Sync the exact locked environment:
```bash
uv sync --locked
```

No shell activation is required. uv creates a local `.venv` and runs project commands inside it with `uv run`.

Before running ChessML, ensure the ignored `./config.local.yaml` exists. If you do not need local overrides, create it with the YAML content `{}`.

As a sanity check, print the merged configuration:
```bash
uv run python scripts/sanity_check.py
```

> Tip: All script entry points are located in the `./scripts` directory. Use `-h` for guidance on how to use these scripts.
````

Extend the Configuration section with this sentence:

```markdown
Machine-specific overrides belong in the ignored `./config.local.yaml` and are merged over `./config.yaml`.
```

Make these exact command substitutions elsewhere in README:

```text
tensorboard --logdir=logs/tensorboard/lightning_logs
-> uv run tensorboard --logdir=logs/tensorboard/lightning_logs

python scripts/train/train_board_detector.py
-> uv run python scripts/train/train_board_detector.py

python scripts/train/train_piece_classifier.py
-> uv run python scripts/train/train_piece_classifier.py

python scripts/train/train_meta_predictor.py
-> uv run python scripts/train/train_meta_predictor.py

python scripts/data/download_pgns.py
-> uv run python scripts/data/download_pgns.py

python scripts/data/export_unique_fens.py
-> uv run python scripts/data/export_unique_fens.py
```

Change the live comment in `scripts/validate/process_video.sh` to:

```bash
# uv run python scripts/validate/validate_board_recognition.py -i "$output_dir"
```

Do not edit historical files under `docs/superpowers`.

- [ ] **Step 3: Verify documentation and all four board-training imports**

Run the documentation search again:

```bash
rg -n -i '\bconda\b|environment\.yml|(^|`)python scripts/|`tensorboard --' README.md scripts/validate/process_video.sh
```

Expected: no matches.

Run each training script in its own process:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python scripts/train/train_board_detector.py --help
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python scripts/train/train_square_classifier.py --help
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python scripts/train/train_piece_classifier.py --help
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python scripts/train/train_meta_predictor.py --help
```

Expected: every command exits zero after printing argparse usage; none starts training.

Run the documented sanity check:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python scripts/sanity_check.py
```

Expected: exit zero and the merged configuration is printed.

- [ ] **Step 4: Review and commit the user workflow**

Run:

```bash
git diff --check
git diff -- README.md scripts/validate/process_video.sh
```

Expected: only live commands and the required config prerequisite changed.

Commit:

```bash
git add README.md scripts/validate/process_video.sh
git commit -m "docs: replace conda commands with uv"
```

Expected: the documentation commit succeeds.

---

### Task 3: Run final package, training, and MPS inference gates

**Files:**

- Verify only: repository files from Tasks 1 and 2
- Generated and ignored: `dist/`, `.venv/`, and validator output directories beside `test_data/input_frames/book_long`

**Interfaces:**

- Consumes: the committed `pyproject.toml`, `uv.lock`, Python 3.11.12 environment, existing checkpoints, test frames, piece assets, and local configuration.
- Produces: evidence that lock integrity, installed compatibility, tests, training imports, checkpoint loading, MPS inference, and FEN outputs all remain functional.

- [ ] **Step 1: Run the broad reproducibility gates from a clean commit state**

Run:

```bash
git status --short --branch
UV_CACHE_DIR=/tmp/chessml-uv-cache uv lock --check
UV_CACHE_DIR=/tmp/chessml-uv-cache uv sync --locked
UV_CACHE_DIR=/tmp/chessml-uv-cache uv pip check
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python -m pytest -q
UV_CACHE_DIR=/tmp/chessml-uv-cache uv build
```

Expected: the branch is clean, the lock is current, sync changes nothing material, package compatibility passes, all 24 tests pass, and the package builds successfully. Task 1 already records the retained-version and wheel-content checks while its temporary verification helpers still exist.

- [ ] **Step 2: Preserve any existing validator outputs before the destructive command**

Inspect:

```bash
find test_data/input_frames -maxdepth 1 -type d -name 'book_long_*' -print
```

Expected in the current checkout: no output directories exist. If directories appear, inspect them and copy non-reproducible content to `/tmp/chessml-validator-backup` before continuing. Do not delete uninspected user data.

- [ ] **Step 3: Run the exact MPS acceptance command**

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run python scripts/validate/validate_board_recognition.py -d mps
```

Expected: exit zero, all four existing checkpoints load, and tqdm completes 167 images without an exception.

- [ ] **Step 4: Validate every generated FEN and output count**

Run:

```bash
UV_CACHE_DIR=/tmp/chessml-uv-cache uv run --locked python -c 'from pathlib import Path; import chess; root = Path("test_data/input_frames"); inputs = sorted((root / "book_long").glob("*.png")); marked = sorted((root / "book_long_marked").glob("*.png")); extracted = sorted((root / "book_long_extracted").glob("*.png")); boards = sorted((root / "book_long_boards").glob("*.png")); fens = sorted((root / "book_long_boards").glob("*.txt")); assert len(inputs) == 167; assert len(marked) == len(extracted) == len(boards) == len(fens) == len(inputs); [chess.Board(path.read_text().strip()) for path in fens]; print(f"validated {len(fens)} FEN outputs")'
```

Expected: `validated 167 FEN outputs`.

- [ ] **Step 5: Review the final branch without creating another commit**

Run:

```bash
git status --short --branch
git log --oneline -4
git diff HEAD~2 --check
```

Expected: tracked files are clean; ignored `.venv`, `dist`, and validation outputs may exist; the latest implementation commits are the package-manager and documentation commits. This task is verification-only and creates no commit.
