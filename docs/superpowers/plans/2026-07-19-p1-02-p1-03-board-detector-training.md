# P1-02 and P1-03 BoardDetector Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the supported dynamic BoardDetector entrypoint train on finite,
disjoint iterable splits with compatible targets and save checkpoints selected
by its real validation metric.

**Architecture:** Partition and shuffle FEN strings before expensive rendering,
keep DataLoader shuffling only for the pregenerated map-style dataset, flatten
dynamic labels at their construction seam, and set BoardDetector's checkpoint
metric explicitly. One synthetic but real-Lightning entrypoint smoke covers
the complete train/validate/checkpoint path.

**Tech Stack:** Python 3.14, PyTorch 2.12, Lightning 2.6, pytest, NumPy, Pillow.

## Global Constraints

- Governing design:
  `docs/superpowers/specs/2026-07-19-p1-02-p1-03-board-detector-training-design.md`.
- Governing findings: `BOARD_RECOGNITION_AUDIT.tmp.md`, P1-02 and P1-03.
- Follow root `AGENTS.md`; no nested instruction file exists.
- Preserve the P2-02 multiprocessing-context behavior in
  `chessml/train/standard_training.py`.
- Do not change `standard_training()`'s default checkpoint monitor, model
  geometry/inference, worker counts, CLI surface, or iterable base classes.
- Do not download assets, regenerate datasets, train a production model, run
  production checkpoints, stage, commit, or publish anything.
- Put every test artifact under pytest `tmp_path` and restore global config,
  module imports, and monkeypatches automatically.

---

### Task 1: Add the end-to-end training regression and prove RED

**Files:**

- Create: `tests/test_board_detector_training.py`
- Read only: `scripts/train/train_board_detector.py`
- Read only: `chessml/train/standard_training.py`
- Read only: `chessml/models/lightning/board_detector_model.py`

- [ ] **Step 1: Build a safe script-entrypoint fixture**

Temporarily replace `chessml.Script.__call__` with an identity decorator,
replace `chessml.script` with a fresh `Script`, remove
`scripts.train.train_board_detector` from `sys.modules`, import it, and remove
it again during fixture teardown. Do not invoke argparse or mutate the shared
script parser.

- [ ] **Step 2: Replace only expensive external seams**

Use four lines (`0` through `3`) in `tmp_path/unique_fens.txt` and point
`config.dataset.path` there. Keep the real `FileLinesDataset`.

Patch the imported renderer and augmentation classes with tiny
`ExtendedIterableDataset` subclasses:

- the renderer consumes the real FEN iterable and emits a one-pixel `Picture`
  whose value identifies the source line;
- the augmentation dataset preserves those identities and emits exactly four
  ordered corner pairs before the production transform runs;
- each fake calls the real iterable base constructor and records its nested FEN
  source for control assertions.

Provide an inert temporary `chessml.data.assets` module. Patch
`MobileViTV2FPN` with a tiny trainable eight-output backbone implementing
`preprocess_image()` and `forward()`; preprocessing must retain the encoded
source identity.

- [ ] **Step 3: Bound real Lightning without mocking the behavior under test**

Keep the real `standard_training`, `BoardDetector`, DataLoaders, optimizer,
backward pass, validation loop, `ModelCheckpoint`, and serialization.

Wrap the entrypoint's imported `standard_training` only to set
`max_epochs=1`, `save_top_k=1`, and `log_steps=1`. Patch the training module's
`Trainer` symbol with a factory around the real Lightning Trainer that forces:

```python
accelerator="cpu"
devices=1
max_epochs=1
limit_train_batches=1
limit_val_batches=1
num_sanity_val_steps=0
enable_progress_bar=False
enable_model_summary=False
default_root_dir=tmp_path
```

Return `False` from the patched `TensorBoardLogger`, point
`config.checkpoints.path` under `tmp_path`, and attach a small real Callback to
record train/validation batch source identities and target shapes. Do not use
`fast_dev_run`, because it disables checkpointing.

- [ ] **Step 4: Invoke the actual dynamic entrypoint and assert outcomes**

Call `train()` with a Namespace selecting the tiny backbone, dynamic dataset,
batch size two, one validation batch, validation interval one, a fixed seed,
and no checkpoint.

Assert:

- the two nested real FEN sources resolve to `(offset=0, limit=2)` and
  `(offset=2, limit=-1)`;
- both sources use the fixed seed and shuffle buffer `20`;
- observed validation identities are `{0, 1}` and training identities are
  `{2, 3}` (order is intentionally unspecified);
- both target batches have shape `(2, 8)` and dtype `torch.float32`;
- the training loader's dataset is an `IterableDataset`;
- `trainer.global_step == 1` and the tiny parameter changed;
- the real `ModelCheckpoint` monitors `val_loss`, has a finite best score, and
  its best path is an existing file below `tmp_path`.

- [ ] **Step 5: Prove current production is RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_training.py
```

Expected before production edits: failure from the real DataLoader rejecting
`shuffle=True` for the dynamic iterable. Record the exact failure. Do not
weaken the test to advance past it; the already-recorded focused reproductions
cover the target-shape and metric failures hidden behind this first exception.

- [ ] **Step 6: Self-review the test boundary**

Confirm the test does not import raw assets, instantiate MobileViT, access a
production dataset/checkpoint, mock Trainer/ModelCheckpoint, rely on shuffled
order, or leave files outside `tmp_path`. Report RED evidence and the diff.

---

### Task 2: Repair the BoardDetector entrypoint and make the regression GREEN

**Files:**

- Modify: `scripts/train/train_board_detector.py`
- Test: `tests/test_board_detector_training.py`

- [ ] **Step 1: Move split and shuffle controls to the FEN source**

Give `use_dynamic_dataset()` an explicit `shuffle_buffer: int = 1` argument.
Construct its `FileLinesDataset` with:

```python
offset=offset
limit=-1 if limit is None else limit
shuffle_buffer=shuffle_buffer
shuffle_seed=shuffle_seed
```

Pass that source into `BoardsImagesFromFENs`. Do not pass offset, limit, or the
buffer to either downstream iterable.

- [ ] **Step 2: Enforce the eight-value target contract**

Change only the dynamic transform so its `torch.float32` corners tensor is
flattened to `(8,)`. Preserve corner order and the pregenerated path.

- [ ] **Step 3: Configure the dynamic iterable at the entrypoint**

When `dataset_type == "dynamic"`, bind
`shuffle_buffer=args.batch_size * 10` into the dynamic factory.

Pass `shuffle=args.dataset_type != "dynamic"` to `standard_training()`. This
must disable DataLoader shuffling for dynamic iterables while preserving the
pregenerated loader's existing shuffle behavior.

- [ ] **Step 4: Configure the exact BoardDetector checkpoint metric**

Pass `checkpoint_monitor="val_loss"` in the same `standard_training()` call.
Do not change the helper default or the model's logged metric.

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_training.py
uv run --locked python -m pytest -q tests/test_standard_training.py tests/test_board_detector_model.py tests/test_board_detector_training.py
uv run --locked python scripts/train/train_board_detector.py --help
```

Expected: all tests and help smoke pass; the integrated test records one real
optimizer step, validation event, finite monitored score, and temporary
checkpoint.

- [ ] **Step 6: Self-review scope**

Inspect the production diff for one-owner split controls, `None` normalization,
dynamic-only loader shuffle disablement, exact target shape, and explicit
metric. Confirm `standard_training.py`, datasets, checkpoints, and the index
remain unchanged.

---

### Task 3: Independently verify and close P1-02/P1-03

**Controller-only file:**

- Modify after task reviews: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] Have a separate reviewer inspect Task 1's oracle independence, real
  Lightning coverage, temporary-output containment, and sensitivity to all
  four production corrections. Resolve every blocking finding.
- [ ] Have a separate reviewer inspect Task 2 for root-cause placement,
  compatibility, pregenerated behavior, and absence of double splitting.
  Resolve every blocking finding.
- [ ] Run a fresh verification package:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_training.py
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
uv run --locked python scripts/train/train_board_detector.py --help
git diff --check
git diff --cached --stat
git status --short
```

- [ ] Mark P1-02 and P1-03 resolved in `BOARD_RECOGNITION_AUDIT.tmp.md`, add
  the actual implementation and verification evidence, and update the required
  order so P2-08/P2-09 are next. Do not claim model quality or production
  retraining.
- [ ] Give the complete production, test, design, plan, and audit diff to a
  final independent reviewer. Resolve all blocking findings, then rerun the
  focused test, whitespace, index, and status checks.
- [ ] Confirm HEAD remains `320fc5d`, the index is empty, no real dataset or
  checkpoint inventory changed, and no file was committed or staged.
