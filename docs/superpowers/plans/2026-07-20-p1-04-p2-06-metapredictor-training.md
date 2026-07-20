# P1-04 and P2-06 MetaPredictor Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MetaPredictor optimize and select checkpoints by the same valid BCE objective while honoring and recording the CLI FEN source.

**Architecture:** Preserve MetaPredictor's probability-valued inference and state-dict contract. Repair the existing model and training entry point directly, with one focused regression file covering the model loss/logging seam and the entry-point dataset/provenance seam.

**Tech Stack:** Python 3.14, PyTorch, Lightning, pytest.

## Global Constraints

- Preserve the final sigmoid, output ordering, prediction threshold, and existing state-dict keys.
- Mean BCE is the single optimization, `train_loss`, `val_loss`, and checkpoint-selection objective.
- Resolve `--path_to_fens` once; use and record that exact absolute path.
- Do not change `standard_training`, train or load a production model, touch datasets/checkpoints, add dependencies, stage, commit, or make a model-quality claim.
- Preserve the architecture-research file and all audit content outside the
  targeted P1-04, P2-06, and required-order summary sections.

---

### Task 1: Repair the MetaPredictor training contract

**Files:**
- Create: `tests/test_meta_predictor_training.py`
- Modify: `chessml/models/lightning/meta_predictor_model.py`
- Modify: `scripts/train/train_meta_predictor.py`

**Interfaces:**
- Consumes: `MetaPredictor(input_shape: tuple)` and `train(args)`.
- Produces: `MetaPredictor(input_shape: tuple, path_to_fens: str | None = None)` with unchanged tensor outputs and state dict; the existing `train_loss` and `val_loss` keys both carry mean BCE.

- [ ] **Step 1: Write failing model-objective tests**

Create a focused test that replaces only `forward()` with fixed probabilities,
captures `log()`, and asserts `training_step()` returns
`F.binary_cross_entropy(outputs, targets)` while both `train_loss` and
`val_loss` equal that exact scalar. Add a state-dict compatibility assertion:

```python
baseline = MetaPredictor(input_shape=(13, 8, 8))
with_provenance = MetaPredictor(
    input_shape=(13, 8, 8), path_to_fens="/data/unique_fens.txt"
)
assert baseline.state_dict().keys() == with_provenance.state_dict().keys()
```

- [ ] **Step 2: Verify the model test is RED for the audited reason**

Run:

```bash
uv run --locked python -m pytest -q tests/test_meta_predictor_training.py
```

Expected: the logged values differ from BCE because current code passes
probabilities to `sigmoid_focal_loss`; the provenance constructor argument is
also not yet accepted.

- [ ] **Step 3: Make BCE the single model objective**

In `meta_predictor_model.py`:

- add optional `path_to_fens: str | None = None` to `__init__`, allowing the
  existing `save_hyperparameters()` call to store it;
- remove the `sigmoid_focal_loss` import;
- make the existing loss helper return only mean
  `F.binary_cross_entropy(outputs, targets)`; and
- return/log that same tensor from `training_step()` and log it from
  `validation_step()` under the existing keys.

Do not change the final `nn.Sigmoid`, `predict()`, layers, or optimizer.

- [ ] **Step 4: Verify the model test is GREEN**

Run the same focused pytest command. Expected: the model-objective and
state-dict tests pass.

- [ ] **Step 5: Write failing CLI-source tests**

Import `scripts.train.train_meta_predictor` with `Script.__call__` temporarily
returning the decorated function. Replace `BoardsFromFEN` and
`standard_training` with recorders, invoke `train()` with a custom relative
`path_to_fens`, and assert separately that:

```python
assert observed_dataset_paths == [requested.resolve(), requested.resolve()]
assert captured_model.hparams.path_to_fens == str(requested.resolve())
```

The recorder must call the supplied dataset factory once with `limit=...`
and once with `offset=...`, without starting Lightning training. It must also
assert the unchanged checkpoint seam directly:

```python
assert captured_training["checkpoint_monitor"] == "val_loss"
assert captured_training["checkpoint_mode"] == "min"
```

- [ ] **Step 6: Verify the CLI tests are RED for the audited reason**

Run the focused pytest command. Expected: dataset paths point to the configured
default rather than the custom argument, and no source hyperparameter exists.

- [ ] **Step 7: Honor and record the CLI source**

At the start of `train(args)`, compute:

```python
path_to_fens = Path(args.path_to_fens).resolve()
```

Use `path_to_fens` in `BoardsFromFEN`, and construct the model with:

```python
MetaPredictor(
    input_shape=board_representation.shape,
    path_to_fens=str(path_to_fens),
)
```

Remove the now-unused `config` import from the script.

- [ ] **Step 8: Verify focused behavior and self-review**

Run:

```bash
uv run --locked python -m pytest -q tests/test_meta_predictor_training.py
uv run --locked python scripts/train/train_meta_predictor.py --help
```

Expected: all focused tests pass and help exits zero. Review the diff for
unrelated changes and confirm no dataset/checkpoint file changed.

No commit or staging step is permitted.

### Task 2: Record verified resolution evidence

**Files:**
- Modify: `BOARD_RECOGNITION_AUDIT.tmp.md`

**Interfaces:**
- Consumes: clean task review plus fresh focused/full verification evidence.
- Produces: resolved headings and evidence for P1-04 and P2-06 without claiming retraining or model-quality improvement.

- [ ] **Step 1: Run final verification before editing the audit**

Run:

```bash
uv run --locked python -m pytest -q tests/test_meta_predictor_training.py
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
uv run --locked python scripts/train/train_meta_predictor.py --help
git diff --check
git diff --cached --quiet
```

Expected: every command exits zero; pytest reports no failures, and the index
remains empty.

Execution note: preflight review missed that this full-suite command conflicts
with the global no-production-checkpoint-load constraint when ignored local
integration assets exist. The command was run once; the existing one-frame CPU
integration smoke strictly loaded BoardDetector, SquareClassifier, and
PieceClassifier checkpoints. It did not load MetaPredictor or invoke the
benchmark evaluator. The three hashes still match the prior benchmark records
and their modification times remain from 2026-07-16. No checkpoint was trained,
written, selected, calibrated, or replaced. Do not rerun the full suite merely
to reconfirm this documentation-only correction.

- [ ] **Step 2: Update both findings and the required-order summary**

Mark P1-04 and P2-06 resolved on 2026-07-20. Record the BCE/objective
alignment, exact CLI-source routing/provenance behavior, checkpoint state-dict
compatibility, test results, and explicit absence of training, checkpoint, or
dataset changes.

- [ ] **Step 3: Review and re-run documentation checks**

Read the changed audit sections in context, run `git diff --check`, and confirm
the only untracked pre-existing files remain the audit and architecture
research plus the intentional design/plan/test additions.

No commit or staging step is permitted.
