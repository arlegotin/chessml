# P2-08 and P2-09 PieceClassifier Metrics/Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PieceClassifier focal loss mathematically correct and select
checkpoints with MCC over the complete validation epoch rather than a mean of
batch MCCs.

**Architecture:** Keep the existing weighted cross-entropy term and read focal
`p_t` directly from softmax. Reuse the repository's established TorchMetrics
validation lifecycle to accumulate multiclass MCC and log it once per epoch
under the unchanged checkpoint key.

**Tech Stack:** Python 3.14, PyTorch 2.12, Lightning 2.6, TorchMetrics 1.9,
scikit-learn, pytest.

## Global Constraints

- Governing design:
  `docs/superpowers/specs/2026-07-19-p2-08-p2-09-piece-classifier-metrics-loss-design.md`.
- Governing findings: `BOARD_RECOGNITION_AUDIT.tmp.md`, P2-08 and P2-09.
- Follow root `AGENTS.md`; no nested instruction file exists.
- Work in the current shared `improvements-2` checkout at baseline `4939b69`;
  required audit/research inputs are untracked and the user prohibited commits.
- Modify only `chessml/models/lightning/piece_classifier_model.py`,
  `tests/test_piece_classifier_model.py`, the design/plan, task scratch reports,
  and finally `BOARD_RECOGNITION_AUDIT.tmp.md`.
- Preserve the public model/loss signatures, `val/mcc` key, checkpoint mode,
  training batch MCC diagnostic, focal reductions, label smoothing, model
  architecture, datasets, and shared trainer.
- Add no dependency or test-only production seam.
- Put test logs and checkpoints only below pytest `tmp_path`.
- Do not regenerate data, retrain a production model, replace a checkpoint,
  run recognition inference, stage, commit, merge, push, or publish.

---

### Task 1: Correct weighted focal probability modulation

**Files:**

- Modify: `tests/test_piece_classifier_model.py`
- Modify: `chessml/models/lightning/piece_classifier_model.py:131-151`

**Interfaces:**

- Consumes: `WeightedFocalLoss(logits, targets)` with existing
  `weight`, `gamma`, and `reduction` constructor arguments.
- Produces: the same scalar/vector reduction contract, using true hard-target
  probability for modulation and class weight in the final loss term.

- [ ] **Step 1: Add the formula regression**

Import `WeightedFocalLoss` in `tests/test_piece_classifier_model.py` and add:

```python
def test_weighted_focal_loss_modulates_with_true_class_probability():
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    logits = torch.tensor(
        [[0.0, 2.0], [1.0, -1.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([1, 0])
    probabilities = torch.softmax(logits, dim=1)
    target_probabilities = probabilities[
        torch.arange(len(targets)), targets
    ]
    expected = (
        weights[targets]
        * (1 - target_probabilities).pow(2)
        * -target_probabilities.log()
    )

    actual = WeightedFocalLoss(
        weight=weights,
        gamma=2,
        reduction="none",
    )(logits, targets)

    torch.testing.assert_close(actual, expected)
```

The oracle expresses the focal definition directly. Do not calculate expected
`p_t` from the criterion or from weighted cross-entropy.

- [ ] **Step 2: Prove RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py::test_weighted_focal_loss_modulates_with_true_class_probability
```

Expected before production edits: one assertion failure because current
`exp(-weighted_ce)` equals `p_t ** class_weight`, not `p_t`. Record actual and
expected values.

- [ ] **Step 3: Make the smallest focal correction**

Keep the existing weighted, unreduced cross-entropy. Replace only the current
`pt = torch.exp(-ce)` line with:

```python
pt = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
```

Keep the existing focal multiplication and all three reduction branches
unchanged.

- [ ] **Step 4: Prove GREEN and compatibility**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py::test_weighted_focal_loss_modulates_with_true_class_probability
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py
```

Expected: the new regression and the complete PieceClassifier test file pass.

- [ ] **Step 5: Self-review and report**

Confirm the production change is the single `p_t` expression, reductions and
signatures are unchanged, and no artifact exists outside pytest temp. Write
`.superpowers/sdd/p2-08-p2-09-task-1-report.md` with RED/GREEN commands and
outputs, files changed, and scope review. Do not stage or commit.

---

### Task 2: Select checkpoints with complete-epoch multiclass MCC

**Files:**

- Modify: `tests/test_piece_classifier_model.py`
- Modify: `chessml/models/lightning/piece_classifier_model.py:153-280`

**Interfaces:**

- Consumes: validation predictions/labels and the existing `val/mcc` monitor
  configured by `scripts/train/train_piece_classifier.py`.
- Produces: exactly one `val/mcc` value per validation loop, computed from the
  aggregate multiclass confusion matrix and reset before the next loop.

- [ ] **Step 1: Add a module-level synthetic backbone**

Add these imports to `tests/test_piece_classifier_model.py`:

```python
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
```

Add a module-level class so Lightning can serialize the model hyperparameters:

```python
class LogitBackbone(torch.nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(output_features))

    def forward(self, logits):
        return logits + self.bias
```

- [ ] **Step 2: Add the real checkpoint regression**

Add a test using this exact equal-batch counterexample:

```python
def test_piece_classifier_checkpoints_complete_epoch_mcc(tmp_path):
    labels = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1])
    predictions = torch.tensor([0, 0, 1, 0, 1, 0, 1, 1])
    logits = torch.full((8, 12), -10.0)
    logits[torch.arange(8), predictions] = 10.0

    model = PieceClassifier(base_model_class=LogitBackbone)
    model.configure_optimizers = lambda: torch.optim.SGD(
        model.parameters(), lr=0.0
    )
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        monitor="val/mcc",
        mode="max",
        save_top_k=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=TensorBoardLogger(tmp_path / "logs"),
        callbacks=[checkpoint],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            TensorDataset(logits[:4], labels[:4]), batch_size=4
        ),
        val_dataloaders=DataLoader(
            TensorDataset(logits, labels), batch_size=4
        ),
    )

    assert trainer.callback_metrics["val/mcc"].item() == pytest.approx(0.0)
    assert checkpoint.best_model_score.item() == pytest.approx(0.0)
    best_path = Path(checkpoint.best_model_path)
    assert best_path.is_file()
    assert best_path.resolve().is_relative_to(tmp_path.resolve())
    assert model.val_mcc.confmat.sum().item() == 0
    assert not any(key.startswith("val_mcc.") for key in model.state_dict())
```

Each batch's sklearn MCC is `-1/3`; the concatenated epoch MCC is `0`.

- [ ] **Step 3: Prove RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py::test_piece_classifier_checkpoints_complete_epoch_mcc
```

Expected before the MCC production correction: failure showing callback metric
and best score approximately `-0.33333334` instead of `0`. Confirm the real
checkpoint file was created during the reproduction; do not weaken or mock the
callback.

- [ ] **Step 4: Add the stateful multiclass metric**

Import and initialize the existing dependency:

```python
from torchmetrics.classification import MulticlassMatthewsCorrCoef

self.val_mcc = MulticlassMatthewsCorrCoef(
    num_classes=PIECE_CLASSES_NUMBER
)
```

Keep `calc_losses()` and `train/mcc` unchanged.

- [ ] **Step 5: Update validation and epoch-end lifecycle**

Delete only the per-batch validation line:

```python
self.log("val/mcc", mcc, prog_bar=True)
```

After the existing validation predictions are produced, add:

```python
self.val_mcc.update(preds, labels)
```

At the start of `on_validation_epoch_end()`, before confusion-matrix rendering,
add:

```python
self.log(
    "val/mcc",
    self.val_mcc.compute(),
    prog_bar=True,
    on_step=False,
    on_epoch=True,
)
self.val_mcc.reset()
```

Do not add `sync_dist=True` or change the existing prediction lists.

- [ ] **Step 6: Prove GREEN and run the combined boundary**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py::test_piece_classifier_checkpoints_complete_epoch_mcc
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py tests/test_standard_training.py
uv run --locked python scripts/train/train_piece_classifier.py --help
```

Expected: the real callback records `0`, its checkpoint exists, metric state is
reset/nonpersistent, all focused tests pass, and the existing CLI help works.

- [ ] **Step 7: Self-review and report**

Confirm only the validation checkpoint metric changed; the training diagnostic,
monitor key/mode, model state keys, focal reductions, and other classifiers are
unchanged. Write `.superpowers/sdd/p2-08-p2-09-task-2-report.md` with exact
RED/GREEN evidence, files changed, checkpoint containment, and scope review.
Do not stage or commit.

---

### Task 3: Independently verify and close P2-08/P2-09

**Controller-only file:**

- Modify after task reviews: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] After each implementation task, give its brief, report, and complete
  unstaged diff/full new-file content to a separate reviewer. Require both
  specification compliance and code-quality approval; resolve every Critical
  or Important finding and re-review.
- [ ] Run a fresh verification package:

```bash
uv run --locked python -m pytest -q tests/test_piece_classifier_model.py
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
uv run --locked python scripts/train/train_piece_classifier.py --help
git diff --check
git diff --cached --stat
git status --short
git rev-parse --short HEAD
```

- [ ] Verify `git status --short -- datasets checkpoints` is empty and no file
  below those directories is newer than the design spec.
- [ ] Mark P2-08 and P2-09 resolved in `BOARD_RECOGNITION_AUDIT.tmp.md`. Record
  the old/new numerical semantics, real callback evidence, compatibility
  boundary, exact checks, no retraining, and that old MCC rankings/nonzero-focal
  runs are not comparable. Remove the stale “next P2-08/P2-09” order text and
  leave P1-04/P2-06 and P2-11 explicitly conditional on their retained paths.
- [ ] Give the complete model, test, design, plan, reports, and audit package to
  a final independent reviewer. Resolve every blocking finding.
- [ ] Rerun the focused test, tracked/untracked whitespace checks, index, status,
  HEAD, and dataset/checkpoint freshness checks after the audit/final review.
  Confirm HEAD remains `4939b69` and no file was staged or committed.
