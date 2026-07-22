# P2-08 and P2-09 PieceClassifier Metrics/Loss Design

## Decision

Repair both PieceClassifier training defects in the existing model file:

1. Replace validation's mean of per-batch MCC values with one stateful
   multiclass MCC accumulated across the full validation loop and logged once
   under the unchanged `val/mcc` checkpoint key.
2. Derive focal-loss `p_t` directly from the unweighted softmax probability,
   while retaining class-weighted cross-entropy as the final weighted loss
   term and preserving all existing reductions.

No trainer, entrypoint, dependency, data, architecture, or public inference
change is needed.

## Root Cause

`PieceClassifier.calc_losses()` computes sklearn MCC independently for every
batch. `validation_step()` logs each scalar as `val/mcc`, so Lightning 2.6.5
reduces them by a sample-weighted mean before `ModelCheckpoint` reads the key.
MCC is nonlinear and cannot be averaged this way. For two equal batches:

```text
targets:     [0, 0, 0, 1]  [0, 1, 1, 1]
predictions: [0, 0, 1, 0]  [1, 0, 1, 1]
batch MCC:       -1/3             -1/3
mean batch MCC:  -1/3
epoch MCC:        0
```

A real one-epoch current-code reproduction confirms that the callback stores
`-1/3` even though the combined epoch confusion matrix has MCC `0`.

`WeightedFocalLoss` currently computes weighted cross-entropy first and then
uses `exp(-weighted_ce)` as `p_t`. For a target class weight `w`, this value is
`p_t ** w`, not `p_t`. Every repository PieceClassifier weight is below one,
so the current modulation systematically makes hard examples appear easier.

## Epoch MCC Contract

Follow the existing SquareClassifier pattern with the already-installed
`torchmetrics.classification.MulticlassMatthewsCorrCoef`:

- construct `self.val_mcc` with
  `num_classes=PIECE_CLASSES_NUMBER`;
- remove only validation's per-batch `val/mcc` log;
- update `self.val_mcc` from each validation batch's predictions and labels;
- in `on_validation_epoch_end()`, compute and log it once as `val/mcc`, then
  reset it for the next validation loop.

Keep the batch-level sklearn `train/mcc` diagnostic and the existing NumPy
prediction lists used by the confusion-matrix figure. Removing their duplicate
forward/CPU work would be unrelated cleanup. Do not add `sync_dist=True`:
TorchMetrics already sums its confusion-matrix state across ranks during
`compute()`.

The metric's state is nonpersistent in the locked TorchMetrics 1.9.0, so it
adds no checkpoint keys and strict loading of existing PieceClassifier model
states remains compatible. The logged key and `mode="max"` training-script
contract remain unchanged.

## Weighted Focal Contract

For logits `z`, hard target `y`, class weight `w_y`, and focal exponent
`gamma`, the per-sample loss is:

```text
p_t = softmax(z)[y]
weighted_ce = w_y * -log(p_t)
focal = (1 - p_t) ** gamma * weighted_ce
```

Keep the existing weighted `F.cross_entropy(..., reduction="none")` result,
but obtain `p_t` with `F.softmax(logits, dim=1)` and gather the hard-target
probability. The existing `none`, arithmetic `mean`, and `sum` branches remain
unchanged. Do not add label smoothing to focal loss; that is not part of the
audited defect or its current contract.

With the default `focal_weight=0`, optimized gradients remain unchanged, but
the logged focal diagnostic becomes correct. Runs that select a nonzero focal
weight intentionally receive the corrected objective.

## Regression Boundary

Extend `tests/test_piece_classifier_model.py` with two tests.

The focal unit regression uses unequal sub-one weights and independently
computes `w_y * (1 - p_t) ** gamma * -log(p_t)` from known logits. It asserts
the per-sample output so the current `p_t ** w_y` implementation fails for the
mathematical reason, without involving PieceClassifier's label-smoothed CE.

The MCC regression uses a tiny trainable identity-logit backbone, synthetic
TensorDatasets, the real Lightning Trainer, real validation loop, real
TensorBoard logger, and real `ModelCheckpoint(monitor="val/mcc", mode="max")`.
Only the unrelated OneCycle optimizer is replaced by zero-learning-rate SGD.
All logs and checkpoints stay below `tmp_path`.

Use the two-batch counterexample above and assert:

- Lightning's final `val/mcc` is exactly `0`, not `-1/3`;
- the callback best score is `0` and its real checkpoint exists;
- the stateful metric is reset after the validation loop; and
- it contributes no persistent model-state keys.

The tests must demonstrate RED against the current tree before production
changes, then GREEN after both minimal corrections.

## Compatibility and Scope

Historical `val/mcc` scores and checkpoint rankings used the invalid batch
mean and are not comparable to corrected epoch scores. Future corrected runs
must use a fresh checkpoint-selection run rather than treating an old callback
score as the same metric. Runs using nonzero focal weight should likewise
restart for clean objective comparability.

Do not retrain a production model, replace a checkpoint, run recognition
inference, regenerate datasets, change classifier sampling, modify the shared
trainer, stage, commit, or make a model-quality claim. Close P2-08/P2-09 only
after focused/full verification and an independent review of the production,
tests, and audit evidence.
