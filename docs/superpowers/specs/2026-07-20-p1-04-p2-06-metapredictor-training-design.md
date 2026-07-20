# P1-04 and P2-06 MetaPredictor Training Design

## Decision

Keep MetaPredictor's existing probability-valued forward and prediction
contract, but use one mean binary cross-entropy (BCE) value for optimization,
training diagnostics, validation, and `val_loss` checkpoint selection.

Resolve `--path_to_fens` once at the training entry point, use that exact path
for both validation and training datasets, and store its absolute string in
the model hyperparameters written to new checkpoints.

## Root cause

MetaPredictor ends in `nn.Sigmoid`, so its outputs are probabilities. The
locked torchvision 0.27.1 `sigmoid_focal_loss` API expects logits and applies
sigmoid internally. Passing perfect probabilities `[0, 1]` therefore produces
loss `0.0678148046`; converting them to logits produces approximately
`5.13e-19`. Training already returns BCE to Lightning, while both diagnostic
and checkpoint keys log the malformed focal value.

The training CLI parses `--path_to_fens`, but its dataset factory ignores the
argument and always constructs `config.dataset.path / "unique_fens.txt"`.
Consequently a successful run can train on a source other than the one named
by the command, and the resulting checkpoint cannot identify its source.

## Compatibility boundary

- Keep the network, final sigmoid, output ordering, prediction threshold, and
  state-dict keys unchanged.
- Add optional constructor hyperparameter `path_to_fens: str | None = None`.
  Old checkpoints omit it and continue to construct with the default; new
  training checkpoints record the resolved source path.
- Delete the unused weighted-BCE and malformed focal calculations rather than
  preserving diagnostics that do not drive the optimizer.
- Keep the existing `train_loss`, `val_loss`, checkpoint monitor, checkpoint
  mode, data split, and dataset transform behavior.
- Do not change the shared trainer.

Changing MetaPredictor to emit logits would require coordinated inference and
checkpoint-behavior changes. Converting probabilities back to logits only for
the diagnostic would retain the mismatch between the optimized and selected
objectives. Retiring MetaPredictor would contradict its documented standalone
training path. Those alternatives are out of scope.

## Regression boundary

Focused tests must demonstrate RED against the current tree and GREEN after
the repair:

1. `training_step()` returns mean BCE, and both `train_loss` and `val_loss`
   log that same value for fixed probability outputs.
2. A custom relative `--path_to_fens` is resolved and supplied to every
   `BoardsFromFEN` construction instead of the configured default.
3. The model passed to training records the same absolute path under
   `hparams.path_to_fens`.
4. Adding the optional provenance hyperparameter does not change model
   state-dict keys.

## Scope and acceptance

Do not train a model, load or replace a production checkpoint, regenerate a
dataset, add a dependency, change core recognition, stage, or commit. After
focused and full-suite verification, update `BOARD_RECOGNITION_AUDIT.tmp.md`
to mark P1-04 and P2-06 resolved with the exact evidence and limitations.

Execution deviation: the full-suite requirement conflicted with the no-load
constraint because all ignored local integration assets were present. The
controller ran the suite once, causing the existing one-frame CPU smoke to
strictly load the BoardDetector, SquareClassifier, and PieceClassifier
checkpoints. It did not load MetaPredictor or run the benchmark evaluator.
Those three hashes still match the previously recorded benchmark hashes and
their modification times remain from 2026-07-16; no checkpoint was written,
selected, calibrated, or replaced. This deviation is retained explicitly
rather than rewriting the original constraint after execution.
