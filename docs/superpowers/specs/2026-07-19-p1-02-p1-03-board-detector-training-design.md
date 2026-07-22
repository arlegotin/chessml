# P1-02 and P1-03 BoardDetector Training Design

## Decision

Repair the BoardDetector entrypoint at the three boundaries where its dynamic
training contract is currently lost:

1. Configure BoardDetector checkpoint selection explicitly with the metric it
   logs, `val_loss`.
2. Apply validation/training offsets, limits, and buffered shuffling to the
   cheap FEN source before rendering or augmentation, and disable DataLoader
   shuffling only for the dynamic iterable path.
3. Flatten the dynamic corner label to the model's eight-value target shape at
   dataset construction.

Prove the combined behavior with one real-Lightning entrypoint smoke that runs
one optimizer step, validates on a disjoint finite source split, and writes a
real top-k checkpoint under a temporary directory. No production model,
dataset, checkpoint, or dependency is added or regenerated.

## Root Cause

`BoardDetector.validation_step()` logs `val_loss`, but its entrypoint lets
`standard_training()` use the unrelated default `val/loss`. Lightning performs
an exact callback-key lookup at validation, so the first checkpoint event
raises instead of ranking or saving a model.

The dynamic factory accepts `offset` and `limit` but consumes neither. Both
validation and training therefore start from the same FEN, validation is
unbounded, and the factory provides no iterable-native shuffle. The shared
trainer then passes `shuffle=True` to a DataLoader whose dataset is an
`IterableDataset`, which current PyTorch rejects before training.

After that loader failure is removed, the dynamic transform still collates
four coordinate pairs as `B x 4 x 2`, while BoardDetector emits `B x 8`.
Huber loss cannot compare those tensors.

## Metric Contract

The BoardDetector entrypoint will pass `checkpoint_monitor="val_loss"`
explicitly. This keeps the model's existing metric and dashboard name and
does not alter the shared helper's default or other trainers. The analogous
EloPredictor mismatch is a separate finding, not implicit scope for P1-02.

The regression must use the real installed Lightning `Trainer` and
`ModelCheckpoint`, reach post-training validation, observe a finite
`val_loss`, and create a real checkpoint in a pytest temporary directory.

## Dynamic Data Contract

`use_dynamic_dataset()` will accept an explicit shuffle-buffer size and build
its `FileLinesDataset` with:

- `offset` supplied by `standard_training()`;
- `limit=-1` when the helper supplies its unbounded `None` training limit;
- the configured shuffle buffer; and
- the entrypoint seed.

The FEN stream is the correct owner because one FEN produces one rendered
board and one augmented board. Splitting there avoids rendering and augmenting
the skipped validation prefix for every training iterator. Applying the same
controls again to `BoardsImagesFromFENs` or `AugmentedBoardsImages` would split
twice and is forbidden.

The dynamic entrypoint will use a buffer of `batch_size * 10`, matching the
repository's existing iterable-training convention. Validation receives a
finite prefix and training receives the remaining suffix; each instance
buffers only its own selected FENs, so reordering cannot cross the split.
`standard_training(..., shuffle=False)` is required for this iterable path.
Pregenerated map-style training retains DataLoader shuffling.

The dynamic transform will convert the four ordered corner pairs to a
`torch.float32` tensor and flatten it to shape `(8,)`. Flattening belongs here,
not in `BoardDetector.calc_losses()`: the model should continue to reject a
malformed target rather than silently reinterpret it.

## Regression Boundary

Add `tests/test_board_detector_training.py`. Import the decorated script with
`Script.__call__` temporarily neutralized. Replace only expensive external
seams: the image renderer/augmentation implementation, raw asset module, and
MobileViT backbone. Keep the entrypoint, real `FileLinesDataset`,
`ExtendedIterableDataset`, BoardDetector, standard training helper,
DataLoaders, optimizer/backward pass, validation loop, ModelCheckpoint, and
checkpoint serialization real.

Use four temporary FEN-like source IDs, a tiny eight-output trainable backbone,
and synthetic pictures. Bound the real Trainer to CPU, one epoch, one train
batch, one validation batch, no sanity validation, and no progress or model
summary. Disable TensorBoard output and point all remaining output at
`tmp_path`; do not use `fast_dev_run`, because it disables checkpointing.

The smoke must prove all of these observable outcomes:

- validation source controls are `offset=0`, `limit=2` and training controls
  are `offset=2`, `limit=-1` for batch size two and one validation batch;
- both source datasets own the requested seed and `batch_size * 10` buffer;
- observed validation and training source identities are finite and disjoint;
- both collated targets have shape `(2, 8)` and dtype `float32`;
- the training dataset is iterable but reaches one optimizer step without a
  DataLoader shuffle error;
- the real callback monitors `val_loss`, records a finite best score, and its
  best checkpoint exists under `tmp_path`.

The test must fail against the current tree before production edits. Focused
factory assertions may be added only if the integrated smoke cannot expose an
individual contract clearly; overlapping mock-only tests are unnecessary.

## Scope and Residual Risk

Do not change BoardDetector geometry or inference, the shared training
helper's default monitor, worker counts, `ExtendedIterableDataset`, model
architecture, CLI options, real assets, or generated datasets. Do not train a
production model, run checkpoint inference, stage, or commit.

BoardDetector currently uses zero DataLoader workers. Iterable worker sharding
would be required before increasing that count, but is not part of P1-03.
This repair proves that the supported training path executes correctly; it
makes no model-quality claim and does not replace the real-screenshot half of
the P2-01 benchmark.
