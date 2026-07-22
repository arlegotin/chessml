# P1-01 and P2-07 Classifier Data/Reproducibility Design

## Decision

Repair P1-01 and P2-07 as one data-contract change:

1. `PiecesImages3x3` will build one 12-label image map per piece set and use a
   local seeded RNG to choose the center label, one piece set, one board theme,
   and every neighbor without structural coupling.
2. The classifier generator will partition piece sets and board themes into
   deterministic, disjoint training and validation groups and write separate
   CSVs with direct source provenance.
3. The shared CLI runner will apply an existing Lightning seed before invoking
   any command that declares `seed` or `shuffle_seed`, so dataset construction,
   model initialization, Torch loader permutations, dropout, and Lightning
   workers start from controlled RNG state.

No dataset, model, manifest framework, custom RNG library, or dependency is
added. Existing tainted classifier artifacts are left untouched and are not a
fallback training source.

## Root Cause

`PiecesImages3x3.generator()` currently uses the same counter to choose the
center and theme and derives all eight neighbors from a multiplication of that
counter and the square index. With the current 58 valid piece-set directories
and 54 themes, this produces all of the following deterministic shortcuts:

- square-classifier occupied and empty centers use disjoint theme parities;
- occupied centers always have four occupied neighbors while empty centers
  always have eight;
- piece class modulo six equals theme index modulo six;
- each piece label has one neighbor signature; and
- almost every 3x3 mixes pieces from different piece sets.

Perspective warp and cropping retain enough context for these shortcuts to
reach the classifier input. Shuffling the CSV afterward changes row order only.
`standard_training()` then uses one prefix for validation and the suffix for
training, so both sides share the same synthetic source shortcuts.

P2-07 has the same upstream defect: affected scripts all pass through
`chessml.Script.run()`, but that boundary currently invokes the command without
applying its parsed seed. The piece generator uses its seed only after pixels
exist; Square/Piece/Meta training never reads its seed; and the BoardDetector
seed does not affect model initialization or its default pregenerated path.

The current raw asset inventory also accepts the macOS `.DS_Store` file as a
piece-set path. Filtering the inventory to real directories is a prerequisite
for exercising the corrected generator; deleting the local file would only
hide the boundary defect.

## Sampling Contract

For every `PiecesImages3x3` iteration:

- Create one `label -> Picture` mapping for each supplied piece-set directory.
- Reset a local `random.Random` from the dataset's resolved `shuffle_seed`, so
  iterating the same dataset again yields the same sequence.
- Keep center targets balanced by shuffling complete cycles:
  - piece mode: one of each of the 12 piece labels;
  - square mode: the 12 piece labels plus 12 empty targets, giving exact 50/50
    occupied/empty balance within each complete cycle.
- For each center target, independently choose one piece set and one board
  theme. Every occupied square in that 3x3 uses the chosen piece set.
- Choose every neighbor independently: empty with `EMPTY_SQUARE_CHANCE`,
  otherwise one of the 12 labels uniformly from the chosen piece set.
- Emit both center-square parities for the sampled context, preserving exact
  light/dark balance within each complete pair without coupling label to theme.
- Preserve the existing first two yielded fields, `Picture` and piece label,
  and append `piece_set`, `dark_color`, and `light_color` provenance. The
  augmentation dataset passes additional fields through unchanged.

Seed `0` is valid. `LoopedList` must therefore distinguish `None` from zero.
An arbitrary dataset `limit` may truncate the final target cycle or parity
pair; that creates at most one unpaired parity and one partial balanced cycle,
not a repeated label/source rule.

## Group-Held-Out Artifacts

The generator retains `--limit` as a total row count and uses a fixed one-fifth
validation allocation. It requires at least two rows, two valid piece sets, and
two themes.

Starting from sorted sources, one local RNG seeded from the CLI seed shuffles
piece sets and themes, then partitions each into non-empty 80/20 complements.
Training renders with the requested seed and validation with
`(seed + 1) % 2**32`; the streams are distinct across the supported seed range,
and the splits cannot share a piece set or theme.

The output layout is:

```text
<square_classifier|piece_classifier>/
  train.csv
  validation.csv
  images/
    train/
    validation/
```

Both CSVs use this exact schema:

```text
image_path,piece_name,piece_set,dark_color,light_color
```

The provenance columns are the split evidence; a separate JSON manifest would
duplicate them and is not needed. Classifier datasets consume the first two
columns and ignore the provenance tail. `SquareClassifierDataset.__getitem__`
and `PieceClassifierDataset.__getitem__` must therefore unpack
`path, target, *_` rather than requiring exactly two fields.

`standard_training()` gains one backward-compatible optional validation
factory. When supplied, it constructs validation from that factory and starts
the training factory at offset zero. Existing one-factory callers retain the
current prefix/offset behavior. Square and Piece training require `train.csv`
and `validation.csv`; they do not fall back to the known-leaky `meta.csv`.

## Reproducibility Contract

Immediately before a parsed command callback runs, `Script.run()` checks for a
`seed` argument, then `shuffle_seed`, and calls the installed
`lightning.seed_everything(value, workers=True)` when present. This is early
enough to cover model and dataset construction. It seeds Python, NumPy, and
Torch and asks Lightning to seed DataLoader workers.

The piece generator additionally passes the same explicit seed into
`PiecesImages3x3` and `AugmentedPiecesImages`; local sample selection must not
depend on unrelated global RNG consumption. No explicit DataLoader generator
or custom worker initializer is added: the seeded Torch global state and
Lightning worker hook already cover identical-command repeatability. Add an
isolated loader generator only if future evidence requires independence from
unrelated Torch draws.

The guarantee is identical stochastic streams for the same command, seed,
inputs, locked dependency versions, and device. It is not a promise of bitwise
identity across dependency versions, hardware backends, or nondeterministic
kernels. Supported CLI seeds are integers from `0` through `2**32 - 1`, matching
Lightning/NumPy; values outside that range fail before the command callback.
Fresh same-seed augmented dataset constructions repeat. Re-iterating one
already-consumed `AugmentedPiecesImages` instance is not a public replay
contract because its existing augmentator owns advancing global RNG streams.

## Regression Boundary

Tests use small synthetic piece assets and temporary output paths; they do not
regenerate the real classifier datasets.

- Prove complete center cycles are balanced, labels span themes, neighbor
  occupancy and labels vary, and every 3x3 uses exactly one piece set. Prove an
  arbitrary truncated limit still emits exactly the requested number of rows.
- Prove raw sequences replay from the same dataset, and fresh augmented dataset
  constructions repeat `(pixels, label, provenance)` for the same seed and
  change for another seed, including seed zero. Prove out-of-range CLI seeds
  fail before callback execution.
- Prove generated training/validation CSVs are deterministic, exhaustive for
  the requested limit, and have disjoint piece-set and theme provenance.
- Prove separate validation factories bypass the legacy offset while the
  existing one-factory `standard_training` behavior is unchanged.
- Prove `Script.run()` resets Python, NumPy, Torch/model/dropout, and shuffled
  loader streams before callbacks for both supported seed names and enables
  Lightning worker seeding.
- Prove non-directory entries in `assets/piece_png` cannot enter `PIECE_SETS`.

Before closure, run the focused tests, the full repository suite, compilation,
one small real-asset temporary canary, and independent code/data-oracle review.

## Alternatives Rejected

- **Only randomize the old index and seed globals:** smaller, but validation
  still shares sources and the old flattened asset list can still mix piece
  sets inside one board context.
- **One CSV with a split column:** `CSVDataset` currently limits rows before a
  split filter and returns positional tuples, so this needs filtering machinery
  plus the same training-factory change. Separate CSVs are shorter and fail
  closed when legacy data is present.
- **Refactor every augmentation to private RNG objects:** stronger isolation,
  but far wider than the demonstrated identical-command requirement. The
  central Lightning seed plus the classifier sampler's local RNG is sufficient.
- **Delete `.DS_Store`:** removes one local symptom and leaves the inventory
  boundary incorrect on the next macOS metadata file.

## Scope

Do not retrain models, regenerate the real classifier datasets, change model
losses/metrics, repair BoardDetector training defects, rerun the recognition
benchmark, stage, or commit. P2-08/P2-09 and the BoardDetector training group
remain subsequent work. Existing `meta.csv` files remain as historical tainted
artifacts until the maintainer explicitly regenerates datasets with the fixed
command.
