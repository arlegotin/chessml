# MetaPredictor Board Representation Compatibility Design

## Goal

Restore MetaPredictor in board recognition without retraining or modifying
`checkpoints/mp-MetaPredictor-v1.ckpt`, while preserving the branch's new
12-class piece classifier and binary square classifier.

## Root cause

`PIECE_CLASSES` intentionally changed from 13 image labels (empty plus twelve
pieces) to twelve piece labels. `OnlyPieces` accidentally continued deriving
its board-plane count from that classifier count. It maps an empty square to
`-1` and then indexes `np.eye(12)`, so NumPy encodes every empty square as the
last plane, which is also the white king.

This accidental coupling changes `OnlyPieces` from its documented
`(13, 8, 8)` contract to `(12, 8, 8)` and `FullPosition` from `(19, 8, 8)` to
`(18, 8, 8)`. The published MetaPredictor checkpoint stores
`input_shape=(13, 8, 8)` and cannot load into the recursively narrower
12-channel architecture.

## Stable contracts

Classifier labels and board tensor planes are separate schemas:

- `PIECE_CLASSES` remains twelve-way, with piece IDs `0..11`. The current
  PieceClassifier checkpoint, constrained decoder, datasets, and
  `INVERTED_PIECE_CLASSES` remain unchanged.
- Square occupancy remains the SquareClassifier's binary responsibility.
- `OnlyPieces` emits thirteen one-hot float32 planes: empty is plane `0`, and a
  piece is plane `PIECE_CLASSES[piece.symbol()] + 1`.
- Every board square activates exactly one plane.
- `FullPosition` emits nineteen float32 planes: the thirteen piece-state
  planes followed by en passant, four castling planes, and side to move.

This board mapping is byte-for-byte equivalent to the `main` contract across
the audited real positions and is the format expected by the published
MetaPredictor checkpoint.

## Recognition data flow

`BoardRecognitionHelper` continues to use the new image pipeline:

1. BoardDetector extracts the board.
2. SquareClassifier identifies occupied squares.
3. PieceClassifier assigns one of twelve piece IDs only to occupied squares.
4. The helper maps IDs back to piece symbols and creates a placement-only
   `chess.Board`.
5. `OnlyPieces` converts that board to the restored thirteen-plane tensor.
6. MetaPredictor returns castling rights, side to move, and source viewpoint.
7. If the source is flipped, the placement is rotated exactly once into
   canonical FEN coordinates before applying canonical turn and castling
   metadata. `RecognitionResult.flipped` remains the source-view flag.

MetaPredictor is a required `BoardRecognitionHelper` dependency. Omitting it
must not silently produce the current hard-coded metadata.

The validator loads the existing MetaPredictor checkpoint strictly, switches
it to evaluation mode, and passes it into the helper. Rendered comparison
boards use `result.flipped`, so canonical FEN output does not change the visual
orientation of validation artifacts.

## Training compatibility

Restoring `OnlyPieces` makes new MetaPredictor training use the coherent
thirteen-channel input again. The current shared training helper also requires
two Meta-specific arguments:

- `shuffle=False`, because `BoardsFromFEN` is an `IterableDataset` and PyTorch
  rejects `shuffle=True` for iterable datasets.
- `checkpoint_monitor="val_loss"`, matching the metric currently logged by
  MetaPredictor.

These arguments belong in `scripts/train/train_meta_predictor.py`; shared
training defaults and other model pipelines remain unchanged.

## Failure behavior

- Checkpoint loading remains strict. Shape mismatches are errors; no
  `strict=False` fallback or weight mutation is allowed.
- A missing checkpoint remains an explicit validation setup error.
- Python-chess continues to normalize structurally impossible castling rights
  when serializing FEN. This change does not claim that recognition always
  produces a legal position.
- MetaPredictor remains responsible only for metadata; image-classification
  failures are not hidden or reinterpreted by the representation layer.

## Verification

Automated regression coverage will establish:

- `OnlyPieces(Board())` is `(13, 8, 8)`, float32, one-hot per square, with 32
  starting-position empties and exactly one white king.
- `FullPosition(Board())` is `(19, 8, 8)` with metadata planes at stable
  indices `13..18`.
- An asymmetric flipped recognition result is canonicalized exactly once and
  retains `result.flipped=True`.
- An unflipped recognition result is not spatially changed.

Manual integration verification will:

- Strictly load `checkpoints/mp-MetaPredictor-v1.ckpt` and run a thirteen-plane
  prediction.
- Run the focused tests in the `chessml` conda environment.
- Run `scripts/validate/validate_board_recognition.py -d cpu` across the eight
  `book_short` images with MetaPredictor enabled.
- Review generated FENs, orientation flags, the final diff, and repository
  status.

The existing repository test that expects version `0.1.0` while the package is
`0.1.1` is an unrelated pre-existing failure and will not be changed as part
of this repair.

## Compatibility and non-goals

No tracked consumer or inspected local artifact expects 12-channel
`OnlyPieces` or 18-channel `FullPosition`. Existing board-model references
predate the regression, while the active 12-output PieceClassifier and binary
SquareClassifier do not consume board tensors.

Unpublished models trained against the broken schema would require an explicit
legacy adapter; preserving the broken empty-as-white-king encoding as the
default is not acceptable. No such artifact is in the approved scope.

This repair does not:

- retrain or patch MetaPredictor weights;
- add empty back to `PIECE_CLASSES`;
- alter piece or square classifier checkpoints, labels, or constraints;
- redesign MetaPredictor losses or thresholds;
- change shared training defaults;
- introduce speculative representation versioning.
