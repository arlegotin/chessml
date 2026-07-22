# Board Recognition Output Contract Design

## Goal

Fix P1-06 and define the public failure semantics needed by P1-07 before
building an evaluator or retraining any model.

Board recognition will report only what a still image can establish: the
piece placement in source-image order, or an explicit per-image failure. It
will not present orientation or history-dependent FEN fields as observed
facts. A caller that knows every missing field may construct a full FEN
through a separate, explicit function.

## Root cause

`BoardRecognitionHelper` currently turns classified pieces into a
`chess.Board`, then asks `MetaPredictor` to guess castling rights, side to move,
and orientation from `OnlyPieces`. That tensor contains only current piece
occupancy and symbols. En-passant and both clocks are hard-coded.

This cannot be made exact:

- the same source pixels can represent opposite canonical orientations;
- identical placements can have different turn, castling, and en-passant
  state because those fields depend on history; and
- move clocks are not present in the image at all.

The existing result nevertheless exposes a full FEN and a definite
`flipped: bool`. No-board, invalid geometry, solver failure, and invalid
placement also lack a stable per-image result contract.

## Public result contract

The old mutable `RecognitionResult` is replaced by a discriminated union:

```python
class BoardOrientation(Enum):
    WHITE_AT_BOTTOM = auto()
    BLACK_AT_BOTTOM = auto()


class RecognitionFailureReason(Enum):
    NO_BOARD = auto()
    INVALID_GEOMETRY = auto()
    DECODING_FAILED = auto()
    INVALID_PLACEMENT = auto()


@dataclass(frozen=True)
class RecognitionSuccess:
    source_placement: str
    board_image: Picture


@dataclass(frozen=True)
class RecognitionFailure:
    reason: RecognitionFailureReason


RecognitionResult = RecognitionSuccess | RecognitionFailure
```

These types and the full-FEN builder stay in the existing
`board_recognition_helper` module; this change does not add an interface or
package layer around one implementation.

`BoardRecognitionHelper.recognize()` returns `RecognitionResult`. It no longer
returns or exposes a `chess.Board`, `get_fen()`, `flipped`, inferred metadata,
or confidence values.

`board_image` is the rectified board in the same orientation as the source
image. It is present only on success; a failure may occur before a usable
rectified image exists.

### Source placement coordinates

`source_placement` uses FEN piece-placement compression, but its coordinate
frame is deliberately not canonical FEN:

- its eight rows describe the rectified image from top to bottom;
- each row describes squares from left to right; and
- piece letters retain their normal chess color and piece meaning.

The string therefore records the observed 8x8 grid without claiming which
edge is White's home rank. Consumers must not pass it off as the first field
of a canonical FEN unless orientation is supplied separately.

## Explicit full-FEN construction

A separate `build_fen()` function accepts a `RecognitionSuccess` and these
required keyword-only arguments, with no defaults:

```text
build_fen(
    result: RecognitionSuccess,
    *,
    orientation: BoardOrientation,
    turn: chess.Color,
    castling: str,
    en_passant: str,
    halfmove_clock: int,
    fullmove_number: int,
) -> str
```

`BoardOrientation` has exactly `WHITE_AT_BOTTOM` and `BLACK_AT_BOTTOM`.
Unknown orientation is intentionally not accepted: when orientation is
unknown, the correct output remains `source_placement`, not a full FEN.

The builder keeps the source grid for `WHITE_AT_BOTTOM` and rotates the
expanded 8x8 grid by 180 degrees for `BLACK_AT_BOTTOM`, then serializes the
canonical placement. It validates all six FEN fields together. Castling must
be `-` or a canonical `KQkq` subset, en-passant must be `-` or a valid target
square, the halfmove clock must be non-negative, the fullmove number must be
positive, and the resulting `chess.Board` must be valid. Invalid context
raises `ValueError`; omitted context is rejected by the required function
signature.

The returned FEN contains exactly the validated caller-supplied context. The
builder does not invoke a model, invent defaults, silently normalize rights,
or claim that those fields came from the image.

## Recognition data flow

The core path is:

1. Detect and rectify the board in source orientation.
2. Return `INVALID_GEOMETRY` when the detector's four points are non-finite,
   non-distinct, self-intersecting, zero-area, or produce a zero-sized warp.
3. Classify occupancy and pieces in source row-major order.
4. Return `DECODING_FAILED` when constrained assignment is infeasible,
   invalid, times out, or cannot prove an optimal assignment.
5. Serialize the decoded source placement and validate that it can represent
   a chess position.
6. Return `INVALID_PLACEMENT` when neither orientation with either side to
   move yields a valid `python-chess` board under neutral internal history
   fields (`- - 0 1`). These neutral values are validation scaffolding only
   and are never returned.
7. Return `RecognitionSuccess` otherwise.

`NO_BOARD` is reserved for a real presence/confidence decision introduced by
the later P1-07 implementation. The current checkpoints have no such output,
so geometry, decoding, or placement invalidity must not be relabeled as
`NO_BOARD`. Until P1-07 is implemented, a sufficiently plausible
hallucination can still pass the structural checks; this contract makes that
limitation explicit rather than inventing a confidence threshold.

## Failure values versus exceptions

Typed failures represent expected outcomes for individual images. They let a
batch or video caller record a failed frame and continue with the next one.
Only exceptions raised at the narrow geometry, constrained-decoder, and
placement-validation boundaries are converted to their corresponding
failure reasons.

Missing or incompatible checkpoints, invalid configuration or caller input,
unsupported devices, model execution failures, and programming errors remain
exceptions. The helper must not catch broad exceptions around the whole
pipeline.

## MetaPredictor boundary

`MetaPredictor` is removed from `BoardRecognitionHelper`'s constructor and
core recognition path. It may remain as a separately invoked experimental
prior, but its output is not automatically applied to placement and is never
presented by recognition as authoritative FEN state.

This design supersedes the following parts of
`2026-07-16-metapredictor-board-representation-design.md`:

- recognition data-flow steps that invoke MetaPredictor and rotate placement;
- the mandatory MetaPredictor helper dependency;
- `RecognitionResult.flipped` and canonical full-FEN output;
- validator behavior based on inferred orientation; and
- verification that asserts guessed metadata or one-time canonicalization.

It preserves that design's independent representation contract: twelve piece
classifier labels, thirteen `OnlyPieces` planes, and nineteen `FullPosition`
planes. MetaPredictor training and checkpoint code may continue to consume
those tensors outside core recognition.

## Migration

This is an intentional clean break in the pre-1.0 API:

- remove the old result's `.board`, `.get_fen()`, `.flipped`, and public square
  iteration behavior;
- remove the required `meta_predictor` constructor argument;
- update tests and callers to branch on `RecognitionSuccess` and
  `RecognitionFailure`;
- update the README so board recognition demonstrates source placement and
  explicit failure handling, while MetaPredictor is labeled as an optional
  experimental prior; and
- update the existing validator only enough to record source placements or
  failure reasons without loading MetaPredictor or pretending to be an
  accuracy gate.

There is no compatibility shim, deprecated alias, optional metadata field, or
silent fallback. The implementation does not publish a release; any later
release containing this pre-1.0 breaking change requires a minor-version bump.

## Verification and acceptance

Focused automated tests must establish:

1. Success and failure are distinct frozen result types, and callers branch
   on the result type.
2. An asymmetric classified grid is returned in source top-to-bottom,
   left-to-right order and is never rotated by core recognition.
3. Two positions with identical placement but different historical state
   produce the same recognition result and expose none of that state.
4. `build_fen()` handles both orientations and preserves explicit turn,
   castling, en-passant, and clock values in the resulting valid FEN.
5. Missing builder arguments fail, and invalid or inconsistent context raises
   `ValueError` rather than being defaulted or normalized.
6. Degenerate detector geometry, each constrained-decoder failure status, and
   structurally invalid placement return their exact failure reasons without
   raising.
7. A failure on one image does not prevent a subsequent image from being
   recognized.
8. An unexpected classifier/model exception propagates unchanged.
9. Core helper construction and recognition neither require nor invoke
   MetaPredictor.

The local-checkpoint integration smoke test may prove that the detector,
square classifier, and piece classifier still run together and produce a
typed result. It does not prove recognition correctness. The unlabeled
167-frame sequence, structural FEN validity, and king-presence assertions are
not acceptance gates.

## Non-goals and next gate

This work does not:

- train or calibrate board-presence, geometry-confidence, or no-board
  rejection;
- promise that every blank, noise, or plausible hallucination is rejected
  before P1-07;
- define uncalibrated metadata confidence or provenance fields;
- create the P2-01 labeled real-image benchmark or redesign the validator into
  that evaluator;
- repair MetaPredictor loss, checkpoint selection, or training;
- repair classifier data, detector training, geometry normalization, or image
  channel handling; or
- retrain or select any checkpoint.

After this contract is implemented, the next required gate is P2-01: freeze a
versioned, grouped, labeled real-image benchmark that scores board presence,
exact source placement, orientation only where ground truth exists, inferred
metadata separately, and negative/degraded inputs. Retraining remains blocked
until that evaluator exists.
