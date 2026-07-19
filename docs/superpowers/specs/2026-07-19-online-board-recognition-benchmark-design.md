# Online Board Recognition Benchmark Design

## Goal

Create the first trustworthy accuracy evaluator for the current board-recognition
contract, together with a deterministic benchmark for rendered online chess
boards.

Benchmark generation is gated on proving the generator and scoring oracle first.
The existing bulk dataset scripts are not part of this design and must not be run.

## Claim boundary

Version 1 is a **synthetic online-render regression benchmark**. It can establish
that a checkpoint recognizes known rendered-board cases, that exact placement and
no-board regressions fail visibly, and that future code changes are compared on a
fixed corpus.

It is not evidence of accuracy on unseen chess sites. The current checkpoints do
not record their training assets, themes, positions, or augmentation manifest, so
holdout from those checkpoints cannot be proved. P2-01 is therefore only fully
closed when a separately captured and manually labeled online-screenshot suite is
added. Future training manifests must exclude the generated images, every base
placement and its 180-degree rotation, all selected piece assets/palettes/layouts,
the negative templates, and every derived variant. Any overlap must be reported
explicitly rather than silently weakening the holdout claim.

This limitation must appear in the generated manifest and evaluator report. Case
count must not be presented as the number of independent positions: render
variants share a smaller set of base positions.

## Why a new generator is required

The existing board generator is unsuitable for benchmark construction:

- `PIECE_SETS` discovers arbitrary directory entries and currently includes
  `.DS_Store`, which makes `BoardsImagesFromFENs` fail during piece-set loading.
- `BoardsImagesFromFENs` couples FEN, piece set, palette, and view by enumeration
  index without returning the style identities needed for grouped scoring.
- `AugmentedBoardsImages` and `Augmentator` use process-global Python and NumPy
  random state. Construction order and repeated iteration change pixels.
- augmentation comes from ignored `config.local.yaml`, so the recipe is not a
  versioned input.
- optional final geometric transforms can invalidate corners computed earlier;
  perspective cases can also leave the frame.
- the bulk script deletes its destination before validation, ignores image-write
  return values, and can leave partial image/TXT pairs with no manifest or hashes.
- its photographic backgrounds model a different domain from online-board
  screenshots.

The clean `fentoboardimage` renderer may be reused only behind the independent
oracle tests below. Its optional coordinate overlay is excluded: version 1.4.1
rotates pieces for a black view but still draws white-view coordinates.

## Public recognition contract being scored

The evaluator follows the implemented P1-06 contract:

- positive ground truth is `source_placement`, read literally from the image's
  top row to bottom row and left to right within each row;
- negative ground truth is `board_present = false`;
- orientation is a generation group, not a predicted field;
- side to move, castling, en-passant, clocks, and other historical metadata are
  neither labeled nor scored; and
- expected per-image failures remain distinct from evaluator or model-execution
  errors.

For a canonical placement `P`, a white-bottom render is labeled `P`. A
black-bottom render is labeled with the expanded 8 by 8 grid rotated 180 degrees
and recompressed. The label is never guessed from `flipped` after rendering.

## Frozen source specification

A versioned source specification defines every input to generation. It contains:

- benchmark and schema versions;
- an explicit catalog of 16 base positions with stable IDs and source
  provenance;
- the selection-time SHA-256 of `unique_fens.txt` if a catalog entry was selected
  from it;
- four explicit piece-set directories, their canonical tree hashes, upstream
  source, author/license metadata, and expected filenames;
- four named board palettes;
- the exact layouts, quality transforms, negative templates, channel modes,
  piece-resampling filter, and PNG encoder settings;
- the required versions of Python, `chess`, `fentoboardimage`, and Pillow, plus
  the runtime PNG/zlib fingerprint recorded when final artifacts are written;
  and
- the case-matrix formula and expected counts.

The local Unique-FEN source presently contains 22,641,843 lines and has SHA-256
`9e6bc34257368bc1518df56dfc4ae42411de932bf265edc17d8bc4ee740d32c7`.
The digest is selection provenance, not a recurring 1.1 GB generation
dependency. The generator does not require or stream that file during normal
preflight or rendering. Selected placements and their original four-field source
records are frozen into the source specification so a later reshuffle or
replacement cannot silently change the benchmark.

The reviewed version-1 source specification has canonical semantic SHA-256
`8393ff76e9f10b4bbe24b558f9b2d0032c493ebb6fe1781a4515b4790018756b`.
The loader rejects duplicate JSON keys and non-standard numeric constants before
computing that digest.

### Position requirements

Before any image is rendered, every base position must pass all of these checks:

1. The frozen four-field source record is normalized by appending exactly
   `0 1`; `python-chess` parses that six-field FEN and reports a valid standard
   chess position. The built-in starting position is already six fields.
2. Its placement expands to exactly 64 squares using only the twelve standard
   piece symbols.
3. It contains exactly one king of each color.
4. Expanded 64-square placements are pairwise unique directly and against every
   other placement's 180-degree rotation.
5. The catalog as a whole includes every piece symbol, asymmetric corner and
   edge cases, and low-, medium-, and high-occupancy positions.
6. Stable IDs, catalog order, and provenance are unique.

Four base positions, one from each position-index residue modulo four, form a
development split. The other twelve form the frozen acceptance split. All render
variants of one position stay in the same split. Later threshold exploration may
use development cases; acceptance cases must not become training or calibration
input. Current-checkpoint holdout still cannot be claimed because its historical
training manifest is missing.

The renderer's own parser is not a validator: it silently ignores malformed
characters and rank widths. Rendering is prohibited until the independent
`python-chess` checks pass.

### Piece assets

Version 1 uses only these already provisioned Lichess-derived sets. Upstream
license/source references are pinned to Lichess `lila` revision
`940e876ef193e8c031f42945f6f7773c71764395`:

| ID | Upstream license | Approved local tree SHA-256 |
| --- | --- | --- |
| `lichess_chessnut` | Apache-2.0 | `ab9628638c8266b8f7e9a3f36f9d518c5de59042bf0c0c5a09c6b8384654ef3b` |
| `lichess_celtic` | MIT | `5fc39e440f5aba5cff5cebce938c6fa32c1f6ac8ef5b3060861f2fea0a4ec397` |
| `lichess_fantasy` | MIT | `d8b894b8f0952cc6c0db654cc47681f818f4c7f7ae00814b54dbff5e1aab52cd` |
| `lichess_spatial` | MIT | `fc033d2aff36ff3f0edee801ac2aa01b1a1d0c2b20cb95dd2fd1065c48646d0d` |

The tree hash covers exactly the twelve allowed PNG paths, sorted by relative
POSIX path: for each, SHA-256 receives the path, a NUL byte, its exact file bytes,
and a trailing NUL byte. Non-PNG metadata such as `.DS_Store` is neither trusted
nor hashed. Each set must contain exactly
`{black,white}/{Pawn,Rook,Knight,Bishop,Queen,King}.png`, with no missing or
extra PNGs. Every image must decode as 100 by 100 RGBA, have a nonempty alpha
channel, and have a unique file digest.

These 48 files passed that automated inventory and a visual contact-sheet review
before this design was written. The source specification remains the enforceable
preflight boundary; arbitrary `PIECE_SETS` discovery is forbidden.

### Palettes and layouts

Four palettes are copied into the source specification with stable IDs rather
than imported from the eager asset module:

- brown: dark `#B58862`, light `#F0D9B5`;
- blue: dark `#4675AB`, light `#C3CDD7`;
- green: dark `#85A666`, light `#FFFFDC`; and
- gray: dark `#6C6360`, light `#CCCDCD`.

Each pair must pass a minimum RGB-distance check and remain distinct after
grayscale conversion.

The first version deliberately uses no perspective, crop, occlusion, arrows,
last-move highlight, coordinate text, or photographic background. It has two
fully visible, axis-aligned online-style presentations:

- `board_crop`: a 512 by 512 board-only RGB image;
- `desktop_ui`: a 960 by 540 RGB canvas with a 480 by 480 board at an explicit
  integer rectangle and deterministic panel/clock shapes outside the board.

Board boxes use half-open `[left, top, right, bottom]` coordinates. Pixel-center
corners are recorded separately in TL, TR, BR, BL order, using `right - 1` and
`bottom - 1`. Tests must prove that cropping the half-open box from a clean
composition reproduces the standalone board pixel for pixel.

Piece PNGs are loaded only from the twelve validated paths and resized with
Pillow's `LANCZOS` filter through a call-local loader; the dependency's
process-global piece cache is not used. Final PNGs use `optimize=false` and
compression level 9 with no metadata.

All non-board pixels come from reviewed declarative `ui_recipes` in the source
specification, not hard-coded implementation choices. Recipe coordinates use a
1000 by 1000 integer space. Half-open rectangle/ellipse boxes scale each x value
as `x * width // 1000` and each y value as `y * height // 1000`; the renderer
subtracts one from the scaled right/bottom endpoints before calling Pillow's
inclusive drawing API. Polygon points scale x as `x * (width - 1) // 1000` and y
as `y * (height - 1) // 1000`. Shapes are painted in listed order. The source
specification's canonical SHA-256 anchors every recipe color, coordinate, shape
kind, and order, and mutation tests prove the renderer consumes those recipes.

Two quality modes are allowed:

- `clean`; and
- `browser_scaled`, a fixed downscale/upscale operation whose dimensions and
  resampling filters are in the source specification.

The quality transform may alter pixels but not dimensions, geometry, visibility,
or labels. Final artifacts are lossless PNGs; image quality is encoded into the
pixels before PNG serialization.

### Case matrix

The main RGB suite contains:

```text
16 positions x 4 piece sets x 2 views x 2 layouts x 2 quality modes
= 512 positive cases
```

The palette index is `(position_index + piece_set_index) % 4`. Because position
count is divisible by four, automated checks must prove that every position,
piece set, view, layout, and quality group sees all four palettes with equal
aggregate counts. Case order or subset generation must not affect a case's
specification or pixels.

The main suite also contains 32 negatives:

```text
8 board-free online-UI templates x 2 layouts x 2 quality modes
= 32 negative cases
```

Negative templates use a separate construction path and never invoke the board
renderer or create an 8 by 8 checker pattern. They include deterministic panel,
clock, dialog, lobby, and analysis-like shapes rather than only blanks or noise.
An empty 8 by 8 board is not used as a negative. The distinct “board present but
structurally invalid placement” boundary is deferred from version 1 rather than
silently labeled no-board.
Two negative templates are assigned to development and the other six to
acceptance; every layout/quality/channel variant of a template stays in that
split.

An input-contract suite adds 16 positive cases (four positions by two views by
RGBA/grayscale) using the clean board-crop presentation and one fixed reviewed
style (`lichess_chessnut` pieces on the brown palette), plus eight negative cases
(four board-free templates by RGBA/grayscale). Holding style constant isolates
the channel-mode boundary that this suite is intended to exercise; the main RGB
matrix already covers all four piece sets and palettes. It is reported separately
so P2-05 failures cannot be mistaken for main RGB accuracy. The full version-1
plan therefore contains 568 cases: 528 positive and 40 negative.

The complete expected count is frozen in the source specification and asserted
before and after generation.

## Manifest contract

The generator builds the entire case plan in memory and validates it before
opening a destination. The generated `manifest.json` includes top-level source,
dependency, schema, benchmark-version, and claim-boundary metadata plus a sorted
case list. Claim-boundary metadata includes the four frozen limitation statements
and `base_position_count = 16`, so the 568 variants cannot be presented as 568
independent positions.

Every positive case records at least:

- case ID and relative PNG path;
- suite, development/acceptance split, and base-position ID;
- `board_present = true`;
- canonical source placement and expected source-oriented placement;
- view, piece set, palette, layout, quality, and channel mode;
- image width, height, and mode;
- half-open board box and ordered pixel-center corners;
- encoded-file SHA-256 and decoded pixel SHA-256. The pixel digest covers image
  mode, dimensions, and raw decoded pixel bytes in that order.

Every negative case records the same applicable generation groups, hashes, and
dimensions, with `board_present = false`, a negative-template ID, and no
placement, view, piece set, palette, or board geometry.

The manifest contains no timestamp, host path, thread order, or mutable local
configuration. JSON is UTF-8 with sorted keys, compact separators, no NaN values,
and one trailing newline. Identical semantic inputs must produce identical
decoded-pixel hashes and canonical manifest fields. Encoded PNG byte identity is
required across clean processes in the validated local runtime; final file hashes
and the recorded Python/Pillow/zlib fingerprint freeze the artifact when native
encoders differ elsewhere.

### Independent trust anchor

The evaluator must not accept semantic labels merely because the generated
manifest is internally self-consistent. It reloads the reviewed source
specification, rebuilds the full case plan, and compares every semantic manifest
field case for case before reading predictions. The source specification is the
oracle for IDs, labels, groups, geometry, and expected counts.

After final generation, the canonical manifest SHA-256 is written to a separate
`benchmarks/board_recognition_v1_manifest.sha256` file outside the ignored dataset
directory, and its filesystem identity must not appear anywhere in the dataset via
a hard link. Evaluation requires both semantic reconstruction and that digest.
Before reading external-digest bytes, verification requires both path containment
and filesystem-identity isolation; it then validates the exact digest before
parsing. Staged and canary verification reads the manifest bytes in memory and
uses the same canonical parsing, semantic, and image checks without manufacturing
a temporary digest.
Because this agent will not commit, the benchmark is locally frozen but does not
gain a repository-history trust anchor until the user later reviews and commits
the source specification and digest.

## Evaluator contract

The pure scorer is implemented and proven before the image generator. It accepts
one prediction per already verified manifest case and rejects missing IDs,
duplicates, unknown IDs, and invalid placement strings. The surrounding evaluator
pipeline rejects tampered manifest or image hashes before invoking the scorer.

Predictions distinguish:

- success with `source_placement`;
- typed recognition failure with its exact reason; and
- evaluator/model execution error.

The checkpoint runner reconstructs cases from the source specification, verifies
the frozen manifest and image hashes, and creates a fresh `Picture` for each case.
After corpus verification and before model loading, it rejects any report path
that resolves inside the dataset.
It calls `BoardRecognitionHelper.recognize()` exactly once per image, never marks
or mutates that image, and writes exactly one prediction with the same case ID.
Typed results map directly to success/failure records. An unexpected exception is
recorded as an execution error, never relabeled as `RecognitionFailure`; the run
continues to produce the full diagnostic report and exits nonzero after writing
it when any execution error occurred.

The report includes overall and grouped counts plus:

- positive recognition-success rate;
- exact 64-square source-placement accuracy, with positive failures counted
  wrong;
- per-square accuracy;
- occupancy precision and recall;
- piece-symbol accuracy conditional on both sides marking a square occupied;
- negative any-rejection rate;
- negative exact-`NO_BOARD` rate;
- positive false-`NO_BOARD` rejection rate;
- failure-reason distribution; and
- execution-error count.

Groups include suite, split, base position, piece set, palette, view, layout,
quality, channel mode, and negative template where applicable. Orientation and
historical FEN metadata are not scored because the public API does not return
them.

Quality thresholds are not invented before a baseline exists. Corpus integrity
and prediction completeness are hard pass/fail gates immediately; checkpoint
acceptance thresholds are frozen only after the current baseline and known
defects are measured.

For all positive quality metrics, a typed failure or execution error is a wrong
case. It contributes zero correct squares out of 64, and every expected occupied
square contributes an occupancy false negative. It contributes no conditional
piece comparison because there is no predicted occupied square. Conditional
piece accuracy is therefore reported together with its numerator/denominator and
occupancy recall; a zero denominator is JSON `null`, never 0 or 1. A successful
placement is scored square by square: occupancy TP/FP/FN use empty versus
occupied, while conditional piece accuracy uses squares both sides mark occupied.

### Anti-tautology tests

Hand-built scorer fixtures must prove, before generator work begins, that metrics
change correctly for:

1. one wrong piece symbol on an occupied square;
2. one occupied/empty error;
3. a 180-degree placement error;
4. a typed failure on a positive case;
5. success on a no-board case;
6. a non-`NO_BOARD` rejection on a negative case;
7. a missing, duplicate, or extra prediction; and
8. a changed image or manifest hash.

No king-presence or FEN-parseability assertion counts as recognition accuracy.

## Independent generator proof

The following gates must all pass before final generation:

1. **Source preflight:** dependency versions, frozen selection-provenance fields,
   exact assets and tree hashes, licenses, palettes, positions, IDs, case counts,
   and group balance validate without opening the large source corpus or writing
   output.
2. **Renderer oracle:** an artificial piece loader gives all twelve symbols
   distinct full-square colors. Across corner, edge, and center squares in both
   views, pixel locations must match an independent `python-chess` coordinate
   calculation. The existing exploratory probe passed 144 such checks; the
   behavior becomes a permanent regression test.
3. **Orientation oracle:** an asymmetric position rendered black-bottom must
   match a white-bottom render of the independently rotated placement, while its
   expected label is the rotated source grid.
4. **Composition oracle:** the clean embedded board crop must equal the clean
   standalone render exactly, and recorded boxes/corners must be finite, ordered,
   convex, in bounds, and consistent with the documented convention.
5. **Label-preserving transform checks:** quality and channel conversions keep
   dimensions, geometry, board visibility, and expected placement unchanged.
6. **Determinism:** two complete canary runs in clean processes, repeated
   generation in one process, reversed case order, and subset generation produce
   identical per-case bytes and manifest content.
7. **Full-matrix binding:** inject a fake renderer whose output encodes the
   independently planned position/view/style identity, run all 568 cases, and
   prove every image remains bound to its own case and label even when generation
   order is reversed. For a real staged corpus, reconstruct every case from the
   source specification and re-render it independently; every positive image (or
   clean board crop where applicable) and every negative template must match the
   expected case-specific pixel hash. A deterministic case swap is a hard failure.
8. **Transactional output:** an existing destination is refused without changing
   a sentinel; an injected write failure leaves no final destination; a successful
   temporary run contains exactly the manifest and expected PNGs; every image
   reopens with its declared mode/size and both hashes match. The staging
   directory's pre-rename device/inode identity is the only rollback token, so a
   substituted destination is never deleted. Disposable canaries
   are passed to the writer by their resolved physical path so platform aliases
   such as macOS `/tmp` do not weaken the symlink-ancestor safety boundary.
9. **Visual canary review:** only after automated gates pass, create a disposable
   stratified contact sheet covering every piece set, palette, view, layout,
   quality, channel mode, and negative template outside the corpus directory.
   Inspect labels and geometry.
10. **Independent code review:** a separate agent reviews generator and scorer
   boundaries, tests, and final diff. Critical or important findings block final
   generation.

The CLI defaults to read-only preflight. Writing requires an explicit flag and a
nonexistent final destination. It writes all artifacts into a unique staging
directory, validates the staged corpus, and atomically renames it only after all
checks pass. It never calls `reset_dir`, never overwrites, never reads
`config.local.yaml`, and never uses global RNG or worker threads.

## Execution sequence

1. Implement and test the strict placement parser and pure scorer.
2. Freeze and review the position/asset source specification.
3. Implement case planning and all read-only preflight checks.
4. Implement rendering/composition behind the proven oracle.
5. Implement transactional writing and corpus verification.
6. Run focused and full repository checks.
7. Run two disposable canaries, compare hashes, and visually inspect the
   stratified contact sheet.
8. Obtain independent review and resolve all material findings.
9. Only then generate `datasets/board_recognition_benchmark/v1`.
10. Verify and freeze the corpus, then stop. P1-05 must decouple model imports
    from ignored raw assets before the current-checkpoint baseline is an accepted
    downstream step. No checkpoint is selected or retrained here.

## Non-goals

This change does not:

- repair the existing training generators;
- use the synthetic benchmark as training data;
- claim a real-site or unseen-style accuracy number;
- infer or score orientation/history metadata;
- add perspective, cropping, occlusion, or coordinate overlays;
- fix P1-05, P2-05, P2-07, or full P1-07 rejection training;
- run or claim the current-checkpoint baseline before P1-05;
- choose quality thresholds before a baseline; or
- commit, publish, upload, or distribute generated artifacts.
