# P2-01 Online Screenshot Benchmark Design

Date: 2026-07-19

## Goal

Close the real-domain half of P2-01 with a frozen, manually labeled acceptance
benchmark made from actual anonymous online chess pages. The benchmark measures
the public recognition contract without using project renderers, local piece
assets, model output, or physical-board images to create its oracle.

The existing `online-render-v1` corpus remains a synthetic diagnostic suite.
This design adds a separate `online-screenshot-v1` acceptance suite whose exact
pixels and review evidence are retained and whose failed requirements make the
evaluator exit nonzero.

## Locked scope

Version 1 contains exactly 40 original full-viewport PNG screenshots:

- 32 positives: 2 sites x 4 positions x 2 source views x 2 capture conditions.
- 8 negatives: 2 board-free pages x 2 capture conditions for each site. These
  are four distinct pages, not eight independent pages.
- Sites: the public logged-out Lichess and PyChess board editors. Both are
  deployed online clients; neither is a project renderer.
- All cases use `suite=captured` and `split=acceptance`. No pixel, crop,
  derivative, position, or label from this corpus may be used for training,
  calibration, checkpoint selection, or acceptance-threshold selection.

The four positive positions are:

| ID | Full FEN |
| --- | --- |
| `start` | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` |
| `najdorf` | `rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6` |
| `nimzo` | `rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7` |
| `pawn_endgame` | `8/5k2/8/3p4/3P4/4K3/8/8 w - - 0 1` |

They are legal and asymmetric as a set, span high/medium/low occupancy, and
collectively contain every piece symbol. A positive label is the visible
top-to-bottom placement: the FEN placement with White at bottom and its 180
degree rotation with Black at bottom.

The two capture conditions use the same locked site-native board and piece
style, so viewport and style are not confounded. Lichess uses its `chessnut`
piece set (Apache-2.0) and brown board selected through the normal anonymous
settings UI; PyChess uses its `firi` pieces (CC BY 4.0) and brown board from the
versioned deployed assets covered by its `COPYING.md`:

- `desktop_clean`: 1440 x 900 CSS pixels, device-pixel ratio 1, quality
  `clean`.
- `compact_scaled`: 800 x 600 CSS pixels, device-pixel ratio 1, quality
  `browser_scaled`. Responsive site layout produces a materially smaller but
  complete visible board; feasibility checks measured about 328 pixels on
  Lichess and 336 pixels on PyChess, versus about 520 and 616 pixels on the
  desktop captures.

The compact condition is the v1 degraded-frame coverage required by the audit.
It uses only native responsive rendering: no browser zoom, CSS injection,
cropping, resizing, compression, blur, overlay, or postprocessing. Both
conditions require the complete board to remain stable and unobscured. More
severe clipping and obstruction require a new corpus version with an explicit
failure oracle; they are not silently mixed into exact-placement cases.

Each site contributes two public board-free pages, captured under both
conditions. A negative must contain no complete 8x8 chess grid or board
thumbnail. Empty complete boards and invalid-position boards are outside v1:
the current success contract requires a legal placement, so v1 does not pretend
to resolve their future presence semantics.

## Rights and retention boundary

Chess.com is excluded because its proprietary terms do not support a durable
redistributable corpus. Lichess and PyChess publish their server/UI sources
under AGPL-3.0-or-later and maintain source `COPYING.md` files describing board,
piece, font, logo, and other asset exceptions. Capture uses only anonymous
editor or project-authored legal/privacy pages, never user content.

Before capture, a tracked attribution file records the exact official site,
source repository, license/COPYING URLs, terms URL, checked date, displayed
piece/board asset identifiers, and required notices. The retained evidence is
not limited to mutable links: it includes path- and SHA-256-bound, commit/tag-
pinned copies of lila's `COPYING.md`, Chessnut's upstream license, PyChess
Variants' `static/COPYING.md`, PyChess's local Firi license, and Chessgroundx's
license, plus dated exact-text snapshots of each site's terms and privacy page.
It also retains the applicable generic license texts. The default displayed
assets must have a redistributable license in those project-specific notices;
an unknown or proprietary asset aborts capture. Attribution uses exact TASL
records for Chessnut and Firi, records that rendering and PNG bytes are
unchanged, and preserves lila's restriction that its logo is used only to refer
to lichess.org. The PNGs stay byte-for-byte unchanged and retain attribution.
Source-code licensing is not treated as a blanket license for arbitrary
live-site or user content.

The attribution file also states that these upstream-covered screenshot files
are not relicensed by the repository's root license and supplies any license
texts required for redistribution.

A separate canonical `RIGHTS.json` is the assembler's only machine-readable
rights input. For each site it records the exact displayed piece and board
asset IDs, piece URL prefix, board-style signature, upstream revision, and
path/SHA-256 records for project COPYING, piece license, terms, and privacy
evidence. Positive receipts repeat the displayed asset ID/prefix/signature;
assembly aborts unless they equal the applicable frozen site record.

The exact corpus, manifest, capture receipts, both annotation passes, source
specification, digest, attribution, and canonical machine-readable rights
record are trackable repository artifacts.
`.gitignore` receives only narrow exceptions for this corpus's PNGs and the
exact `.txt` rights-evidence files otherwise hidden by repository-wide rules.
The user-required no-commit rule still applies during this task; repository
history protection begins when the user later commits the reviewed files.

## Capture boundary and receipts

Capture uses an isolated anonymous browser context and the ordinary public site
interface. It rejects optional tracking where offered. The browser writes a
full visible-viewport PNG; the project does not render or transform pixels.
Immediately before capture, the live terms are compared with the frozen terms
evidence. Capture stops on a material change, access block, rate limit, or a
policy classification that would require prior permission.

Every capture is a separately and manually triggered sequential action. There
is no concurrent capture, crawler, scripted capture loop, automated retry,
bulk DOM extraction, or navigation outside the exact URL allowlist. DOM reads
are limited to the controlled FEN/orientation/style state and the metadata
needed for the receipt. Screenshots use visible-viewport capture only
(`fullPage=false`).

Every positive must contain exactly one stable, fully visible target board.
Every image must exclude usernames, ratings, avatars, chat, notifications,
private URLs, session identifiers, or tokens. Allowed URLs are HTTPS and
host/path allow-listed. Lichess may carry only the non-sensitive `fen` query;
PyChess and negative pages carry no query or fragment.

Immediately around each screenshot, a capture receipt records:

- opaque case ID and PNG SHA-256;
- UTC timestamp, final safe URL, document title, site, page kind, and locale;
- browser user agent/platform, viewport, DPR, scroll position, visibility, and
  navigation state;
- board rectangle, orientation, displayed style identifiers, and displayed
  asset URLs or hashes for positives;
- the absence of a complete board for negatives.

Receipts bind provenance assertions to exact PNG hashes. Mutable public URLs
remain provenance, not reproducible upstream revisions; the retained PNG bytes
are the benchmark input. The freezer parses the PNG byte stream itself and
permits only a valid signature, one `IHDR`, one-or-more contiguous `IDAT`
chunks, and one `IEND`, with correct lengths/order/CRCs and no trailing bytes;
Pillow `verify()` and a full decode must also pass. Thus textual, EXIF, color
profile, and unknown ancillary chunks all fail instead of relying on incomplete
`image.info`. The tested browser screenshot path emits exactly this minimal
chunk sequence. Unsafe URLs, receipt/image hash mismatches, or receipt/spec
semantic mismatches also fail.

## Manual oracle

No production prediction may seed, select, correct, or adjudicate a label.

1. Freeze opaque IDs and capture all images plus receipts into staging.
2. Two independent reviewers receive only the images and IDs, not the capture
   map, input FEN/view, the other pass, the source spec, or model output.
3. Each reviewer records board presence and, for positives, all 64 visible
   squares in source orientation in a canonical annotation file.
4. Retain both annotation files. Their schemas, complete inventories, contents,
   and hashes are verified; a hash without its bytes is not evidence.
5. Positive labels require exact pass-to-pass agreement. Negatives require
   agreement that no complete target grid is present. Only then compare the
   consensus with the controlled capture input. Any mismatch rejects and
   recaptures the case; it is never silently relabeled.
6. Both reviewers inspect every one of the 40 PNGs at original resolution for
   board/content correctness, privacy, metadata, and capture-condition rules.
7. A tracked review attestation records review isolation, both annotation
   hashes, every-case full-resolution/privacy completion, and agreement.

The agreed annotation, not the capture formula, supplies the scoring oracle.
The later controlled-input comparison proves capture correctness independently.

## Tracked artifacts

- `benchmarks/board_recognition_online_screenshot_v1.json`
- `benchmarks/board_recognition_online_screenshot_v1_manifest.sha256`
- `benchmarks/board_recognition_online_screenshot_v1_capture_receipts.json`
- `benchmarks/board_recognition_online_screenshot_v1_annotations_a.json`
- `benchmarks/board_recognition_online_screenshot_v1_annotations_b.json`
- `benchmarks/board_recognition_online_screenshot_v1_review.json`
- `benchmarks/board_recognition_online_screenshot_v1_freeze.sha256`
- `benchmarks/board_recognition_online_screenshot_v1_ATTRIBUTION.md`
- `benchmarks/board_recognition_online_screenshot_v1_RIGHTS.json`
- any upstream license text named by that attribution file under
  `benchmarks/licenses/`
- `benchmarks/board_recognition_online_screenshot_v1/manifest.json`
- `benchmarks/board_recognition_online_screenshot_v1/images/*.png`
- freezer/verifier/evaluator code, tests, README, and audit evidence

Only disposable capture mapping/staging and the checkpoint report live under
ignored `output/`. Staging is removed after the trackable corpus verifies; no
capture, annotation, contact sheet, or derivative is uploaded elsewhere.

## Source specification and cases

The canonical tracked source specification has these top-level fields:

- `schema_version=1`, `benchmark_version=online-screenshot-v1`, and
  `claim=captured_online_screenshot_acceptance`;
- `base_position_count=4` and exact counts `all=40`, `positive=32`,
  `negative=8`;
- nonempty frozen `limitations`;
- the strict `acceptance_policy` below;
- capture protocol, allowed sources, capture conditions, and SHA-256 values for
  receipts, both annotations, review attestation, attribution, the canonical
  rights record, and an exact
  path/SHA-256 inventory of every required upstream license text;
- 40 explicit consensus-derived semantic cases.

Every case contains a safe opaque ID and `images/<id>.png` path plus
`suite=captured`, `split=acceptance`, `board_present`, `site`, `viewport`,
`layout=full_page`, `quality`, `channel_mode=RGB`, `image_mode=RGB`, exact
`image_size`, safe public `source_url`, and page kind.

Positive cases also contain base position, the full controlled legal FEN,
legal canonical placement, literal
consensus `source_placement`, view, locked site-native piece style, and board
rectangle. Negatives omit placement/view/style/board geometry and contain a
negative-page identifier. Validation rejects unknown/missing fields, unsafe
URLs, duplicate IDs or paths, invalid placements, illegal canonical positions,
orientation or annotation mismatches, wrong counts/matrix coverage,
non-acceptance splits, viewport/style confounding, and positive/negative field
leakage.

## Freezing and exact-input verification

A small dedicated byte-preserving freezer accepts the reviewed staging
directory and a nonexistent destination. It:

1. validates source spec, receipts, annotations, review attestation,
   attribution and canonical-rights-record hashes, exact referenced-license
   inventory/hashes, and exact PNG inventory;
2. opens each image once without following symlinks, retains its bytes, and
   rejects extra/missing/malformed images, metadata, wrong mode/size, receipt
   mismatch, and duplicate encoded-file or decoded-pixel hashes from those
   retained buffers;
3. writes those exact retained PNG buffers transactionally without
   decoding/re-encoding or reopening a source pathname;
4. adds encoded-file and decoded-pixel SHA-256 values to a canonical manifest;
5. verifies staging before one atomic no-replace rename publishes it and never
   overwrites a destination, including if the destination races publication;
6. writes an external manifest digest and a canonical pre-inference freeze
   commitment over source spec, receipts, annotations, review, attribution,
   canonical rights record, every required license text, manifest, digest, and
   every PNG.

The existing canonical JSON, containment, hash, scorer, and report code is
reused. `build_case_plan()` dispatches this captured claim to its explicit
reviewed cases while retaining the synthetic path unchanged. `site` and
`viewport` are added to score grouping.

Before model loading, the evaluator verifies the source, manifest digest,
freeze commitment, complete inventory, and every PNG. It then reads each PNG
once into a byte buffer, verifies file and decoded-pixel hashes from that same
buffer, and later constructs `Picture` from an in-memory Pillow copy. Inference
therefore cannot reopen a replaced pathname after verification. The report
records the freeze commitment.

Because commits are forbidden in this task, two independent reviews must attest
the exact pre-inference commitment in session review messages outside the
frozen file set; they do not edit the already-hashed `review.json`. The
evaluator must reproduce the commitment immediately before and after inference.
It also retains all three checkpoint hashes before model loading, recomputes
them after inference, and reports only those retained values. No
corpus/evidence or checkpoint byte may change between those checks. This is the
session-local temporal anchor; the final user commit is the durable history
anchor.

## Acceptance policy

The policy is declared before any checkpoint access and cannot be selected from
the baseline:

- exact placement: 32/32 positives;
- negative `NO_BOARD`: 8/8 negatives;
- false positive `NO_BOARD`: 0/32 positives;
- execution errors: 0.

The report contains `acceptance.passed` and exact failed requirements. For this
captured claim, any failed requirement makes the evaluator exit nonzero after
writing the report. The synthetic claim preserves its current diagnostic
error-only exit behavior. A failing current checkpoint is valid baseline
evidence, not a reason to relax the policy or alter the corpus.

## Verification gates

Automated tests must prove captured schema/count/matrix/orientation rules,
annotation and receipt binding, byte-preserving transactional freeze, metadata
and malformed-input rejection, full tamper detection, exact-input buffering,
site/viewport groups, strict acceptance exit behavior, and unchanged synthetic
behavior. Focused and full tests, compilation, CLI help, anchored synthetic
verification, whitespace, status, index, and corpus-immutability checks must
remain green.

Whitespace checks apply to every hand-authored attribution, rights, code, test,
and design/plan file. The exact thirteen third-party rights-evidence files
enumerated in the implementation plan are preserved byte-for-byte instead:
their gate is SHA-256 and byte equality with the recorded immutable-source or
live-response retrieval evidence, plus nonempty plain text with no NUL byte or
HTML document wrapper. Original upstream line endings and end-of-file bytes are
not normalized to satisfy whitespace diagnostics.

Before checkpoint access, independent review must confirm all 40 original-size
privacy/content inspections, exact annotation agreement, controlled-input
agreement, rights/attribution evidence, frozen-corpus verification, and the
pre-inference commitment. After checkpoint inference, recompute that commitment
and independently recompute every report score and acceptance result.

## Closure and limitations

P2-01 is resolved when the corpus/evidence and evaluator pass all construction
gates, even if the current checkpoint honestly fails the recognition acceptance
policy. That model failure becomes the frozen baseline for P1-07 rather than a
false-green P2-01 result. P1-07 remains open until a real presence decision
returns exact `NO_BOARD` for negatives without rejecting positives.

`online-screenshot-v1` is deliberately narrow: two open-source deployed
clients, four positions, their two native orientations, one locked style per
site, two viewport qualities, and four distinct negative pages. It is a frozen
regression/acceptance smoke gate, not a statistically precise product-accuracy
estimate or proof of holdout from legacy checkpoint training. Expand by making
a new version, never by mutating v1.
