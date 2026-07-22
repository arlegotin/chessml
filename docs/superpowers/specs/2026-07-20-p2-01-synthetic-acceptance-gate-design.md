# P2-01 Synthetic Acceptance Gate Design

Date: 2026-07-20

## Decision

Promote the existing frozen `online-render-v1` acceptance split from a
diagnostic report into an enforceable correctness gate. Do not create or
capture another dataset.

This closes P2-01 for the repository's generated online-board domain: wrong
piece placement, wrong source orientation, missing `NO_BOARD`, false
`NO_BOARD`, and execution errors must make the evaluator exit nonzero. It does
not establish holdout from the historical checkpoints or accuracy on unseen
chess sites.

The previously planned live-browser `online-screenshot-v1` path remains dormant.
No browser, network request, real-board imagery, or live-site capture is part of
this change.

## Existing evidence and root cause

The existing corpus already supplies the required oracle:

- 568 deterministic cases: 528 positives and 40 negatives;
- 16 independently validated base positions, four piece sets, both source
  views, two layouts, clean and browser-scaled quality, and RGB/RGBA/L inputs;
- exact source-oriented placement labels, with black-bottom labels derived by
  independently tested 180-degree grid rotation;
- development and acceptance splits grouped by base source;
- exact file/pixel hashes, canonical manifest bytes, an external manifest
  digest, transactional generation, and independent rerender verification; and
- scores for exact placement, square occupancy and symbols, exact `NO_BOARD`,
  false `NO_BOARD`, failure reasons, execution errors, and generation groups.

Fresh baseline evidence before this design is `487 passed` across the benchmark
construction/evaluator suites and preflight counts of `568/528/40`. Corpus
verification reached all manifest file/pixel checks and failed only because the
ignored corpus contains an extra Finder-created `.DS_Store`; the strict
inventory check must remain unchanged and that single extraneous file must be
removed.

The root defect is the synthetic evaluator's final status rule. It returns
nonzero only for execution errors, so the anchored post-P2-05 run exited zero at
245/384 exact acceptance placements and 0/28 acceptance negatives returning
`NO_BOARD`. Scoring is correct; pass/fail interpretation is missing.

## Frozen policy

Add this policy to `benchmarks/board_recognition_v1.json` and validate it as part
of the source specification:

```json
{
  "exact_placement": {"correct": 384, "total": 384},
  "negative_no_board_rate": {"correct": 28, "total": 28},
  "positive_false_no_board_rate": {"correct": 0, "total": 384},
  "execution_error_count": 0
}
```

Placement and rejection requirements apply only to
`scores.groups.split.acceptance`. The `execution_error_count` requirement applies
to `scores.overall`, so a crash in a development case still fails the run.
Development accuracy remains available for diagnosis and future calibration and
cannot compensate for an acceptance failure. Perfect aggregate requirements
also imply perfect results for both source views and every included acceptance
variant; one wrong acceptance case fails the gate.

Replace the obsolete P2-01-open source limitation with exactly:
`This gate enforces generated-domain regression correctness only.` The policy
and this limitation produce source semantic digest
`524e3ee10e93624b295ad120f2f510a64979e6a3fc35895aca2c4884e14b305a`;
source validation must pin both.

Source orientation is tested through exact `source_placement` for both views.
Turn, castling, en-passant, clocks, and inferred orientation metadata are not
part of the P1-06 recognition contract and remain unscored.

## Evaluator behavior

Reuse the existing acceptance-policy comparator already used by the captured
path. For synthetic runs:

1. verify the source, manifest, exact image bytes, and checkpoints as today;
2. run and retain all predictions and grouped scores as today;
3. evaluate placement/rejection requirements against the acceptance-split score
   group and the execution-error requirement against the complete run;
4. always publish the canonical report with an `acceptance` object; and
5. return zero only when every policy requirement passes.

A failed gate still writes the full report. Validation, model-loading, corpus,
or report-publication errors remain exceptions and must not be converted into
ordinary accuracy failures.

Captured behavior and freeze checks remain unchanged. The unused captured
corpus machinery is not removed or expanded in this task.

## Corpus and trust-anchor update

Changing the source policy and limitation text changes the semantic source
digest, so regenerate two disposable normal/reverse corpora with the existing
reviewed generator. They must be byte-identical to each other, and all 568 PNG
bytes must remain identical to the current corpus. Only source-bound manifest
metadata is expected to change.

After those checks, replace the ignored corpus `manifest.json` through generated
output, update `benchmarks/board_recognition_v1_manifest.sha256`, remove only
`datasets/board_recognition_benchmark/v1/.DS_Store`, and run anchored full corpus
verification. Do not weaken inventory validation or edit PNGs manually.

## Tests and documentation

Use TDD at two observable seams:

- source validation rejects any acceptance-policy mutation and proves the
  frozen totals equal the generated acceptance split; and
- a synthetic evaluator run with a wrong but executable acceptance-split
  prediction writes a failed `acceptance` result and exits 1, a development-only
  execution error also exits 1, and perfect acceptance with no execution errors
  exits 0.

Update README and only the P2-01/required-order portions of
`BOARD_RECOGNITION_AUDIT.tmp.md`. Record that P2-01 is resolved as an enforceable
generated-domain gate, the historical post-P2-05 predictions fail its new
policy, and unseen-site accuracy remains unproven. P1-07 may proceed to
implementation and retraining, but it is not resolved by this change.

## Scope constraints

- Do not touch `scripts/data/download_piece_sets.py` or P2-11 content.
- Do not use a browser or network, capture online pages, or create a real-board
  dataset.
- Do not train, calibrate, select, replace, or load a checkpoint and do not rerun
  benchmark inference; recompute the historical acceptance result from the
  existing anchored report.
- Do not add a dependency, acceptance framework, policy class, or second scorer.
- Do not stage or commit.

## Acceptance

- A wrong acceptance-split prediction with zero execution errors makes the
  evaluator publish a failed acceptance result and return 1.
- An execution error in either split makes the evaluator publish a failed
  acceptance result and return 1.
- A perfect synthetic acceptance result returns 0.
- The source policy, case-plan totals, regenerated manifest, external digest,
  and all 568 image bytes verify.
- The historical post-P2-05 report recomputes to 245/384 exact acceptance
  placements, 0/28 exact `NO_BOARD`, 0/384 false `NO_BOARD`, and zero execution
  errors, therefore failing the gate without inference.
- Focused and broad checkpoint-excluding tests, compilation, whitespace, scope,
  and empty-index checks pass.
- README and the audit state the closure and limitations without claiming
  unseen-site accuracy or P1-07 completion.
