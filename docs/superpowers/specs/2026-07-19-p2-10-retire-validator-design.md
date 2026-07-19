# P2-10 Retired Board-Detector Validator Design

## Decision

Delete `scripts/validate/validate_board_detector.py`. The supported board-recognition benchmark evaluator remains the single evaluation path.

## Context

The detector-only script is an abandoned pre-contract tool. It fails before model loading because its decorated `main(args, config)` signature conflicts with the current `Script` wrapper, has no callers or documentation, hard-codes stale input/checkpoint/output paths, passes a Pillow image where `BoardDetector` requires `Picture`, and calls `save` on `Picture` even though `Picture` has no such method.

The repository already has `scripts/validate/evaluate_board_recognition_benchmark.py`, with direct evaluator tests, frozen benchmark validation, typed recognition results, and canonical report publication. Repairing the old script would create a second evaluation authority without adding coverage that the benchmark evaluator lacks.

## Alternatives Rejected

- Repair the old CLI and add a detector fixture: retains duplicate evaluation paths and hard-coded detector-only semantics.
- Turn the old path into an evaluator wrapper: preserves a misleading obsolete entry point and adds forwarding code with no consumer.

## Acceptance Criteria

- `scripts/validate/validate_board_detector.py` no longer exists.
- Runtime code, tests, configuration, and README contain no reference to `validate_board_detector`.
- The canonical benchmark evaluator tests and full repository suite still pass.
- P2-10 is marked resolved without claiming that a current-checkpoint baseline or the real-screenshot half of P2-01 was completed.
- No dependency, replacement wrapper, absence-only test, stage, or commit is added.

## Verification Strategy

Capture the existing CLI failure before deletion. After deletion, verify file/reference absence and run the existing benchmark-evaluator tests. A new test asserting that a repository path stays absent would test layout rather than supported behavior, so the durable regression coverage remains the canonical evaluator suite. Finish with the full suite, compilation, whitespace, index, and independent review gates.
