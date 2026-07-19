# P2-03 Pixel-Space Geometry Regression Design

## Decision

Strengthen the existing valid-quadrilateral extraction test so it locks the already-correct pixel-space edge calculation on a non-square, rotated board. Do not change production code.

## Context

The historical implementation measured Euclidean edge lengths in normalized coordinates, then multiplied a complete width edge by image width and a complete height edge by image height. That is wrong when an edge contains both x and y components on a non-square image.

Commit `eb1c18d` already corrected the production path by converting all four coordinates to pixels before measuring widths and heights. The audit remained open because the existing test uses a square image and asserts only that both output dimensions are positive, so it cannot distinguish the historical formula from the corrected one.

## Fixture and Oracle

Use a black OpenCV image with shape `(800, 1600, 3)` and normalized corners:

```python
[0.25, 0.125, 0.5, 0.5, 0.40625, 0.75, 0.15625, 0.375]
```

These map exactly to pixel points `(400, 100)`, `(800, 400)`, `(650, 600)`, and `(250, 300)`. The convex parallelogram has two 500-pixel width edges and two 250-pixel height edges, so the extracted OpenCV image must have shape `(250, 500, 3)`. The historical normalized-coordinate formula instead produces `(213, 721, 3)`.

All normalized values are binary-exact, all scaled points are integers, and the pixel-space edges are exact 3-4-5 triangles. This makes an exact shape assertion stable.

## Test Boundary

Reuse `detector_with_coords()` to replace only model coordinate prediction. The test continues through the real geometry validation, pixel-space length calculation, perspective transform, OpenCV warp, and `Picture` conversion.

Rename and strengthen the existing valid-quadrilateral test rather than adding a second overlapping test. Retain its `Picture` return assertion and replace the two weak positive-dimension assertions with the exact output shape.

## Alternatives Rejected

- Add a second extraction test: duplicates the existing valid-path setup without increasing behavioral coverage.
- Extract a target-size helper for unit testing: adds production API solely for a test.
- Update only the audit: leaves the historical bug unprotected.

## Acceptance Criteria

- `tests/test_board_detector_model.py` contains one exact rectangular/rotated extraction regression.
- The regression fails with actual shape `(213, 721, 3)` when the historical formula is temporarily restored and passes with `(250, 500, 3)` after the current formula is restored.
- `chessml/models/lightning/board_detector_model.py` has no final diff.
- P2-03 is marked resolved without claiming that the checkpoint baseline ran.
- No dependency, helper, duplicate test, stage, or commit is added.
