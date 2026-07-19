# P2-03 Pixel-Space Geometry Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the existing pixel-space board-extraction correction with an exact non-square rotated regression and close P2-03.

**Architecture:** Replace the weak valid-quadrilateral assertions with a deterministic exact-shape oracle. Prove the regression detects the historical formula through a temporary mutation, restore the existing production implementation, and leave no production diff.

**Tech Stack:** Python 3.14, NumPy, OpenCV, pytest.

## Global Constraints

- Governing design: `docs/superpowers/specs/2026-07-19-p2-03-pixel-space-geometry-regression-design.md`.
- Governing finding: `BOARD_RECOGNITION_AUDIT.tmp.md`, P2-03.
- Final production behavior and `chessml/models/lightning/board_detector_model.py` must remain unchanged from `HEAD`.
- Reuse `detector_with_coords()` and the existing valid-quadrilateral test; do not add a helper or overlapping test.
- The fixture is exactly a `(800, 1600, 3)` black image with normalized corners `[0.25, 0.125, 0.5, 0.5, 0.40625, 0.75, 0.15625, 0.375]`.
- The corrected output shape is exactly `(250, 500, 3)`; the historical formula produces `(213, 721, 3)`.
- Do not run checkpoint inference or claim the current-checkpoint baseline was completed.
- Preserve unrelated work. Do not stage or commit any file.

---

### Task 1: Strengthen the existing extraction regression

**Files:**

- Modify: `tests/test_board_detector_model.py`
- Temporarily mutate, then restore with no final diff: `chessml/models/lightning/board_detector_model.py`

- [ ] **Step 1: Replace the weak valid-quadrilateral fixture and assertions**

Replace `test_extract_board_image_accepts_a_valid_quadrilateral` with:

```python
def test_extract_board_image_scales_rotated_edges_in_pixel_space(monkeypatch):
    source = Picture(np.zeros((800, 1600, 3), dtype=np.uint8))
    detector = detector_with_coords(
        monkeypatch,
        [0.25, 0.125, 0.5, 0.5, 0.40625, 0.75, 0.15625, 0.375],
    )

    extracted = detector.extract_board_image(source)

    assert isinstance(extracted, Picture)
    assert extracted.cv2.shape == (250, 500, 3)
```

- [ ] **Step 2: Confirm the current correction is GREEN**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py::test_extract_board_image_scales_rotated_edges_in_pixel_space
```

Expected: `1 passed` with pristine output.

- [ ] **Step 3: Prove RED with the historical formula**

Temporarily replace only the current four edge-length calculations and two target-size assignments with the historical normalized-coordinate calculation:

```python
width_a = np.sqrt(((br_x - bl_x) ** 2 + (br_y - bl_y) ** 2)) * width
width_b = np.sqrt(((tr_x - tl_x) ** 2 + (tr_y - tl_y) ** 2)) * width
target_width = max(int(width_a), int(width_b))

height_a = np.sqrt(((tr_x - br_x) ** 2 + (tr_y - br_y) ** 2)) * height
height_b = np.sqrt(((tl_x - bl_x) ** 2 + (tl_y - bl_y) ** 2)) * height
target_height = max(int(height_a), int(height_b))
```

Run the exact test command from Step 2. Expected: assertion failure showing actual `(213, 721, 3)` versus expected `(250, 500, 3)`.

- [ ] **Step 4: Restore the pixel-space calculation and verify GREEN**

Restore exactly:

```python
width_a = np.linalg.norm(source_points[2] - source_points[3])
width_b = np.linalg.norm(source_points[1] - source_points[0])
target_width = max(int(width_a), int(width_b))

height_a = np.linalg.norm(source_points[1] - source_points[2])
height_b = np.linalg.norm(source_points[0] - source_points[3])
target_height = max(int(height_a), int(height_b))
```

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
git diff --exit-code HEAD -- chessml/models/lightning/board_detector_model.py
```

Expected: all seven detector tests pass; the production-file comparison exits zero with no output.

- [ ] **Step 5: Self-review scope and report evidence**

Confirm the final tracked diff changes only the existing test, the index remains empty, and no production code differs from `HEAD`. Report the initial GREEN, mutation RED, restored GREEN, exact shapes, and commands.

---

### Task 2: Independently review, verify, and close P2-03

**Controller-only files:**

- Modify after review and verification: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] Have a separate subagent review Task 1 for oracle independence, mutation sensitivity, real extraction coverage, minimality, and restored production state.
- [ ] Resolve every blocking review finding and re-review if necessary.
- [ ] Have a separate verifier run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
git diff --exit-code HEAD -- chessml/models/lightning/board_detector_model.py
git diff --check
git diff --cached --stat
git status --short
```

- [ ] Mark P2-03 resolved in `BOARD_RECOGNITION_AUDIT.tmp.md`, record that `eb1c18d` already contained the production correction, and update the required-order item so P2-05 is the remaining preprocessing fix after the still-unrun baseline.
- [ ] Supply the exact audit patch and complete change package to a final read-only reviewer; resolve any findings.
- [ ] Re-run the focused test, production-file identity, whitespace, index, and status checks. Do not commit, stage, run checkpoint inference, or publish a report.
