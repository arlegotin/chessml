# P2-04 Non-Mutating Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BoardDetector.mark_board_on_image()` return a marked diagnostic image without changing the caller's `Picture`.

**Architecture:** Preserve the public method and drawing logic. Copy the source PIL image at the shared method boundary, draw on that copy, and return it as a new `Picture`.

**Tech Stack:** Python 3.14, Pillow, NumPy, pytest.

## Global Constraints

- Governing finding and correction boundary: `BOARD_RECOGNITION_AUDIT.tmp.md`, P2-04.
- The source `Picture` must have identical pixels before and after diagnostic marking.
- The returned `Picture` must contain the grid overlay.
- Keep prediction, coordinates, colors, line widths, and public signatures unchanged.
- Add no dependency or abstraction; the production fix should be the image copy at the existing boundary.
- Preserve unrelated work. Do not stage or commit any file.

---

### Task 1: Copy before drawing and lock the behavior with a regression

**Files:**

- Modify: `tests/test_board_detector_model.py`
- Modify: `chessml/models/lightning/board_detector_model.py`
- Modify: `README.md`

**Behavior:**

- `mark_board_on_image(source)` does not mutate `source.pil`.
- Its returned `Picture` contains the existing green board grid.

- [ ] **Step 1: Add a fresh-source behavioral regression**

Add a test to `tests/test_board_detector_model.py` that creates its own black `Picture`, uses `detector_with_coords()` with a valid inset quadrilateral, snapshots the source pixels, and calls `mark_board_on_image()`.

Assert both observable invariants:

```python
np.testing.assert_array_equal(np.asarray(source.pil), before)
assert np.any(np.asarray(marked.pil) != before)
```

Mock only `predict_coords`; the real `Picture`, Pillow copy/drawing behavior, and production marking method remain under test.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
```

Expected before implementation: the new test fails because the source pixels contain the green grid after the call.

- [ ] **Step 3: Make the minimal shared-boundary fix**

In `BoardDetector.mark_board_on_image()`, change only the image acquisition from the cached source object to a Pillow copy:

```python
image = original_image.pil.copy()
```

Leave the remaining prediction and drawing code unchanged.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
```

Expected: all focused tests pass with pristine output.

- [ ] **Step 5: Synchronize the public example**

Change the README wording from saying the method marks the original image to saying it returns a marked copy. Do not change the example API.

- [ ] **Step 6: Self-review scope and report evidence**

Confirm the implementation is the one-line production fix, the test uses a fresh source rather than the module-level `IMAGE`, and no unrelated files were changed. Report the exact RED and GREEN commands/results. Do not update the audit finding; the controller closes it only after independent review and broad verification.

---

### Task 2: Independently review, verify, and close P2-04

**Controller-only files:**

- Modify after review and verification: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] Have a separate subagent review Task 1 for the two behavior invariants, minimality, and test quality.
- [ ] Resolve every blocking review finding and re-review if necessary.
- [ ] Run fresh focused and full verification:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
git diff --check
git diff --cached --stat
```

- [ ] Mark P2-04 resolved in `BOARD_RECOGNITION_AUDIT.tmp.md`, replacing stale validator-call evidence with the current public-contract impact and recording the regression/full-suite evidence. Do not claim the unrun benchmark baseline was completed.
- [ ] Re-run `git diff --check`, inspect the final diff/status, and confirm the index is empty.
