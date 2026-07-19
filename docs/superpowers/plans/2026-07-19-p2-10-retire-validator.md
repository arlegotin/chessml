# P2-10 Retired Board-Detector Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the broken, unused detector-only validator and keep the benchmark evaluator as the sole supported evaluation path.

**Architecture:** Delete the obsolete script instead of migrating a second CLI. Preserve the existing benchmark evaluator and its tests unchanged, then close P2-10 only after independent review and full verification.

**Tech Stack:** Python 3.14, pytest, existing benchmark evaluator.

## Global Constraints

- Governing design: `docs/superpowers/specs/2026-07-19-p2-10-retire-validator-design.md`.
- Governing finding: `BOARD_RECOGNITION_AUDIT.tmp.md`, P2-10.
- `scripts/validate/evaluate_board_recognition_benchmark.py` remains the single evaluation authority and is not modified.
- Do not add a replacement wrapper, dependency, fixture, or test that merely asserts a file is absent.
- Do not run checkpoint inference, generate a benchmark report, or claim the P2-01 real-screenshot benchmark is complete.
- Preserve unrelated work. Do not stage or commit any file.

---

### Task 1: Retire the obsolete validator

**Files:**

- Delete: `scripts/validate/validate_board_detector.py`

**Supported replacement:**

- Existing: `scripts/validate/evaluate_board_recognition_benchmark.py`
- Existing coverage: `tests/test_board_recognition_benchmark.py`

- [ ] **Step 1: Reproduce the broken entry point**

Run:

```bash
uv run --locked python scripts/validate/validate_board_detector.py
```

Expected before deletion: exit nonzero with `TypeError: main() missing 1 required positional argument: 'config'` before model loading.

- [ ] **Step 2: Delete the obsolete script**

Delete `scripts/validate/validate_board_detector.py` in full. Do not copy its hard-coded paths, model loading, drawing, or output behavior elsewhere.

- [ ] **Step 3: Verify retirement and canonical coverage**

Run:

```bash
test ! -e scripts/validate/validate_board_detector.py
rg -n "validate_board_detector" README.md pyproject.toml chessml scripts tests
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py
```

Expected: the absence check exits zero; `rg` exits one with no matches; the existing benchmark-evaluator suite passes with pristine output.

- [ ] **Step 4: Self-review scope and report evidence**

Confirm the tracked task diff deletes exactly the obsolete script, the index remains empty, and the canonical evaluator/tests are unchanged. Explain that no new test was added because the change removes an unsupported path and the supported replacement already has direct coverage. Report the exact before-deletion failure and after-deletion results.

---

### Task 2: Independently review, verify, and close P2-10

**Controller-only files:**

- Modify after review and verification: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] Have a separate subagent review Task 1 for complete retirement, canonical-path preservation, minimality, and test evidence.
- [ ] Resolve every blocking review finding and re-review if necessary.
- [ ] Have a separate verifier run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
git diff --check
git diff --cached --stat
git status --short
```

- [ ] Mark P2-10 resolved in `BOARD_RECOGNITION_AUDIT.tmp.md`; update the required-order and P2-01 status references so the current-checkpoint baseline is next. Preserve the statement that P2-01 still lacks independently captured, manually labeled online screenshots.
- [ ] Supply the exact audit patch to a final read-only reviewer, resolve any findings, and confirm no unrelated audit issue changed.
- [ ] Re-run file/reference absence, whitespace, index, and status checks. Do not commit, stage, run checkpoint inference, or publish a report.
