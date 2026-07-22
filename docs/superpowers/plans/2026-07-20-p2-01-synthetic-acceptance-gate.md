# P2-01 Synthetic Acceptance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Replace commit-based transport with shared-working-tree reports because the user forbids staging and commits. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing frozen generated-online-board benchmark fail on recognition inaccuracy instead of only execution errors, then accurately close P2-01 for that domain.

**Architecture:** Add one exact policy to the existing source specification and reuse the existing acceptance comparator. Synthetic placement/rejection requirements consume the frozen acceptance-split score group, while execution errors consume the complete run; no new scorer, corpus, or abstraction is introduced. Regenerate only source-bound manifest metadata through the reviewed generator and prove every PNG byte is unchanged.

**Tech Stack:** Python 3.14, pytest, Pillow, existing benchmark generator/scorer/evaluator, stdlib JSON/hash utilities.

## Global Constraints

- Follow root `AGENTS.md`; no nested instruction file applies to the target paths.
- Preserve the pre-existing untracked `BOARD_RECOGNITION_ARCHITECTURE_RESEARCH.tmp.md` and all work outside P2-01.
- Do not touch `scripts/data/download_piece_sets.py` or any P2-11 content.
- Do not use a browser or network, capture online pages, or create a real-board dataset.
- Do not train, calibrate, select, replace, or load a checkpoint; do not rerun benchmark inference.
- Recompute historical acceptance only from the existing anchored post-P2-05 report.
- Keep the strict corpus inventory check; remove only `datasets/board_recognition_benchmark/v1/.DS_Store`.
- Do not alter any benchmark PNG byte. Generate disposable corpora through the existing generator and update the ignored final `manifest.json` only after exact image comparison succeeds.
- Do not add a dependency, policy class, second scorer, new dataset, or new evaluator CLI option.
- Run broad pytest with `tests/test_board_recognition_integration.py` excluded because that test loads local production checkpoints.
- Do not stage or commit.

---

### Task 1: Freeze and validate the generated-domain acceptance policy

**Files:**
- Modify: `benchmarks/board_recognition_v1.json`
- Modify: `benchmarks/board_recognition.py`
- Modify: `tests/test_board_recognition_benchmark.py`

**Interfaces:**
- Consumes: the existing `online-render-v1` source specification and `build_case_plan(spec) -> list[dict]`.
- Produces: `spec["acceptance_policy"]` with exact acceptance-split totals, validated by `validate_source_spec()` and bound by `EXPECTED_SOURCE_SPEC_SHA256`.

- [x] **Step 1: Write the policy-presence and derived-count test**

Add this exact policy fixture near the version-1 source tests:

```python
SYNTHETIC_ACCEPTANCE_POLICY = {
    "exact_placement": {"correct": 384, "total": 384},
    "negative_no_board_rate": {"correct": 28, "total": 28},
    "positive_false_no_board_rate": {"correct": 0, "total": 384},
    "execution_error_count": 0,
}


def test_version_1_acceptance_policy_matches_generated_acceptance_split():
    spec = _source_spec()
    assert spec.get("acceptance_policy") == SYNTHETIC_ACCEPTANCE_POLICY

    acceptance = [case for case in _case_plan() if case["split"] == "acceptance"]
    positives = sum(case["board_present"] for case in acceptance)
    negatives = len(acceptance) - positives
    assert (len(acceptance), positives, negatives) == (412, 384, 28)
    assert {case["view"] for case in acceptance if case["board_present"]} == {
        "white_bottom",
        "black_bottom",
    }
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_board_recognition_benchmark.py::test_version_1_acceptance_policy_matches_generated_acceptance_split
```

Expected: FAIL because `acceptance_policy` is absent.

- [x] **Step 3: Add the frozen policy and corrected limitation**

Add this top-level source field to `benchmarks/board_recognition_v1.json`:

```json
"acceptance_policy": {
  "exact_placement": {"correct": 384, "total": 384},
  "negative_no_board_rate": {"correct": 28, "total": 28},
  "positive_false_no_board_rate": {"correct": 0, "total": 384},
  "execution_error_count": 0
}
```

Replace only the limitation that says P2-01 remains open with:

```text
This gate enforces generated-domain regression correctness only.
```

Update `EXPECTED_SOURCE_SPEC_SHA256` to
`524e3ee10e93624b295ad120f2f510a64979e6a3fc35895aca2c4884e14b305a`
and update `_EXPECTED_LIMITATIONS` with the exact corrected final string. Do not
add a dedicated policy check yet. Run the policy/count test plus
`test_version_1_source_spec_is_semantically_valid`; both must pass without
changing the case plan.

- [x] **Step 4: Add the explicit mutation regression and verify RED**

Add:

```python
def test_source_spec_rejects_acceptance_policy_mutation():
    _validate_mutation(
        ("acceptance_policy", "exact_placement", "correct"),
        383,
        "acceptance_policy",
    )
```

Run:

```bash
.venv/bin/python -m pytest -q tests/test_board_recognition_benchmark.py::test_source_spec_rejects_acceptance_policy_mutation
```

Expected: FAIL because validation reaches only the generic source-digest mismatch rather than the policy boundary.

- [x] **Step 5: Lock the policy before the semantic-digest fallback**

In `benchmarks/board_recognition.py`:

```python
_EXPECTED_ACCEPTANCE_POLICY = {
    "exact_placement": {"correct": 384, "total": 384},
    "negative_no_board_rate": {"correct": 28, "total": 28},
    "positive_false_no_board_rate": {"correct": 0, "total": 384},
    "execution_error_count": 0,
}
```

Add this check immediately after limitations validation:

```python
if spec.get("acceptance_policy") != _EXPECTED_ACCEPTANCE_POLICY:
    raise BenchmarkValidationError("acceptance_policy differs")
```

Confirm the digest independently:

```bash
.venv/bin/python -c 'from pathlib import Path; from benchmarks.board_recognition import load_json, source_spec_sha256; print(source_spec_sha256(load_json(Path("benchmarks/board_recognition_v1.json"))))'
```

Expected: `524e3ee10e93624b295ad120f2f510a64979e6a3fc35895aca2c4884e14b305a`.

- [x] **Step 6: Run Task 1 GREEN checks**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_board_recognition_benchmark.py -k 'source_spec or acceptance_policy or case_plan'
.venv/bin/python -m scripts.data.generate_board_recognition_benchmark --preflight
```

Expected: selected tests pass; preflight prints
`cases=568 positive=528 negative=40`.

Write `.superpowers/sdd/p2-01-acceptance-task-1-report.md` with exact RED/GREEN
commands, results, changed paths, and self-review. Do not stage or commit.

---

### Task 2: Enforce synthetic acceptance in report and exit status

**Files:**
- Modify: `scripts/validate/evaluate_board_recognition_benchmark.py`
- Modify: `tests/test_board_recognition_benchmark.py`

**Interfaces:**
- Consumes: `source["acceptance_policy"]`, `scores["groups"]["split"]["acceptance"]`, `scores["overall"]["execution_error_count"]`, and existing `evaluate_captured_acceptance(policy, scores)`.
- Produces: a canonical `report["acceptance"]` for both claims and exit `0` only when every requirement passes.

- [x] **Step 1: Strengthen the existing synthetic CLI regression**

Change `test_evaluator_cli_verifies_before_loading_and_publishes_complete_report`
to use two negative cases, one per split:

```python
manifest["cases"] = [
    {
        "id": "acceptance",
        "path": "images/acceptance.png",
        "board_present": False,
        "split": "acceptance",
    },
    {
        "id": "development",
        "path": "images/development.png",
        "board_present": False,
        "split": "development",
    },
]
source["acceptance_policy"] = {
    "exact_placement": {"correct": 0, "total": 0},
    "negative_no_board_rate": {"correct": 1, "total": 1},
    "positive_false_no_board_rate": {"correct": 0, "total": 0},
    "execution_error_count": 0,
}


class RecognitionSuccess:
    source_placement = "8/8/8/8/8/8/8/8"
```

Create distinct one-pixel PNG bytes for the two case IDs and keep `Picture.pixel`
as the discriminator. Parameterize the helper outcome as:

```python
@pytest.mark.parametrize(
    ("outcome", "expected_status", "failed_metric"),
    [
        ("no_board", 0, None),
        ("wrong_acceptance", 1, "negative_no_board_rate"),
        ("development_error", 1, "execution_error_count"),
    ],
)
```

Make the helper produce these outcomes:

- `no_board`: return `RecognitionFailure()` for both cases;
- `wrong_acceptance`: return `RecognitionSuccess()` only for the acceptance
  pixel and `RecognitionFailure()` for the development pixel;
- `development_error`: return `RecognitionFailure()` for the acceptance pixel
  and raise `RuntimeError("model exploded")` only for the development pixel.

Have `load_images` return both buffers keyed by case ID. Require every report to
contain `acceptance`; assert its only failed requirement is `failed_metric` (or
that none failed for `no_board`) and that the report is already complete and
canonical when `main()` returns either status:

```python
assert evaluator.main(args) == expected_status
report = json.loads(report_path.read_bytes())
assert set(report) == {
    "benchmark_version",
    "claim",
    "base_position_count",
    "limitations",
    "manifest_sha256",
    "checkpoints",
    "device",
    "inference_fingerprint",
    "predictions",
    "scores",
    "acceptance",
}
source_report_fields = (
    "benchmark_version",
    "claim",
    "base_position_count",
    "limitations",
)
assert {key: report[key] for key in source_report_fields} == {
    key: source[key] for key in source_report_fields
}
assert [
    item["metric"]
    for item in report["acceptance"]["requirements"]
    if not item["passed"]
] == ([] if failed_metric is None else [failed_metric])
assert report["acceptance"]["passed"] is (failed_metric is None)
assert report["scores"]["overall"]["execution_error_count"] == int(
    outcome == "development_error"
)
assert report_path.read_bytes() == canonical_json_bytes(report)
```

Replace the old four-key-report comparison to the whole `source` with the
two-sided `source_report_fields` projection above, because `source` now also
contains `acceptance_policy`. Remove every remaining reference to the old
`execution_error` parameter.

- [x] **Step 2: Run the strengthened CLI test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_board_recognition_benchmark.py -k 'verifies_before_loading'
```

Expected: the current evaluator omits `report["acceptance"]`; it also falsely
returns zero for `wrong_acceptance`. The `development_error` path proves that a
development-only execution error must remain fatal even though acceptance-split
accuracy is perfect.

- [x] **Step 3: Implement the minimal shared-policy call**

After scoring and checkpoint hash rechecks, retain captured freeze revalidation
as-is. Then build the comparator input:

```python
policy_scores = (
    scores["overall"]
    if captured
    else scores["groups"]["split"]["acceptance"]
)
policy_scores = dict(policy_scores)
policy_scores["execution_error_count"] = scores["overall"][
    "execution_error_count"
]
acceptance = evaluate_captured_acceptance(
    source["acceptance_policy"], {"overall": policy_scores}
)
```

Always add:

```python
report["acceptance"] = acceptance
```

Keep `freeze_sha256` captured-only, write the complete report as today, and
replace claim-specific status returns with:

```python
return int(not acceptance["passed"])
```

- [x] **Step 4: Run Task 2 GREEN checks**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_board_recognition_benchmark.py -k 'captured or acceptance or verifies_before_loading'
.venv/bin/python -m pytest -q tests/test_online_board_screenshot_benchmark.py tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py
.venv/bin/python -m compileall -q benchmarks scripts tests
```

Expected: all commands exit zero. Existing captured freeze and acceptance
semantics remain unchanged.

Write `.superpowers/sdd/p2-01-acceptance-task-2-report.md` with exact RED/GREEN
commands, results, changed paths, and self-review. Do not stage or commit.

---

### Task 3: Refresh the generated trust anchor and close P2-01 documentation

**Files:**
- Generated/modify: `datasets/board_recognition_benchmark/v1/manifest.json`
- Delete ignored drift: `datasets/board_recognition_benchmark/v1/.DS_Store`
- Modify: `benchmarks/board_recognition_v1_manifest.sha256`
- Modify: `README.md`
- Modify: `BOARD_RECOGNITION_AUDIT.tmp.md`

**Interfaces:**
- Consumes: the reviewed generator, updated source semantic digest, existing frozen PNG corpus, and anchored historical post-P2-05 report.
- Produces: an anchored manifest bound to the new policy plus accurate P2-01 closure evidence.

- [x] **Step 1: Snapshot exact current generated state**

Confirm the only extra corpus entry is the known file:

```bash
find datasets/board_recognition_benchmark/v1 -name '.DS_Store' -print
```

Expected exactly:

```text
datasets/board_recognition_benchmark/v1/.DS_Store
```

Copy the current ignored manifest and tracked digest to controller-owned rollback
files under `.superpowers/sdd/p2-01-acceptance-backup/`. Do not copy PNGs.

```bash
test ! -e .superpowers/sdd/p2-01-acceptance-backup
mkdir -p .superpowers/sdd/p2-01-acceptance-backup
cp datasets/board_recognition_benchmark/v1/manifest.json .superpowers/sdd/p2-01-acceptance-backup/manifest.json
cp benchmarks/board_recognition_v1_manifest.sha256 .superpowers/sdd/p2-01-acceptance-backup/board_recognition_v1_manifest.sha256
```

- [x] **Step 2: Generate two disposable corpora**

Confirm the task-owned path is absent, create it, and generate normal and reverse
children:

```bash
test ! -e /private/tmp/chessml-p2-01-acceptance-20260720
mkdir -p /private/tmp/chessml-p2-01-acceptance-20260720
.venv/bin/python -m scripts.data.generate_board_recognition_benchmark --write /private/tmp/chessml-p2-01-acceptance-20260720/normal
.venv/bin/python -m scripts.data.generate_board_recognition_benchmark --write /private/tmp/chessml-p2-01-acceptance-20260720/reverse --case-order reverse
.venv/bin/python -m scripts.data.generate_board_recognition_benchmark --compare /private/tmp/chessml-p2-01-acceptance-20260720/normal /private/tmp/chessml-p2-01-acceptance-20260720/reverse
cmp /private/tmp/chessml-p2-01-acceptance-20260720/normal/manifest.json /private/tmp/chessml-p2-01-acceptance-20260720/reverse/manifest.json
```

Expected: both writes report 568 cases, compare reports 568 cases/569 files, and
`cmp` exits zero.

- [x] **Step 3: Prove all PNG bytes are unchanged**

Run a recursive byte comparison that excludes only `manifest.json` and
`.DS_Store`:

```bash
diff -qr -x manifest.json -x .DS_Store datasets/board_recognition_benchmark/v1 /private/tmp/chessml-p2-01-acceptance-20260720/normal
```

Expected: exit zero with no output. Stop without changing the final corpus if
any PNG differs.

- [x] **Step 4: Publish only generated metadata and remove exact drift**

Compute SHA-256 of the generated normal manifest and require
`8a16341a7739870f56b69e1ac066f5a6de0785174e0ba6ed9d06e0faaa9806b3`.
Use `apply_patch` to set `benchmarks/board_recognition_v1_manifest.sha256` to
exactly:

```bash
shasum -a 256 /private/tmp/chessml-p2-01-acceptance-20260720/normal/manifest.json
```

```text
8a16341a7739870f56b69e1ac066f5a6de0785174e0ba6ed9d06e0faaa9806b3  manifest.json
```

Copy
`/private/tmp/chessml-p2-01-acceptance-20260720/normal/manifest.json` over only
`datasets/board_recognition_benchmark/v1/manifest.json`, then delete exactly
`datasets/board_recognition_benchmark/v1/.DS_Store`. Do not replace or rewrite
the `images/` directory.

```bash
cp /private/tmp/chessml-p2-01-acceptance-20260720/normal/manifest.json datasets/board_recognition_benchmark/v1/manifest.json
rm datasets/board_recognition_benchmark/v1/.DS_Store
```

- [x] **Step 5: Verify the refreshed corpus and historical failure**

Run:

```bash
.venv/bin/python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256
```

Expected: `verified cases=568`.

Run a read-only Python check that loads the final manifest and existing
`output/board_recognition_online-render-v1_post-p2-05_2a1e5e0_mps.json`, filters
both cases and predictions to `split == "acceptance"`, calls
`score_predictions()`, replaces only the comparator input's
`execution_error_count` with the complete report's overall count, and calls
`evaluate_captured_acceptance()` with the source policy. Assert and print:

```text
exact=245/384
no_board=0/28
false_no_board=0/384
execution_errors=0
acceptance_passed=False
```

Do not load a model or create a new inference report.

- [x] **Step 6: Update README and only P2-01/order audit text**

README must state:

- `online-render-v1` is now an enforceable generated-domain gate;
- its perfect acceptance policy and acceptance-split scope;
- overall execution errors remain fatal;
- the historical post-P2-05 predictions fail at the exact values above; and
- passing it does not prove checkpoint holdout or unseen-site accuracy.

In `BOARD_RECOGNITION_AUDIT.tmp.md`:

- mark P2-01 `resolved 2026-07-20` for the generated online-board domain;
- record source/manifest policy binding, exact gate requirements, and the
  historical failed result without claiming fresh inference;
- update the prominent required order and critical chain so P2-01 no longer
  blocks starting P1-07, while P1-07 itself remains open; and
- use `rg -n 'P2-01|real-screenshot|real-board' BOARD_RECOGNITION_AUDIT.tmp.md`
  to correct every stale P2-01-open cross-reference, changing only the clauses
  needed to state the generated-domain closure and its unseen-site limitation;
  and
- leave every P2-11 line untouched.

- [x] **Step 7: Run final verification without checkpoint loading**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_online_board_screenshot_benchmark.py tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py
.venv/bin/python -m pytest -q --ignore=tests/test_board_recognition_integration.py
.venv/bin/python -m compileall -q benchmarks chessml scripts tests
.venv/bin/python -m scripts.data.generate_board_recognition_benchmark --preflight
.venv/bin/python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256
git diff --check
git diff --cached --quiet
git diff -- scripts/data/download_piece_sets.py
```

Expected: all tests, compilation, preflight, corpus verification, whitespace,
empty-index, and P2-11 scope checks pass. The final command has no output.

Run `git diff --no-index --check /dev/null` for the audit, design, and plan;
exit 1 with no diagnostic means an untracked file differs cleanly.

Write `.superpowers/sdd/p2-01-acceptance-task-3-report.md` with generated
digests, exact comparisons, tests, documentation scope, and self-review. Retain
rollback files until independent final verification succeeds. Do not stage or
commit.
