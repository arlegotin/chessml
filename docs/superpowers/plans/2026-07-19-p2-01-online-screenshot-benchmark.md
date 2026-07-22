# P2-01 Online Screenshot Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task, but replace its commit-based transport with shared-working-tree snapshots and read-only diff review because this task forbids staging and commits. Steps use checkbox (`- [ ]`) syntax for tracking.

> **EXECUTION STOP — 2026-07-19:** Tasks 1–4 are implemented, reviewed, and
> repository-tested. Task 5 has the exact 13-file primary-source rights bundle
> and attribution, but the permitted in-app browser was unavailable for the
> required live five-field Lichess/PyChess asset observation. Consequently
> `benchmarks/board_recognition_online_screenshot_v1_RIGHTS.json` is
> intentionally absent. Do not capture, label, freeze, or load a checkpoint
> until a permitted fresh anonymous browser selects Chessnut/brown and
> Firi/brown through the normal UI, records both five-field identities, writes
> canonical `RIGHTS.json`, and the complete Task 5 construction/review gate is
> clean. Never infer or placeholder those identity fields from source alone.

**Goal:** Add a durable, manually labeled 40-case online-screenshot acceptance benchmark whose exact inputs are verified, whose wrong predictions fail the evaluator, and whose construction is proven before any screenshot is frozen or checkpoint is loaded.

**Architecture:** Keep the synthetic `online-render-v1` path intact. Add one focused captured-benchmark module for the `captured_online_screenshot_acceptance` source/evidence contract, minimal-PNG validation, verified byte snapshots, strict acceptance, and freeze commitments; add one thin CLI for capture-plan assembly/freezing; reuse the existing manifest verifier, scorer, evaluator, and report writer. Capture Lichess and PyChess through their anonymous live editors only after the construction code and tests pass independent review.

**Tech Stack:** Python 3.14, Pillow, python-chess, stdlib `hashlib/json/os/pathlib/struct/zlib`, pytest, the existing benchmark/evaluator APIs, and the permitted anonymous in-app browser capture surface.

## Global Constraints

- Follow root `AGENTS.md`; no nested instruction file exists for the target paths.
- Do not stage or commit any file. The user will supply the durable git-history anchor later.
- Preserve both untracked audit/research files and every unrelated working-tree change.
- Do not train, calibrate, select, or replace a checkpoint. Run the three trusted checkpoints once only after the corpus is frozen and independently attested.
- Do not capture final benchmark images until Tasks 1-4 pass focused tests and independent construction review.
- Use exactly 40 original full-viewport RGB PNGs: 32 positives and 8 negatives.
- Use only Lichess `chessnut`/brown and PyChess `firi`/brown through their normal anonymous settings UI.
- Use `desktop_clean=1440x900x1` and `compact_scaled=800x600x1`; never crop, resize, re-encode, blur, inject CSS, or otherwise transform screenshot pixels.
- Retain exact PNGs, receipts, both blind annotations, review, attribution, canonical rights record, licenses, source spec, manifest, manifest digest, and freeze commitment as trackable artifacts.
- Require exact 32/32 placement, 8/8 negative `NO_BOARD`, 0/32 positive `NO_BOARD`, and zero execution errors. A current-checkpoint failure is evidence, not permission to weaken this policy.
- Keep reports and the reviewer-hidden controlled capture plan under ignored `output/`.
- A separate worktree is intentionally not used: the user requires an uncommitted shared-workspace result, and worktree integration would require commits while hiding the active untracked audit files from workers.
- Disable the subagent skill's commit, commit-range review-package, durable
  ledger, branch-finishing, and merge mechanics. Each implementer must make no
  commit and report its exact touched paths. The controller records a
  pre-task `git status --short` and copies the task's pre-edit file bytes into
  an ignored `.superpowers/sdd/<task>/before/` snapshot; task review receives a
  `git diff --no-index` package between that snapshot and the working tree.
  Delete only those controller-owned review snapshots after final review.

## File map

- Create `benchmarks/captured_board_recognition.py`: captured source/evidence validation, capture-plan construction, minimal PNG parser, acceptance evaluation, and freeze creation/verification.
- Modify `benchmarks/board_recognition.py`: dispatch the captured claim, group scores by `site` and `viewport`, and load hash-verified immutable byte snapshots without changing synthetic semantics.
- Create `scripts/data/freeze_online_board_screenshot_benchmark.py`: thin CLI over the captured module.
- Modify `scripts/validate/evaluate_board_recognition_benchmark.py`: consume verified byte buffers, require/recheck the captured freeze commitment, and enforce captured acceptance.
- Create `tests/test_online_board_screenshot_benchmark.py`: captured contract, evidence, PNG, plan, freezer, and commitment regressions.
- Modify `tests/test_board_recognition_benchmark.py`: evaluator byte-buffer and acceptance/exit regressions plus unchanged synthetic behavior.
- Modify `.gitignore`: a narrow exception for only the frozen screenshot PNG directory.
- Create the tracked `benchmarks/board_recognition_online_screenshot_v1*` evidence/corpus files, including the canonical rights record, and `benchmarks/licenses/*` license texts only after construction review.
- Modify `README.md` and `BOARD_RECOGNITION_AUDIT.tmp.md` after the verified baseline.

---

### Task 1: Captured semantic source contract and scorer dispatch

**Files:**
- Create: `benchmarks/captured_board_recognition.py`
- Modify: `benchmarks/board_recognition.py`
- Create: `tests/test_online_board_screenshot_benchmark.py`

**Interfaces:**
- Produces: `CAPTURED_CLAIM`, `CAPTURED_VERSION`, `CAPTURED_ACCEPTANCE_POLICY`, `build_capture_plan() -> dict`, `validate_captured_source_spec(spec: dict, repo_root: Path, *, verify_local_assets: bool) -> None`, `build_captured_case_plan(spec: dict) -> list[dict]`, and `evaluate_captured_acceptance(policy: dict, scores: dict) -> dict`.
- `benchmarks.board_recognition.validate_source_spec()` and `build_case_plan()` dispatch only when `spec["claim"] == CAPTURED_CLAIM`.

- [ ] **Step 1: Write the captured source tests first**

Add fixtures that derive a source specimen from `build_capture_plan()` and replace only capture-dependent hashes/geometry with valid test values. Lock these exact invariants in separate tests:

```python
def test_capture_plan_is_exact_deterministic_40_case_matrix():
    first = captured.build_capture_plan()
    second = captured.build_capture_plan()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert len(first["cases"]) == 40
    assert Counter(case["board_present"] for case in first["cases"]) == {
        True: 32,
        False: 8,
    }
    positives = [case for case in first["cases"] if case["board_present"]]
    assert {
        (case["site"], case["base_position"], case["view"], case["viewport"])
        for case in positives
    } == set(product(
        ("lichess", "pychess"),
        ("start", "najdorf", "nimzo", "pawn_endgame"),
        ("white_bottom", "black_bottom"),
        ("desktop_clean", "compact_scaled"),
    ))


def test_captured_source_is_explicit_and_dispatches_without_synthetic_assets(valid_spec):
    validate_source_spec(valid_spec, Path("/does/not/exist"), verify_local_assets=False)
    assert build_case_plan(valid_spec) == sorted(valid_spec["cases"], key=itemgetter("id"))


def test_captured_source_rejects_matrix_orientation_and_field_leakage(valid_spec):
    mutated = copy.deepcopy(valid_spec)
    positive = next(case for case in mutated["cases"] if case["board_present"])
    positive["source_placement"] = positive["canonical_placement"]
    positive["view"] = "black_bottom"
    with pytest.raises(BenchmarkValidationError, match="source placement"):
        validate_source_spec(mutated, ROOT, verify_local_assets=False)

    mutated = copy.deepcopy(valid_spec)
    negative = next(case for case in mutated["cases"] if not case["board_present"])
    negative["piece_set"] = "lichess_chessnut"
    with pytest.raises(BenchmarkValidationError, match="negative fields"):
        validate_source_spec(mutated, ROOT, verify_local_assets=False)
```

Parametrize mutations for schema/version/claim/counts, missing/unknown keys, unsafe IDs/paths/URLs, wrong host/path/query, invalid/legal-noncanonical FEN, duplicate/rotated positions, viewport dimensions, style/site mapping, board rectangles, split/suite/layout/channel/mode, positive/negative fields, and all matrix cells.

- [ ] **Step 2: Run the new tests and record the expected red state**

Run:

```bash
uv run --locked python -m pytest -q tests/test_online_board_screenshot_benchmark.py \
  -k 'capture_plan or captured_source or captured_acceptance or captured_scores_group'
```

Expected: collection fails because `benchmarks.captured_board_recognition` does not exist.

- [ ] **Step 3: Implement the exact capture constants and plan**

Use immutable constants for the four full FENs, two sites, two viewports, site styles, allowed editor/negative paths, counts, and strict acceptance policy. IDs must be opaque but deterministic:

```python
CAPTURED_VERSION = "online-screenshot-v1"
CAPTURED_CLAIM = "captured_online_screenshot_acceptance"
POSITIONS = {
    "start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "najdorf": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "nimzo": "rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7",
    "pawn_endgame": "8/5k2/8/3p4/3P4/4K3/8/8 w - - 0 1",
}
SITES = {
    "lichess": {
        "editor_url": "https://lichess.org/editor",
        "negative_urls": (
            "https://lichess.org/terms-of-service",
            "https://lichess.org/privacy",
        ),
        "piece_set": "lichess_chessnut",
        "board_theme": "brown",
    },
    "pychess": {
        "editor_url": "https://www.pychess.org/editor/chess",
        "negative_urls": (
            "https://www.pychess.org/terms",
            "https://www.pychess.org/privacy",
        ),
        "piece_set": "pychess_firi",
        "board_theme": "brown",
    },
}
VIEWPORTS = {
    "desktop_clean": {"width": 1440, "height": 900, "dpr": 1, "quality": "clean"},
    "compact_scaled": {"width": 800, "height": 600, "dpr": 1, "quality": "browser_scaled"},
}
CAPTURED_ACCEPTANCE_POLICY = {
    "exact_placement": {"correct": 32, "total": 32},
    "negative_no_board_rate": {"correct": 8, "total": 8},
    "positive_false_no_board_rate": {"correct": 0, "total": 32},
    "execution_error_count": 0,
}


def _opaque_id(index: int) -> str:
    value = hashlib.sha256(f"{CAPTURED_VERSION}:{index:02d}".encode("ascii"))
    return f"capture_{value.hexdigest()[:16]}"
```

The exact top-level source keys are `schema_version`, `benchmark_version`,
`claim`, `base_position_count`, `limitations`, `expected_counts`, `acceptance_policy`,
`capture_protocol`, `allowed_sources`, `capture_conditions`, `evidence`, and
`cases`. Evidence keys are `receipts`, `annotations_a`, `annotations_b`,
`review`, `attribution`, `rights`, and `licenses`; each singleton is exactly `{path,
sha256}` and `licenses` is a sorted nonempty list of those records. Every case
has exactly `id`, `path`, `suite`, `split`, `board_present`, `site`, `viewport`,
`layout`, `quality`, `channel_mode`, `image_mode`, `image_size`, `source_url`,
and `page_kind`. Positive-only keys are `base_position`, `fen`,
`canonical_placement`, `source_placement`, `view`, `piece_set`, `board_theme`,
and `board_rect`; negative-only key is `negative_page`.

Use these exact remaining values:

```python
LIMITATIONS = [
    "This suite covers anonymous Lichess and PyChess pages only.",
    "The compact condition is native responsive browser scaling, not postprocessing.",
    "Empty and invalid complete boards are outside online-screenshot-v1.",
    "No benchmark image or label may be used for training, calibration, checkpoint selection, or threshold selection.",
]
EXPECTED_COUNTS = {"all": 40, "positive": 32, "negative": 8}
EVIDENCE_PATHS = {
    "receipts": "benchmarks/board_recognition_online_screenshot_v1_capture_receipts.json",
    "annotations_a": "benchmarks/board_recognition_online_screenshot_v1_annotations_a.json",
    "annotations_b": "benchmarks/board_recognition_online_screenshot_v1_annotations_b.json",
    "review": "benchmarks/board_recognition_online_screenshot_v1_review.json",
    "attribution": "benchmarks/board_recognition_online_screenshot_v1_ATTRIBUTION.md",
    "rights": "benchmarks/board_recognition_online_screenshot_v1_RIGHTS.json",
}
CAPTURE_PROTOCOL = {
    "browser_context": "fresh_anonymous_isolated",
    "capture_mode": "visible_viewport",
    "full_page": False,
    "manual_sequential": True,
    "max_concurrency": 1,
    "automated_retry": False,
    "postprocessing": "none",
    "dom_reads": ["controlled_fen", "orientation", "style", "receipt_metadata"],
}
LICENSE_PATHS = [
    "benchmarks/licenses/AGPL-3.0-or-later.txt",
    "benchmarks/licenses/Apache-2.0.txt",
    "benchmarks/licenses/CC-BY-4.0.txt",
    "benchmarks/licenses/GPL-3.0-or-later.txt",
    "benchmarks/licenses/chessgroundx-GPL-3.0.txt",
    "benchmarks/licenses/lichess-chessnut-LICENSE.txt",
    "benchmarks/licenses/lichess-lila-COPYING.md",
    "benchmarks/licenses/lichess-privacy-2026-07-19.txt",
    "benchmarks/licenses/lichess-terms-2026-07-19.txt",
    "benchmarks/licenses/pychess-firi-LICENSE.txt",
    "benchmarks/licenses/pychess-privacy-2026-05-22.md",
    "benchmarks/licenses/pychess-terms-2026-05-22.md",
    "benchmarks/licenses/pychess-variants-COPYING.md",
]
```

`allowed_sources` is exactly the JSON-native expansion of `SITES`, with tuple
values serialized as lists. `capture_conditions` is exactly the JSON-native
expansion of `VIEWPORTS`. Placeholder evidence paths are the canonical tracked
paths declared by Tasks 5 and 7 and every placeholder SHA-256 is 64 zeroes;
`licenses` contains the sorted `LICENSE_PATHS` records. Positive placeholder
`board_rect` is `[0, 0, 0, 0]` and is the only case field replaced after
capture. `fen` is the literal full `POSITIONS[base_position]` value and
`canonical_placement` is its placement field. Final evidence assembly replaces
every zero evidence hash and every positive placeholder rectangle; it changes
no other semantic field.

`build_capture_plan()` emits canonical semantic capture instructions in a stable site/positive-or-negative/position/view/viewport order. Positive controlled inputs include full FEN and expected view; negative inputs include only the exact public page path. The plan is ignored evidence and must not be copied into blind review packets.

- [ ] **Step 4: Implement strict source validation and captured case planning**

Validate exact top-level/source-policy values and exact per-case key sets. Positive source placement must be derived with the existing `rotate_placement()` helper. Validate legal canonical boards with `chess.Board(fen).is_valid()`. Require one case for every locked matrix cell, exact style mapping (`lichess_chessnut`/`pychess_firi`), and exact quality/size mapping.

Keep the public boundary small:

```python
def build_captured_case_plan(spec: dict) -> list[dict]:
    validate_captured_source_spec(spec, Path(), verify_local_assets=False)
    return sorted(spec["cases"], key=lambda case: case["id"])
```

- [ ] **Step 5: Add claim dispatch and score groups without changing synthetic constants**

In `benchmarks/board_recognition.py`, add `site` and `viewport` to `GROUP_FIELDS`.
In `validate_source_spec()`, preserve the existing non-dictionary guard, then
dispatch before the synthetic schema/version checks:

```python
if spec.get("claim") == "captured_online_screenshot_acceptance":
    from benchmarks.captured_board_recognition import validate_captured_source_spec
    validate_captured_source_spec(spec, repo_root, verify_local_assets=verify_local_assets)
    return
```

In `build_case_plan()`, keep `validate_source_spec(spec, Path(),
verify_local_assets=False)` as its first statement. Only after that succeeds,
branch on `spec["claim"]`, locally import `build_captured_case_plan`, and return
it. All existing synthetic code remains byte-for-byte below that branch.

- [ ] **Step 6: Implement and test strict acceptance evaluation**

Return an auditable ordered result rather than a bare boolean:

```python
def evaluate_captured_acceptance(policy: dict, scores: dict) -> dict:
    overall = scores["overall"]
    actual = {
        metric: {
            "correct": overall[metric]["correct"],
            "total": overall[metric]["total"],
        }
        for metric in (
            "exact_placement",
            "negative_no_board_rate",
            "positive_false_no_board_rate",
        )
    } | {
        "execution_error_count": overall["execution_error_count"],
    }
    requirements = [
        {"metric": metric, "required": required, "actual": actual[metric],
         "passed": actual[metric] == required}
        for metric, required in policy.items()
    ]
    return {
        "passed": all(item["passed"] for item in requirements),
        "requirements": requirements,
    }
```

Tests must cover a perfect result and one failure for each of the four requirements.

Add `test_captured_scores_group_by_site_and_viewport`, asserting the exact two
keys under each group and the correct 20/20 and 16/16 case totals.

- [ ] **Step 7: Run focused and synthetic-regression tests**

Run:

```bash
uv run --locked python -m pytest -q \
  tests/test_online_board_screenshot_benchmark.py \
  tests/test_board_recognition_benchmark.py \
  tests/test_online_board_benchmark_generator.py
```

Expected: all pass; the existing 568-case synthetic plan and source digest tests remain unchanged.

- [ ] **Step 8: Request an independent contract review**

Reviewer must check exact matrix coverage, oracle independence, strict policy, no synthetic-path weakening, no new dependency, and Ponytail minimality. Resolve every Critical/Important finding before Task 2.

---

### Task 2: Evidence binding, minimal PNG parsing, and exact verified input bytes

**Files:**
- Modify: `benchmarks/captured_board_recognition.py`
- Modify: `benchmarks/board_recognition.py`
- Modify: `tests/test_online_board_screenshot_benchmark.py`

**Interfaces:**
- Produces: `validate_captured_evidence(spec: dict, repo_root: Path) -> dict`, `validate_browser_png(encoded: bytes, *, mode: str, size: Sequence[int]) -> str`, and shared `benchmarks.board_recognition.load_verified_image_bytes(dataset_dir: Path, cases: Sequence[dict], *, require_minimal_png: bool) -> dict[str, bytes]`.
- `validate_browser_png()` returns the decoded pixel SHA-256.

- [ ] **Step 1: Add red tests for evidence and byte-level PNG validation**

Build RGB PNG bytes with Pillow, then parse/rebuild chunks in the test using `struct` and `zlib.crc32`. Lock these cases:

```python
def test_browser_png_accepts_only_ihdr_contiguous_idat_iend(rgb_png):
    assert captured.validate_browser_png(rgb_png, mode="RGB", size=(8, 8))


@pytest.mark.parametrize("chunk_type", [b"tEXt", b"eXIf", b"iCCP", b"vpAg"])
def test_browser_png_rejects_known_and_unknown_ancillary_chunks(rgb_png, chunk_type):
    mutated = insert_valid_crc_chunk_before_iend(rgb_png, chunk_type, b"sensitive")
    with pytest.raises(BenchmarkValidationError, match="PNG chunk sequence"):
        captured.validate_browser_png(mutated, mode="RGB", size=(8, 8))
```

Also test invalid signature, bad length, bad CRC, duplicate/misordered IHDR, no/fragmented IDAT, nonzero IEND, trailing bytes, decode failure, RGBA/L mode, and wrong size.

Evidence tests must prove canonical JSON, exact path/hash inventory, safe repository-relative paths, no symlinks/hardlink identity aliases, exact receipt/annotation/review inventories, annotation agreement, privacy/full-resolution booleans, controlled-input agreement, attribution hash, and exact required license inventory.

- [ ] **Step 2: Run the tests and confirm they fail on missing APIs**

Run:

```bash
uv run --locked python -m pytest -q tests/test_online_board_screenshot_benchmark.py \
  -k 'browser_png or captured_evidence or verified_image_bytes or import_order'
```

Expect `AttributeError` for the new functions.

- [ ] **Step 3: Implement the byte-level PNG allowlist**

Parse from the 8-byte PNG signature to EOF. For every chunk, validate bounds and `zlib.crc32(chunk_type + payload)`. Accept exactly:

```text
IHDR, IDAT[, IDAT...], IEND
```

Require IHDR length 13, IEND length 0, at least one contiguous IDAT, and no bytes after IEND. Then call Pillow `verify()`, reopen from `BytesIO`, fully load, validate PNG/RGB/size, and compute the existing `pixel_sha256()` over the loaded image.

- [ ] **Step 4: Implement exact evidence validation**

The source spec's `evidence` object contains exact `{path, sha256}` records for receipts, annotations A/B, review, attribution, canonical rights record, plus a sorted list of required license records. Resolve each under `repo_root`, reject symlink components/non-regular files/identity aliases, compare raw file hashes, require canonical JSON for JSON evidence, and validate exact schemas. Return the parsed JSON objects under exact `receipts`, `annotations_a`, `annotations_b`, `review`, and `rights` keys; attribution and license files are hash-validated raw evidence only.

Annotation records must contain only opaque ID, `board_present`, optional `source_placement`, and all-true `full_resolution_pass`, `privacy_pass`, and `capture_condition_pass`. A/B must agree exactly on board presence and placement. Review bytes must bind both annotation hashes, exact IDs, review isolation strings, and controlled-input agreement.

Use these exact evidence shapes. Receipts are an object with only
`schema_version=1`, `benchmark_version`, and ID-sorted `captures`. Each capture
has common keys `id`, `png_sha256`, `captured_at_utc`, `final_url`, `title`,
`site`, `page_kind`, `viewport`, `viewport_size`, `dpr`, `scroll`,
`visibility_state`, `navigation_state`, `user_agent`, `platform`, and `language`. Positive-only
receipt keys are `fen`, `view`, `board_rect`, `piece_set`, `board_theme`,
`piece_asset_id`, `board_asset_id`, `piece_asset_url_prefix`,
`board_style_signature`, and `upstream_revision`; negative-only key is
`no_complete_board=true`. Each annotation object has only
`schema_version=1`, `benchmark_version`, `reviewer` (`A` or `B`),
`isolation="opaque_images_only"`, and ID-sorted `cases`. Review has only
`schema_version=1`, `benchmark_version`, `annotation_sha256` (exact `a`/`b`
keys), `case_ids`, `review_isolation` (exact two reviewer strings),
`a_b_agreement=true`, `controlled_input_agreement=true`, and
`model_output_used=false`.

`review_isolation` is exactly:

```python
[
    "reviewer_a_received_only_opaque_images_and_schema",
    "reviewer_b_received_only_opaque_images_and_schema",
]
```

The rights object has exact keys `schema_version=1`,
`benchmark_version=online-screenshot-v1`, `checked_at_utc`, and `sites`.
`sites` has exact `lichess` and `pychess` keys; each site has exactly
`piece_asset_id`, `board_asset_id`, `piece_asset_url_prefix`,
`board_style_signature`, `upstream_revision`, `project_copying`,
`piece_license`, `terms`, and `privacy`. The first five are nonempty strings;
the final four are exact `{path, sha256}` records. Require those four records
per site to identify members of the source's exact license inventory, require
their referenced hashes to match, and require every positive receipt's five
asset-identity strings to equal its site's rights record.
Receipts and annotations have exactly the same ID inventory as the 40 source
cases; each receipt's site/page/viewport/FEN/view/rectangle/style fields match
its source case, and A/B placement consensus equals each final positive
`source_placement`. The review's case IDs and annotation hashes bind those exact
bytes.

- [ ] **Step 5: Implement single-buffer verified image loading**

Before opening a case, walk the dataset without following links and require its
exact inventory to be `manifest.json`, the `images/` directory, and precisely
the case paths declared by the manifest; reject extra/missing files,
directories, symlinks, and identity aliases. This preserves the current
synthetic `verify_images=True` inventory semantics when the evaluator switches
to `verify_images=False` plus this loader. For each manifest case, validate the safe contained path, open with `os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))`, require a regular `fstat`, read to EOF, and retain that exact `bytes` object. Check the encoded hash and decoded pixel/mode/size from that buffer. When `require_minimal_png=True`, use a function-local import of `benchmarks.captured_board_recognition.validate_browser_png()` and call it; no module-level board-to-captured edge is allowed. Return an ID-keyed mapping. Never construct a later path-backed image from these cases. The evaluator uses this loader for both claims; only captured browser bytes receive the stricter chunk allowlist.

- [ ] **Step 6: Add race regressions**

Tests replace or unlink the pathname after `load_verified_image_bytes()` returns and assert that decoding the retained mapping still yields the original pixel. Another test mutates the file before loading and requires a hash failure. A symlink and a file swap before `os.open` must fail or produce bytes whose hash fails; neither may be inferred from the verified pathname. Exact-inventory regressions add one extra PNG, one extra directory/file, and one symlink entry and require rejection for both synthetic and captured mode. Two subprocess regressions import `benchmarks.board_recognition` first and `benchmarks.captured_board_recognition` first; both must exit zero.

- [ ] **Step 7: Run focused tests and independent security review**

Run the captured suite plus current manifest tamper tests. Reviewer checks chunk parsing, CRC/order, unknown-chunk rejection, evidence containment/identity, annotation non-tautology, and same-buffer semantics. Resolve all Critical/Important findings.

---

### Task 3: Transactional capture-plan, reviewed-source, freezer, and commitment CLI

**Files:**
- Modify: `benchmarks/captured_board_recognition.py`
- Create: `scripts/data/freeze_online_board_screenshot_benchmark.py`
- Modify: `tests/test_online_board_screenshot_benchmark.py`

**Interfaces:**
- Produces: `write_capture_plan(path: Path) -> dict`; `verify_capture_staging(capture_plan: dict, receipts_path: Path, capture_dir: Path) -> dict`; `assemble_reviewed_source(capture_plan_path: Path, receipts_path: Path, capture_dir: Path, annotations_a_path: Path, annotations_b_path: Path, attribution_path: Path, rights_path: Path, license_paths: Sequence[Path], review_path: Path, source_spec_path: Path, repo_root: Path) -> dict`; `freeze_captured_corpus(source_spec_path: Path, capture_dir: Path, destination: Path, manifest_digest_path: Path, freeze_digest_path: Path, repo_root: Path) -> dict`; and `verify_freeze_commitment(source_spec_path: Path, dataset_dir: Path, manifest_digest_path: Path, freeze_digest_path: Path, repo_root: Path) -> str`.
- CLI modes: `--write-plan`, `--verify-captures`, `--assemble-source`, `--freeze`, and `--verify`; every written output must be nonexistent.

- [ ] **Step 1: Add CLI and transaction failure tests first**

Test exact argument dependencies and all five successful modes (`--write-plan`, `--verify-captures`, `--assemble-source`, `--freeze`, and `--verify`). Inject failure at image copy, manifest write, digest write, freeze write, and rename; assert that each invocation removes only outputs it owns, leaves every pre-existing source/evidence input intact, leaves no owned destination/digest/freeze/source sibling, and preserves unrelated sentinels. Test destination races and symlinked parents.

Lock byte preservation explicitly:

```python
def test_freezer_preserves_every_png_byte(tmp_path, reviewed_fixture):
    result = captured.freeze_captured_corpus(**reviewed_fixture)
    for case in result["manifest"]["cases"]:
        source = reviewed_fixture["capture_dir"] / case["path"]
        frozen = reviewed_fixture["destination"] / case["path"]
        assert frozen.read_bytes() == source.read_bytes()
```

- [ ] **Step 2: Run new CLI/freezer tests and confirm the red state**

Run:

```bash
uv run --locked python -m pytest -q tests/test_online_board_screenshot_benchmark.py \
  -k 'write_capture_plan or verify_capture_staging or assemble_reviewed_source or freeze_captured_corpus or freeze_commitment or cli_mode'
```

Expect missing API/module failures.

- [ ] **Step 3: Implement safe canonical writers**

Reuse the existing owned-temp validation, but do not use ordinary `os.rename`
for publication because it can replace a target created after the pre-check.
For regular files, atomically publish with `os.link(owned_temp, destination)`
and then unlink the owned temp; `os.link` must fail if the destination exists.
For the corpus directory, add one private `_rename_directory_noreplace()` that
uses the platform's no-replace rename primitive (`renameatx_np` with
`RENAME_EXCL` on Darwin, `renameat2` with `RENAME_NOREPLACE` on Linux) through
stdlib `ctypes`, and raises a validation error on an unsupported platform.
Writers must reject an existing output, real/symlink identity aliases,
non-directory parents, and target appearance at publication. A callback-driven
racer creates each target immediately before publication; the operation must
fail, preserve the racer's bytes/directory, and remove only its owned temp.
Use the same regular-file publisher for evaluator reports in Task 4. Do not
introduce a generic filesystem abstraction.

- [ ] **Step 4: Implement reviewed source assembly**

`assemble_reviewed_source()` accepts the hidden capture plan, tracked receipt file, two tracked annotation files, attribution, canonical rights record, and exact license paths. It validates captures and both blind passes, validates the rights record and requires every positive receipt's asset identity to equal its site's record, creates canonical `review.json`, then creates the source spec from annotation consensus. It compares consensus to controlled input only after A/B agreement and refuses any mismatch. It hashes every evidence byte into the source spec and writes review/source atomically to nonexistent paths.

- [ ] **Step 5: Implement transactional byte-preserving freeze**

Validate the final source/evidence and exact capture inventory first. Open each
staging PNG once with no symlink following, read it into one retained `bytes`
object, and validate receipt hash, minimal PNG, mode, size, and pixel hash from
that object. Compute both encoded-file and decoded-pixel hashes for the 40
retained buffers and reject unless each set has exactly 40 members. Write those
same retained buffers (never reopen the source paths) into an owned staging
directory, write the existing manifest shape with the existing
`runtime_fingerprint()` field set, call the shared unanchored verifier, then
publish by one atomic no-replace directory rename. Write the external digest in
exact existing format:

```text
<64 lowercase hex>  manifest.json\n
```

The source, receipt, annotations, review, attribution, canonical rights record,
every license, manifest, digest, and every PNG become the exact sorted
commitment inventory. Write `<sha256>  <repo-relative-path>\n` lines; the
commitment does not list itself.

- [ ] **Step 6: Implement read-only freeze verification**

`verify_freeze_commitment()` derives the only allowed inventory from the source spec and dataset, requires canonical sorted unique lines, rejects unsafe/missing/extra/symlink/identity-aliased paths, hashes each byte, calls shared manifest/evidence/image verification, and returns the SHA-256 of the commitment file.

Regressions must provide (a) two byte-identical PNGs and (b) two differently encoded valid PNGs with identical decoded RGB pixels. Both staging verification and freezing reject before any destination appears.

Add a source-swap regression through an injected post-read callback: after all
source buffers are retained, replace one capture pathname with a different valid
PNG. The frozen file and manifest must still describe the reviewed retained
buffer, proving validation and writing use the same bytes.

- [ ] **Step 7: Implement the thin CLI**

The module parser must make mode requirements explicit and return zero only after each mode's full postcondition verifies. `--verify-captures` requires plan, receipts, and capture directory but no source spec. `--verify` requires source, dataset, manifest digest, and freeze commitment; `--freeze` additionally requires capture directory and nonexistent outputs.

Use these exact mutually exclusive modes/arguments and reject every missing or
mode-irrelevant argument:

```text
--write-plan PATH
--verify-captures --capture-plan PATH --receipts PATH --capture-dir PATH
--assemble-source --capture-plan PATH --receipts PATH --capture-dir PATH
  --annotations-a PATH --annotations-b PATH --attribution PATH --rights PATH
  --licenses-dir PATH --review-output PATH --source-output PATH
--freeze --source-spec PATH --capture-dir PATH --dataset PATH
  --manifest-digest PATH --freeze-digest PATH
--verify --source-spec PATH --dataset PATH --manifest-digest PATH
  --freeze-digest PATH
```

The CLI's repository root is `Path(__file__).resolve().parents[2]` and
`--licenses-dir` must resolve to that root's `benchmarks/licenses` directory;
the assembler consumes exactly the sorted `LICENSE_PATHS` inventory, not every
incidental file in that directory.

- [ ] **Step 8: Run CLI help, focused suite, and independent construction review**

Run:

```bash
uv run --locked python -m scripts.data.freeze_online_board_screenshot_benchmark --help
uv run --locked python -m pytest -q tests/test_online_board_screenshot_benchmark.py
```

Reviewer must specifically verify oracle assembly order, no model imports, exact byte preservation, failure cleanup ownership, no overwrite, commitment completeness, and that no capture has occurred yet.

---

### Task 4: Evaluator exact-input integration and enforceable captured exit

**Files:**
- Modify: `scripts/validate/evaluate_board_recognition_benchmark.py`
- Modify: `tests/test_board_recognition_benchmark.py`

**Interfaces:**
- `run_predictions(dataset_dir: Path, cases: Sequence[dict], helper: object, *, picture_cls: type, success_cls: type, failure_cls: type, image_bytes_by_id: Mapping[str, bytes] | None = None) -> list[dict]`.
- Captured CLI requires `--freeze-digest`; synthetic CLI rejects that argument when supplied and preserves existing behavior when it is omitted.

- [ ] **Step 1: Add failing evaluator tests**

Extend the existing CLI event-order test to require:

```text
source -> freeze-pre -> manifest -> verified-bytes -> model-load -> recognize
       -> freeze-post -> report
```

Add four captured score cases with one violated acceptance requirement each; every one writes a complete report and returns 1. A perfect captured report returns 0. The synthetic execution-error parametrization remains exactly 0/1 as before and has no `acceptance` field.

- [ ] **Step 2: Add the path-swap regression**

Supply retained PNG bytes and replace the on-disk case before recognition. The fake `Picture` must receive a Pillow image whose pixels equal the retained bytes, never a `Path`:

```python
class Picture:
    def __init__(self, value):
        assert isinstance(value, Image.Image)
        self.pixel = value.getpixel((0, 0))
```

Add `test_production_in_memory_picture_preserves_rgb_rgba_and_l_preprocessing`.
For one deterministic 8x8 PNG in each Pillow mode `RGB`, `RGBA`, and `L`, call
`run_predictions()` once through its legacy path input and once through
`image_bytes_by_id` using the real `chessml.data.images.picture.Picture`.
The helper records `picture.as_3_channels.cv2`; assert equal shape, dtype, and
`numpy.array_equal` pixels for both calls. This is an integration regression,
not a fake-`Picture` assertion.

- [ ] **Step 3: Run targeted tests and confirm current behavior fails**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py \
  -k 'captured or path_swap or production_in_memory_picture or cli_event_order'
```

Expect missing `--freeze-digest`, missing verified-byte loading, and false-zero captured acceptance.

- [ ] **Step 4: Implement in-memory picture construction**

When `image_bytes_by_id` is present, open `BytesIO`, load, copy the Pillow image, and pass that copy to `picture_cls`. Preserve path behavior only for direct legacy unit callers that omit the mapping; the production CLI supplies verified bytes for both benchmark claims.

- [ ] **Step 5: Integrate pre/post commitment verification and acceptance**

For captured sources: require the freeze argument and verify it before manifest/model work. For both claims, verify the manifest with `verify_images=False`, derive the report's manifest hash as `sha256(canonical_json_bytes(manifest)).hexdigest()` from the verified returned dictionary (the verifier already requires byte-identical canonical input), and load all exact case bytes before loading models; captured cases set `require_minimal_png=True`. Hash all three checkpoint files before `load_helper()` and retain those hashes for the report. After scoring, recompute every checkpoint hash and require equality with its retained pre-inference value. Then recompute the captured commitment before report writing and require equality with the pre-inference commitment. Any mismatch aborts before report publication. Add:

```python
acceptance = evaluate_captured_acceptance(source["acceptance_policy"], scores)
post_freeze_sha256 = verify_freeze_commitment(
    source_spec_path, dataset_dir, manifest_digest_path, freeze_digest_path, ROOT
)
if post_freeze_sha256 != freeze_sha256:
    raise BenchmarkValidationError("freeze commitment changed during inference")
report["freeze_sha256"] = freeze_sha256
report["acceptance"] = acceptance
```

Return `int(not acceptance["passed"])`. Synthetic sources retain `int(execution_error_count > 0)` and their existing report schema.

Add a mutation regression whose fake recognizer replaces one committed evidence file with another internally valid committed corpus between prediction and the post-check. It must raise `freeze commitment changed during inference` and leave the report nonexistent; restore the fixture in `finally`.

Add one checkpoint mutation regression: change a checkpoint after `load_helper`
and restore it in `finally`; the evaluator must raise `checkpoint changed during
inference` and leave the report nonexistent. Assert the report uses the retained
manifest/checkpoint hashes rather than reopening either provenance path after
inference.

- [ ] **Step 6: Run focused evaluator, captured, and synthetic tests**

Run both benchmark test files and the synthetic generator tests. Expected: all pass, including the old report/order/exit assertions after their deliberately scoped update.

- [ ] **Step 7: Request independent evaluator review**

Reviewer checks false-green elimination, report-before-exit behavior, exact byte reuse, verification before model loading, post-run commitment, and no synthetic compatibility regression.

---

### Task 5: Pre-capture full verification, licenses, and attribution

**Files:**
- Modify: `.gitignore`
- Create: `benchmarks/board_recognition_online_screenshot_v1_ATTRIBUTION.md`
- Create: `benchmarks/board_recognition_online_screenshot_v1_RIGHTS.json`
- Create: `benchmarks/licenses/AGPL-3.0-or-later.txt`
- Create: `benchmarks/licenses/GPL-3.0-or-later.txt`
- Create: `benchmarks/licenses/Apache-2.0.txt`
- Create: `benchmarks/licenses/CC-BY-4.0.txt`
- Create: `benchmarks/licenses/lichess-lila-COPYING.md`
- Create: `benchmarks/licenses/lichess-chessnut-LICENSE.txt`
- Create: `benchmarks/licenses/lichess-terms-2026-07-19.txt`
- Create: `benchmarks/licenses/lichess-privacy-2026-07-19.txt`
- Create: `benchmarks/licenses/pychess-variants-COPYING.md`
- Create: `benchmarks/licenses/pychess-firi-LICENSE.txt`
- Create: `benchmarks/licenses/pychess-terms-2026-05-22.md`
- Create: `benchmarks/licenses/pychess-privacy-2026-05-22.md`
- Create: `benchmarks/licenses/chessgroundx-GPL-3.0.txt`

**Interfaces:**
- The attribution file inventories every license path and SHA-256 and states that screenshot files retain their upstream terms rather than the repository root license.
- `RIGHTS.json` is canonical JSON and the assembler's only machine-readable
  asset/license identity source. Its exact keys are `schema_version=1`,
  `benchmark_version=online-screenshot-v1`, `checked_at_utc`, and `sites`.
  `sites` has exact `lichess` and `pychess` keys; each site has exactly
  `piece_asset_id`, `board_asset_id`, `piece_asset_url_prefix`,
  `board_style_signature`, `upstream_revision`, `project_copying`,
  `piece_license`, `terms`, and `privacy`. The final four fields are exact
  `{path, sha256}` records. The other five identity strings are nonempty,
  site-specific, and copied literally into every positive receipt for that
  site.

- [ ] **Step 1: Add only narrow corpus-artifact ignore exceptions**

Append rules that continue ignoring every other PNG and text file while exposing exactly the corpus PNG directory and the nine required `.txt` evidence files:

```gitignore
!benchmarks/board_recognition_online_screenshot_v1/
!benchmarks/board_recognition_online_screenshot_v1/images/
!benchmarks/board_recognition_online_screenshot_v1/images/*.png
!benchmarks/licenses/AGPL-3.0-or-later.txt
!benchmarks/licenses/GPL-3.0-or-later.txt
!benchmarks/licenses/Apache-2.0.txt
!benchmarks/licenses/CC-BY-4.0.txt
!benchmarks/licenses/lichess-chessnut-LICENSE.txt
!benchmarks/licenses/lichess-terms-2026-07-19.txt
!benchmarks/licenses/lichess-privacy-2026-07-19.txt
!benchmarks/licenses/pychess-firi-LICENSE.txt
!benchmarks/licenses/chessgroundx-GPL-3.0.txt
```

After creating them, run `git check-ignore -q <path>` for each exception and
require exit 1. Assert each appears in `git status --short
--untracked-files=all` without staging it.

- [ ] **Step 2: Freeze project-specific rights evidence from primary official sources**

Use GNU, Apache, and Creative Commons official canonical texts. Also retain commit/tag-pinned exact copies of lila `COPYING.md`, the Chessnut upstream license, PyChess Variants `static/COPYING.md`, PyChess's local Firi license, and Chessgroundx's GPL license. Retain the exact official Lichess and PyChess terms/privacy text displayed for the negative captures, recording upstream commit/tag or live retrieval timestamp and response identity. Do not execute downloaded code. Confirm every file is plain text, compute SHA-256, and include every path/hash in source evidence and the freeze commitment.

- [ ] **Step 3: Write exact per-site attribution**

Record Lichess live editor/terms/privacy, lila source/COPYING, Chessnut exact title/author/source/license (TASL), capture date, and no endorsement. Reproduce lila's logo restriction that the logo is used only to refer to lichess.org. Record PyChess editor/terms/privacy, source tag/observed deployed asset version, `COPYING.md`, Chessgroundx GPL-3.0-or-later, Firi exact TASL, capture date, and no endorsement. Explicitly state full visible viewport, unchanged rendering/pixels, anonymous synthetic positions, no user content, and that the project-specific COPYING/license snapshots—not generic license links alone—govern upstream-covered components.

Write `RIGHTS.json` from the same verified evidence. Populate both site records
with the exact displayed IDs, resolved piece-asset URL prefix, computed board
style signature, and upstream revision observed during the pre-capture source/UI
check. Its evidence records must match the retained file hashes. Re-read it
through the canonical JSON loader and reject any mismatch before capture.

- [ ] **Step 4: Run all construction verification before any final capture**

Run:

```bash
uv run --locked python -m pytest -q \
  tests/test_online_board_screenshot_benchmark.py \
  tests/test_board_recognition_benchmark.py \
  tests/test_online_board_benchmark_generator.py
uv run --locked python -m compileall -q benchmarks scripts tests
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight
uv run --locked python -m scripts.data.generate_board_recognition_benchmark \
  --verify datasets/board_recognition_benchmark/v1 \
  --digest benchmarks/board_recognition_v1_manifest.sha256
git diff --check
git diff --cached --quiet
```

Because `git diff --check` omits untracked files, enumerate untracked task files
with `git ls-files --others --exclude-standard` under `benchmarks/`,
`scripts/data/`, `tests/`, and `docs/superpowers/`; run
`git diff --no-index --check /dev/null <path>` on every regular file, accept
status 0 or the normal content-difference status 1, and require empty check
diagnostics. Binary PNG differences are acceptable; whitespace errors are not
for hand-authored attribution, rights, code, test, and design/plan files.

The following exact thirteen third-party evidence files are raw-byte exceptions
to that whitespace check and must not be normalized:

```text
benchmarks/licenses/AGPL-3.0-or-later.txt
benchmarks/licenses/GPL-3.0-or-later.txt
benchmarks/licenses/Apache-2.0.txt
benchmarks/licenses/CC-BY-4.0.txt
benchmarks/licenses/lichess-lila-COPYING.md
benchmarks/licenses/lichess-chessnut-LICENSE.txt
benchmarks/licenses/lichess-terms-2026-07-19.txt
benchmarks/licenses/lichess-privacy-2026-07-19.txt
benchmarks/licenses/pychess-variants-COPYING.md
benchmarks/licenses/pychess-firi-LICENSE.txt
benchmarks/licenses/pychess-terms-2026-05-22.md
benchmarks/licenses/pychess-privacy-2026-05-22.md
benchmarks/licenses/chessgroundx-GPL-3.0.txt
```

For each exception, instead require byte-for-byte equality and SHA-256 equality
with its recorded immutable-source or live-response retrieval evidence, and
require a nonempty plain-text file with no NUL byte or HTML document wrapper.
Preserve original line endings and end-of-file bytes. At these frozen versions,
`CC-BY-4.0.txt` and `pychess-firi-LICENSE.txt` produce `new blank line at EOF`,
while the exact CRLF response bodies in the two Lichess terms/privacy snapshots
produce `trailing whitespace` diagnostics; these four upstream-byte diagnostics
must be reported rather than edited away.

Expected: every command succeeds; synthetic preflight still prints `cases=568 positive=528 negative=40`; index is empty.

- [ ] **Step 5: Obtain two independent pre-capture reviews**

One reviewer validates code/tests and one validates oracle/rights/capture protocol. Both must report no Critical/Important findings. This is the user-required proof that the benchmark creator is trustworthy before generation. If either is not Ready, fix and repeat this gate; do not capture.

---

### Task 6: Capture the 40 original online screenshots and receipts

**Files:**
- Generate ignored: `output/p2_01_online_screenshot_capture_plan.json`
- Generate ignored: `output/p2_01_online_screenshot_staging/images/*.png`
- Create: `benchmarks/board_recognition_online_screenshot_v1_capture_receipts.json`

**Interfaces:**
- Capture plan is reviewer-hidden and controlled-input-bearing.
- Receipts bind browser metadata and exact PNG SHA-256 to every opaque case ID.

- [ ] **Step 1: Generate and inspect the controlled capture plan**

Run:

```bash
uv run --locked python -m scripts.data.freeze_online_board_screenshot_benchmark \
  --write-plan output/p2_01_online_screenshot_capture_plan.json
```

Then independently count the 32 positive and 8 negative instructions and exact matrix. Do not expose this plan to annotation reviewers.

- [ ] **Step 2: Prepare one fresh anonymous isolated browser context**

Re-read the live Lichess and PyChess terms immediately before capture and compare them to the frozen terms evidence. Open only `https://lichess.org/editor` and `https://www.pychess.org/editor/chess`. Keep login/settings menus closed during screenshots. Reject optional tracking. Select Lichess chessnut/brown and PyChess Firi/brown through normal UI, then resolve all five displayed identity strings and require literal equality with the applicable canonical `RIGHTS.json` site record; an unknown, changed, proprietary, or mismatched asset aborts before any capture. Stop without capture if terms changed materially, the site rate-limits/blocks the session, or a site policy classifies these 20 manually triggered PyChess screenshots as prohibited automation; maintainer permission would then be required before continuing.

- [ ] **Step 3: Capture positives at a low manual rate**

Trigger one capture at a time, sequentially, with no concurrency, crawler, scripted loop, automated retry, or navigation outside the allowlist. Restrict DOM reads to the controlled FEN/orientation/style and receipt metadata; do not extract page or user content. For each controlled positive instruction:

1. Set the viewport/DPR exactly.
2. Load the full FEN through the site's editor input/allowed FEN URL.
3. Use the site's visible flip control for `black_bottom`.
4. Wait for fonts/assets and a stable board; close all menus.
5. Evaluate final URL/title/UTC/user-agent/platform/language/scroll/visibility, board rectangle, orientation, style classes, and asset identifiers immediately around capture.
6. Take one visible-viewport PNG with browser `fullPage=false` and decode the returned browser bytes directly to `images/<opaque-id>.png` without Pillow or any transform.
7. Check the screenshot dimensions, minimal PNG chunk sequence, exact board rectangle, and file SHA before continuing.

- [ ] **Step 4: Capture four distinct negative pages at both viewports**

Use only `https://lichess.org/terms-of-service`, `https://lichess.org/privacy`, `https://www.pychess.org/terms`, and `https://www.pychess.org/privacy`, with no query, fragment, or redirect. At scroll position zero, wait for first-party assets and reject network/error banners. Confirm at original resolution that each has no complete 8x8 grid or board thumbnail and no user identity. Capture exactly once per viewport, sequentially.

- [ ] **Step 5: Write canonical receipts and verify staging**

Receipts contain exact common browser metadata plus per-case hash/time/final safe URL/title/site/page/viewport/DPR/scroll/visibility/navigation state. Positives include controlled FEN/view, board rectangle, style, and assets; negatives assert no complete grid. At this pre-evidence stage, manually compare every distinct positive receipt asset identifier with the frozen attribution/license evidence and record the result in the capture-inventory review. `assemble_reviewed_source()`, which receives attribution and licenses in Task 7, must enforce that binding automatically before it can write review/source bytes; a mismatch fails before assembly/freezing. Run:

```bash
uv run --locked python -m scripts.data.freeze_online_board_screenshot_benchmark \
  --verify-captures \
  --capture-plan output/p2_01_online_screenshot_capture_plan.json \
  --receipts benchmarks/board_recognition_online_screenshot_v1_capture_receipts.json \
  --capture-dir output/p2_01_online_screenshot_staging
```

Expected: 40 unique encoded hashes, 40 unique decoded-pixel hashes, and 40 valid minimal PNGs.

- [ ] **Step 6: Independently inspect the capture inventory without labels**

A reviewer checks counts, byte hashes, no transformation, live-source receipts, matrix, and privacy without running any project model. Reject and recapture a case on any uncertainty.

---

### Task 7: Blind annotation, reviewed source assembly, freeze, and pre-inference attestation

**Files:**
- Create: `benchmarks/board_recognition_online_screenshot_v1_annotations_a.json`
- Create: `benchmarks/board_recognition_online_screenshot_v1_annotations_b.json`
- Create: `benchmarks/board_recognition_online_screenshot_v1_review.json`
- Create: `benchmarks/board_recognition_online_screenshot_v1.json`
- Create: `benchmarks/board_recognition_online_screenshot_v1/manifest.json`
- Create: `benchmarks/board_recognition_online_screenshot_v1/images/*.png`
- Create: `benchmarks/board_recognition_online_screenshot_v1_manifest.sha256`
- Create: `benchmarks/board_recognition_online_screenshot_v1_freeze.sha256`

**Interfaces:**
- Two reviewers receive only a directory of opaque-ID images and the annotation schema.
- The assembler derives the source oracle only from exact A/B consensus, then compares it to the hidden plan.

- [ ] **Step 1: Dispatch two blind review passes with no inherited task context**

Use fresh subagents with `fork_turns="none"`. Tell each to inspect every original PNG at full resolution, not to read other repository files, and to write only their assigned annotation file. Each record contains ID, board presence, source-oriented placement for positives, and three all-true inspection gates. They must not receive FEN/view/site/matrix/model output or the other annotation.

- [ ] **Step 2: Verify pass isolation and exact agreement before controlled comparison**

Run the evidence assembler. It must first validate each full inventory and A/B equality, then compare consensus to the hidden capture plan. Any difference aborts without a source spec and identifies only the mismatching opaque ID for recapture/re-review.

- [ ] **Step 3: Assemble review and source spec**

After exact agreement, write canonical review/source bytes to nonexistent tracked paths:

```bash
uv run --locked python -m scripts.data.freeze_online_board_screenshot_benchmark \
  --assemble-source \
  --capture-plan output/p2_01_online_screenshot_capture_plan.json \
  --receipts benchmarks/board_recognition_online_screenshot_v1_capture_receipts.json \
  --capture-dir output/p2_01_online_screenshot_staging \
  --annotations-a benchmarks/board_recognition_online_screenshot_v1_annotations_a.json \
  --annotations-b benchmarks/board_recognition_online_screenshot_v1_annotations_b.json \
  --attribution benchmarks/board_recognition_online_screenshot_v1_ATTRIBUTION.md \
  --rights benchmarks/board_recognition_online_screenshot_v1_RIGHTS.json \
  --licenses-dir benchmarks/licenses \
  --review-output benchmarks/board_recognition_online_screenshot_v1_review.json \
  --source-output benchmarks/board_recognition_online_screenshot_v1.json
```

Re-read the outputs, verify evidence hashes/license inventory/cases/acceptance, and assert neither contains a model prediction or checkpoint identifier.

- [ ] **Step 4: Freeze byte-for-byte into the trackable corpus**

Run the freezer with nonexistent destination, manifest digest, and freeze commitment, then verify it:

```bash
uv run --locked python -m scripts.data.freeze_online_board_screenshot_benchmark \
  --freeze \
  --source-spec benchmarks/board_recognition_online_screenshot_v1.json \
  --capture-dir output/p2_01_online_screenshot_staging \
  --dataset benchmarks/board_recognition_online_screenshot_v1 \
  --manifest-digest benchmarks/board_recognition_online_screenshot_v1_manifest.sha256 \
  --freeze-digest benchmarks/board_recognition_online_screenshot_v1_freeze.sha256
uv run --locked python -m scripts.data.freeze_online_board_screenshot_benchmark \
  --verify \
  --source-spec benchmarks/board_recognition_online_screenshot_v1.json \
  --dataset benchmarks/board_recognition_online_screenshot_v1 \
  --manifest-digest benchmarks/board_recognition_online_screenshot_v1_manifest.sha256 \
  --freeze-digest benchmarks/board_recognition_online_screenshot_v1_freeze.sha256
```

Expected output reports exactly 40/32/8 and the commitment SHA-256.

- [ ] **Step 5: Perform two independent frozen-corpus attestations**

Both reviewers independently verify every tracked evidence hash, license inventory, manifest/case/hash, PNG chunk stream/pixels, blind agreement, controlled agreement, all-40 privacy gates, and freeze commitment. Their Ready messages are external session attestations; do not edit the frozen `review.json` afterward.

- [ ] **Step 6: Remove only the disposable reviewed capture staging**

After both frozen-corpus attestations, confirm `output/p2_01_online_screenshot_staging` is a real non-symlink directory and that every PNG is byte-identical to its tracked frozen counterpart. Remove exactly that directory and the reviewer-hidden `output/p2_01_online_screenshot_capture_plan.json`; the trackable corpus/evidence remains the recovery copy. Record this cleanup in the final handoff.

- [ ] **Step 7: Record the pre-inference immutability snapshot**

Record the freeze-file SHA-256 and `git status --short`. Run read-only verification once more immediately before checkpoint access. No evidence/corpus byte may change after this point.

---

### Task 8: Current-checkpoint baseline, independent report verification, docs, and final gates

**Files:**
- Generate ignored: `output/board_recognition_online-screenshot-v1_baseline_35e27c1_mps.json`
- Modify: `README.md`
- Modify: `BOARD_RECOGNITION_AUDIT.tmp.md`

**Interfaces:**
- The report records source claim/version, freeze/manifest/checkpoint hashes, fingerprints, all predictions/scores, and strict acceptance failures.

- [ ] **Step 1: Run the frozen corpus once with trusted checkpoints**

Run:

```bash
uv run --locked python -m scripts.validate.evaluate_board_recognition_benchmark \
  --dataset benchmarks/board_recognition_online_screenshot_v1 \
  --source-spec benchmarks/board_recognition_online_screenshot_v1.json \
  --manifest-digest benchmarks/board_recognition_online_screenshot_v1_manifest.sha256 \
  --freeze-digest benchmarks/board_recognition_online_screenshot_v1_freeze.sha256 \
  --board-detector-checkpoint checkpoints/bd-MobileViTV2FPN-v1.ckpt \
  --square-classifier-checkpoint checkpoints/sc-9-bs=64-step=23296.ckpt \
  --piece-classifier-checkpoint checkpoints/pc-48-bs=128-step=18944.ckpt \
  --report output/board_recognition_online-screenshot-v1_baseline_35e27c1_mps.json \
  --device mps
```

Expected: the command writes a canonical report. Exit 1 is expected if any strict acceptance requirement fails; do not rerun to tune the corpus or policy.

- [ ] **Step 2: Recompute the freeze commitment immediately after inference**

Run the read-only verifier and compare its SHA exactly with the pre-inference value and report field. Any change invalidates the baseline and must be investigated before proceeding.

- [ ] **Step 3: Independently verify every report value**

A fresh reviewer parses canonical bytes, checks exactly 40 IDs, hashes source/manifest/freeze/checkpoints, recomputes every overall/group score from predictions, recomputes every acceptance requirement and exit expectation, checks no sibling temp, and reports exact metrics plus report SHA-256.

- [ ] **Step 4: Update README and audit from measured evidence only**

README documents both corpora, captured verification/evaluation commands, strict policy, frozen limitations, and the exact baseline/report hash. In the audit:

- mark P1-06 resolved by commit `eb1c18d` and current typed observed-placement contract;
- mark P2-01 resolved as a trustworthy corpus/evaluator even if the model gate is red;
- record exact captured metrics, acceptance failures, corpus/freeze/manifest/checkpoint/report hashes;
- keep P1-07 open and name it the next logical fix if negatives do not return `NO_BOARD` or positives are falsely rejected.

- [ ] **Step 5: Run broad final verification**

Run:

```bash
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml benchmarks scripts tests
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight
uv run --locked python -m scripts.data.generate_board_recognition_benchmark \
  --verify datasets/board_recognition_benchmark/v1 \
  --digest benchmarks/board_recognition_v1_manifest.sha256
uv run --locked python -m scripts.data.freeze_online_board_screenshot_benchmark \
  --verify \
  --source-spec benchmarks/board_recognition_online_screenshot_v1.json \
  --dataset benchmarks/board_recognition_online_screenshot_v1 \
  --manifest-digest benchmarks/board_recognition_online_screenshot_v1_manifest.sha256 \
  --freeze-digest benchmarks/board_recognition_online_screenshot_v1_freeze.sha256
git diff --check
git diff --cached --quiet
git status --short
```

Expected: tests/compilation/preflights/diff/index pass; status contains only intentional task files plus the two pre-existing untracked audit/research files.

- [ ] **Step 6: Request final correctness and scope reviews**

One reviewer checks code/tests/docs against the design; a second independently validates corpus/report immutability and exact metrics. Resolve every Critical/Important finding, rerun affected checks, and do not stage or commit.

- [ ] **Step 7: Final handoff**

Report what changed, exact checks and results, exact baseline acceptance state/metrics/hashes, the no-commit durability limitation, and P1-07 as the next logical issue when supported by the measured baseline.
