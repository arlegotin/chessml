# P1-01 and P2-07 Classifier Data/Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove classifier label/source shortcuts, create source-group-held-out classifier splits, and make every existing seeded CLI control construction and training RNG streams.

**Architecture:** Rebuild `PiecesImages3x3` around one piece map per set and a local seeded sampler, then generate separate training/validation CSVs from disjoint piece-set/theme groups. Seed all commands once in `Script.run()` before callback construction, reusing Lightning's installed worker-aware seeding.

**Tech Stack:** Python 3.14, stdlib `random`/`csv`, NumPy, OpenCV, Pillow, PyTorch 2.12, Lightning 2.6, pytest.

## Global Constraints

- Governing design: `docs/superpowers/specs/2026-07-19-p1-01-p2-07-classifier-data-reproducibility-design.md`.
- Governing findings: `BOARD_RECOGNITION_AUDIT.tmp.md`, P1-01 and P2-07.
- Preserve the first two `PiecesImages3x3`/`AugmentedPiecesImages` yielded fields and append exact provenance `(piece_set, dark_color, light_color)`.
- Every rendered 3x3 uses one piece set; center label, piece set, theme, and every neighbor choice have no index-derived coupling.
- Use `EMPTY_SQUARE_CHANCE` for neighbor occupancy even when empty centers are disabled.
- Center targets are balanced per complete cycle; arbitrary limits may truncate one final cycle/pair.
- Generated artifacts are exactly `train.csv`, `validation.csv`, `images/train/`, and `images/validation/`; CSV header is exactly `image_path,piece_name,piece_set,dark_color,light_color`.
- Training and validation piece-set names and `(dark_color, light_color)` pairs must be disjoint. Never fall back to legacy `meta.csv`.
- CLI seeds are integers in `[0, 2**32 - 1]`; seed before callback/model/dataset construction through installed `lightning.seed_everything(seed, workers=True)`.
- Do not add a dependency, custom RNG framework, dataset manifest, explicit DataLoader generator, or custom worker initializer.
- Preserve unrelated `BOARD_RECOGNITION_ARCHITECTURE_RESEARCH.tmp.md` and every existing dataset/checkpoint artifact.
- Do not regenerate the real classifier datasets, retrain, run checkpoint inference/the recognition benchmark, stage, commit, merge, push, or publish.
- Baseline: `1de97ae` on `improvements-2`; full suite `318 passed` before changes.

---

### Task 1: Replace index-coupled 3x3 sampling

**Files:**

- Create: `tests/test_piece_image_generation.py`
- Modify: `chessml/data/images/pieces_images.py`
- Modify: `chessml/data/utils/looped_list.py`
- Modify: `chessml/data/assets.py`
- Modify: `scripts/data/visualize_augmented_pieces.py`
- Modify: `tests/test_chessml.py`

**Interfaces:**

- Consumes: supplied piece-set directories containing every path in `PIECE_FILE_NAMES`; supplied `(dark, light)` themes; `EMPTY_SQUARE_CHANCE`.
- Produces: `PiecesImages3x3 -> (Picture, str | None, str, str, str)` and `AugmentedPiecesImages` preserving the same tuple tail.

- [ ] **Step 1: Add synthetic asset helpers and sampling regressions**

In `tests/test_piece_image_generation.py`, create two temporary piece-set trees. Each must contain all 12 `PIECE_FILE_NAMES` paths as opaque 1x1 RGBA PNGs. Give every set a disjoint BGR palette so a rendered non-background pixel identifies its set.

Add a helper with this contract:

```python
def make_piece_set(
    root: Path,
    name: str,
    base: int,
) -> tuple[Path, dict[str, tuple[int, int, int]]]:
    directory = root / name
    palette = {}
    for index, (label, location) in enumerate(PIECE_FILE_NAMES.items()):
        rgb = (base + index, base + index + 1, base + index + 2)
        path = directory / f"{location}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (1, 1), (*rgb, 255)).save(path)
        palette[label] = rgb[::-1]
    return directory, palette
```

Use two visibly distinct themes and `square_size=1`. Add tests that:

```python
piece_set_a, palette_a = make_piece_set(tmp_path, "set-a", 10)
piece_set_b, palette_b = make_piece_set(tmp_path, "set-b", 100)
dataset_kwargs = {
    "piece_sets": [piece_set_a, piece_set_b],
    "board_colors": [("#010203", "#040506"), ("#111213", "#141516")],
    "square_size": 1,
}

piece_rows = list(
    islice(
        PiecesImages3x3(
            **dataset_kwargs,
            shuffle_seed=0,
            with_empty_squares=False,
        ),
        24,
    )
)
assert Counter(row[1] for row in piece_rows) == Counter({name: 2 for name in PIECE_FILE_NAMES})

square_rows = list(
    islice(
        PiecesImages3x3(
            **dataset_kwargs,
            shuffle_seed=0,
            with_empty_squares=True,
        ),
        48,
    )
)
assert sum(row[1] is None for row in square_rows) == 24
assert Counter(row[1] for row in square_rows if row[1] is not None) == Counter(
    {name: 2 for name in PIECE_FILE_NAMES}
)
```

Across 32 complete square cycles, convert every raw pixel to a semantic label:

```python
backgrounds = {
    hex_to_bgr(color)
    for theme in dataset_kwargs["board_colors"]
    for color in theme
}
palettes = {"set-a": palette_a, "set-b": palette_b}

def semantic_grid(row):
    picture, center_label, piece_set_name, *_ = row
    reverse_palette = {color: label for label, color in palettes[piece_set_name].items()}
    return tuple(
        None if tuple(pixel) in backgrounds else reverse_palette[tuple(pixel)]
        for pixel in picture.cv2.reshape(-1, 3)
    )
```

Derive each neighbor signature from semantic indexes `0, 1, 2, 3, 5, 6, 7, 8`, excluding center index `4`, so dark/light parity cannot create a false variation. Assert:

- each non-empty center label occurs with both themes;
- more than one neighbor signature occurs for every center label;
- both background/empty and occupied neighbor pixels occur; and
- every semantic occupied pixel belongs to the palette named by that row's
  `piece_set` provenance; and
- semantic center index `4` equals the yielded center label for occupied rows
  and is `None` for empty rows.

Add explicit truncated-prefix checks:

```python
assert len(list(PiecesImages3x3(**dataset_kwargs, shuffle_seed=0,
                               with_empty_squares=False, limit=25))) == 25
assert len(list(PiecesImages3x3(**dataset_kwargs, shuffle_seed=0,
                               with_empty_squares=True, limit=49))) == 49
```

The 25th/49th rows may start a new cycle; assert only that the completed
24/48-row prefixes have the exact balance shown above and the extra row does
not change earlier rows.

Add same-object replay and different-seed checks:

```python
def capture(dataset, count):
    return [(row[0].cv2.copy(), row[1:]) for row in islice(dataset, count)]


dataset = PiecesImages3x3(
    **dataset_kwargs,
    shuffle_seed=0,
    with_empty_squares=False,
)
first = capture(dataset, 96)
second = capture(dataset, 96)
different = capture(
    PiecesImages3x3(
        **dataset_kwargs,
        shuffle_seed=1,
        with_empty_squares=False,
    ),
    96,
)

assert all(left[1] == right[1] and np.array_equal(left[0], right[0])
           for left, right in zip(first, second, strict=True))
assert any(left[1] != right[1] or not np.array_equal(left[0], right[0])
           for left, right in zip(first, different, strict=True))
```

- [ ] **Step 2: Add inventory and seed-zero regressions**

In `tests/test_chessml.py`, add a fresh subprocess test that creates:

```text
<assets>/piece_png/valid-set/
<assets>/piece_png/.DS_Store
<assets>/bg/512/
```

Set `chessml.config.assets.path` before importing `chessml.data.assets` and assert `PIECE_SETS` contains only the real directory.

In `tests/test_piece_image_generation.py`, assert `LoopedList([0, 1, 2], shuffle_seed=0)` uses `Random(0)` exactly, proving zero is treated as an explicit seed rather than `None`.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_image_generation.py tests/test_chessml.py
```

Expected: nonzero. The current generator has only two yielded fields, fixed label/theme/neighbor relations and mixed piece sets; `LoopedList` ignores seed zero; the raw inventory includes `.DS_Store`.

- [ ] **Step 4: Implement the minimal sampler**

In `PiecesImages3x3.__init__`, replace the flattened `LoopedList` inputs with:

```python
self.piece_sets = [
    (
        piece_set.name,
        {
            piece_name: Picture(piece_set / f"{piece_location}.png")
            for piece_name, piece_location in PIECE_FILE_NAMES.items()
        },
    )
    for piece_set in piece_sets
]
self.backgrounds = [
    (
        dark,
        light,
        Picture(np.full((1, 1, 3), hex_to_bgr(dark), dtype=np.uint8)),
        Picture(np.full((1, 1, 3), hex_to_bgr(light), dtype=np.uint8)),
    )
    for dark, light in board_colors
]
self.center_labels = list(PIECE_FILE_NAMES)
if with_empty_squares:
    self.center_labels += [None] * len(PIECE_FILE_NAMES)
self.empty_picture = Picture(np.zeros((1, 1, 4), dtype=np.uint8))
```

In `generator()`, create `rng = random.Random(self.shuffle_seed)`. For every complete center-label cycle, shuffle a copy, choose one `(piece_set_name, pieces)` and one theme for each target, independently choose eight neighbor labels with `EMPTY_SQUARE_CHANCE`, and emit the same context once with a dark center and once with a light center. Select every occupied pixel from `pieces[label]`; never select from a flattened cross-set pool.

Yield exactly:

```python
yield Picture(grid), center_label, piece_set_name, dark_color, light_color
```

Update `AugmentedPiecesImages.generator()` to unpack `original_picture, piece_name, *provenance` and yield `(Picture(augmented_image), piece_name, *provenance)`.

Delete the obsolete `LoopedList`/`itertools` imports from `pieces_images.py`.

Change `LoopedList` to:

```python
if shuffle_seed is not None:
    Random(shuffle_seed).shuffle(self.data)
```

Filter `PIECE_SETS` with `path.is_dir()` at its inventory comprehension. Update the visualization caller to pass `shuffle_seed=args.seed` to `AugmentedPiecesImages` and unpack `(picture, name, *_)`.

- [ ] **Step 5: Verify GREEN and task scope**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_image_generation.py tests/test_chessml.py tests/test_iterable_dataset.py
git diff --check
git diff --cached --stat
```

Expected: all focused tests pass; index remains empty.

---

### Task 2: Generate and consume source-group-held-out splits

**Files:**

- Modify: `tests/test_piece_image_generation.py`
- Create: `tests/test_standard_training.py`
- Modify: `scripts/data/generate_piece_classifier_dataset.py`
- Modify: `chessml/train/standard_training.py`
- Modify: `scripts/train/train_square_classifier.py`
- Modify: `scripts/train/train_piece_classifier.py`

**Interfaces:**

- Consumes: Task 1's five-field augmented samples.
- Produces: disjoint `train.csv`/`validation.csv` artifacts and optional `make_val_dataset: Callable[..., Dataset] | None` routing in `standard_training`.

- [ ] **Step 1: Add grouped-artifact RED tests**

Import `scripts.data.generate_piece_classifier_dataset` with a fresh `chessml.Script` whose `__call__` returns the decorated function, then call its pure generation function with temporary output, synthetic piece sets, themes, a total limit of 50, and seed zero.

Assert:

```python
assert train_header == validation_header == [
    "image_path", "piece_name", "piece_set", "dark_color", "light_color"
]
assert len(train_rows) + len(validation_rows) == 50
assert set(train_piece_sets).isdisjoint(validation_piece_sets)
assert set(train_themes).isdisjoint(validation_themes)
assert all(Path(row[0]).is_file() for row in train_rows + validation_rows)
```

Generate into a second temporary root with the same seed and compare ordered labels/provenance plus per-image SHA-256 values. Generate with seed one and assert at least one provenance or image hash changes. Parameterize once with `with_empty_squares=False` and once with `True`; square labels must be only `"0"` or `"1"`.

Run one additional two-row generation with seed `2**32 - 1` and assert both
split CSVs contain one row. This exercises validation's wrapped
`(seed + 1) % 2**32 == 0` stream.

Add validation cases for total limit below two and fewer than two sources/themes.

- [ ] **Step 2: Add separate-factory routing RED tests**

In `tests/test_standard_training.py`, replace only the external `Trainer`, logger, and checkpoint callback with recording stubs; use real tiny Torch datasets/DataLoaders and force the MPS availability probe false so unrelated P2-02 does not enter this test.

For the legacy one-factory path, assert calls remain:

```python
[{"limit": batch_size * val_batches}, {"offset": batch_size * val_batches}]
```

For a separate validation factory, assert:

```python
validation_calls == [{"limit": batch_size * val_batches}]
training_calls == [{"offset": 0}]
```

Also assert `Trainer.fit()` receives loaders backed by the returned training and validation datasets.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_image_generation.py tests/test_standard_training.py
```

Expected: nonzero because the generator still writes one `meta.csv` and `standard_training` has no validation factory.

- [ ] **Step 4: Implement deterministic grouped artifacts**

In `generate_piece_classifier_dataset.py`, delete unused model/Torch imports and the post-generation CSV reread/shuffle. Add:

```python
def validation_count(length: int) -> int:
    if length < 2:
        raise ValueError("at least two items are required")
    return max(1, round(length / 5))


def split_sources(values, rng: random.Random):
    values = sorted(values)
    rng.shuffle(values)
    count = validation_count(len(values))
    return values[count:], values[:count]
```

Add this exact boundary:

```python
def write_split(
    dataset_dir: Path,
    split: str,
    piece_sets: list[Path],
    board_colors: list[tuple[str, str]],
    limit: int,
    seed: int,
    with_empty_squares: bool,
) -> Path:
```

It creates `images/<split>/`, constructs both image datasets with the supplied
split seed/limit, writes `<split>.csv` once, and saves each PNG. Before choosing
the filename or writing the CSV row, normalize the raw target explicitly:

```python
target = int(piece_name is not None) if with_empty_squares else piece_name
```

Then write the exact five columns `image_path,target,piece_set,dark_color,light_color`.
This is required because square-mode raw samples contain a piece symbol or
`None`, while their classifier artifact contract is `"1"` or `"0"`.

Add this exact pure entry point:

```python
def generate_classifier_dataset(
    dataset_dir: Path,
    piece_sets: list[Path],
    board_colors: list[tuple[str, str]],
    limit: int,
    seed: int,
    with_empty_squares: bool,
) -> tuple[Path, Path]:
```

It:

1. rejects `limit < 2`;
2. uses one `random.Random(seed)` to partition sorted piece sets and themes;
3. allocates `validation_count(limit)` rows to validation and the remainder to training;
4. calls `write_split` for training with `seed` and validation with
   `(seed + 1) % 2**32`; and
5. returns the two CSV paths for focused verification.

The decorated CLI `main(args)` selects the square/piece directory and calls this function with `PIECE_SETS`, `BOARD_COLORS`, `args.limit`, `args.seed`, and `args.with_empty_squares`.

- [ ] **Step 5: Route classifier training to the two CSVs**

Append `make_val_dataset: Callable[..., Dataset] | None = None` to `standard_training`. Implement:

```python
val_factory = make_dataset if make_val_dataset is None else make_val_dataset
val_dataset = val_factory(limit=batch_size * val_batches)
train_offset = batch_size * val_batches if make_val_dataset is None else 0
train_dataset = make_dataset(offset=train_offset)
```

Leave every other loader/trainer argument unchanged.

In both classifier dataset `__getitem__` methods, unpack `path, target, *_`. Change each training entrypoint to define one factory accepting `path_to_csv`, then pass `partial(factory, path_to_csv=train.csv)` as `make_dataset` and `partial(factory, path_to_csv=validation.csv)` as `make_val_dataset`. Do not reference `meta.csv`.

- [ ] **Step 6: Verify GREEN and task scope**

Run:

```bash
uv run --locked python -m pytest -q tests/test_piece_image_generation.py tests/test_standard_training.py tests/test_chessml.py
uv run --locked python scripts/train/train_square_classifier.py --help
uv run --locked python scripts/train/train_piece_classifier.py --help
git diff --check
git diff --cached --stat
```

Expected: all focused tests and both CLI help smokes pass; index remains empty.

---

### Task 3: Seed every affected command before construction

**Files:**

- Create: `tests/test_script_seed.py`
- Modify: `chessml/__init__.py`

**Interfaces:**

- Consumes: parsed `Namespace.seed` or `Namespace.shuffle_seed`.
- Produces: callback execution after `lightning.seed_everything(value, workers=True)`; callback is not invoked for a Lightning-invalid seed.

- [ ] **Step 1: Add the CLI-seed RED regression**

Parameterize `seed` and `shuffle_seed`. For each name, create a fresh `Script`, replace only `parser.parse_args` with a zero-I/O callable returning the test namespace, then let a real callback construct/draw:

```python
(
    random.random(),
    np.random.random(),
    torch.nn.Linear(4, 4).weight.detach().clone(),
    torch.nn.functional.dropout(torch.ones(16), p=0.5, training=True),
    next(iter(DataLoader(TensorDataset(torch.arange(12)), batch_size=12, shuffle=True)))[0],
)
```

Invoke twice with each supported boundary seed, `0` and `2**32 - 1`, and assert
every value/order repeats and `os.environ["PL_SEED_WORKERS"] == "1"`. Invoke
with seed one after the seed-zero case and assert the tuple differs.

Add a parameterized test for `-1` and `2**32` that records whether the callback ran, expects Lightning's `ValueError`, and asserts the callback remained untouched.

- [ ] **Step 2: Run RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_script_seed.py
```

Expected: repeatability and invalid-seed tests fail because `Script.run()` does not seed.

- [ ] **Step 3: Implement the single shared seed boundary**

Immediately before `fn(parsed_args)` in `Script.run()`:

```python
seed = getattr(parsed_args, "seed", None)
if seed is None:
    seed = getattr(parsed_args, "shuffle_seed", None)

if seed is not None:
    from lightning import seed_everything

    seed_everything(seed, workers=True)
```

Keep the import lazy so unseeded utility scripts do not import Lightning solely for the runner.

- [ ] **Step 4: Verify GREEN and seeded entrypoint smokes**

Run:

```bash
uv run --locked python -m pytest -q tests/test_script_seed.py tests/test_piece_image_generation.py tests/test_iterable_dataset.py
uv run --locked python scripts/data/generate_piece_classifier_dataset.py --help
uv run --locked python scripts/train/train_board_detector.py --help
uv run --locked python scripts/train/train_square_classifier.py --help
uv run --locked python scripts/train/train_piece_classifier.py --help
uv run --locked python scripts/train/train_meta_predictor.py --help
git diff --check
git diff --cached --stat
```

Expected: focused tests and help smokes pass; index remains empty.

---

### Task 4: Review, verify, and close P1-01/P2-07

**Files:**

- Modify: `README.md`
- Modify: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] Have an independent task reviewer inspect each completed task against its brief, RED/GREEN evidence, exact diff, sampling oracle, compatibility, and scope. Resolve every Critical or Important finding and re-review.

- [ ] Run a real-asset canary without writing under `datasets/`: construct the fixed raw and augmented datasets from filtered `PIECE_SETS`/`BOARD_COLORS`, consume at least 48 samples for seeds zero and one, and verify same-seed fresh-construction pixel/provenance hashes match while another seed differs. Use a `mktemp -d` path only if grouped CSV output is included; remove no repository artifact.

- [ ] Update README's generation section with the two exact classifier-generation commands, the new CSV/image layout, deterministic seed behavior, source-group-held-out validation, and the explicit warning that legacy `meta.csv` is not accepted.

- [ ] Mark P1-01 and P2-07 resolved in the audit only after all tests/reviews pass. Record the pre-fix invariants, one-set/local-RNG correction, exact source-group separation, centralized seed timing, artifact compatibility break, and the fact that existing datasets/checkpoints remain tainted until explicit regeneration/retraining. Update required-order step 4 so the BoardDetector training group (P1-02/P1-03/P2-02) is next.

- [ ] Have a fresh verifier run:

```bash
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256
git diff --check
while IFS= read -r -d '' file; do git diff --no-index --check /dev/null "$file" >/dev/null; status=$?; if test "$status" -gt 1; then exit "$status"; fi; done < <(git ls-files --others --exclude-standard -z)
git diff --cached --stat
git status --short --untracked-files=all
```

- [ ] Supply the exact complete change/evidence package to a final read-only reviewer. Resolve all findings, then rerun the focused sampling/seed/training tests, whitespace scans for tracked and new files, index/status, and `git rev-parse --short HEAD`.

- [ ] Do not commit. Leave P2-08/P2-09, P1-02/P1-03/P2-02, P1-07, and the real-screenshot half of P2-01 open.
