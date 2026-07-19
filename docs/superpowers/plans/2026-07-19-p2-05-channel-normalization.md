# P2-05 Channel Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every public BoardDetector image path accept RGB, RGBA, grayscale, and OpenCV inputs, then measure the fix against the frozen benchmark.

**Architecture:** Reuse `Picture.as_3_channels` at the start of the three public BoardDetector image methods. Keep `Picture` itself channel-preserving, retain the existing prediction seam, and prove both preprocessing and warp output contracts before rerunning identical checkpoints.

**Tech Stack:** Python 3.14, Pillow, OpenCV, NumPy, PyTorch/Lightning, pytest.

## Global Constraints

- Governing design: `docs/superpowers/specs/2026-07-19-p2-05-channel-normalization-design.md`.
- Governing finding: `BOARD_RECOGNITION_AUDIT.tmp.md`, P2-05.
- Normalize through the existing `Picture.as_3_channels`; do not change `Picture`, shared timm transforms, training data, or checkpoints.
- Cover Pillow RGB, Pillow RGBA, Pillow L, OpenCV BGR, file-backed RGBA/L extraction, and marked-copy non-mutation.
- Preserve the existing public `predict_coords` override/monkeypatch seam.
- Before report: `output/board_recognition_online-render-v1_baseline_b1862d3_mps.json`, SHA-256 `a5f3132a32a4b6973262693000246473f10b5b5c6010eee44832ee112333d5bd`.
- After report must be new: `output/board_recognition_online-render-v1_post-p2-05_2a1e5e0_mps.json`.
- The same corpus, manifest digest, three checkpoint files/hashes, and MPS device must be used for before/after comparison.
- Preserve the synthetic-only claim and keep P2-01 open for independently captured, manually labeled online screenshots.
- Preserve unrelated work. Do not stage or commit any file.

---

### Task 1: Normalize BoardDetector public image inputs with TDD

**Files:**

- Modify: `tests/test_board_detector_model.py`
- Modify: `chessml/models/lightning/board_detector_model.py`

**Interfaces:**

- Consumes: `Picture.as_3_channels -> Picture`.
- Produces: unchanged public signatures for `predict_coords`, `mark_board_on_image`, and `extract_board_image`; each consumes a three-channel local copy without mutating its caller.

- [ ] **Step 1: Extend the existing model stub for the real prediction path**

Add `from PIL import Image`, then extend `ModelStub` without creating another stub:

```python
class ModelStub(torch.nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.output_features = output_features
        self.preprocessed_images = []

    def preprocess_image(self, image):
        self.preprocessed_images.append(image.copy())
        return torch.zeros((3, image.height, image.width), dtype=torch.float32)

    def forward(self, batch):
        return torch.zeros(
            (batch.shape[0], self.output_features),
            dtype=batch.dtype,
            device=batch.device,
        )
```

- [ ] **Step 2: Add the four-way preprocessing regression**

Add one parameterized test using these exact cases:

```python
@pytest.mark.parametrize(
    ("image_input", "expected_rgb"),
    [
        (Image.new("RGB", (4, 3), (10, 20, 30)), (10, 20, 30)),
        (Image.new("RGBA", (4, 3), (10, 20, 30, 255)), (10, 20, 30)),
        (Image.new("L", (4, 3), 10), (10, 10, 10)),
        (np.full((3, 4, 3), (30, 20, 10), dtype=np.uint8), (10, 20, 30)),
    ],
    ids=("rgb", "rgba", "grayscale", "opencv-bgr"),
)
def test_predict_coords_normalizes_input_to_rgb(image_input, expected_rgb):
    source = Picture(image_input)
    source_before = source.cv2.copy()
    detector = BoardDetector(base_model_class=ModelStub)

    coords = detector.predict_coords(source)

    assert len(detector.model.preprocessed_images) == 1
    preprocessed = detector.model.preprocessed_images[0]
    assert preprocessed.mode == "RGB"
    assert preprocessed.size == (4, 3)
    assert preprocessed.getpixel((0, 0)) == expected_rgb
    np.testing.assert_array_equal(source.cv2, source_before)
    np.testing.assert_array_equal(coords, np.full(8, 0.5, dtype=np.float32))
```

- [ ] **Step 3: Strengthen marking and file-backed extraction regressions**

Replace the existing marked-copy test with:

```python
@pytest.mark.parametrize(
    ("mode", "color"),
    [
        ("RGB", (0, 0, 0)),
        ("RGBA", (0, 0, 0, 255)),
        ("L", 0),
    ],
)
def test_mark_board_on_image_returns_marked_copy(monkeypatch, mode, color):
    source = Picture(Image.new(mode, (100, 100), color))
    before = np.asarray(source.pil).copy()
    unmarked_rgb = np.asarray(source.as_3_channels.pil).copy()
    detector = detector_with_coords(
        monkeypatch,
        [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
    )

    marked = detector.mark_board_on_image(source)

    assert source.pil.mode == mode
    np.testing.assert_array_equal(np.asarray(source.pil), before)
    assert marked.pil.mode == "RGB"
    assert np.any(np.asarray(marked.pil) != unmarked_rgb)
```

Add this file-backed regression so mocked prediction cannot hide a raw-channel warp:

```python
@pytest.mark.parametrize(
    ("mode", "color", "raw_shape"),
    [
        ("RGBA", (10, 20, 30, 255), (100, 100, 4)),
        ("L", 10, (100, 100)),
    ],
)
def test_extract_board_image_normalizes_file_channels(
    monkeypatch, tmp_path, mode, color, raw_shape
):
    image_path = tmp_path / f"source-{mode}.png"
    Image.new(mode, (100, 100), color).save(image_path)
    source = Picture(image_path)
    assert source.cv2.shape == raw_shape
    detector = detector_with_coords(
        monkeypatch,
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    )

    extracted = detector.extract_board_image(source)

    assert extracted.cv2.shape == (100, 100, 3)
```

- [ ] **Step 4: Run RED before production changes**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
```

Expected: nonzero; six non-RGB parameter cases fail for raw `L`/`RGBA` preprocessing, marking, or warp channels, while the existing RGB/OpenCV cases remain green.

- [ ] **Step 5: Apply the minimal production fix**

Add one assignment at the beginning of each method body, before any use of the image:

```python
def predict_coords(self, img: Picture) -> np.ndarray:
    img = img.as_3_channels
```

```python
def mark_board_on_image(self, original_image: Picture) -> Picture:
    original_image = original_image.as_3_channels
```

```python
def extract_board_image(self, original_image: Picture) -> Picture:
    original_image = original_image.as_3_channels
```

Do not add a helper, change signatures, alter `Picture`, or bypass public `predict_coords` calls.

- [ ] **Step 6: Verify GREEN and self-review**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_detector_model.py
uv run --locked python -m pytest -q tests/test_image_compatibility.py tests/test_board_recognition_helper.py tests/test_board_detector_model.py
git diff --check
git diff --cached --stat
git status --short --untracked-files=all
```

Expected: 15 detector cases pass; relevant image/helper tests pass; only the two task code/test files plus approved untracked documentation/audit work differ; index is empty.

---

### Task 2: Review, verify, benchmark, and close P2-05

**Files:**

- Modify after verified benchmark: `README.md`
- Modify after verified benchmark: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] Have an independent reviewer verify Task 1 against the exact test and production diff, including caller non-mutation, preprocessing color order, file-backed warp channels, public API coverage, minimality, and unchanged training behavior.
- [ ] Resolve every Critical or Important finding and re-review.
- [ ] Have a fresh verifier run:

```bash
uv run --locked python -m pytest -q
uv run --locked python -m compileall -q chessml scripts tests
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256
git diff --check
git diff --cached --stat
```

- [ ] Confirm the post-fix report path does not exist, then run:

```bash
uv run --locked python -m scripts.validate.evaluate_board_recognition_benchmark \
  --dataset datasets/board_recognition_benchmark/v1 \
  --source-spec benchmarks/board_recognition_v1.json \
  --manifest-digest benchmarks/board_recognition_v1_manifest.sha256 \
  --board-detector-checkpoint checkpoints/bd-MobileViTV2FPN-v1.ckpt \
  --square-classifier-checkpoint checkpoints/sc-9-bs=64-step=23296.ckpt \
  --piece-classifier-checkpoint checkpoints/pc-48-bs=128-step=18944.ckpt \
  --report output/board_recognition_online-render-v1_post-p2-05_2a1e5e0_mps.json \
  --device mps
```

Expected: exit zero and a complete report with zero execution errors. If not, preserve the report, diagnose the newly exposed failure, and do not mark P2-05 resolved.

- [ ] Independently verify canonical report bytes/schema, exact 568 IDs, source/manifest/checkpoint hashes, runtime fingerprint, score recomputation, and no sibling temp file. Compare both reports and measure all metric changes; separately compare the 544 RGB main predictions and classify any drift.
- [ ] Update README and audit with exact before/after results, both report SHA-256 values, zero-error evidence, claim boundary, and next required work. Mark P2-05 resolved only if every acceptance gate passes.
- [ ] Supply the complete change/evidence package to a final read-only reviewer; resolve findings.
- [ ] Re-run the focused detector suite, anchored corpus verification, post-report SHA-256, whitespace, index, status, and HEAD checks. Do not stage, commit, retrain, regenerate the corpus, or claim P2-01 complete.
