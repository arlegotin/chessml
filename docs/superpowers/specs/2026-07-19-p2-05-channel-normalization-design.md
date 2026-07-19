# P2-05 Channel Normalization Design

## Decision

Normalize every image entering a public `BoardDetector` image operation with the existing `Picture.as_3_channels` property. Keep `Picture` channel-preserving for rendering and asset composition, add no helper or dependency, and do not change training transforms.

## Context

`BoardDetector.predict_coords()` currently passes `Picture.pil` directly to timm's three-channel normalization. The frozen baseline reproduced 24 execution errors: all 12 grayscale cases failed with a one-versus-three broadcast mismatch and all 12 RGBA cases failed with a four-versus-three tensor mismatch. All 544 RGB cases executed.

Changing only the preprocessing argument would be incomplete. `extract_board_image()` also warps `original_image.cv2`; a grayscale file would remain two-dimensional and fail later when the recognition helper converts extracted squares through `Picture.bw`. `mark_board_on_image()` would still try to draw an RGB line on a grayscale Pillow image.

## Boundary

At the beginning of each public BoardDetector image method:

```python
image = image.as_3_channels
```

Specifically:

- `predict_coords()` normalizes `img` before timm preprocessing.
- `mark_board_on_image()` normalizes `original_image` before prediction and drawing.
- `extract_board_image()` normalizes `original_image` before prediction, geometry, and warping.

`Picture.as_3_channels` returns a new `Picture`, so caller-owned objects and pixels remain unchanged. RGB/BGR inputs preserve their colors, grayscale is replicated to three channels, and alpha is dropped. The intended contract is RGB/BGR content; alpha compositing, CMYK/palette handling, and unusual two-or-five-channel arrays remain outside P2-05.

Marking and extraction continue to call the public `predict_coords()` seam. That method sees an already-three-channel `Picture` and wraps the same OpenCV array again; later Pillow access can therefore perform a second OpenCV-to-Pillow conversion in those nested calls. This bounded cost is preferred to bypassing a public override/monkeypatch seam or changing `Picture` semantics for an unmeasured optimization.

## Rejected Alternatives

- Normalize in `Picture.__init__`: breaks its source-preserving role and alpha-dependent piece rendering.
- Convert only in `Backboned.preprocess_image`: changes every model and training caller while leaving the extraction warp unnormalized.
- Fix only `BoardRecognitionHelper.recognize`: leaves the README-advertised `BoardDetector.predict_coords`, extraction, and marking APIs inconsistent.
- Add a private pre-normalized prediction path: bypasses the existing public `predict_coords` monkeypatch/override seam and adds machinery for a three-line normalization fix.

## Regression Boundary

Use `tests/test_board_detector_model.py` only:

- Extend the existing model stub so real `predict_coords()` records the Pillow image passed to preprocessing and returns deterministic coordinates.
- One four-way parameterized regression covers Pillow RGB, Pillow RGBA, Pillow L, and an OpenCV BGR ndarray. Preprocessing must receive one RGB image with preserved size/color, return eight coordinates, and leave the source unchanged.
- Parameterize the existing marked-copy regression over RGB, RGBA, and L. The source remains identical and the marked result is RGB.
- Add one two-way file-backed extraction regression for RGBA and L PNGs. The real geometry/warp path must return a three-channel image; only coordinate prediction is replaced.

The current code must fail the non-RGB cases before the production edit and all focused tests must pass afterward.

## Benchmark Acceptance

Rerun the same anchored 568-case corpus with the same three checkpoint hashes and MPS runtime to a new, non-overwriting report. Require:

- zero execution errors for all 24 input-contract cases;
- zero execution errors for the unchanged 544-case RGB main suite;
- disappearance of both historical channel-mismatch messages;
- canonical report integrity and exact score recomputation.

Measure every accuracy, rejection, and failure-reason change from the new report. Do not predict an improvement or call the synthetic corpus unseen-site accuracy. Compare the 544 RGB predictions with the baseline and investigate any drift rather than assuming stability.

## Documentation and Scope

Update the README baseline note and P2-01/P2-05 audit evidence with exact before/after metrics and report hashes. Mark P2-05 resolved only after focused/full tests, compilation, anchored corpus verification, the real benchmark rerun, independent report validation, and final review pass.

Do not stage, commit, retrain, replace checkpoints, regenerate the dataset, or claim P2-01 complete.
