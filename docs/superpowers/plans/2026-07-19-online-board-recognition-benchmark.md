# Online Board Recognition Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove a deterministic 568-case synthetic online-board regression benchmark and exact-placement evaluator, then generate the final corpus only after every pre-generation gate passes.

**Architecture:** A top-level `benchmarks` package stays independent of `chessml` so generation never imports ignored `config.local.yaml` or eager raw-asset discovery. A reviewed JSON source specification is the semantic oracle; the generated manifest is reconstructed from it and anchored by a separate digest. Rendering is a pure Pillow/fentoboardimage path, while a thin validation CLI is the only layer that imports the current model stack.

**Tech Stack:** Python 3.14.6, standard-library JSON/hashlib/tempfile, python-chess 1.11.2, fenToBoardImage 1.4.1, Pillow 12.3.0, pytest 9.1.1.

## Global Constraints

- Governing design: `docs/superpowers/specs/2026-07-19-online-board-recognition-benchmark-design.md`.
- Version 1 is named `online-render-v1` and claims only synthetic online-render regression coverage, never unseen-site accuracy.
- Do not run `generate_augmented_boards.py`, `AugmentedBoardsImages`, `BoardsImagesFromFENs`, `export_unique_fens.py`, `download_piece_sets.py`, or `reset_dir`.
- Do not import `chessml`, `chessml.data.assets`, or `config.local.yaml` from benchmark source validation or generation code.
- Do not use global RNG, threads, perspective, cropping, occlusion, arrows, last-move highlights, coordinate overlays, photographic backgrounds, or lossy output files.
- The full plan is exactly 568 cases: 528 positive and 40 negative. Main RGB contains 512 positive and 32 negative; input-contract contains 16 positive and 8 negative.
- A positive black-bottom image is labeled with the expanded canonical placement rotated 180 degrees and recompressed.
- JSON is canonical UTF-8: `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False`, plus one trailing newline.
- Existing destinations are refused. Writes use a unique sibling staging directory and one atomic final rename.
- No final benchmark generation is permitted before Task 6 is complete and both Task 6 reviewers report no Critical or Important findings.
- P1-05 remains a prerequisite for an accepted current-checkpoint baseline. This plan implements the runner but does not run or claim that baseline.
- Do not stage or commit any file.
- `git diff --check` does not inspect new untracked files. Every task review must
  also read each new file directly and search it for trailing whitespace; final
  reviewers must use `git diff --no-index /dev/null <file>` or direct file reads.
  Do not use `git add -N` to make untracked files visible.

## File Map

- Create `benchmarks/__init__.py`: makes the benchmark tooling importable without importing `chessml`.
- Create `benchmarks/board_recognition.py`: strict placement oracle, source/case validation, hashes, manifest verification, prediction scoring.
- Create `benchmarks/render_online_boards.py`: deterministic rendering, composition, canary selection, transactional writer.
- Create `benchmarks/board_recognition_v1.json`: reviewed source positions, asset hashes/licenses, layouts, and exact counts.
- Create `benchmarks/board_recognition_v1_manifest.sha256`: only in Task 7, after final generation and verification.
- Create `scripts/data/generate_board_recognition_benchmark.py`: read-only-by-default preflight/generation CLI.
- Create `scripts/validate/evaluate_board_recognition_benchmark.py`: fresh-`Picture` checkpoint runner and report writer.
- Create `tests/test_board_recognition_benchmark.py`: oracle, source, case-plan, scorer, manifest-trust tests.
- Create `tests/test_online_board_benchmark_generator.py`: renderer, full-matrix binding, determinism, and transactional-output tests.
- Modify `README.md`: document the synthetic claim, preflight/generation/verification commands, and P1-05 baseline prerequisite.
- Modify `BOARD_RECOGNITION_AUDIT.tmp.md`: record the delivered synthetic half of P2-01 without claiming the independent screenshot half or baseline is complete.

---

### Task 1: Freeze Sources and Implement the Independent Placement Oracle

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/board_recognition.py`
- Create: `benchmarks/board_recognition_v1.json`
- Create: `tests/test_board_recognition_benchmark.py`

**Interfaces:**
- Produces: `BenchmarkValidationError`, `load_json()`, `canonical_json_bytes()`, `expand_placement()`, `compress_placement()`, `rotate_placement()`, `file_sha256()`, `pixel_sha256()`, `piece_set_tree_sha256()`, `load_source_spec()`, and `validate_source_spec()`.
- Also produces: `source_spec_sha256()` and `validate_piece_set()` for later
  planner/generator gates.
- Consumes: no `chessml` modules and no mutable configuration.

- [ ] **Step 1: Write strict placement and canonical-JSON tests**

Add tests that establish the independent label grammar:

```python
import json
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from benchmarks.board_recognition import (
    BenchmarkValidationError,
    canonical_json_bytes,
    compress_placement,
    expand_placement,
    file_sha256,
    pixel_sha256,
    rotate_placement,
)


ASYMMETRIC = "r3k2r/8/8/3p4/8/2N5/8/R3K2R"


def test_placement_round_trip_and_rotation_are_source_ordered():
    squares = expand_placement(ASYMMETRIC)
    assert len(squares) == 64
    assert compress_placement(squares) == ASYMMETRIC
    assert rotate_placement(rotate_placement(ASYMMETRIC)) == ASYMMETRIC
    assert rotate_placement(ASYMMETRIC) == "R2K3R/8/5N2/8/4p3/8/8/r2k3r"


@pytest.mark.parametrize(
    "placement",
    [None, "", "8/8", "9/8/8/8/8/8/8/8", "44/8/8/8/8/8/8/8", "11111111/8/8/8/8/8/8/8", "K/8/8/8/8/8/8/8", "x7/8/8/8/8/8/8/8"],
)
def test_placement_parser_rejects_non_64_square_inputs(placement):
    with pytest.raises(BenchmarkValidationError):
        expand_placement(placement)


def test_canonical_json_is_stable_and_has_one_newline():
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_file_and_decoded_pixel_hashes_have_independent_fixtures(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    assert file_sha256(payload) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    image = Image.new("RGB", (2, 1), (1, 2, 3))
    expected = hashlib.sha256(b"RGB\0" + b"2x1\0" + bytes((1, 2, 3, 1, 2, 3)))
    assert pixel_sha256(image) == expected.hexdigest()
```

- [ ] **Step 2: Run the placement tests and confirm RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py
```

Expected: collection fails because `benchmarks.board_recognition` does not exist.

- [ ] **Step 3: Add the source specification with the exact reviewed inputs**

Create `benchmarks/__init__.py` empty, then create `benchmarks/board_recognition_v1.json` with this exact semantic content (canonical formatting is not required for the reviewed source file):

```json
{
  "base_position_count": 16,
  "benchmark_version": "online-render-v1",
  "claim": "synthetic_online_render_regression_only",
  "dependencies": {
    "Pillow": "12.3.0",
    "chess": "1.11.2",
    "fentoboardimage": "1.4.1",
    "python": "3.14.6"
  },
  "expected_counts": {
    "all": 568,
    "input_contract_negative": 8,
    "input_contract_positive": 16,
    "main_negative": 32,
    "main_positive": 512,
    "negative": 40,
    "positive": 528
  },
  "case_matrix": {
    "input_contract": {
      "channel_modes": ["RGBA", "L"],
      "layout_id": "board_crop",
      "negative_template_ids": ["lobby_cards", "modal_dialog", "analysis_panels", "profile_cards"],
      "palette_id": "brown",
      "piece_set_id": "lichess_chessnut",
      "position_ids": ["pos_01", "pos_06", "pos_11", "pos_16"],
      "quality_id": "clean"
    },
    "main": {
      "channel_mode": "RGB",
      "layout_ids": ["board_crop", "desktop_ui"],
      "palette_index_formula": "(position_index + piece_set_index) % 4",
      "quality_ids": ["clean", "browser_scaled"]
    },
    "views": ["white_bottom", "black_bottom"]
  },
  "layouts": [
    {"board_box": [0, 0, 512, 512], "canvas_size": [512, 512], "id": "board_crop"},
    {"board_box": [24, 30, 504, 510], "canvas_size": [960, 540], "id": "desktop_ui"}
  ],
  "limitations": [
    "Variants share 16 base positions; 568 is not an independent-position count.",
    "Current-checkpoint holdout cannot be proved because its training manifest is unavailable.",
    "This synthetic corpus does not measure accuracy on unseen chess sites.",
    "P2-01 remains open until an independently captured and manually labeled online-screenshot suite exists."
  ],
  "negative_templates": [
    {"id": "lobby_cards", "split": "development"},
    {"id": "modal_dialog", "split": "development"},
    {"id": "analysis_panels", "split": "acceptance"},
    {"id": "profile_cards", "split": "acceptance"},
    {"id": "settings_panel", "split": "acceptance"},
    {"id": "loading_skeleton", "split": "acceptance"},
    {"id": "connection_error", "split": "acceptance"},
    {"id": "tournament_table", "split": "acceptance"}
  ],
  "palettes": [
    {"dark": "#B58862", "id": "brown", "light": "#F0D9B5"},
    {"dark": "#4675AB", "id": "blue", "light": "#C3CDD7"},
    {"dark": "#85A666", "id": "green", "light": "#FFFFDC"},
    {"dark": "#6C6360", "id": "gray", "light": "#CCCDCD"}
  ],
  "piece_sets": [
    {
      "author": "Alexis Luengas",
      "id": "lichess_chessnut",
      "license": "Apache-2.0",
      "path": "assets/piece_png/lichess_chessnut",
      "tree_sha256": "ab9628638c8266b8f7e9a3f36f9d518c5de59042bf0c0c5a09c6b8384654ef3b",
      "upstream": "https://github.com/lichess-org/lila/tree/940e876ef193e8c031f42945f6f7773c71764395/public/piece/chessnut"
    },
    {
      "author": "Maurizio Monge",
      "id": "lichess_celtic",
      "license": "MIT",
      "path": "assets/piece_png/lichess_celtic",
      "tree_sha256": "5fc39e440f5aba5cff5cebce938c6fa32c1f6ac8ef5b3060861f2fea0a4ec397",
      "upstream": "https://github.com/lichess-org/lila/tree/940e876ef193e8c031f42945f6f7773c71764395/public/piece/celtic"
    },
    {
      "author": "Maurizio Monge",
      "id": "lichess_fantasy",
      "license": "MIT",
      "path": "assets/piece_png/lichess_fantasy",
      "tree_sha256": "d8b894b8f0952cc6c0db654cc47681f818f4c7f7ae00814b54dbff5e1aab52cd",
      "upstream": "https://github.com/lichess-org/lila/tree/940e876ef193e8c031f42945f6f7773c71764395/public/piece/fantasy"
    },
    {
      "author": "Maurizio Monge",
      "id": "lichess_spatial",
      "license": "MIT",
      "path": "assets/piece_png/lichess_spatial",
      "tree_sha256": "fc033d2aff36ff3f0edee801ac2aa01b1a1d0c2b20cb95dd2fd1065c48646d0d",
      "upstream": "https://github.com/lichess-org/lila/tree/940e876ef193e8c031f42945f6f7773c71764395/public/piece/spatial"
    }
  ],
  "positions": [
    {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "id": "pos_01", "source": {"kind": "python_chess_starting_fen"}, "split": "development"},
    {"fen": "2b5/1p4k1/2p1pq1r/3p2p1/r5P1/N3PB1p/4QP1K/R3R3 w - - 0 1", "id": "pos_02", "source": {"kind": "unique_fens", "line": 1, "record": "2b5/1p4k1/2p1pq1r/3p2p1/r5P1/N3PB1p/4QP1K/R3R3 w - -"}, "split": "acceptance"},
    {"fen": "4rbk1/1b3ppp/p7/3pP1P1/pPrN2P1/P7/1BP1R2P/R5K1 b - - 0 1", "id": "pos_03", "source": {"kind": "unique_fens", "line": 2, "record": "4rbk1/1b3ppp/p7/3pP1P1/pPrN2P1/P7/1BP1R2P/R5K1 b - -"}, "split": "acceptance"},
    {"fen": "8/5k2/1B4p1/7p/R7/7P/5PPK/b2r4 b - - 0 1", "id": "pos_04", "source": {"kind": "unique_fens", "line": 3, "record": "8/5k2/1B4p1/7p/R7/7P/5PPK/b2r4 b - -"}, "split": "acceptance"},
    {"fen": "8/5k2/2R5/3p4/4p3/2K1P2N/r5PP/8 b - - 0 1", "id": "pos_05", "source": {"kind": "unique_fens", "line": 4, "record": "8/5k2/2R5/3p4/4p3/2K1P2N/r5PP/8 b - -"}, "split": "acceptance"},
    {"fen": "3r3k/5npp/4b3/Q3p3/1p2B2P/8/P7/6K1 b - - 0 1", "id": "pos_06", "source": {"kind": "unique_fens", "line": 5, "record": "3r3k/5npp/4b3/Q3p3/1p2B2P/8/P7/6K1 b - -"}, "split": "development"},
    {"fen": "1K6/8/8/8/1R5p/5k2/5P2/8 b - - 0 1", "id": "pos_07", "source": {"kind": "unique_fens", "line": 6, "record": "1K6/8/8/8/1R5p/5k2/5P2/8 b - -"}, "split": "acceptance"},
    {"fen": "5q2/1np4k/1p1p3p/1P5n/2P3pB/2Q5/5PP1/4R1K1 w - - 0 1", "id": "pos_08", "source": {"kind": "unique_fens", "line": 7, "record": "5q2/1np4k/1p1p3p/1P5n/2P3pB/2Q5/5PP1/4R1K1 w - -"}, "split": "acceptance"},
    {"fen": "3r4/2R2b1k/5P2/1p2KP1p/p2B4/P7/1P6/8 b - - 0 1", "id": "pos_09", "source": {"kind": "unique_fens", "line": 8, "record": "3r4/2R2b1k/5P2/1p2KP1p/p2B4/P7/1P6/8 b - -"}, "split": "acceptance"},
    {"fen": "r4rk1/pbqn1ppp/2Pb1n2/4p3/1p2P3/1BN2N2/PPQ2PPP/R1BR2K1 b - - 0 1", "id": "pos_10", "source": {"kind": "unique_fens", "line": 9, "record": "r4rk1/pbqn1ppp/2Pb1n2/4p3/1p2P3/1BN2N2/PPQ2PPP/R1BR2K1 b - -"}, "split": "acceptance"},
    {"fen": "8/8/8/1B4p1/4np2/4k2P/6K1/8 b - - 0 1", "id": "pos_11", "source": {"kind": "unique_fens", "line": 10, "record": "8/8/8/1B4p1/4np2/4k2P/6K1/8 b - -"}, "split": "development"},
    {"fen": "4q1k1/4r1b1/1p1p2pp/prpPnp2/Q7/6PP/RP1BPPB1/1R4K1 w - - 0 1", "id": "pos_12", "source": {"kind": "unique_fens", "line": 12, "record": "4q1k1/4r1b1/1p1p2pp/prpPnp2/Q7/6PP/RP1BPPB1/1R4K1 w - -"}, "split": "acceptance"},
    {"fen": "8/1k4p1/5b2/5P1P/8/pN1K4/8/8 w - - 0 1", "id": "pos_13", "source": {"kind": "unique_fens", "line": 14, "record": "8/1k4p1/5b2/5P1P/8/pN1K4/8/8 w - -"}, "split": "acceptance"},
    {"fen": "1r3rk1/2qb2pp/8/p1n1p3/2P1B3/PPQp2P1/7P/2NRR1K1 b - - 0 1", "id": "pos_14", "source": {"kind": "unique_fens", "line": 16, "record": "1r3rk1/2qb2pp/8/p1n1p3/2P1B3/PPQp2P1/7P/2NRR1K1 b - -"}, "split": "acceptance"},
    {"fen": "r4rk1/3bqppp/3b1nn1/p3p3/Pp2P3/1P1BBP2/3NNQPP/2R2RK1 b - - 0 1", "id": "pos_15", "source": {"kind": "unique_fens", "line": 18, "record": "r4rk1/3bqppp/3b1nn1/p3p3/Pp2P3/1P1BBP2/3NNQPP/2R2RK1 b - -"}, "split": "acceptance"},
    {"fen": "8/4Q3/8/K7/1p6/k2q4/8/8 b - - 0 1", "id": "pos_16", "source": {"kind": "unique_fens", "line": 48, "record": "8/4Q3/8/K7/1p6/k2q4/8/8 b - -"}, "split": "development"}
  ],
  "qualities": [
    {"id": "clean"},
    {"downsample": [3, 4], "downsample_filter": "BILINEAR", "id": "browser_scaled", "upsample_filter": "BICUBIC"}
  ],
  "renderer": {
    "piece_resampling": "LANCZOS",
    "png_compress_level": 9,
    "png_optimize": false
  },
  "schema_version": 1,
  "source_selection": {
    "unique_fens_lines": 22641843,
    "unique_fens_sha256": "9e6bc34257368bc1518df56dfc4ae42411de932bf265edc17d8bc4ee740d32c7",
    "upstream_copying": "https://github.com/lichess-org/lila/blob/940e876ef193e8c031f42945f6f7773c71764395/COPYING.md"
  },
  "ui_recipes": {
    "coordinate_space": [1000, 1000],
    "negative_templates": {
      "analysis_panels": {
        "background": "#211F1D",
        "shapes": [
          {"box": [30, 70, 290, 930], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [320, 70, 660, 930], "fill": "#262421", "kind": "rectangle"},
          {"box": [690, 70, 970, 930], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [370, 180, 610, 230], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [370, 310, 560, 360], "fill": "#6B8E4E", "kind": "rectangle"},
          {"box": [370, 440, 625, 490], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [370, 570, 520, 620], "fill": "#B58862", "kind": "rectangle"}
        ]
      },
      "connection_error": {
        "background": "#262421",
        "shapes": [
          {"box": [170, 170, 830, 830], "fill": "#312E2B", "kind": "rectangle"},
          {"fill": "#D59020", "kind": "polygon", "points": [[500, 260], [350, 540], [650, 540]]},
          {"box": [485, 355, 515, 455], "fill": "#262421", "kind": "rectangle"},
          {"box": [485, 480, 515, 510], "fill": "#262421", "kind": "ellipse"},
          {"box": [360, 650, 640, 745], "fill": "#6B8E4E", "kind": "rectangle"}
        ]
      },
      "loading_skeleton": {
        "background": "#262421",
        "shapes": [
          {"box": [0, 0, 1000, 90], "fill": "#171513", "kind": "rectangle"},
          {"box": [80, 170, 720, 250], "fill": "#3A3734", "kind": "rectangle"},
          {"box": [80, 310, 900, 390], "fill": "#34312E", "kind": "rectangle"},
          {"box": [80, 450, 610, 530], "fill": "#3A3734", "kind": "rectangle"},
          {"box": [80, 590, 820, 670], "fill": "#34312E", "kind": "rectangle"},
          {"box": [80, 730, 500, 810], "fill": "#3A3734", "kind": "rectangle"}
        ]
      },
      "lobby_cards": {
        "background": "#262421",
        "shapes": [
          {"box": [0, 0, 1000, 90], "fill": "#171513", "kind": "rectangle"},
          {"box": [70, 140, 930, 340], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [100, 180, 220, 300], "fill": "#6B8E4E", "kind": "ellipse"},
          {"box": [270, 195, 850, 235], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [70, 400, 930, 600], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [100, 440, 220, 560], "fill": "#B58862", "kind": "ellipse"},
          {"box": [270, 455, 760, 495], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [70, 660, 930, 860], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [100, 700, 220, 820], "fill": "#4675AB", "kind": "ellipse"},
          {"box": [270, 715, 810, 755], "fill": "#4A4744", "kind": "rectangle"}
        ]
      },
      "modal_dialog": {
        "background": "#1B1917",
        "shapes": [
          {"box": [0, 0, 1000, 1000], "fill": "#24211F", "kind": "rectangle"},
          {"box": [160, 200, 840, 800], "fill": "#3A3734", "kind": "rectangle"},
          {"box": [220, 270, 780, 340], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [220, 410, 700, 465], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [220, 510, 650, 565], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [250, 650, 470, 735], "fill": "#5A5652", "kind": "rectangle"},
          {"box": [530, 650, 750, 735], "fill": "#6B8E4E", "kind": "rectangle"}
        ]
      },
      "profile_cards": {
        "background": "#262421",
        "shapes": [
          {"box": [50, 90, 950, 330], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [90, 135, 240, 285], "fill": "#4675AB", "kind": "ellipse"},
          {"box": [290, 155, 850, 205], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [290, 235, 690, 275], "fill": "#3A3734", "kind": "rectangle"},
          {"box": [50, 390, 950, 630], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [90, 435, 240, 585], "fill": "#B58862", "kind": "ellipse"},
          {"box": [290, 455, 790, 505], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [290, 535, 620, 575], "fill": "#3A3734", "kind": "rectangle"},
          {"box": [50, 690, 950, 930], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [90, 735, 240, 885], "fill": "#6B8E4E", "kind": "ellipse"},
          {"box": [290, 755, 830, 805], "fill": "#4A4744", "kind": "rectangle"}
        ]
      },
      "settings_panel": {
        "background": "#211F1D",
        "shapes": [
          {"box": [130, 70, 870, 930], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [200, 170, 580, 220], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [690, 155, 790, 235], "fill": "#6B8E4E", "kind": "ellipse"},
          {"box": [200, 350, 640, 400], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [690, 335, 790, 415], "fill": "#5A5652", "kind": "ellipse"},
          {"box": [200, 530, 530, 580], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [690, 515, 790, 595], "fill": "#6B8E4E", "kind": "ellipse"},
          {"box": [200, 710, 610, 760], "fill": "#4A4744", "kind": "rectangle"},
          {"box": [690, 695, 790, 775], "fill": "#5A5652", "kind": "ellipse"}
        ]
      },
      "tournament_table": {
        "background": "#262421",
        "shapes": [
          {"box": [70, 90, 930, 210], "fill": "#171513", "kind": "rectangle"},
          {"box": [70, 210, 930, 340], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [70, 340, 930, 470], "fill": "#3A3734", "kind": "rectangle"},
          {"box": [70, 470, 930, 600], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [70, 600, 930, 730], "fill": "#3A3734", "kind": "rectangle"},
          {"box": [70, 730, 930, 860], "fill": "#312E2B", "kind": "rectangle"},
          {"box": [270, 90, 278, 860], "fill": "#5A5652", "kind": "rectangle"},
          {"box": [500, 90, 508, 860], "fill": "#5A5652", "kind": "rectangle"},
          {"box": [730, 90, 738, 860], "fill": "#5A5652", "kind": "rectangle"}
        ]
      }
    },
    "positive_desktop": {
      "background": "#262421",
      "shapes": [
        {"box": [0, 0, 1000, 45], "fill": "#171513", "kind": "rectangle"},
        {"box": [550, 56, 980, 944], "fill": "#312E2B", "kind": "rectangle"},
        {"box": [590, 120, 940, 180], "fill": "#4A4744", "kind": "rectangle"},
        {"box": [590, 240, 880, 300], "fill": "#3A3734", "kind": "rectangle"},
        {"box": [590, 360, 920, 420], "fill": "#4A4744", "kind": "rectangle"},
        {"box": [590, 760, 760, 860], "fill": "#B58862", "kind": "rectangle"},
        {"box": [790, 760, 960, 860], "fill": "#6B8E4E", "kind": "rectangle"},
        {"box": [0, 960, 1000, 1000], "fill": "#171513", "kind": "rectangle"}
      ]
    }
  }
}
```

- [ ] **Step 4: Implement the strict oracle and deterministic hashes**

In `benchmarks/board_recognition.py`, implement these exact boundaries:

```python
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import zlib
from importlib.metadata import version
from pathlib import Path
from typing import Iterable, Sequence

import chess
from PIL import Image


PIECE_SYMBOLS = frozenset("prnbqkPRNBQK")
PIECE_NAMES = ("Pawn", "Rook", "Knight", "Bishop", "Queen", "King")
EXPECTED_SOURCE_SPEC_SHA256 = "8393ff76e9f10b4bbe24b558f9b2d0032c493ebb6fe1781a4515b4790018756b"
ALLOWED_PIECE_PATHS = tuple(
    f"{color}/{piece}.png"
    for color in ("black", "white")
    for piece in PIECE_NAMES
)


class BenchmarkValidationError(ValueError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def expand_placement(placement: str) -> tuple:
    if not isinstance(placement, str):
        raise BenchmarkValidationError("placement must be a string")
    rows = placement.split("/")
    if len(rows) != 8:
        raise BenchmarkValidationError("placement must contain eight ranks")
    squares: list[str | None] = []
    for row in rows:
        expanded: list[str | None] = []
        for symbol in row:
            if symbol in "12345678":
                expanded.extend([None] * int(symbol))
            elif symbol in PIECE_SYMBOLS:
                expanded.append(symbol)
            else:
                raise BenchmarkValidationError(f"invalid placement symbol: {symbol!r}")
        if len(expanded) != 8:
            raise BenchmarkValidationError("every rank must expand to eight squares")
        squares.extend(expanded)
    result = tuple(squares)
    if compress_placement(result) != placement:
        raise BenchmarkValidationError("placement must use canonical FEN compression")
    return result


def compress_placement(squares: Sequence[str | None]) -> str:
    if len(squares) != 64:
        raise BenchmarkValidationError("source grid must contain 64 squares")
    rows: list[str] = []
    for start in range(0, 64, 8):
        row: list[str] = []
        empty = 0
        for symbol in squares[start : start + 8]:
            if symbol is None:
                empty += 1
            else:
                if symbol not in PIECE_SYMBOLS:
                    raise BenchmarkValidationError(f"invalid piece symbol: {symbol!r}")
                if empty:
                    row.append(str(empty))
                    empty = 0
                row.append(symbol)
        if empty:
            row.append(str(empty))
        rows.append("".join(row))
    return "/".join(rows)


def rotate_placement(placement: str) -> str:
    return compress_placement(tuple(reversed(expand_placement(placement))))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(b"\0")
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


```

- [ ] **Step 5: Run the strict-oracle tests and confirm GREEN**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py -k 'placement or canonical_json or hash'
```

Expected: the strict parser/hash primitives now pass; source validation is not
implemented yet.

- [ ] **Step 6: Write source-spec and asset-boundary tests before validation code**

```python
from benchmarks.board_recognition import (
    EXPECTED_SOURCE_SPEC_SHA256,
    load_source_spec,
    source_spec_sha256,
    validate_source_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = ROOT / "benchmarks/board_recognition_v1.json"


def test_version_1_source_spec_is_semantically_valid():
    spec = load_source_spec(SOURCE_SPEC)
    assert source_spec_sha256(spec) == EXPECTED_SOURCE_SPEC_SHA256
    validate_source_spec(spec, ROOT, verify_local_assets=False)


def test_provisioned_version_1_assets_match_reviewed_hashes():
    if not (ROOT / "assets/piece_png/lichess_chessnut").is_dir():
        pytest.skip("requires ignored, locally provisioned benchmark assets")
    spec = load_source_spec(SOURCE_SPEC)
    validate_source_spec(spec, ROOT, verify_local_assets=True)
```

Before implementing validation, add deep-copy mutation cases for every frozen
boundary: schema/dependency version, base count/limitation, case-matrix literal,
one UI-recipe color/coordinate/kind/order, duplicate ID, malformed/noncanonical
FEN, source-record mismatch, missing king, direct/rotated duplicate placement,
low-contrast palette, expected count, renderer setting, license/hash/path, and
unexpected recipe/template key. Assert a boundary-specific error-message fragment
for every semantically invalid mutation, proving the corresponding check runs
before the final digest gate. For semantically well-formed but frozen-value
changes (such as an alternate valid UI color/order), assert the specific source
digest mismatch. A no-op or digest-only semantic validator must fail these tests.
Write raw JSON fixtures proving `load_json()` rejects a nested duplicate key,
`NaN`, `Infinity`, and a non-object root even when the parsed canonical object
could otherwise match.

Add `validate_piece_set(directory, expected_tree_sha256)` tests under `tmp_path`.
Build a hermetic valid 12-file tree of unique deterministic 100-by-100 RGBA images
with nonempty alpha, compute its expected tree hash, then independently prove
rejection for a wrong expected hash, missing/extra/nested/uppercase PNG,
symlinked color directory or PNG, undecodable PNG, wrong size, non-RGBA mode,
all-zero alpha, and duplicate file bytes. For shape/mode/alpha/duplicate
mutations, calculate the mutated tree hash first so the test reaches
decoded-asset validation rather than stopping at the hash mismatch. Keep the
separate real four-set preflight above and require it not to skip in this
workspace; only that integration gate may skip on an unprovisioned clean clone.
Monkeypatch `platform.python_version()` and the imported metadata `version()`
lookup separately to return wrong values, and require local-asset preflight to
reject each; mutating only the source dependency dictionary does not prove the
runtime comparison exists.

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py -k 'source_spec or asset or piece_set'
```

Expected: failures because `load_source_spec()`, `source_spec_sha256()`,
`validate_piece_set()`, and `validate_source_spec()` are absent.

- [ ] **Step 7: Implement source and asset validation**

Implement `load_source_spec(path)` through `load_json(path)`,
`source_spec_sha256(spec)` through SHA-256 of `canonical_json_bytes(spec)`, and
`validate_source_spec(spec, repo_root, *, verify_local_assets)`.

Implement strict JSON loading and the exact tree hash now that their failure
tests are RED:

```python
def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise BenchmarkValidationError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise BenchmarkValidationError(f"invalid JSON constant: {value}")


def load_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise BenchmarkValidationError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{path} must contain a JSON object")
    return value


def piece_set_tree_sha256(directory: Path) -> str:
    if directory.is_symlink() or not directory.is_dir():
        raise BenchmarkValidationError(f"{directory} must be a real directory")
    for color in ("black", "white"):
        color_dir = directory / color
        if color_dir.is_symlink() or not color_dir.is_dir():
            raise BenchmarkValidationError(f"{color_dir} must be a real directory")
    pngs = tuple(
        path for path in directory.rglob("*") if path.suffix.lower() == ".png"
    )
    if any(path.is_symlink() or not path.is_file() for path in pngs):
        raise BenchmarkValidationError(f"{directory} contains a non-regular PNG")
    allowed = set(ALLOWED_PIECE_PATHS)
    actual = {path.relative_to(directory).as_posix() for path in pngs}
    if actual != allowed:
        raise BenchmarkValidationError(
            f"{directory} PNG inventory differs: "
            f"missing={sorted(allowed - actual)}, extra={sorted(actual - allowed)}"
        )
    digest = hashlib.sha256()
    for relative in sorted(ALLOWED_PIECE_PATHS):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((directory / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
```

Perform explicit semantic checks first so diagnostics identify the violated
boundary, then require `source_spec_sha256(spec) ==
EXPECTED_SOURCE_SPEC_SHA256` before returning. Require exact
schema/version/claim/dependencies,
`case_matrix`, `renderer`, `base_position_count`, limitations, recipe key sets,
coordinate space, background/fill colors, allowed shape kinds, geometry key sets,
coordinate integer ranges/order, and shape list order as represented by the
anchored JSON. Require negative-recipe keys to equal the negative-template ID
set exactly. Require numeric counts/coordinates to have `type(value) is int` so
JSON booleans are rejected. Validate all IDs for uniqueness; append `0 1` only when checking
the frozen four-field `source.record`; require it equals the stored six-field FEN;
require `chess.Board(fen).is_valid()`; require exactly one king per color; require
direct/rotated pairwise placement uniqueness; require all twelve symbols and
piece-count buckets `2..8`, `9..20`, and `21..32`; require development position
indexes `[0, 5, 10, 15]`; require palette RGB distance at least 64 and grayscale
luminance difference at least 40. Do not evaluate the formula string.

`validate_piece_set()` first calls the exact inventory/tree hash function, then
decodes every allowed file and requires `(100, 100)`, mode `RGBA`, nonempty alpha,
and unique file digests. With `verify_local_assets=True`, verify exact installed
package versions and all four resolved real, non-symlink piece-set paths through
that helper. Do not open `datasets/unique_fens.txt`.

- [ ] **Step 8: Run Task 1 tests and confirm GREEN**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py
```

Expected: every Task 1 test passes; the local-source test must run rather than skip in this workspace.

- [ ] **Step 9: Review Task 1 without staging or committing**

Run `git diff --check` and inspect only the four Task 1 files. Confirm importing
`benchmarks.board_recognition` does not add `chessml` to `sys.modules`.

---

### Task 2: Implement the Anti-Tautological Pure Scorer

**Files:**
- Modify: `benchmarks/board_recognition.py`
- Modify: `tests/test_board_recognition_benchmark.py`

**Interfaces:**
- Consumes: strict placement/hash functions from Task 1.
- Produces: `score_predictions()` for the downstream checkpoint runner.

- [ ] **Step 1: Write metric mutation tests before scorer code**

Use a hand-built positive and negative case independent of model constraints:

```python
EXPECTED = "8/8/8/3p4/4P3/8/8/4K2k"
WRONG_PIECE = "8/8/8/3q4/4P3/8/8/4K2k"
MISSING_PIECE = "8/8/8/8/4P3/8/8/4K2k"
EXTRA_PIECE = "N7/8/8/3p4/4P3/8/8/4K2k"


def cases():
    return [
        {
            "id": "positive",
            "board_present": True,
            "source_placement": EXPECTED,
            "suite": "main",
            "split": "development",
        },
        {
            "id": "negative",
            "board_present": False,
            "suite": "main",
            "split": "development",
            "negative_template": "dialog",
        },
    ]


def test_one_wrong_symbol_fails_exact_board_without_hiding_occupancy():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "success", "source_placement": WRONG_PIECE},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["exact_placement"]["value"] == 0
    assert report["overall"]["square_accuracy"]["value"] == pytest.approx(63 / 64)
    assert report["overall"]["occupancy_precision"]["value"] == 1
    assert report["overall"]["occupancy_recall"]["value"] == 1
    assert report["overall"]["conditional_piece_accuracy"] == {
        "correct": 3,
        "total": 4,
        "value": 0.75,
    }


def test_occupied_to_empty_error_reduces_recall_not_precision():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "success", "source_placement": MISSING_PIECE},
            {"id": "negative", "outcome": "failure", "reason": "INVALID_GEOMETRY"},
        ],
    )
    assert report["overall"]["occupancy_precision"]["value"] == 1
    assert report["overall"]["occupancy_recall"]["value"] == 0.75
    assert report["overall"]["negative_any_rejection_rate"]["value"] == 1
    assert report["overall"]["negative_no_board_rate"]["value"] == 0


def test_empty_to_occupied_error_reduces_precision_not_recall():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "success", "source_placement": EXTRA_PIECE},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["square_accuracy"]["value"] == pytest.approx(63 / 64)
    assert report["overall"]["occupancy_precision"]["value"] == 0.8
    assert report["overall"]["occupancy_recall"]["value"] == 1
    assert report["overall"]["occupancy"] == {"tp": 4, "fp": 1, "fn": 0}


def test_failure_on_positive_penalizes_every_square_and_no_board_false_rejection():
    report = score_predictions(
        cases(),
        [
            {"id": "positive", "outcome": "failure", "reason": "NO_BOARD"},
            {"id": "negative", "outcome": "failure", "reason": "NO_BOARD"},
        ],
    )
    assert report["overall"]["square_accuracy"] == {
        "correct": 0,
        "total": 64,
        "value": 0,
    }
    assert report["overall"]["occupancy_recall"]["value"] == 0
    assert report["overall"]["conditional_piece_accuracy"]["value"] is None
    assert report["overall"]["positive_false_no_board_rate"]["value"] == 1
```

Also add tests that a rotated placement fails exact placement; success on the
negative yields zero rejection; execution error increments the error count and
penalizes all positive squares; missing, duplicate, extra, or invalid predictions
raise `BenchmarkValidationError`; a noncanonical success such as ranks encoded
with `44` is invalid; grouped results include `suite` and `split`.

- [ ] **Step 2: Run the scorer tests and confirm RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py -k 'wrong_symbol or occupied_to_empty or empty_to_occupied or failure_on_positive or prediction'
```

Expected: import or attribute failure because `score_predictions()` is absent.

- [ ] **Step 3: Implement scoring with explicit numerators and denominators**

Use these exact prediction outcomes and metric helpers:

```python
PREDICTION_OUTCOMES = frozenset({"success", "failure", "error"})
FAILURE_REASONS = frozenset(
    {"NO_BOARD", "INVALID_GEOMETRY", "DECODING_FAILED", "INVALID_PLACEMENT"}
)


def _ratio(correct: int, total: int) -> dict:
    return {
        "correct": correct,
        "total": total,
        "value": None if total == 0 else correct / total,
    }


GROUP_FIELDS = (
    "suite",
    "split",
    "base_position",
    "piece_set",
    "palette",
    "view",
    "layout",
    "quality",
    "channel_mode",
    "negative_template",
)
```

Implement `_score_group(cases, prediction_by_id)` as one direct loop. For a
positive success, expand expected and predicted placements and accumulate exact
matches, 64 square comparisons, occupancy TP/FP/FN, and symbol matches only where
both grids are occupied. For a positive failure or error, add 64 square attempts
with zero correct and add one FN per expected occupied square. For a negative,
count only typed failures as rejections, and count only `NO_BOARD` as exact
no-board rejections. Count a positive `NO_BOARD` failure as a false rejection;
errors never count as rejections. Return exactly these keys:

```python
{
    "counts": {"all": int, "positive": int, "negative": int,
               "success": int, "failure": int, "error": int},
    "positive_success_rate": _ratio(positive_successes, positive_count),
    "exact_placement": _ratio(exact_positive_placements, positive_count),
    "square_accuracy": _ratio(correct_positive_squares, positive_count * 64),
    "occupancy": {"tp": int, "fp": int, "fn": int},
    "occupancy_precision": _ratio(occupancy_tp, occupancy_tp + occupancy_fp),
    "occupancy_recall": _ratio(occupancy_tp, occupancy_tp + occupancy_fn),
    "conditional_piece_accuracy": _ratio(correct_symbols, both_occupied),
    "negative_any_rejection_rate": _ratio(negative_failures, negative_count),
    "negative_no_board_rate": _ratio(negative_no_board, negative_count),
    "positive_false_no_board_rate": _ratio(positive_no_board, positive_count),
    "failure_reasons": {"DECODING_FAILED": int, "INVALID_GEOMETRY": int,
                        "INVALID_PLACEMENT": int, "NO_BOARD": int},
    "execution_error_count": int,
}
```

`score_predictions()` first validates that prediction IDs are unique and equal
the case-ID set. A success must contain one valid `source_placement`; a failure
must contain one allowed `reason`; an error must contain string `error_type` and
`error_message`. It returns the `_score_group` result under `overall` and a
`groups` mapping. For each `GROUP_FIELDS` entry, group only cases that contain
that field, sort group values, and call `_score_group` on that subset. Do not
introduce a metrics framework or dependency.

- [ ] **Step 4: Run all Task 1-2 tests and confirm GREEN**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py
```

Expected: all tests pass with no skipped local-source check in this workspace.

- [ ] **Step 5: Review Task 2 without staging or committing**

Run `git diff --check`. Mutate one expected placement in a temporary fixture and
confirm exact placement falls; restore only the temporary fixture by letting
pytest clean its `tmp_path`.

---

### Task 3: Build and Validate the Complete Semantic Case Plan

**Files:**
- Modify: `benchmarks/board_recognition.py`
- Modify: `tests/test_board_recognition_benchmark.py`

**Interfaces:**
- Consumes: the `source_spec_sha256()` anchor from Task 1.
- Produces: `build_case_plan()`, `validate_case_plan()`,
  `runtime_fingerprint()`, and `verify_manifest()`.
- Still performs no rendering and no writes outside pytest temporary directories.

- [ ] **Step 1: Write exact-matrix and label-binding tests**

Load the reviewed source and assert all of the following before implementing the
planner:

```python
plan = build_case_plan(load_source_spec(SOURCE_SPEC))
assert len(plan) == 568
assert [case["id"] for case in plan] == sorted(case["id"] for case in plan)
assert len({case["id"] for case in plan}) == 568
assert len({case["path"] for case in plan}) == 568
assert sum(case["board_present"] for case in plan) == 528
assert sum(not case["board_present"] for case in plan) == 40
```

Assert the suite counts are main positive 512, main negative 32,
input-contract positive 16, and input-contract negative 8. For every main
positive, recompute the palette from the indexes in the frozen source lists and
assert it equals `(position_index + piece_set_index) % 4`. Assert the matrix is
the exact Cartesian product of position, piece set, view, layout, and quality;
no tuple may be absent or duplicated. Assert each position/piece-set/view/layout/
quality aggregate sees each palette equally.

For every positive case, derive `canonical_placement` from `fen.split()[0]` in
the source spec and independently assert:

```python
expected = canonical_placement
if case["view"] == "black_bottom":
    expected = compress_placement(tuple(reversed(expand_placement(expected))))
assert case["source_placement"] == expected
```

Require `board_crop` geometry to be `[0, 0, 512, 512]` with pixel-center corners
`[[0, 0], [511, 0], [511, 511], [0, 511]]`; require `desktop_ui` geometry to be
`[24, 30, 504, 510]` with corners
`[[24, 30], [503, 30], [503, 509], [24, 509]]`. Check convexity, TL/TR/BR/BL
ordering, positive area, and bounds independently in the test.

Assert main cases are RGB; input-contract cases use only RGBA or L, clean,
board-crop, brown, and chessnut exactly as frozen. Assert negative cases omit
`canonical_placement`, `source_placement`, `base_position`, `view`, `piece_set`,
`palette`, `board_box`, and `corners`. Assert input-contract template and position
sets exactly match the reviewed source. Add mutation tests for one wrong expected
count, an invalid view, a changed palette formula, and a duplicate ID.

- [ ] **Step 2: Run the new plan tests and confirm RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py -k 'case_plan or matrix or geometry or palette'
```

Expected: import or attribute failures for the absent planner.

- [ ] **Step 3: Implement stable IDs and the direct matrix loops**

Use only the frozen list order to calculate the palette. IDs and paths are
semantic, not sequence numbers:

```python
f"main_pos_{position_id}_{piece_set_id}_{view}_{layout_id}_{quality_id}_rgb"
f"main_neg_{template_id}_{layout_id}_{quality_id}_rgb"
f"input_pos_{position_id}_{view}_{channel_mode.lower()}"
f"input_neg_{template_id}_{channel_mode.lower()}"
f"images/{case_id}.png"
```

Lowercase only the channel suffix used in the ID; retain the exact PIL mode in
`channel_mode` and `image_mode`. Each positive case contains exactly `id`,
`path`, `suite`, `split`, `board_present`, `base_position`,
`canonical_placement`, `source_placement`, `view`, `piece_set`, `palette`,
`layout`, `quality`, `channel_mode`, `image_mode`, `image_size`, `board_box`, and
`corners`. Each negative contains exactly `id`, `path`, `suite`, `split`,
`board_present`, `negative_template`, `layout`, `quality`, `channel_mode`,
`image_mode`, and `image_size`, with no board-only fields. Return cases sorted by
ID.
The source validator has already matched the formula string exactly; implement
the arithmetic directly and never evaluate source text as Python.

`validate_case_plan(plan, spec)` must enforce the exact field sets, safe relative
POSIX PNG paths, uniqueness, counts, all products, all balance rules, group split
inheritance, image modes/sizes, label rotation, and geometry. It must reject a
semantically altered plan even when its declared counts are changed to agree.

`source_spec_sha256(spec)` is SHA-256 over `canonical_json_bytes(spec)`. The
runtime fingerprint contains no host path or timestamp and is exactly:

```python
{
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "Pillow": version("Pillow"),
    "chess": version("chess"),
    "fentoboardimage": version("fenToBoardImage"),
    "zlib_build": zlib.ZLIB_VERSION,
    "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
}
```

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py -k 'case_plan or matrix or geometry or palette'
```

Expected: all case-planning tests pass before manifest tests are introduced.

- [ ] **Step 4: Write manifest semantic-anchor tests**

Build a full manifest fixture from the 568 independently planned cases, adding
dummy lowercase 64-hex `file_sha256` and `pixel_sha256` values. Its top-level
fields are exactly `schema_version`, `benchmark_version`, `claim`,
`base_position_count`, `limitations`, `source_spec_sha256`,
`generation_runtime`, `expected_counts`, and `cases`. The base-position count and
limitations are copied verbatim from the reviewed source; add mutations proving
either cannot be omitted or changed. Write it canonically under `tmp_path`, then
write a digest containing exactly:

```text
<manifest SHA-256>  manifest.json
```

Assert `verify_manifest(dataset_dir, source_spec, digest_path,
verify_images=False)` accepts it. Then independently
assert rejection for a missing digest, malformed digest, changed top-level claim,
changed semantic label with refreshed manifest digest, missing/extra/duplicate
case, changed relative path, and invalid generated hash. This proves the source
specification, rather than manifest self-consistency, owns labels and groups.

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py -k 'manifest or digest'
```

Expected: failures because `verify_manifest()` is not implemented.

- [ ] **Step 5: Implement semantic reconstruction and the digest boundary**

Implement:

```python
def verify_manifest(
    dataset_dir: Path,
    source_spec: dict,
    digest_path: Path,
    *,
    verify_images: bool = True,
) -> dict:
```

Require the digest file to match the exact lowercase-hex/two-space/filename
format `^[0-9a-f]{64}  manifest\.json\n$` and the encoded `manifest.json`
SHA-256 before parsing; reject symlinked digest, dataset, or manifest paths.
After lexical symlink and regular-file checks but before reading it, require the
resolved external digest path to be outside the resolved dataset and its `lstat`
device/inode identity to differ from every dataset entry found by a traversal that
does not follow symlinks. Convert digest `lstat`, dataset traversal, and entry
`lstat` failures to `BenchmarkValidationError`.
Rebuild the full expected plan with
`build_case_plan(source_spec)`. Compare the source-owned top-level values
(`schema_version`, `benchmark_version`, `claim`, `source_spec_sha256`, and
`base_position_count`, `limitations`, `expected_counts`) and every case field
after removing only `file_sha256` and
`pixel_sha256` from the actual case. `generation_runtime` is digest-anchored
artifact metadata rather than a current-machine expectation: require its exact
key set and nonempty string values, but do not require an evaluator on another
machine to reproduce the generation runtime. Require both generated hashes to be
lowercase 64-hex strings.

When `verify_images=True`, require the exact planned PNG inventory, reject
absolute paths, `..`, and symlinked dataset directories/files, decode every image
fully, check declared mode and size, then compare encoded-file and decoded-pixel
hashes. Do not follow a path that resolves outside `dataset_dir`.
`verify_images=False` skips only filesystem image inventory/decoding; it never
skips source reconstruction or digest validation.
After parsing, require the original manifest bytes to equal
`canonical_json_bytes(manifest)`. A refreshed digest must not make pretty-printed
JSON or an extra newline acceptable.

- [ ] **Step 6: Run and review Task 3**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py
git diff --check
```

Expected: all Task 1–3 tests pass, including the provisioned-asset preflight;
there are still no generated benchmark files.

---

### Task 4: Prove the Renderer, Then Add Transactional Materialization

**Files:**
- Create: `benchmarks/render_online_boards.py`
- Create: `scripts/data/generate_board_recognition_benchmark.py`
- Create: `tests/test_online_board_benchmark_generator.py`
- Modify: `benchmarks/board_recognition.py`

**Interfaces:**
- Produces: `render_case()`, `write_dataset()`, `verify_rendered_corpus()`,
  `compare_corpora()`, `select_contact_sheet_cases()`, and
  `write_contact_sheet()`.
- The CLI is read-only preflight unless `--write` is supplied.

The two test seams are explicit and keyword-only:

- `verify_rendered_corpus(dataset_dir: Path, spec: dict, repo_root: Path, *,
  render_case_fn=render_case) -> dict`
- `write_dataset(destination: Path, spec: dict, repo_root: Path, *,
  ordered_cases: Sequence[dict] | None = None, render_case_fn=render_case,
  image_writer=_save_png) -> dict`

`write_dataset()` must pass the same injected `render_case_fn` to its staged
full re-render verification. Production CLI calls omit these keywords and can
therefore use only the real renderer/writer.

- [ ] **Step 1: Add the permanent 144-check renderer oracle and orientation test**

Define this independent expectation literally in the test without importing a
production mapping:

```python
PIECE_PATH_BY_SYMBOL = {
    "p": "black/Pawn.png",
    "r": "black/Rook.png",
    "n": "black/Knight.png",
    "b": "black/Bishop.png",
    "q": "black/Queen.png",
    "k": "black/King.png",
    "P": "white/Pawn.png",
    "R": "white/Rook.png",
    "N": "white/Knight.png",
    "B": "white/Bishop.png",
    "Q": "white/Queen.png",
    "K": "white/King.png",
}
```

For each of the four reviewed real sets, all twelve symbols, and both production
square sizes (64 and 60), call the production call-local loader and pixel-compare
its returned RGBA image with a separately opened expected named file independently
resized with `Image.Resampling.LANCZOS`: 96 binding/filter checks. A swapped
production symbol/filename, ignored resize, or wrong resampling filter must fail.

Create an artificial piece loader that returns one opaque, full-square RGBA color
per standard piece symbol. For every 12 symbols, squares `a8`, `h8`, `a1`, `h1`,
`d5`, and `e4`, and both views, render a single-piece placement through
`fentoboardimage.fen_to_image`. Calculate the expected display row/column in the
test from `chess.square_file()` and `chess.square_rank()`:

```python
if view == "white_bottom":
    column, row = file_index, 7 - rank_index
else:
    column, row = 7 - file_index, rank_index
```

Assert the center of exactly that square has the symbol color: 144 independent
mapping checks. Also assert an asymmetric black-bottom render pixel-matches a
white-bottom render of `rotate_placement(placement)`, and that the planned black
case label is the rotated placement. Do not use the dependency's coordinate
overlay.

- [ ] **Step 2: Add composition, transform, and negative-path tests**

First add a fast resolver-spy test over all 528 positive cases. Monkeypatch the
renderer module's call-local piece-loader factory and imported `fen_to_image`
symbol, then call the real `render_case()` for every case. Independently look up
the source position/style/layout and assert every call receives the exact frozen
six-field FEN, resolved piece-set directory, dark/light colors, square size (64
or 60), `flipped` value, and `arrows=None`, `last_move=None`,
`coordinates=None`. Require 528 calls and ensure the 40 negatives make zero
calls. The spies return correctly sized solid images so this remains cheap. A
renderer that ignores or mis-resolves FEN, piece set, palette, view, or layout
must fail even if generation and re-render share the same bug.

Add an independent declarative-recipe test with a tiny hand-built rectangle,
ellipse, and polygon recipe. Assert exact boundary pixels from the frozen scaling
rules: rectangle/ellipse coordinates are half-open in 1000-space and converted to
Pillow-inclusive right/bottom coordinates; polygon points map 1000 to the last
pixel; listed shape order controls overlap. Spy on `draw_ui_recipe()` to require
the exact source `positive_desktop` object for every desktop positive and the
matching source template object for every negative. Mutate one copied source
color, coordinate, kind, and shape order in turn and assert the direct recipe
pixel hash changes.

Test with reviewed real assets that:

- a clean desktop board crop equals the standalone board rendered at 480 by 480
  pixel for pixel;
- board-crop and desktop output sizes, modes, boxes, and corners match the plan;
- `browser_scaled` downscales each full canvas to exactly three quarters of each
  dimension with `Image.Resampling.BILINEAR`, then restores the original size
  with `Image.Resampling.BICUBIC`;
- RGBA conversion adds a fully opaque alpha channel and L conversion equals
  Pillow's direct RGB-to-L conversion, without changing geometry or labels; and
- all eight negative templates render in both layouts, contain multiple colors,
  have pairwise-distinct pixel hashes within a layout, and never call the
  positive board renderer (monkeypatch it to raise).

The reviewed source recipes describe board-free online-UI shapes: three lobby
cards, a centered modal with buttons, analysis panes with bars, profile rows with
avatar circles, settings toggles, loading skeleton bars, a connection warning,
and a five-row/four-column tournament table. The implementation must consume
that data through one generic painter rather than restating colors or coordinates
in code. Use no text, fonts, noise, RNG, or 8-by-8 repeated pattern.

- [ ] **Step 3: Run renderer tests and confirm RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_online_board_benchmark_generator.py -k 'renderer or orientation or composition or channel or negative'
```

Expected: collection fails because the renderer module does not exist.

- [ ] **Step 4: Implement the smallest pure rendering pipeline**

In `benchmarks/render_online_boards.py`, implement direct functions in this
order:

1. Add a production `PIECE_PATH_BY_SYMBOL` literal matching the twelve bindings
   shown in Step 1; do not derive it from filename order or color/name casing.
   Look up a positive case's frozen position/piece set/palette/layout from the
   source spec. Open exactly the twelve already validated PNG paths with Pillow,
   copy/convert them to RGBA, and build a call-local piece-loader closure that
   resizes them with the frozen `Image.Resampling.LANCZOS` filter for the
   requested board. Do not use the dependency's process-wide
   piece cache. Call `fen_to_image(fen=fen, square_length=square_size,
   piece_set=piece_loader, dark_color=dark, light_color=light, arrows=None,
   last_move=None, coordinates=None, flipped=view == "black_bottom")` with a
   square size of 64 for board crop and 60 for desktop.
2. Implement one `draw_ui_recipe(recipe, canvas_size, coordinate_space)` painter
   for only `rectangle`, `ellipse`, and `polygon`. Validate every color, shape
   key set, coordinate bound/order, and kind before drawing; scale exactly as the
   design specifies and paint only in listed order. Return the 512 board directly
   for board crop. For desktop, draw the source `positive_desktop` recipe, then
   paste the 480 board at `[24, 30, 504, 510]`.
3. Render negatives by passing the source recipe selected by the frozen template
   ID to the same Pillow-only painter; never restate recipe geometry in code.
4. Apply the quality transform to the completed canvas.
5. Convert to the case's final RGB, RGBA, or L mode and assert the planned size.

No rendering function reads configuration, discovers assets, uses a global
cache/RNG, mutates the case, or writes a file. `render_case(case, spec, repo_root)`
returns a newly owned, fully loaded `PIL.Image.Image`.

Run:

```bash
uv run --locked python -m pytest -q tests/test_online_board_benchmark_generator.py -k 'renderer or orientation or composition or channel or negative or recipe or resolver'
```

Expected: every pure renderer/resolver/recipe test passes before writer tests are
introduced.

- [ ] **Step 5: Add full-matrix binding and determinism tests**

Inject a fake `render_case_fn` that returns a correctly sized/mode image whose
pixel stripes encode SHA-256 of the canonical semantic case (before generated
hashes). Materialize all 568 cases twice in one process, once in normal order and
once reversed. Assert:

- identical relative inventories and byte-for-byte files;
- canonical manifest case order remains sorted;
- every file/pixel hash matches the independently encoded case;
- `verify_rendered_corpus()` reconstructs all 568 cases from the source spec and
  catches any two swapped files; and
- rendering a deterministic subset alone produces the same pixel hashes as that
  subset inside the full run.

Also render a small real-asset subset twice in the same process and assert exact
PNG bytes, pixel hashes, and semantic fields.

- [ ] **Step 6: Add transactional-failure tests**

Under `tmp_path`, prove:

1. An existing destination containing a sentinel is refused and unchanged.
   Repeat with a dangling destination symlink; `Path.exists()` alone is not an
   acceptable refusal check.
2. An image-writer test double that raises after several writes leaves neither a
   final dataset nor a sibling staging directory.
3. A successful fake-renderer run contains exactly `manifest.json` and 568
   planned PNGs, every image reopens, and full verification passes.
4. A bad mode, short write, corrupted PNG, extra file, and deterministic case
   swap each fail verification.
5. An injected post-rename verification failure removes only the just-created
   destination and leaves an unrelated sibling sentinel unchanged.

Run:

```bash
uv run --locked python -m pytest -q tests/test_online_board_benchmark_generator.py -k 'matrix or deterministic or transaction or corruption or swap'
```

Expected: failures because materialization and verification are not implemented.

- [ ] **Step 7: Implement canonical PNG writing and atomic publication**

Save images as PNG using the already exact-validated source renderer values
`optimize=False` and `compress_level=9` (plus `format="PNG"` and no metadata),
reopen and load each file, then add `file_sha256` and `pixel_sha256` to its case.
Add a save-spy test that proves those exact source values reach Pillow. Write the
canonical manifest only after every image succeeds. `write_dataset()` must:

- validate source, local assets, and the full case plan before touching the
  destination;
- require a real, non-symlink destination parent and refuse any destination for
  which `os.path.lexists(destination)` is true, including a dangling symlink;
- create one `tempfile.mkdtemp()` sibling staging directory;
- accept an ordered-case and renderer/writer injection seam only for tests, while
  still requiring the exact 568-case semantic set;
- verify the staged corpus semantically, by hashes, and by full case-specific
  re-render before publication, propagating its injected `render_case_fn` only
  through the private test seam;
- recheck `os.path.lexists(destination)` immediately before one same-filesystem
  atomic rename;
- verify the published directory again, rolling back only the directory created
  by this invocation if that post-rename verification unexpectedly fails, using
  only the staging directory's pre-rename device/inode identity; and
- delete only its resolved staging directory after an exception.

`verify_rendered_corpus()` rebuilds the plan from the source spec, renders each
expected case again, and compares mode, size, and `pixel_sha256` to the file bound
to that exact ID. During staging and disposable-canary verification, read the
manifest bytes in memory and use the same canonical parsing, semantic, and image
checks as `verify_manifest()` without creating a temporary digest. External
verification still validates its separate digest before invoking those shared
checks.
`compare_corpora()` independently validates both source-bound corpora and then
requires identical relative inventories and file bytes. The permanent external
digest is deliberately created only in Task 7 after independent canary review.

- [ ] **Step 8: Implement the safe CLI**

The generator CLI uses mutually exclusive modes:

```text
--preflight
--write DESTINATION [--case-order normal|reverse]
--verify DATASET --digest PATH
--compare FIRST SECOND
--contact-sheet DATASET OUTPUT
```

No mode defaults to writing; no arguments and `--preflight` both run only source,
asset, and full case-plan validation and print counts. `--write` refuses existing
dataset paths and calls the transactional writer. `--verify` performs external
digest, inventory/hash, and full re-render verification. `--compare` performs
source-bound validation plus exact corpus comparison. Contact-sheet selection is a
deterministic greedy cover over every base position, piece set, palette, view,
layout, quality, channel mode, negative template, suite, and split; tests assert
complete value coverage and collective coverage of all twelve piece symbols.
The disposable sheet may use Pillow's built-in font for case IDs, but the corpus
renderer may not use fonts. The sheet output must be a nonexistent path that
resolves outside the verified dataset.

- [ ] **Step 9: Run and review Task 4**

Run:

```bash
uv run --locked python -m pytest -q tests/test_online_board_benchmark_generator.py
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight
git diff --check
```

Expected: all tests pass, preflight reports 568/528/40, and
`datasets/board_recognition_benchmark/v1` still does not exist.

---

### Task 5: Add the Checkpoint Runner Without Claiming a Baseline

**Files:**
- Create: `scripts/validate/evaluate_board_recognition_benchmark.py`
- Modify: `tests/test_board_recognition_benchmark.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `run_predictions()` plus a CLI that verifies the frozen corpus before
  importing/loading the current model stack.
- Does not run a checkpoint in this task because P1-05 is still unresolved.

Use these exact deferred-import seams:

- `load_helper(board_detector_checkpoint: Path, square_classifier_checkpoint:
  Path, piece_classifier_checkpoint: Path, device: str) -> tuple[object, type,
  type, type]` returns `(helper, Picture, RecognitionSuccess,
  RecognitionFailure)` after local imports and model loading.
- `run_predictions(dataset_dir: Path, cases: Sequence[dict], helper: object, *,
  picture_cls: type, success_cls: type, failure_cls: type) -> list[dict]` performs
  the one-call-per-case loop. Tests provide all four stubs directly; production
  unpacks `load_helper()` into the keyword arguments.

- [ ] **Step 1: Write runner contract tests with stubs**

Load the evaluator module without importing `chessml`. Use stub `Picture`, helper,
success, and failure classes to prove:

- a new `Picture(image_path)` is constructed for every case;
- `helper.recognize()` is called exactly once per case;
- success maps to `{id, outcome: "success", source_placement}`;
- typed failure maps to `{id, outcome: "failure", reason:
  result.reason.name}` after requiring that name is in `FAILURE_REASONS`;
- an unexpected exception maps to `{id, outcome: "error", error_type,
  error_message}`, processing continues, and the final error count is nonzero;
- neither input pictures nor successful `board_image` results are marked, saved,
  or reused; and
- predictions remain in sorted manifest-case order and score with the pure scorer.

Add report-publication tests proving an existing sentinel report and a dangling
report symlink are refused unchanged, a symlinked/non-directory parent is
rejected, and an injected canonical-write/rename failure removes its sibling temp
file without creating or overwriting the report.

Add a CLI-level stub test proving manifest verification occurs before the helper
factory is invoked, a tampered corpus prevents model loading, and a report path
resolving inside the dataset is rejected after verification but before model
loading or publication.

- [ ] **Step 2: Run the runner tests and confirm RED**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py -k 'runner or evaluator'
```

Expected: the evaluator module or runner is absent.

- [ ] **Step 3: Implement deferred model loading and report writing**

Keep all `chessml` imports inside `load_helper()`/`main()` after source and corpus
verification. The CLI requires explicit paths:

```text
--dataset DATASET
--source-spec SOURCE_SPEC
--manifest-digest DIGEST
--board-detector-checkpoint FILE
--square-classifier-checkpoint FILE
--piece-classifier-checkpoint FILE
--report NEW_JSON_FILE
[--device cpu]
```

Load the same concrete classes/base-model kwargs as
`scripts/validate/validate_board_recognition.py`, set all three models to eval,
and construct one `BoardRecognitionHelper`. For each sorted case, instantiate a
fresh `Picture` from its PNG and make exactly one recognition call. Test actual
result types with `isinstance`; do not catch or relabel `RecognitionFailure`.
Validate a success placement with the strict 64-square parser and require a
failure's `result.reason.name` to be a string in `FAILURE_REASONS`; store that
name, never the enum object, in the prediction. An unknown result type or invalid typed
payload becomes an `InvalidRecognitionResult` execution-error record. Catch
unexpected `Exception` only around that one case, store class name and message as
an execution error, and continue.

The canonical report copies benchmark version, claim, `base_position_count`, and
every limitation verbatim from the source, then contains the manifest digest,
SHA-256 and basename for all three checkpoint files, selected device, and an
inference fingerprint containing Python plus installed `torch`, `torchvision`,
`timm`, `lightning`, `opencv-python`, and `numpy` versions. It also contains the
complete predictions and `score_predictions()` output. Refuse an existing
report using `os.path.lexists()` so dangling symlinks are covered; require a real
existing non-symlink parent. Write through one sibling temp file, recheck
`lexists` immediately before an atomic rename that never intentionally
overwrites, and clean only that temp path in `finally`. Exit 1 after writing if
any execution error occurred, otherwise exit 0. Never create annotated images or
mutate the benchmark.

- [ ] **Step 4: Document the exact claim and safe commands**

Add a compact README section with:

```bash
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --write datasets/board_recognition_benchmark/v1
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256
```

State prominently that `online-render-v1` is synthetic rendered-online-board
regression evidence, not unseen-site accuracy; base positions and all derivatives
must never enter training; P2-01 still needs an independently captured/manually
labeled online-screenshot suite; and no accepted current-checkpoint baseline may
be claimed until P1-05 removes ignored raw-asset import coupling. Also state that
the local source/digest pair gains repository-history protection only after the
user reviews and commits it; this task does not commit anything. Mark `--write`
as a maintainer-only gated command that requires all pre-generation proof/review
steps and a nonexistent destination; ordinary users should use preflight/verify.

- [ ] **Step 5: Run and review Task 5**

Run:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight
git diff --check
```

Expected: all focused tests and preflight pass. Do not invoke the evaluator with
real checkpoints.

---

### Task 6: Clear Every Pre-Generation Correctness Gate

**Files:**
- Review only: all files from Tasks 1–5
- Disposable outputs only: a unique `/tmp` canary parent and contact-sheet PNG

**Hard gate:** Any failed automated check, visual mismatch, or Critical/Important
review finding blocks Task 7. Fix the cause, add a regression test, rerun all of
Task 6, and obtain fresh review before proceeding.

- [ ] **Step 1: Run focused and broad repository checks**

Run exactly:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py
uv run --locked python -m pytest -q
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight
git diff --check
rg -n '[ \t]+$' benchmarks/__init__.py benchmarks/board_recognition.py benchmarks/render_online_boards.py benchmarks/board_recognition_v1.json scripts/data/generate_board_recognition_benchmark.py scripts/validate/evaluate_board_recognition_benchmark.py tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py docs/superpowers/specs/2026-07-19-online-board-recognition-benchmark-design.md docs/superpowers/plans/2026-07-19-online-board-recognition-benchmark.md README.md BOARD_RECOGNITION_AUDIT.tmp.md
```

Record actual pass/fail/skip counts. The local asset test may not skip in this
workspace. If an unrelated broad-suite failure is proven pre-existing, report it
to both reviewers; it does not become permission to weaken or skip benchmark
coverage. The `rg` search must print no matches (its no-match exit status is
expected); inspect final-newline state separately while reading each untracked
file.

- [ ] **Step 2: Generate two complete disposable canaries in clean processes**

Create one new temporary parent, resolve the path physically, then invoke the CLI
twice as separate processes. The physical-path step is mandatory: macOS exposes
`/tmp` through a symlink, while the writer intentionally rejects every lexical
symlink ancestor.

```bash
mktemp -d /tmp/chessml-board-benchmark.XXXXXX
uv run --locked python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' <mktemp-output>
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --write <physical-temp-parent>/normal --case-order normal
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --write <physical-temp-parent>/reverse --case-order reverse
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --compare <physical-temp-parent>/normal <physical-temp-parent>/reverse
```

Both runs must contain all 568 cases. Comparison must prove identical manifests,
PNG inventories, and bytes despite fresh interpreter state and reversed order.
Keep the normal canary until Task 7 so the final corpus can be compared against
it; remove only the validated temporary parent after final verification.

- [ ] **Step 3: Generate and inspect the stratified visual canary**

Run:

```bash
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --contact-sheet <physical-temp-parent>/normal <physical-temp-parent>/contact-sheet.png
```

Open the PNG with the available image viewer. Check that all twelve piece
identities are visually plausible, white/black views match labels, palettes and
piece sets are distinct, boards are fully visible and aligned in both layouts,
browser scaling does not change geometry, RGBA/L previews remain recognizable,
and every negative template is board-free. Record the selected case IDs and the
automated coverage map. A visual concern is a blocker even if hashes pass.

- [ ] **Step 4: Obtain two independent reviews**

Dispatch fresh subagents with no implementation ownership:

1. A specification/correctness reviewer maps every design gate and case count to
   code/tests, checks the source oracle, orientation, source-to-image binding,
   scorer semantics, transactional behavior, and P1-05 claim boundary.
2. A code/test reviewer inspects the complete diff for defects, unsafe paths,
   accidental configuration/import coupling, nondeterminism, over-engineering,
   missing failure tests, and unrelated changes.

They must review tracked diffs plus every untracked file directly (using
`git diff --no-index /dev/null <file>` where useful) and inspect test evidence,
not summaries alone. Resolve every
Critical or Important finding with TDD, rerun Steps 1–3, and request re-review.
Minor findings may be deferred only with a concrete written reason. Neither
reviewer may stage or commit.

- [ ] **Step 5: Declare the generation gate cleared in evidence, not prose**

Before Task 7, require all of these artifacts simultaneously: green focused/full
tests, green preflight, two verified byte-identical full canaries, inspected
coverage-complete contact sheet, clean `git diff --check`, and two reviews with no
open Critical/Important findings. If any item is absent, stop without creating the
final dataset or digest.

---

### Task 7: Generate, Freeze, and Re-Verify the Final Local Corpus

**Files:**
- Create: `datasets/board_recognition_benchmark/v1/` (ignored local corpus)
- Create: `benchmarks/board_recognition_v1_manifest.sha256`
- Modify: `BOARD_RECOGNITION_AUDIT.tmp.md`

- [ ] **Step 1: Recheck targets and Task 6 evidence**

Confirm the final dataset and digest do not exist, the destination parent is the
resolved repository path `datasets/board_recognition_benchmark`, and Task 6 has
no open blocker. Create only that narrow parent if needed. Never delete or
overwrite an existing target.

- [ ] **Step 2: Publish from a fresh process**

Run:

```bash
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --write datasets/board_recognition_benchmark/v1 --case-order normal
```

The command must preflight before writing, stage as a sibling, verify all 568
case-specific re-renders, atomically rename the corpus, and verify it again. A
failure must not be described as success.

- [ ] **Step 3: Compare to the retained canary, then create the external digest**

First compare the unanchored but source-validated final bytes:

```bash
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --compare datasets/board_recognition_benchmark/v1 <physical-temp-parent>/normal
shasum -a 256 datasets/board_recognition_benchmark/v1/manifest.json
```

Require 568 total/528 positive/40 negative, exact inventory and hashes, semantic
reconstruction from the reviewed source spec, full re-render equality, and
byte-for-byte equality with the independently generated normal canary. Take the
reported lowercase digest and use `apply_patch` to create
`benchmarks/board_recognition_v1_manifest.sha256` with exactly
`<digest>  manifest.json\n`. Then run:

```bash
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256
```

- [ ] **Step 4: Record the intentionally partial P2-01 status**

Update the P2-01 entry in `BOARD_RECOGNITION_AUDIT.tmp.md` to point to the source
spec, generator, evaluator, and frozen digest. State that the synthetic online
render regression half now exists, while P2-01 remains open for an independently
captured/manually labeled online-screenshot suite and an accepted current-model
baseline remains gated by P1-05. Do not change priorities or mark unrelated
issues fixed.

- [ ] **Step 5: Perform final verification and handoff review**

Run fresh:

```bash
uv run --locked python -m pytest -q tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py
uv run --locked python -m pytest -q
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --preflight
uv run --locked python -m scripts.data.generate_board_recognition_benchmark --verify datasets/board_recognition_benchmark/v1 --digest benchmarks/board_recognition_v1_manifest.sha256
git diff --check
rg -n '[ \t]+$' benchmarks/__init__.py benchmarks/board_recognition.py benchmarks/render_online_boards.py benchmarks/board_recognition_v1.json benchmarks/board_recognition_v1_manifest.sha256 scripts/data/generate_board_recognition_benchmark.py scripts/validate/evaluate_board_recognition_benchmark.py tests/test_board_recognition_benchmark.py tests/test_online_board_benchmark_generator.py docs/superpowers/specs/2026-07-19-online-board-recognition-benchmark-design.md docs/superpowers/plans/2026-07-19-online-board-recognition-benchmark.md README.md BOARD_RECOGNITION_AUDIT.tmp.md
git status --short
```

Inspect the final diff for scope, accidental generated PNGs outside the ignored
dataset, secrets, mutable paths/timestamps, and any staged files. Confirm no real
checkpoint evaluation, commit, staging, upload, or publication occurred. Report
the exact checks and counts, the frozen manifest digest, generated corpus size,
the P1-05 baseline blocker, and the remaining real-online-screenshot limitation.

---
