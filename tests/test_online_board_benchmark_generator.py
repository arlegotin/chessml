from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import chess
import pytest
from fentoboardimage import fen_to_image
from PIL import Image, ImageDraw

import benchmarks.render_online_boards as renderer
import scripts.data.generate_board_recognition_benchmark as generator
from benchmarks.board_recognition import (
    BenchmarkValidationError,
    build_case_plan,
    canonical_json_bytes,
    file_sha256,
    load_source_spec,
    pixel_sha256,
    rotate_placement,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = ROOT / "benchmarks/board_recognition_v1.json"
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


def _spec() -> dict:
    return load_source_spec(SOURCE_SPEC)


def _plan(spec: dict) -> list[dict]:
    return build_case_plan(spec)


def _case(plan: list[dict], **fields) -> dict:
    matches = [
        case
        for case in plan
        if all(case.get(field) == value for field, value in fields.items())
    ]
    assert len(matches) == 1, fields
    return matches[0]


def _single_piece_placement(symbol: str, square_name: str) -> str:
    board = chess.Board(None)
    board.set_piece_at(
        chess.parse_square(square_name), chess.Piece.from_symbol(symbol)
    )
    return board.board_fen()


def test_renderer_piece_loader_binds_all_real_assets_and_lanczos_sizes():
    spec = _spec()
    for piece_set in spec["piece_sets"]:
        directory = ROOT / piece_set["path"]
        loader = renderer._make_piece_loader(directory)
        for square_size in (64, 60):
            actual = loader(Image.new("RGB", (square_size * 8, square_size * 8)))
            assert set(actual) == set(PIECE_PATH_BY_SYMBOL)
            for symbol, relative_path in PIECE_PATH_BY_SYMBOL.items():
                with Image.open(directory / relative_path) as source:
                    expected = source.convert("RGBA").resize(
                        (square_size, square_size), Image.Resampling.LANCZOS
                    )
                assert actual[symbol].mode == "RGBA"
                assert actual[symbol].size == (square_size, square_size)
                assert actual[symbol].tobytes() == expected.tobytes()


def test_orientation_maps_every_piece_square_and_view_independently():
    colors = {
        symbol: ((index * 19 + 17) % 256, (index * 43 + 29) % 256, 101, 255)
        for index, symbol in enumerate(PIECE_PATH_BY_SYMBOL, 1)
    }

    def piece_loader(board: Image.Image) -> dict[str, Image.Image]:
        square_size = board.width // 8
        return {
            symbol: Image.new("RGBA", (square_size, square_size), color)
            for symbol, color in colors.items()
        }

    square_size = 20
    for symbol in PIECE_PATH_BY_SYMBOL:
        for square_name in ("a8", "h8", "a1", "h1", "d5", "e4"):
            square = chess.parse_square(square_name)
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            placement = _single_piece_placement(symbol, square_name)
            for view in ("white_bottom", "black_bottom"):
                if view == "white_bottom":
                    column, row = file_index, 7 - rank_index
                else:
                    column, row = 7 - file_index, rank_index
                image = fen_to_image(
                    fen=placement,
                    square_length=square_size,
                    piece_set=piece_loader,
                    dark_color="#202020",
                    light_color="#E0E0E0",
                    arrows=None,
                    last_move=None,
                    coordinates=None,
                    flipped=view == "black_bottom",
                )
                center = (
                    column * square_size + square_size // 2,
                    row * square_size + square_size // 2,
                )
                assert image.getpixel(center) == colors[symbol][:3]


def test_orientation_rotation_matches_dependency_and_planned_black_label():
    placement = "r3k2r/8/8/3p4/8/2N5/8/R3K2R"
    colors = {
        symbol: ((index * 31 + 11) % 256, 73, (index * 47 + 5) % 256, 255)
        for index, symbol in enumerate(PIECE_PATH_BY_SYMBOL, 1)
    }

    def piece_loader(board: Image.Image) -> dict[str, Image.Image]:
        length = board.width // 8
        return {
            symbol: Image.new("RGBA", (length, length), color)
            for symbol, color in colors.items()
        }

    common = {
        "square_length": 16,
        "piece_set": piece_loader,
        "dark_color": "#222222",
        "light_color": "#DDDDDD",
        "arrows": None,
        "last_move": None,
        "coordinates": None,
    }
    black = fen_to_image(fen=placement, flipped=True, **common)
    rotated_white = fen_to_image(
        fen=rotate_placement(placement), flipped=False, **common
    )
    assert black.tobytes() == rotated_white.tobytes()

    spec = _spec()
    black_case = next(
        case
        for case in _plan(spec)
        if case.get("board_present")
        and case["view"] == "black_bottom"
        and case["base_position"] == "pos_02"
    )
    canonical = next(
        position["fen"].split()[0]
        for position in spec["positions"]
        if position["id"] == "pos_02"
    )
    assert black_case["source_placement"] == rotate_placement(canonical)


def test_renderer_resolver_binds_every_positive_and_skips_every_negative(
    monkeypatch,
):
    spec = _spec()
    plan = _plan(spec)
    piece_sets = {item["id"]: item for item in spec["piece_sets"]}
    palettes = {item["id"]: item for item in spec["palettes"]}
    positions = {item["id"]: item for item in spec["positions"]}
    path_calls = []
    fen_calls = []

    def fake_loader_factory(directory: Path):
        loader = object()
        path_calls.append((directory, loader))
        return loader

    def fake_fen_to_image(**kwargs):
        fen_calls.append(kwargs)
        length = kwargs["square_length"] * 8
        return Image.new("RGB", (length, length), "#123456")

    monkeypatch.setattr(renderer, "_make_piece_loader", fake_loader_factory)
    monkeypatch.setattr(renderer, "fen_to_image", fake_fen_to_image)
    monkeypatch.setattr(
        renderer,
        "draw_ui_recipe",
        lambda recipe, canvas_size, coordinate_space: Image.new(
            "RGB", tuple(canvas_size), "#654321"
        ),
    )

    for case in plan:
        image = renderer.render_case(case, spec, ROOT)
        assert image.size == tuple(case["image_size"])
        assert image.mode == case["image_mode"]

    positive_cases = [case for case in plan if case["board_present"]]
    assert len(positive_cases) == len(path_calls) == len(fen_calls) == 528
    assert len(plan) - len(fen_calls) == 40
    for case, (directory, loader), call in zip(
        positive_cases, path_calls, fen_calls, strict=True
    ):
        palette = palettes[case["palette"]]
        square_size = 64 if case["layout"] == "board_crop" else 60
        assert directory == (ROOT / piece_sets[case["piece_set"]]["path"]).resolve()
        assert call == {
            "fen": positions[case["base_position"]]["fen"],
            "square_length": square_size,
            "piece_set": loader,
            "dark_color": palette["dark"],
            "light_color": palette["light"],
            "arrows": None,
            "last_move": None,
            "coordinates": None,
            "flipped": case["view"] == "black_bottom",
        }


def test_recipe_renderer_uses_half_open_boxes_last_pixel_polygons_and_order():
    recipe = {
        "background": "#000000",
        "shapes": [
            {"kind": "rectangle", "fill": "#FF0000", "box": [100, 100, 600, 600]},
            {"kind": "ellipse", "fill": "#00FF00", "box": [400, 400, 900, 900]},
            {
                "kind": "polygon",
                "fill": "#0000FF",
                "points": [[0, 1000], [1000, 1000], [500, 500]],
            },
        ],
    }
    image = renderer.draw_ui_recipe(recipe, [10, 10], [1000, 1000])
    assert image.mode == "RGB"
    assert image.getpixel((0, 1)) == (0, 0, 0)
    assert image.getpixel((1, 1)) == (255, 0, 0)
    assert image.getpixel((5, 1)) == (255, 0, 0)
    assert image.getpixel((6, 1)) == (0, 0, 0)
    assert image.getpixel((8, 6)) == (0, 255, 0)
    assert image.getpixel((9, 6)) == (0, 0, 0)
    assert image.getpixel((4, 4)) == (0, 0, 255)
    assert image.getpixel((0, 9)) == (0, 0, 255)
    assert image.getpixel((9, 9)) == (0, 0, 255)


@pytest.mark.parametrize(
    "mutation",
    ["background", "shape-key", "kind", "range", "order"],
)
def test_recipe_renderer_rejects_invalid_declarations(mutation):
    recipe = {
        "background": "#000000",
        "shapes": [
            {"kind": "rectangle", "fill": "#FF0000", "box": [0, 0, 500, 500]}
        ],
    }
    if mutation == "background":
        recipe["background"] = "red"
    elif mutation == "shape-key":
        recipe["shapes"][0]["extra"] = True
    elif mutation == "kind":
        recipe["shapes"][0]["kind"] = "line"
    elif mutation == "range":
        recipe["shapes"][0]["box"][0] = -1
    else:
        recipe["shapes"][0]["box"] = [500, 0, 100, 500]
    with pytest.raises(BenchmarkValidationError):
        renderer.draw_ui_recipe(recipe, [10, 10], [1000, 1000])


def test_recipe_renderer_consumes_exact_source_objects(monkeypatch):
    spec = _spec()
    plan = _plan(spec)
    calls = []

    def draw_spy(recipe, canvas_size, coordinate_space):
        calls.append(recipe)
        return Image.new("RGB", tuple(canvas_size), "#202020")

    def fake_board(case, source_spec, repo_root, square_size):
        return Image.new("RGB", (square_size * 8, square_size * 8), "#404040")

    monkeypatch.setattr(renderer, "draw_ui_recipe", draw_spy)
    monkeypatch.setattr(renderer, "_render_board", fake_board)
    selected = [
        case
        for case in plan
        if not case["board_present"] or case["layout"] == "desktop_ui"
    ]
    for case in selected:
        renderer.render_case(case, spec, ROOT)

    negative_recipes = spec["ui_recipes"]["negative_templates"]
    expected = [
        spec["ui_recipes"]["positive_desktop"]
        if case["board_present"]
        else negative_recipes[case["negative_template"]]
        for case in selected
    ]
    assert len(calls) == len(expected) == 296
    assert all(actual is wanted for actual, wanted in zip(calls, expected, strict=True))


def test_recipe_source_color_coordinate_kind_and_order_control_pixels():
    source = _spec()["ui_recipes"]["negative_templates"]["modal_dialog"]
    baseline = pixel_sha256(renderer.draw_ui_recipe(source, [120, 80], [1000, 1000]))
    mutations = []

    changed_color = copy.deepcopy(source)
    changed_color["shapes"][1]["fill"] = "#FFFFFF"
    mutations.append(changed_color)

    changed_coordinate = copy.deepcopy(source)
    changed_coordinate["shapes"][1]["box"][0] += 50
    mutations.append(changed_coordinate)

    changed_kind = copy.deepcopy(source)
    changed_kind["shapes"][1]["kind"] = "ellipse"
    mutations.append(changed_kind)

    changed_order = copy.deepcopy(source)
    changed_order["shapes"][0], changed_order["shapes"][1] = (
        changed_order["shapes"][1],
        changed_order["shapes"][0],
    )
    mutations.append(changed_order)

    assert all(
        pixel_sha256(renderer.draw_ui_recipe(recipe, [120, 80], [1000, 1000]))
        != baseline
        for recipe in mutations
    )


def test_renderer_composition_matches_standalone_board_and_planned_geometry():
    spec = _spec()
    plan = _plan(spec)
    common = {
        "suite": "main",
        "board_present": True,
        "base_position": "pos_01",
        "piece_set": "lichess_chessnut",
        "palette": "brown",
        "view": "white_bottom",
        "quality": "clean",
    }
    crop_case = _case(plan, layout="board_crop", **common)
    desktop_case = _case(plan, layout="desktop_ui", **common)
    crop = renderer.render_case(crop_case, spec, ROOT)
    desktop = renderer.render_case(desktop_case, spec, ROOT)
    standalone_480 = renderer._render_board(desktop_case, spec, ROOT, 60)

    assert crop.mode == desktop.mode == standalone_480.mode == "RGB"
    assert crop.size == (512, 512)
    assert desktop.size == (960, 540)
    assert desktop.crop(tuple(desktop_case["board_box"])).tobytes() == (
        standalone_480.tobytes()
    )
    assert crop_case["board_box"] == [0, 0, 512, 512]
    assert crop_case["corners"] == [[0, 0], [511, 0], [511, 511], [0, 511]]
    assert desktop_case["board_box"] == [24, 30, 504, 510]
    assert desktop_case["corners"] == [
        [24, 30],
        [503, 30],
        [503, 509],
        [24, 509],
    ]


@pytest.mark.parametrize("layout", ["board_crop", "desktop_ui"])
def test_renderer_browser_scaled_transform_matches_frozen_filters(layout):
    spec = _spec()
    plan = _plan(spec)
    fields = {
        "suite": "main",
        "board_present": True,
        "base_position": "pos_01",
        "piece_set": "lichess_chessnut",
        "palette": "brown",
        "view": "white_bottom",
        "layout": layout,
    }
    clean_case = _case(plan, quality="clean", **fields)
    scaled_case = _case(plan, quality="browser_scaled", **fields)
    clean = renderer.render_case(clean_case, spec, ROOT)
    actual = renderer.render_case(scaled_case, spec, ROOT)
    expected = clean.resize(
        (clean.width * 3 // 4, clean.height * 3 // 4),
        Image.Resampling.BILINEAR,
    ).resize(clean.size, Image.Resampling.BICUBIC)
    assert actual.size == clean.size
    assert actual.tobytes() == expected.tobytes()


def test_channel_renderer_matches_rgb_conversion_and_preserves_labels():
    spec = _spec()
    plan = _plan(spec)
    input_fields = {
        "suite": "input_contract",
        "board_present": True,
        "base_position": "pos_01",
        "view": "white_bottom",
    }
    rgba_case = _case(plan, channel_mode="RGBA", **input_fields)
    gray_case = _case(plan, channel_mode="L", **input_fields)
    rgb_case = _case(
        plan,
        suite="main",
        board_present=True,
        base_position="pos_01",
        piece_set="lichess_chessnut",
        palette="brown",
        view="white_bottom",
        layout="board_crop",
        quality="clean",
        channel_mode="RGB",
    )
    originals = [copy.deepcopy(case) for case in (rgba_case, gray_case, rgb_case)]
    rgb = renderer.render_case(rgb_case, spec, ROOT)
    rgba = renderer.render_case(rgba_case, spec, ROOT)
    gray = renderer.render_case(gray_case, spec, ROOT)

    assert rgba.mode == "RGBA"
    assert rgba.getchannel("A").getextrema() == (255, 255)
    assert rgba.convert("RGB").tobytes() == rgb.tobytes()
    assert gray.mode == "L"
    assert gray.tobytes() == rgb.convert("L").tobytes()
    assert rgba.size == gray.size == rgb.size == (512, 512)
    assert [rgba_case, gray_case, rgb_case] == originals
    for field in ("source_placement", "board_box", "corners", "image_size"):
        assert rgba_case[field] == gray_case[field] == rgb_case[field]


def test_negative_renderer_covers_templates_and_layouts_without_board_path(
    monkeypatch,
):
    spec = _spec()
    plan = _plan(spec)

    def forbidden_board(*args, **kwargs):
        raise AssertionError("negative rendering called the positive board renderer")

    monkeypatch.setattr(renderer, "_render_board", forbidden_board)
    template_ids = [template["id"] for template in spec["negative_templates"]]
    for layout in ("board_crop", "desktop_ui"):
        hashes = []
        for template_id in template_ids:
            case = _case(
                plan,
                suite="main",
                board_present=False,
                negative_template=template_id,
                layout=layout,
                quality="clean",
            )
            image = renderer.render_case(case, spec, ROOT)
            assert image.mode == "RGB"
            assert image.size == tuple(case["image_size"])
            assert len(image.getcolors(maxcolors=image.width * image.height) or []) > 1
            hashes.append(pixel_sha256(image))
        assert len(set(hashes)) == len(template_ids)


def _semantic_digest(case: dict) -> bytes:
    return hashlib.sha256(canonical_json_bytes(case)).digest()


def _fake_render_case(case: dict, spec: dict, repo_root: Path) -> Image.Image:
    del spec, repo_root
    mode = case["image_mode"]
    size = tuple(case["image_size"])
    background = 0 if mode == "L" else ((0, 0, 0, 255) if mode == "RGBA" else (0, 0, 0))
    image = Image.new(mode, size, background)
    draw = ImageDraw.Draw(image)
    for x, value in enumerate(_semantic_digest(case)):
        fill = value if mode == "L" else ((value, value, value, 255) if mode == "RGBA" else (value, value, value))
        draw.rectangle((x, 0, x, image.height - 1), fill=fill)
    return image


def _relative_inventory(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def _assert_semantic_stripes(image: Image.Image, case: dict) -> None:
    actual = []
    for x in range(32):
        value = image.getpixel((x, 0))
        actual.append(value if isinstance(value, int) else value[0])
    assert bytes(actual) == _semantic_digest(case)


@pytest.fixture(scope="module")
def matrix_corpora(tmp_path_factory):
    parent = tmp_path_factory.mktemp("online-board-matrix")
    spec = _spec()
    plan = _plan(spec)
    normal = parent / "normal"
    reverse = parent / "reverse"
    generator.write_dataset(
        normal, spec, ROOT, ordered_cases=plan, render_case_fn=_fake_render_case
    )
    generator.write_dataset(
        reverse,
        spec,
        ROOT,
        ordered_cases=list(reversed(plan)),
        render_case_fn=_fake_render_case,
    )
    return parent, normal, reverse, spec, plan


def test_full_matrix_deterministic_order_bytes_hashes_and_bindings(
    monkeypatch, matrix_corpora
):
    parent, normal, reverse, spec, plan = matrix_corpora
    normal_inventory = _relative_inventory(normal)
    reverse_inventory = _relative_inventory(reverse)
    assert normal_inventory == reverse_inventory
    assert normal_inventory == sorted(
        ["manifest.json", *(case["path"] for case in plan)]
    )
    for relative in normal_inventory:
        assert (normal / relative).read_bytes() == (reverse / relative).read_bytes()

    manifest = json.loads((normal / "manifest.json").read_bytes())
    assert [case["id"] for case in manifest["cases"]] == sorted(
        case["id"] for case in plan
    )
    for semantic, materialized in zip(plan, manifest["cases"], strict=True):
        assert {
            key: value
            for key, value in materialized.items()
            if key not in {"file_sha256", "pixel_sha256"}
        } == semantic
        path = normal / semantic["path"]
        assert materialized["file_sha256"] == file_sha256(path)
        with Image.open(path) as image:
            image.load()
            _assert_semantic_stripes(image, semantic)
            assert materialized["pixel_sha256"] == pixel_sha256(image)

    verified = generator.verify_rendered_corpus(
        normal, spec, ROOT, render_case_fn=_fake_render_case
    )
    assert len(verified["cases"]) == 568
    real_verify = generator.verify_rendered_corpus

    def fake_corpus_verify(dataset_dir, source_spec, repo_root):
        return real_verify(
            dataset_dir,
            source_spec,
            repo_root,
            render_case_fn=_fake_render_case,
        )

    monkeypatch.setattr(generator, "verify_rendered_corpus", fake_corpus_verify)
    assert generator.compare_corpora(normal, reverse, spec, ROOT) == {
        "cases": 568,
        "files": 569,
    }
    assert not list(parent.glob("*.sha256"))


def test_rendered_corpus_verification_is_read_only_without_ephemeral_digest(
    monkeypatch, matrix_corpora
):
    parent, normal, _reverse, spec, _plan = matrix_corpora
    before = _relative_inventory(parent)
    manifest_before = (normal / "manifest.json").read_bytes()
    attempts = []

    def forbid_mkstemp(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("rendered-corpus verification created a tempfile")

    monkeypatch.setattr(generator.tempfile, "mkstemp", forbid_mkstemp)

    manifest = generator.verify_rendered_corpus(
        normal, spec, ROOT, render_case_fn=_fake_render_case
    )

    assert len(manifest["cases"]) == 568
    assert attempts == []
    assert _relative_inventory(parent) == before
    assert (normal / "manifest.json").read_bytes() == manifest_before


def test_rendered_corpus_rejects_symlink_before_any_tempfile_write(
    monkeypatch, tmp_path, matrix_corpora
):
    _parent, normal, _reverse, spec, _plan = matrix_corpora
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    linked_dataset = linked_root / "dataset"
    (real_root / "dataset").symlink_to(normal, target_is_directory=True)
    attempts = []

    def forbid_mkstemp(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("symlink rejection happened after a tempfile write")

    monkeypatch.setattr(generator.tempfile, "mkstemp", forbid_mkstemp)

    with pytest.raises(BenchmarkValidationError, match="symlink"):
        generator.verify_rendered_corpus(
            linked_dataset, spec, ROOT, render_case_fn=_fake_render_case
        )

    assert attempts == []


def test_compare_corpora_contract_has_no_test_seam_and_dispatches(
    monkeypatch, tmp_path
):
    assert tuple(inspect.signature(generator.compare_corpora).parameters) == (
        "first",
        "second",
        "spec",
        "repo_root",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "manifest.json").write_bytes(b"same")
    (second / "manifest.json").write_bytes(b"same")
    calls = []
    spec = {}

    def verify_spy(dataset_dir, source_spec, repo_root):
        calls.append((dataset_dir, source_spec, repo_root))
        return {"cases": [{"id": "only"}]}

    monkeypatch.setattr(generator, "verify_rendered_corpus", verify_spy)
    assert generator.compare_corpora(first, second, spec, ROOT) == {
        "cases": 1,
        "files": 1,
    }
    assert calls == [(first, spec, ROOT), (second, spec, ROOT)]


def test_deterministic_subset_matches_its_full_matrix_hashes(matrix_corpora):
    _, normal, _, spec, plan = matrix_corpora
    manifest = json.loads((normal / "manifest.json").read_bytes())
    by_id = {case["id"]: case for case in manifest["cases"]}
    subset = plan[::67]
    assert len(subset) > 1
    for case in subset:
        alone = _fake_render_case(case, spec, ROOT)
        assert pixel_sha256(alone) == by_id[case["id"]]["pixel_sha256"]


def test_deterministic_real_asset_subset_has_exact_png_bytes_and_semantics(tmp_path):
    spec = _spec()
    plan = _plan(spec)
    subset = [
        _case(
            plan,
            suite="main",
            board_present=True,
            base_position="pos_01",
            piece_set="lichess_chessnut",
            palette="brown",
            view="white_bottom",
            layout="board_crop",
            quality="clean",
        ),
        _case(
            plan,
            suite="main",
            board_present=True,
            base_position="pos_02",
            piece_set="lichess_celtic",
            view="black_bottom",
            layout="desktop_ui",
            quality="browser_scaled",
        ),
        _case(
            plan,
            suite="main",
            board_present=False,
            negative_template="connection_error",
            layout="desktop_ui",
            quality="clean",
        ),
        _case(
            plan,
            suite="input_contract",
            board_present=True,
            base_position="pos_06",
            view="black_bottom",
            channel_mode="RGBA",
        ),
    ]
    originals = copy.deepcopy(subset)
    for index, case in enumerate(subset):
        first = renderer.render_case(case, spec, ROOT)
        second = renderer.render_case(case, spec, ROOT)
        first_path = tmp_path / "first" / f"{index}.png"
        second_path = tmp_path / "second" / f"{index}.png"
        first_path.parent.mkdir(exist_ok=True)
        second_path.parent.mkdir(exist_ok=True)
        generator._save_png(first, first_path, spec["renderer"])
        generator._save_png(second, second_path, spec["renderer"])
        assert first_path.read_bytes() == second_path.read_bytes()
        assert pixel_sha256(first) == pixel_sha256(second)
    assert subset == originals


def test_deterministic_png_save_uses_exact_frozen_options(monkeypatch, tmp_path):
    calls = []

    def save_spy(self, destination, *args, **kwargs):
        calls.append((destination, args, kwargs))

    monkeypatch.setattr(Image.Image, "save", save_spy)
    generator._save_png(
        Image.new("RGB", (2, 2)), tmp_path / "unused.png", _spec()["renderer"]
    )
    assert calls == [
        (
            tmp_path / "unused.png",
            (),
            {"format": "PNG", "optimize": False, "compress_level": 9},
        )
    ]


def test_transaction_refuses_existing_directory_and_dangling_symlink(tmp_path):
    spec = _spec()
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(BenchmarkValidationError):
        generator.write_dataset(existing, spec, ROOT, render_case_fn=_fake_render_case)
    assert sentinel.read_bytes() == b"keep"
    assert _relative_inventory(existing) == ["sentinel"]

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    assert not dangling.exists() and os.path.lexists(dangling)
    with pytest.raises(BenchmarkValidationError):
        generator.write_dataset(dangling, spec, ROOT, render_case_fn=_fake_render_case)
    assert dangling.is_symlink()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_transaction_writer_failure_removes_only_staging(tmp_path):
    spec = _spec()
    destination = tmp_path / "dataset"
    writes = 0

    def exploding_writer(image, path, options):
        nonlocal writes
        writes += 1
        if writes == 4:
            raise RuntimeError("injected write failure")
        generator._save_png(image, path, options)

    with pytest.raises(RuntimeError, match="injected write failure"):
        generator.write_dataset(
            destination,
            spec,
            ROOT,
            render_case_fn=_fake_render_case,
            image_writer=exploding_writer,
        )
    assert writes == 4
    assert not os.path.lexists(destination)
    assert not list(tmp_path.glob(".dataset.staging-*"))


@pytest.mark.parametrize("failure", ["short-write", "corrupted-png", "bad-mode"])
def test_transaction_rejects_invalid_render_or_write_and_cleans_up(tmp_path, failure):
    spec = _spec()
    destination = tmp_path / failure
    first_id = next(
        case["id"] for case in _plan(spec) if case["image_mode"] == "RGB"
    )

    def image_writer(image, path, options):
        if failure == "short-write":
            path.write_bytes(b"short")
        else:
            generator._save_png(image, path, options)
            if failure == "corrupted-png":
                path.write_bytes(b"not a png")

    def render(case, source_spec, repo_root):
        image = _fake_render_case(case, source_spec, repo_root)
        return image.convert("L") if failure == "bad-mode" and case["id"] == first_id else image

    kwargs = {"render_case_fn": render}
    if failure != "bad-mode":
        kwargs["image_writer"] = image_writer
    with pytest.raises(BenchmarkValidationError):
        generator.write_dataset(destination, spec, ROOT, **kwargs)
    assert not os.path.lexists(destination)
    assert not list(tmp_path.glob(f".{failure}.staging-*"))


def test_transaction_success_has_exact_inventory_and_reopens(matrix_corpora):
    _, normal, _, spec, plan = matrix_corpora
    assert len(_relative_inventory(normal)) == 569
    for case in plan:
        with Image.open(normal / case["path"]) as image:
            image.load()
            assert image.format == "PNG"
            assert image.mode == case["image_mode"]
            assert image.size == tuple(case["image_size"])
    assert len(
        generator.verify_rendered_corpus(
            normal, spec, ROOT, render_case_fn=_fake_render_case
        )["cases"]
    ) == 568


def _rewrite_manifest(dataset: Path, manifest: dict) -> None:
    (dataset / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def _refresh_case_hashes(dataset: Path, case: dict) -> None:
    path = dataset / case["path"]
    case["file_sha256"] = file_sha256(path)
    with Image.open(path) as image:
        image.load()
        case["pixel_sha256"] = pixel_sha256(image)


@pytest.mark.parametrize(
    "corruption", ["bad-mode", "corrupted-png", "extra-file", "swap"]
)
def test_corruption_and_case_swap_fail_full_verification(
    tmp_path, matrix_corpora, corruption
):
    _, source, _, spec, _ = matrix_corpora
    dataset = tmp_path / corruption
    shutil.copytree(source, dataset)
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    cases = manifest["cases"]

    if corruption == "bad-mode":
        case = next(case for case in cases if case["image_mode"] == "RGB")
        path = dataset / case["path"]
        with Image.open(path) as image:
            changed = image.convert("L")
        changed.save(path, format="PNG", optimize=False, compress_level=9)
        _refresh_case_hashes(dataset, case)
        _rewrite_manifest(dataset, manifest)
    elif corruption == "corrupted-png":
        (dataset / cases[0]["path"]).write_bytes(b"corrupted")
    elif corruption == "extra-file":
        (dataset / "images" / "extra.png").write_bytes(b"extra")
    else:
        candidates = [
            case
            for case in cases
            if case["image_mode"] == "RGB" and case["image_size"] == [512, 512]
        ]
        first, second = candidates[:2]
        first_path = dataset / first["path"]
        second_path = dataset / second["path"]
        first_bytes, second_bytes = first_path.read_bytes(), second_path.read_bytes()
        first_path.write_bytes(second_bytes)
        second_path.write_bytes(first_bytes)
        _refresh_case_hashes(dataset, first)
        _refresh_case_hashes(dataset, second)
        _rewrite_manifest(dataset, manifest)

    with pytest.raises(BenchmarkValidationError):
        generator.verify_rendered_corpus(
            dataset, spec, ROOT, render_case_fn=_fake_render_case
        )
    assert not list(tmp_path.glob("*.sha256"))


def test_transaction_post_rename_failure_rolls_back_only_created_destination(
    monkeypatch, tmp_path
):
    spec = _spec()
    destination = tmp_path / "dataset"
    sibling = tmp_path / "unrelated"
    sibling.mkdir()
    sentinel = sibling / "sentinel"
    sentinel.write_bytes(b"keep")
    real_verify = generator.verify_rendered_corpus
    calls = 0

    def fail_after_rename(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BenchmarkValidationError("injected published verification failure")
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(generator, "verify_rendered_corpus", fail_after_rename)
    with pytest.raises(BenchmarkValidationError, match="published verification"):
        generator.write_dataset(
            destination, spec, ROOT, render_case_fn=_fake_render_case
        )
    assert calls == 2
    assert not os.path.lexists(destination)
    assert sentinel.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_transaction_substitution_after_rename_preserves_substitute(
    monkeypatch, tmp_path
):
    spec = _spec()
    destination = tmp_path / "dataset"
    displaced = tmp_path / "displaced-published-dataset"
    sentinel = destination / "sentinel"
    real_rename = Path.rename
    injected = False

    def substitute_after_publication(path, target):
        nonlocal injected
        result = real_rename(path, target)
        if Path(target) == destination:
            injected = True
            real_rename(destination, displaced)
            destination.mkdir()
            sentinel.write_bytes(b"substitute-owned")
        return result

    monkeypatch.setattr(Path, "rename", substitute_after_publication)

    with pytest.raises(BenchmarkValidationError, match="identity|manifest"):
        generator.write_dataset(
            destination, spec, ROOT, render_case_fn=_fake_render_case
        )

    assert injected is True
    assert destination.is_dir() and not destination.is_symlink()
    assert sentinel.read_bytes() == b"substitute-owned"
    assert (displaced / "manifest.json").is_file()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_contact_sheet_selection_is_deterministic_and_covers_every_dimension():
    spec = _spec()
    plan = _plan(spec)
    selected = generator.select_contact_sheet_cases(plan, spec)
    reversed_selected = generator.select_contact_sheet_cases(
        list(reversed(plan)), spec
    )
    assert [case["id"] for case in selected] == [
        case["id"] for case in reversed_selected
    ]
    assert len({case["id"] for case in selected}) == len(selected)

    fields = (
        "base_position",
        "piece_set",
        "palette",
        "view",
        "layout",
        "quality",
        "channel_mode",
        "negative_template",
        "suite",
        "split",
    )
    for field in fields:
        expected = {case[field] for case in plan if field in case}
        actual = {case[field] for case in selected if field in case}
        assert actual == expected, field
    assert set("prnbqkPRNBQK") <= {
        symbol
        for case in selected
        if case["board_present"]
        for symbol in case["canonical_placement"]
        if symbol.isalpha()
    }


def test_contact_sheet_writer_is_deterministic_and_does_not_mutate_corpus(
    tmp_path, matrix_corpora
):
    _, dataset, _, spec, plan = matrix_corpora
    selected = generator.select_contact_sheet_cases(plan, spec)
    before = _relative_inventory(dataset)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first_result = generator.write_contact_sheet(dataset, first, selected)
    second_result = generator.write_contact_sheet(dataset, second, selected)
    assert first_result == second_result
    assert first_result["cases"] == len(selected)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == tuple(first_result["size"])
        assert len(image.getcolors(maxcolors=image.width * image.height) or []) > 1
    assert _relative_inventory(dataset) == before


def _contact_sheet_fixture(tmp_path: Path) -> tuple[Path, dict]:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), "#123456").save(
        images / "valid.png", format="PNG"
    )
    return dataset, {"id": "valid", "path": "images/valid.png"}


def test_contact_sheet_direct_api_accepts_real_contained_png(tmp_path):
    dataset, case = _contact_sheet_fixture(tmp_path)
    output = tmp_path / "sheet.png"
    before = _relative_inventory(dataset)
    result = generator.write_contact_sheet(dataset, output, [case])
    assert result["cases"] == 1
    with Image.open(output) as image:
        image.load()
        assert image.format == "PNG"
    assert _relative_inventory(dataset) == before


@pytest.mark.parametrize("location", ["root", "images", "lexical-alias"])
def test_contact_sheet_output_must_resolve_outside_dataset(tmp_path, location):
    dataset, case = _contact_sheet_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if location == "root":
        output = dataset / "sheet.png"
    elif location == "images":
        output = dataset / "images/sheet.png"
    else:
        output = outside / ".." / dataset.name / "sheet.png"
    before = _relative_inventory(dataset)

    with pytest.raises(BenchmarkValidationError, match="outside.*dataset"):
        generator.write_contact_sheet(dataset, output, [case])

    assert not os.path.lexists(output)
    assert _relative_inventory(dataset) == before


@pytest.mark.parametrize("unsafe_path", ["parent", "absolute", "non-png"])
def test_contact_sheet_direct_api_rejects_unsafe_lexical_paths(
    tmp_path, unsafe_path
):
    dataset, case = _contact_sheet_fixture(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8), "#654321").save(outside, format="PNG")
    if unsafe_path == "parent":
        case["path"] = "../outside.png"
    elif unsafe_path == "absolute":
        case["path"] = outside.as_posix()
    else:
        non_png = dataset / "images/valid.jpg"
        non_png.write_bytes((dataset / case["path"]).read_bytes())
        case["path"] = "images/valid.jpg"

    output = tmp_path / "sheet.png"
    with pytest.raises(BenchmarkValidationError):
        generator.write_contact_sheet(dataset, output, [case])
    assert not os.path.lexists(output)


@pytest.mark.parametrize("symlink_kind", ["file", "directory", "dataset", "ancestor"])
def test_contact_sheet_direct_api_rejects_symlinked_paths(
    tmp_path, symlink_kind
):
    dataset, case = _contact_sheet_fixture(tmp_path)
    dataset_argument = dataset
    if symlink_kind == "file":
        outside = tmp_path / "outside.png"
        Image.new("RGB", (8, 8), "#654321").save(outside, format="PNG")
        linked = dataset / "images/linked.png"
        linked.symlink_to(outside)
        case["path"] = "images/linked.png"
    elif symlink_kind == "directory":
        linked = dataset / "linked"
        linked.symlink_to(dataset / "images", target_is_directory=True)
        case["path"] = "linked/valid.png"
    elif symlink_kind == "dataset":
        linked = tmp_path / "linked-dataset"
        linked.symlink_to(dataset, target_is_directory=True)
        dataset_argument = linked
    else:
        real_parent = tmp_path / "real-parent"
        dataset.rename(real_parent)
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(tmp_path, target_is_directory=True)
        dataset_argument = linked_parent / "real-parent"

    output = tmp_path / "sheet.png"
    with pytest.raises(BenchmarkValidationError):
        generator.write_contact_sheet(dataset_argument, output, [case])
    assert not os.path.lexists(output)


def test_cli_no_arguments_and_preflight_are_read_only(monkeypatch, capsys):
    def forbidden_write(*args, **kwargs):
        raise AssertionError("preflight attempted to write a dataset")

    monkeypatch.setattr(generator, "write_dataset", forbidden_write)
    assert generator.main([]) == 0
    assert generator.main(["--preflight"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "cases=568 positive=528 negative=40",
        "cases=568 positive=528 negative=40",
    ]


def test_cli_modes_are_mutually_exclusive_and_options_are_scoped(tmp_path):
    with pytest.raises(SystemExit) as raised:
        generator.main(["--preflight", "--write", str(tmp_path / "dataset")])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        generator.main(["--case-order", "reverse"])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        generator.main(["--verify", str(tmp_path / "dataset")])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        generator.main(["--preflight", "--digest", str(tmp_path / "digest")])
    assert raised.value.code == 2


@pytest.mark.parametrize("case_order", ["normal", "reverse"])
def test_cli_write_dispatches_exact_full_order_without_test_seams(
    monkeypatch, tmp_path, case_order
):
    calls = []

    def write_spy(destination, spec, repo_root, *, ordered_cases=None):
        calls.append((destination, spec, repo_root, ordered_cases))
        return {"cases": [{"id": case["id"]} for case in ordered_cases]}

    monkeypatch.setattr(generator, "write_dataset", write_spy)
    destination = tmp_path / case_order
    assert generator.main(
        ["--write", str(destination), "--case-order", case_order]
    ) == 0
    assert len(calls) == 1
    actual_destination, spec, repo_root, ordered = calls[0]
    expected = _plan(spec)
    if case_order == "reverse":
        expected.reverse()
    assert actual_destination == destination
    assert repo_root == ROOT
    assert [case["id"] for case in ordered] == [case["id"] for case in expected]


def test_cli_verify_checks_external_digest_before_full_rerender(
    monkeypatch, tmp_path
):
    calls = []
    dataset = tmp_path / "dataset"
    digest = tmp_path / "manifest.sha256"

    def manifest_spy(dataset_dir, spec, digest_path, *, verify_images=True):
        calls.append(("external", dataset_dir, digest_path, verify_images))
        return {"cases": _plan(spec)}

    def corpus_spy(dataset_dir, spec, repo_root):
        calls.append(("rendered", dataset_dir, repo_root))
        return {"cases": _plan(spec)}

    monkeypatch.setattr(generator, "verify_manifest", manifest_spy)
    monkeypatch.setattr(generator, "verify_rendered_corpus", corpus_spy)
    assert generator.main(
        ["--verify", str(dataset), "--digest", str(digest)]
    ) == 0
    assert calls == [
        ("external", dataset, digest, True),
        ("rendered", dataset, ROOT),
    ]


def test_cli_compare_and_contact_sheet_dispatch_without_test_seams(
    monkeypatch, tmp_path
):
    calls = []
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "sheet.png"

    def compare_spy(first_dir, second_dir, spec, repo_root):
        calls.append(("compare", first_dir, second_dir, repo_root))
        return {"cases": 568, "files": 569}

    def corpus_spy(dataset_dir, spec, repo_root):
        calls.append(("verify", dataset_dir, repo_root))
        return {"cases": _plan(spec)}

    def sheet_spy(dataset_dir, output_path, selected):
        calls.append(("sheet", dataset_dir, output_path, len(selected)))
        return {"cases": len(selected), "size": [1, 1]}

    monkeypatch.setattr(generator, "compare_corpora", compare_spy)
    monkeypatch.setattr(generator, "verify_rendered_corpus", corpus_spy)
    monkeypatch.setattr(generator, "write_contact_sheet", sheet_spy)
    assert generator.main(["--compare", str(first), str(second)]) == 0
    assert generator.main(["--contact-sheet", str(first), str(output)]) == 0
    assert calls[0] == ("compare", first, second, ROOT)
    assert calls[1] == ("verify", first, ROOT)
    assert calls[2][:3] == ("sheet", first, output)
    assert calls[2][3] > 0


def test_cli_preflight_subprocess_is_read_only():
    dataset = ROOT / "datasets/board_recognition_benchmark/v1"

    def snapshot(path):
        if not os.path.lexists(path):
            return ("absent",)

        entries = []

        def visit(current, relative):
            name = relative.as_posix()
            if current.is_symlink():
                entries.append((name, "symlink", os.readlink(current)))
            elif current.is_dir():
                entries.append((name, "directory"))
                for child in sorted(current.iterdir(), key=lambda item: item.name):
                    visit(child, relative / child.name)
            elif current.is_file():
                contents = current.read_bytes()
                entries.append(
                    (name, "file", len(contents), hashlib.sha256(contents).hexdigest())
                )
            else:
                entries.append((name, "other", os.lstat(current).st_mode))

        visit(path, Path("."))
        return tuple(entries)

    before = snapshot(dataset)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.data.generate_board_recognition_benchmark",
            "--preflight",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cases=568 positive=528 negative=40"
    assert snapshot(dataset) == before
