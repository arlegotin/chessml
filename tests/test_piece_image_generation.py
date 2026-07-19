from collections import Counter, defaultdict
import csv
from hashlib import sha256
import importlib
from itertools import islice
from pathlib import Path
from random import Random
import sys

import numpy as np
from PIL import Image
import pytest

from chessml.data.images.pieces_images import (
    PIECE_FILE_NAMES,
    PiecesImages3x3,
    hex_to_bgr,
)
from chessml.data.utils.looped_list import LoopedList


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


def make_dataset_inputs(tmp_path: Path):
    piece_set_a, palette_a = make_piece_set(tmp_path, "set-a", 10)
    piece_set_b, palette_b = make_piece_set(tmp_path, "set-b", 100)
    dataset_kwargs = {
        "piece_sets": [piece_set_a, piece_set_b],
        "board_colors": [("#010203", "#040506"), ("#304050", "#607080")],
        "square_size": 1,
    }
    return dataset_kwargs, {"set-a": palette_a, "set-b": palette_b}


def capture(dataset, count):
    return [(row[0].cv2.copy(), row[1:]) for row in islice(dataset, count)]


def assert_same_rows(left, right):
    assert len(left) == len(right)
    assert all(
        left_row[1:] == right_row[1:]
        and np.array_equal(left_row[0].cv2, right_row[0].cv2)
        for left_row, right_row in zip(left, right, strict=True)
    )


@pytest.fixture
def classifier_generator(monkeypatch):
    import chessml

    monkeypatch.setattr(
        chessml.Script, "__call__", lambda self, function: function
    )
    monkeypatch.setattr(chessml, "script", chessml.Script())
    module_name = "scripts.data.generate_piece_classifier_dataset"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def read_csv(path: Path):
    with path.open(newline="") as csvfile:
        rows = list(csv.reader(csvfile))
    return rows[0], rows[1:]


def artifact_signature(paths):
    return [
        [
            (*row[1:], sha256(Path(row[0]).read_bytes()).hexdigest())
            for row in read_csv(path)[1]
        ]
        for path in paths
    ]


def test_center_cycles_are_balanced_and_limits_only_truncate_final_cycle(tmp_path):
    dataset_kwargs, _ = make_dataset_inputs(tmp_path)

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
    expected_piece_counts = Counter({name: 2 for name in PIECE_FILE_NAMES})
    assert Counter(row[1] for row in piece_rows) == expected_piece_counts

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
    assert Counter(row[1] for row in square_rows if row[1] is not None) == (
        expected_piece_counts
    )

    limited_piece_rows = list(
        PiecesImages3x3(
            **dataset_kwargs,
            shuffle_seed=0,
            with_empty_squares=False,
            limit=25,
        )
    )
    assert len(limited_piece_rows) == 25
    assert Counter(row[1] for row in limited_piece_rows[:24]) == expected_piece_counts
    assert_same_rows(piece_rows, limited_piece_rows[:24])

    limited_square_rows = list(
        PiecesImages3x3(
            **dataset_kwargs,
            shuffle_seed=0,
            with_empty_squares=True,
            limit=49,
        )
    )
    assert len(limited_square_rows) == 49
    assert sum(row[1] is None for row in limited_square_rows[:48]) == 24
    assert Counter(
        row[1] for row in limited_square_rows[:48] if row[1] is not None
    ) == expected_piece_counts
    assert_same_rows(square_rows, limited_square_rows[:48])


def test_sampling_varies_without_mixing_piece_sets(tmp_path):
    dataset_kwargs, palettes = make_dataset_inputs(tmp_path)
    backgrounds = {
        hex_to_bgr(color)
        for theme in dataset_kwargs["board_colors"]
        for color in theme
    }

    def semantic_grid(row):
        picture, center_label, piece_set_name, *_ = row
        reverse_palette = {
            color: label for label, color in palettes[piece_set_name].items()
        }
        semantic_pixels = []
        for pixel in picture.cv2.reshape(-1, 3):
            color = tuple(pixel)
            if color in backgrounds:
                semantic_pixels.append(None)
            else:
                assert color in reverse_palette
                semantic_pixels.append(reverse_palette[color])
        return tuple(semantic_pixels)

    themes_by_center = defaultdict(set)
    signatures_by_center = defaultdict(set)
    has_empty_neighbor = False
    has_occupied_neighbor = False
    neighbor_indexes = (0, 1, 2, 3, 5, 6, 7, 8)

    rows = islice(
        PiecesImages3x3(
            **dataset_kwargs,
            shuffle_seed=0,
            with_empty_squares=True,
        ),
        32 * 48,
    )
    for row in rows:
        semantic = semantic_grid(row)
        center_label = row[1]
        assert semantic[4] == center_label

        neighbors = tuple(semantic[index] for index in neighbor_indexes)
        signatures_by_center[center_label].add(neighbors)
        has_empty_neighbor |= any(label is None for label in neighbors)
        has_occupied_neighbor |= any(label is not None for label in neighbors)

        if center_label is not None:
            themes_by_center[center_label].add(row[3:5])

    expected_themes = set(dataset_kwargs["board_colors"])
    assert all(
        themes_by_center[label] == expected_themes for label in PIECE_FILE_NAMES
    )
    assert all(
        len(signatures_by_center[label]) > 1
        for label in (*PIECE_FILE_NAMES, None)
    )
    assert has_empty_neighbor
    assert has_occupied_neighbor


def test_seeded_raw_samples_replay_and_different_seed_changes(tmp_path):
    dataset_kwargs, _ = make_dataset_inputs(tmp_path)
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

    assert all(
        left[1] == right[1] and np.array_equal(left[0], right[0])
        for left, right in zip(first, second, strict=True)
    )
    assert any(
        left[1] != right[1] or not np.array_equal(left[0], right[0])
        for left, right in zip(first, different, strict=True)
    )


def test_looped_list_treats_zero_as_an_explicit_seed():
    expected = [0, 1, 2]
    Random(0).shuffle(expected)

    values = LoopedList([0, 1, 2], shuffle_seed=0)

    assert [values[index] for index in range(3)] == expected


@pytest.mark.parametrize("with_empty_squares", [False, True])
def test_classifier_artifacts_are_grouped_and_reproducible(
    tmp_path, classifier_generator, with_empty_squares
):
    dataset_kwargs, _ = make_dataset_inputs(tmp_path / "assets")

    def generate(root: Path, seed: int):
        return classifier_generator.generate_classifier_dataset(
            dataset_dir=root,
            limit=50,
            seed=seed,
            with_empty_squares=with_empty_squares,
            **{
                key: dataset_kwargs[key]
                for key in ("piece_sets", "board_colors")
            },
        )

    paths = generate(tmp_path / "first", 0)
    replay_paths = generate(tmp_path / "replay", 0)
    different_paths = generate(tmp_path / "different", 1)

    train_header, train_rows = read_csv(paths[0])
    validation_header, validation_rows = read_csv(paths[1])
    expected_header = [
        "image_path",
        "piece_name",
        "piece_set",
        "dark_color",
        "light_color",
    ]
    assert train_header == validation_header == expected_header
    assert len(train_rows) + len(validation_rows) == 50
    assert {row[2] for row in train_rows}.isdisjoint(
        row[2] for row in validation_rows
    )
    assert {(row[3], row[4]) for row in train_rows}.isdisjoint(
        (row[3], row[4]) for row in validation_rows
    )
    assert all(Path(row[0]).is_file() for row in train_rows + validation_rows)
    assert not (tmp_path / "first" / "meta.csv").exists()

    signature = artifact_signature(paths)
    assert signature == artifact_signature(replay_paths)
    assert signature != artifact_signature(different_paths)

    if with_empty_squares:
        assert {row[1] for row in train_rows + validation_rows} == {"0", "1"}


def test_classifier_artifact_wraps_validation_seed(
    tmp_path, classifier_generator
):
    dataset_kwargs, _ = make_dataset_inputs(tmp_path / "assets")
    paths = classifier_generator.generate_classifier_dataset(
        dataset_dir=tmp_path / "dataset",
        limit=2,
        seed=2**32 - 1,
        with_empty_squares=False,
        **{
            key: dataset_kwargs[key]
            for key in ("piece_sets", "board_colors")
        },
    )

    assert [len(read_csv(path)[1]) for path in paths] == [1, 1]


@pytest.mark.parametrize("limit", [0, 1])
def test_classifier_artifact_requires_two_rows(
    tmp_path, classifier_generator, limit
):
    dataset_kwargs, _ = make_dataset_inputs(tmp_path / "assets")

    with pytest.raises(ValueError, match="at least two items are required"):
        classifier_generator.generate_classifier_dataset(
            dataset_dir=tmp_path / "dataset",
            limit=limit,
            seed=0,
            with_empty_squares=False,
            **{
                key: dataset_kwargs[key]
                for key in ("piece_sets", "board_colors")
            },
        )


@pytest.mark.parametrize("short_key", ["piece_sets", "board_colors"])
def test_classifier_artifact_requires_two_sources_per_group(
    tmp_path, classifier_generator, short_key
):
    dataset_kwargs, _ = make_dataset_inputs(tmp_path / "assets")
    dataset_kwargs[short_key] = dataset_kwargs[short_key][:1]

    with pytest.raises(ValueError, match="at least two items are required"):
        classifier_generator.generate_classifier_dataset(
            dataset_dir=tmp_path / "dataset",
            limit=2,
            seed=0,
            with_empty_squares=False,
            **{
                key: dataset_kwargs[key]
                for key in ("piece_sets", "board_colors")
            },
        )
