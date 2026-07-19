from argparse import Namespace
import os
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chessml import Script


def capture_streams(seed_name, seed):
    script = Script()
    script.parser.parse_args = lambda: Namespace(**{seed_name: seed})
    captured = []

    def callback(_):
        captured.append(
            (
                random.random(),
                np.random.random(),
                torch.nn.Linear(4, 4).weight.detach().clone(),
                torch.nn.functional.dropout(
                    torch.ones(16), p=0.5, training=True
                ),
                next(
                    iter(
                        DataLoader(
                            TensorDataset(torch.arange(12)),
                            batch_size=12,
                            shuffle=True,
                        )
                    )
                )[0],
            )
        )

    script.run(callback)
    return captured[0]


def streams_equal(left, right):
    return left[:2] == right[:2] and all(
        torch.equal(left_value, right_value)
        for left_value, right_value in zip(
            left[2:], right[2:], strict=True
        )
    )


@pytest.mark.parametrize("seed_name", ["seed", "shuffle_seed"])
@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_script_seeds_every_random_stream(seed_name, seed, monkeypatch):
    monkeypatch.setenv("PL_SEED_WORKERS", "0")

    first = capture_streams(seed_name, seed)
    second = capture_streams(seed_name, seed)

    assert streams_equal(first, second)
    assert os.environ["PL_SEED_WORKERS"] == "1"


@pytest.mark.parametrize("seed_name", ["seed", "shuffle_seed"])
def test_different_script_seed_changes_random_streams(seed_name):
    seed_zero = capture_streams(seed_name, 0)
    seed_one = capture_streams(seed_name, 1)

    assert not streams_equal(seed_zero, seed_one)


@pytest.mark.parametrize("seed_name", ["seed", "shuffle_seed"])
@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_script_seed_stops_before_callback(seed_name, seed):
    script = Script()
    script.parser.parse_args = lambda: Namespace(**{seed_name: seed})
    callback_calls = []

    with pytest.raises(ValueError):
        script.run(callback_calls.append)

    assert callback_calls == []
