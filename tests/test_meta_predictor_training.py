from argparse import Namespace
import importlib
from pathlib import Path
import sys

import chessml
import pytest
import torch
import torch.nn.functional as F

from chessml.models.lightning.meta_predictor_model import MetaPredictor


def test_meta_predictor_returns_and_logs_mean_bce(monkeypatch):
    outputs = torch.tensor(
        [
            [0.9, 0.2, 0.7, 0.4, 0.8, 0.1],
            [0.3, 0.6, 0.1, 0.9, 0.2, 0.7],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    model = MetaPredictor(input_shape=(13, 8, 8))
    logged = {}
    monkeypatch.setattr(model, "forward", lambda inputs: outputs)
    monkeypatch.setattr(
        model, "log", lambda name, value: logged.setdefault(name, value)
    )

    expected = F.binary_cross_entropy(outputs, targets)
    training_loss = model.training_step((torch.empty(0), targets), 0)
    model.validation_step((torch.empty(0), targets), 0)

    assert torch.equal(training_loss, expected)
    assert logged["train_loss"] is training_loss
    assert torch.equal(logged["val_loss"], expected)


def test_meta_predictor_provenance_preserves_state_dict_keys():
    baseline = MetaPredictor(input_shape=(13, 8, 8))
    with_provenance = MetaPredictor(
        input_shape=(13, 8, 8), path_to_fens="/data/unique_fens.txt"
    )

    assert baseline.state_dict().keys() == with_provenance.state_dict().keys()


@pytest.fixture
def meta_predictor_training_script(monkeypatch):
    module_name = "scripts.train.train_meta_predictor"
    monkeypatch.setattr(
        chessml.Script, "__call__", lambda self, function: function
    )
    monkeypatch.setattr(chessml, "script", chessml.Script())
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture
def recorded_meta_training(meta_predictor_training_script, monkeypatch):
    module = meta_predictor_training_script
    requested = Path("custom/meta-predictor-fens.txt")
    observed_dataset_paths = []
    captured_training = {}

    def recording_boards_from_fen(*, path, **kwargs):
        observed_dataset_paths.append(Path(path))
        return object()

    def recording_standard_training(
        *,
        model,
        make_dataset,
        batch_size,
        val_batches,
        checkpoint_monitor,
        checkpoint_mode="min",
        **kwargs,
    ):
        captured_training.update(
            model=model,
            checkpoint_monitor=checkpoint_monitor,
            checkpoint_mode=checkpoint_mode,
        )
        split = batch_size * val_batches
        make_dataset(limit=split)
        make_dataset(offset=split)

    monkeypatch.setattr(module, "BoardsFromFEN", recording_boards_from_fen)
    monkeypatch.setattr(module, "standard_training", recording_standard_training)
    module.train(
        Namespace(
            path_to_fens=str(requested),
            batch_size=4,
            val_batches=2,
            val_interval=3,
            shuffle_seed=68,
        )
    )

    return requested, observed_dataset_paths, captured_training


def test_meta_predictor_training_uses_requested_dataset_path(recorded_meta_training):
    requested, observed_dataset_paths, _ = recorded_meta_training

    assert observed_dataset_paths == [requested.resolve(), requested.resolve()]


def test_meta_predictor_training_records_dataset_provenance(recorded_meta_training):
    requested, _, captured_training = recorded_meta_training
    captured_model = captured_training["model"]

    assert captured_model.hparams.path_to_fens == str(requested.resolve())


def test_meta_predictor_training_preserves_checkpoint_seam(recorded_meta_training):
    _, _, captured_training = recorded_meta_training

    assert captured_training["checkpoint_monitor"] == "val_loss"
    assert captured_training["checkpoint_mode"] == "min"
