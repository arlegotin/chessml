import importlib

import pytest
import torch
from torch.utils.data import Dataset


class TinyDataset(Dataset):
    def __init__(self, name):
        self.name = name

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor(index)


@pytest.fixture
def training_harness(monkeypatch):
    module = importlib.import_module("chessml.train.standard_training")
    fit_calls = []

    class RecordingTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, model, **kwargs):
            fit_calls.append((model, kwargs))

    monkeypatch.setattr(module, "Trainer", RecordingTrainer)
    monkeypatch.setattr(module, "TensorBoardLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "ModelCheckpoint", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.torch.backends.mps, "is_available", lambda: False)
    return module.standard_training, fit_calls


def training_kwargs(make_dataset):
    return {
        "model": object(),
        "make_dataset": make_dataset,
        "batch_size": 2,
        "val_batches": 1,
        "val_interval": 3,
        "checkpoint_name": "test-{step}",
    }


def test_standard_training_preserves_legacy_single_factory_routing(
    training_harness,
):
    standard_training, fit_calls = training_harness
    calls = []
    datasets = []

    def make_dataset(**kwargs):
        calls.append(kwargs)
        datasets.append(TinyDataset(str(kwargs)))
        return datasets[-1]

    kwargs = training_kwargs(make_dataset)
    model = kwargs["model"]
    standard_training(**kwargs)

    assert calls == [{"limit": 2}, {"offset": 2}]
    assert fit_calls[0][0] is model
    assert fit_calls[0][1]["val_dataloaders"].dataset is datasets[0]
    assert fit_calls[0][1]["train_dataloaders"].dataset is datasets[1]


def test_standard_training_routes_a_separate_validation_factory(
    training_harness,
):
    standard_training, fit_calls = training_harness
    training_calls = []
    validation_calls = []
    training_dataset = TinyDataset("training")
    validation_dataset = TinyDataset("validation")

    def make_training_dataset(**kwargs):
        training_calls.append(kwargs)
        return training_dataset

    def make_validation_dataset(**kwargs):
        validation_calls.append(kwargs)
        return validation_dataset

    kwargs = training_kwargs(make_training_dataset)
    model = kwargs["model"]
    standard_training(**kwargs, make_val_dataset=make_validation_dataset)

    assert validation_calls == [{"limit": 2}]
    assert training_calls == [{"offset": 0}]
    assert fit_calls[0][0] is model
    assert fit_calls[0][1]["val_dataloaders"].dataset is validation_dataset
    assert fit_calls[0][1]["train_dataloaders"].dataset is training_dataset


@pytest.mark.parametrize(
    ("mps_available", "num_workers", "expected_context"),
    [(True, 0, None), (True, 1, "fork"), (False, 1, None)],
)
def test_standard_training_sets_multiprocessing_context_only_for_mps_workers(
    training_harness,
    monkeypatch,
    mps_available,
    num_workers,
    expected_context,
):
    standard_training, fit_calls = training_harness
    monkeypatch.setattr(
        torch.backends.mps, "is_available", lambda: mps_available
    )

    standard_training(
        **training_kwargs(lambda **kwargs: TinyDataset(str(kwargs))),
        num_workers=num_workers,
    )

    loaders = (
        fit_calls[0][1]["val_dataloaders"],
        fit_calls[0][1]["train_dataloaders"],
    )
    for loader in loaders:
        if expected_context is None:
            assert loader.multiprocessing_context is None
        else:
            assert loader.multiprocessing_context.get_start_method() == expected_context
