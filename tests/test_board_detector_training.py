from argparse import Namespace
import importlib
from pathlib import Path
import sys
from types import ModuleType

from lightning import Callback, Trainer as LightningTrainer
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

import chessml
from chessml.data.images.picture import Picture
from chessml.data.iterable_dataset import ExtendedIterableDataset


@pytest.fixture
def board_detector_training_script(monkeypatch):
    module_name = "scripts.train.train_board_detector"
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


def test_dynamic_entrypoint_trains_validates_and_checkpoints_disjoint_sources(
    board_detector_training_script,
    monkeypatch,
    tmp_path,
):
    module = board_detector_training_script
    (tmp_path / "unique_fens.txt").write_text("0\n1\n2\n3\n")

    renderers = []

    class TinyRenderer(ExtendedIterableDataset):
        def __init__(
            self,
            fens,
            piece_sets,
            board_colors,
            square_size,
            shuffle_seed,
            *args,
            **kwargs,
        ):
            super().__init__(
                shuffle_seed=shuffle_seed,
                *args,
                **kwargs,
            )
            self.fens = fens
            renderers.append(self)

        def generator(self):
            for source_id in self.fens:
                image = np.full((1, 1, 3), int(source_id), dtype=np.uint8)
                yield Picture(image), source_id, False

    class TinyAugmentation(ExtendedIterableDataset):
        def __init__(
            self,
            boards_with_data,
            bg_images,
            shuffle_seed,
            *args,
            **kwargs,
        ):
            super().__init__(shuffle_seed=shuffle_seed, *args, **kwargs)
            self.boards_with_data = boards_with_data

        def generator(self):
            corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            for picture, source_id, flipped in self.boards_with_data:
                yield picture, corners, source_id, flipped

    backbones = []

    class TinyBackbone(torch.nn.Module):
        def __init__(self, output_features):
            super().__init__()
            self.coordinates = torch.nn.Parameter(torch.zeros(output_features))
            self.initial_coordinates = self.coordinates.detach().clone()
            backbones.append(self)

        def preprocess_image(self, image):
            source_id = np.asarray(image)[0, 0, 0]
            return torch.tensor([source_id], dtype=torch.float32)

        def forward(self, images):
            return self.coordinates.expand(images.shape[0], -1)

    fake_assets = ModuleType("chessml.data.assets")
    fake_assets.BG_IMAGES = []
    fake_assets.PIECE_SETS = []
    monkeypatch.setitem(sys.modules, "chessml.data.assets", fake_assets)
    monkeypatch.setattr(module, "BoardsImagesFromFENs", TinyRenderer)
    monkeypatch.setattr(module, "AugmentedBoardsImages", TinyAugmentation)
    monkeypatch.setattr(module, "MobileViTV2FPN", TinyBackbone)
    monkeypatch.setattr(chessml.config.dataset, "path", str(tmp_path))
    monkeypatch.setattr(
        chessml.config.checkpoints, "path", str(tmp_path / "checkpoints")
    )

    class BatchRecorder(Callback):
        def __init__(self):
            self.train_identities = []
            self.validation_identities = []
            self.train_targets = []
            self.validation_targets = []

        @staticmethod
        def identities(batch):
            return {int(value) for value in batch[0][:, 0].detach().cpu()}

        @staticmethod
        def target_contract(batch):
            return tuple(batch[1].shape), batch[1].dtype

        def on_train_batch_start(
            self, trainer, pl_module, batch, batch_idx
        ):
            self.train_identities.append(self.identities(batch))
            self.train_targets.append(self.target_contract(batch))

        def on_validation_batch_start(
            self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
        ):
            self.validation_identities.append(self.identities(batch))
            self.validation_targets.append(self.target_contract(batch))

    recorder = BatchRecorder()
    trainers = []
    training_module = importlib.import_module("chessml.train.standard_training")

    def bounded_trainer(**kwargs):
        kwargs.update(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path),
        )
        kwargs["callbacks"] = [*kwargs["callbacks"], recorder]
        trainer = LightningTrainer(**kwargs)
        trainers.append(trainer)
        return trainer

    monkeypatch.setattr(training_module, "Trainer", bounded_trainer)
    monkeypatch.setattr(
        training_module, "TensorBoardLogger", lambda *args, **kwargs: False
    )

    real_standard_training = module.standard_training

    def bounded_standard_training(**kwargs):
        return real_standard_training(
            **kwargs,
            max_epochs=1,
            save_top_k=1,
            log_steps=1,
        )

    monkeypatch.setattr(module, "standard_training", bounded_standard_training)

    module.train(
        Namespace(
            batch_size=2,
            val_batches=1,
            val_interval=1,
            seed=65,
            dataset_type="dynamic",
            dataset_path=str(tmp_path / "unused"),
            model="MobileViTV2FPN",
            checkpoint=None,
        )
    )

    assert len(renderers) == 2
    assert all(
        (renderer.offset, renderer.limit, renderer.shuffle_buffer) == (0, -1, 1)
        for renderer in renderers
    )
    validation_source, training_source = [renderer.fens for renderer in renderers]
    assert (
        validation_source.offset,
        validation_source.limit,
        validation_source.shuffle_seed,
        validation_source.shuffle_buffer,
    ) == (0, 2, 65, 20)
    assert (
        training_source.offset,
        training_source.limit,
        training_source.shuffle_seed,
        training_source.shuffle_buffer,
    ) == (2, -1, 65, 20)

    assert recorder.validation_identities == [{0, 1}]
    assert recorder.train_identities == [{2, 3}]
    assert recorder.validation_targets == [((2, 8), torch.float32)]
    assert recorder.train_targets == [((2, 8), torch.float32)]

    assert len(trainers) == 1
    trainer = trainers[0]
    assert isinstance(trainer.train_dataloader.dataset, IterableDataset)
    assert trainer.global_step == 1
    assert len(backbones) == 1
    assert not torch.equal(
        backbones[0].coordinates.detach(), backbones[0].initial_coordinates
    )

    checkpoint = next(
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    )
    assert checkpoint.monitor == "val_loss"
    assert checkpoint.best_model_score is not None
    assert torch.isfinite(checkpoint.best_model_score)
    best_path = Path(checkpoint.best_model_path)
    assert best_path.is_file()
    assert best_path.resolve().is_relative_to(tmp_path.resolve())
