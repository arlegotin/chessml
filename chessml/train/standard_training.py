from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from lightning import LightningModule
from typing import Type, Callable
from torch.utils.data import Dataset, IterableDataset
from chessml import config
import logging

logger = logging.getLogger(__name__)

def get_num_workers(dataset: Dataset) -> int:
    if isinstance(dataset, IterableDataset):
        return 1

    return config.num_workers

def standard_training(
        model: Type[LightningModule],
        make_dataset: Callable[..., Dataset],
        batch_size: int,
        val_batches: int,
        val_interval: int,
        checkpoint_name: str,
        prefetch_factor: int = 2,
        max_epochs: int = 10_000,
        save_top_k: int = 10,
        log_steps: int = 32,
):
    logger.info(f"run training: {batch_size=}, {val_batches=}, {val_interval=}")

    val_dataset = make_dataset(
        limit=batch_size * val_batches,
    )
    val_num_workers = get_num_workers(val_dataset)
    logger.info(f"{val_num_workers=}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=val_num_workers,
        prefetch_factor=prefetch_factor,
    )

    train_dataset = make_dataset(
        offset=batch_size * val_batches,
    )
    train_num_workers = get_num_workers(train_dataset)
    logger.info(f"{train_num_workers=}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=train_num_workers,
        prefetch_factor=prefetch_factor,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=TensorBoardLogger(config.logs.tensorboard_path),
        callbacks=[
            ModelCheckpoint(
                dirpath=config.checkpoints.path,
                every_n_train_steps=None,
                filename=checkpoint_name,
                save_top_k=save_top_k,
                monitor="val_loss",
                mode="min"
            ),
        ],
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=log_steps,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
