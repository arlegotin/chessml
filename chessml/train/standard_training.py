from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from lightning import LightningModule
from typing import Type, Callable
from torch.utils.data import Dataset


def standard_training(
        model: Type[LightningModule],
        make_dataset: Callable[..., Dataset],
        batch_size: int,
        val_batches: int,
        val_interval: int,
        shuffle_buffer: int,
        checkpoint_name: str,
        config,
):
    val_dataloader = DataLoader(
        make_dataset(
            limit=batch_size * val_batches,
        ),
        batch_size=batch_size,
    )

    train_dataloader = DataLoader(
        make_dataset(
            offset=batch_size * val_batches,
            shuffle_buffer=shuffle_buffer,
        ),
        batch_size=batch_size,
    )

    trainer = Trainer(
        max_epochs=1000,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=TensorBoardLogger(config.logs.tensorboard_path),
        callbacks=[
            ModelCheckpoint(
                dirpath=config.checkpoints.path,
                every_n_train_steps=None,
                filename=checkpoint_name,
                save_top_k=3,
                monitor="val_loss",
                mode="min"
            ),
        ],
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=32,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )
