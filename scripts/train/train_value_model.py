from chessml import script
from chessml.models.lightning.value_model import ValueModel
from chessml.models.lightning.conv_vae import ConvVAE
from chessml.data.boards.boards_from_fen import BoardsFromFEN
from chessml.data.boards.board_representation import OnlyPieces, FullPosition
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from chessml.data.values.values_from_file import ValuesFromFile
from torch_exid import ExtendedIterableDataset

script.add_argument(
    "-pf", dest="path_to_fens", type=str, default="./datasets/unique_fens.txt"
)
script.add_argument(
    "-pe",
    dest="path_to_evaluations",
    type=str,
    default="./datasets/tmp_eval_stockfish_7_10.txt",
)
script.add_argument("-bs", dest="batch_size", type=int, default=128)
script.add_argument("-vb", dest="val_batches", type=int, default=10)
script.add_argument("-vi", dest="val_interval", type=int, default=50)
script.add_argument("-ci", dest="checkpoint_interval", type=int, default=50)
script.add_argument("-ss", dest="shuffle_seed", type=int, default=42)
script.add_argument("-sb", dest="shuffle_batches", type=int, default=2)
script.add_argument("-me", dest="max_epochs", type=int, default=1500)


class BoardsAndValues(ExtendedIterableDataset):
    def __init__(self, boards, values, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boards = boards
        self.values = values

    def generator(self):
        for board, value in zip(self.boards, self.values):
            if value is None:
                self.skip_next()

            yield board, value


@script
def train(args, config):
    board_representation = FullPosition()

    boards = BoardsFromFEN(
        path=Path(args.path_to_fens), transforms=[board_representation]
    )

    values = ValuesFromFile(path=Path(args.path_to_evaluations))

    val_dataloader = DataLoader(
        BoardsAndValues(
            boards=boards, values=values, limit=args.batch_size * args.val_batches
        ),
        batch_size=args.batch_size,
    )

    train_dataloader = DataLoader(
        BoardsAndValues(
            boards=boards,
            values=values,
            offset=args.batch_size * args.val_batches,
            # shuffle_buffer=args.batch_size * args.shuffle_batches,
            # shuffle_seed=args.shuffle_seed,
        ),
        batch_size=args.batch_size,
    )

    ae = ConvVAE.load_from_checkpoint(
        "/var/www/chessml/checkpoints/vae-full_position-ld=32-ks=2-cm=1.7-step=400.ckpt"
    )
    ae.eval()

    model = ValueModel(
        input_shape=board_representation.shape,
        encoder=ae.encoder,
        encoder_features_mult=0.1,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=TensorBoardLogger(config.logs.tensorboard_path),
        callbacks=[
            ModelCheckpoint(
                dirpath=config.checkpoints.path,
                every_n_train_steps=args.checkpoint_interval,
                filename=f"value_movel-{{step}}",
                save_top_k=3,
                monitor="train_loss",
            )
        ],
        val_check_interval=args.val_interval,
        check_val_every_n_epoch=None,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
