from chessml import script
from chessml.models.lightning.conv_vae import ConvVAE
from chessml.data.boards.boards_from_fen import BoardsFromFEN
from chessml.data.boards.board_representation import OnlyPieces, FullPosition
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

div = 8

script.add_argument(
    "-p", dest="path_to_fens", type=str, default="./datasets/unique_fens.txt"
)
script.add_argument("-bs", dest="batch_size", type=int, default=2 ** 12 // div)
script.add_argument("-vb", dest="val_batches", type=int, default=10 * div)
script.add_argument("-vi", dest="val_interval", type=int, default=50 * div)
script.add_argument("-ci", dest="checkpoint_interval", type=int, default=50 * div)
script.add_argument("-ld", dest="latent_dim", type=int, default=32)
script.add_argument("-ks", dest="kernel_size", type=int, default=2)
script.add_argument("-cm", dest="channels_mult", type=float, default=1.7)
script.add_argument("-ss", dest="shuffle_seed", type=int, default=42)
script.add_argument("-sb", dest="shuffle_batches", type=int, default=10 * div)
script.add_argument("-me", dest="max_epochs", type=int, default=1500)
script.add_argument(
    "-br",
    dest="board_representation",
    type=str,
    choices=["only_pieces", "full_position"],
    default="only_pieces",
)


@script
def train(args, config):
    if args.board_representation == "only_pieces":
        board_representation = OnlyPieces()
    elif args.board_representation == "full_position":
        board_representation = FullPosition()
    else:
        raise ValueError(f"unknown board representation {args.board_representation}")

    val_dataloader = DataLoader(
        BoardsFromFEN(
            path=Path(args.path_to_fens),
            limit=args.batch_size * args.val_batches,
            transforms=[board_representation],
        ),
        batch_size=args.batch_size,
    )

    train_dataloader = DataLoader(
        BoardsFromFEN(
            path=Path(args.path_to_fens),
            offset=args.batch_size * args.val_batches,
            shuffle_buffer=args.batch_size * args.shuffle_batches,
            shuffle_seed=args.shuffle_seed,
            transforms=[board_representation],
        ),
        batch_size=args.batch_size,
    )

    model = ConvVAE(
        input_shape=board_representation.shape,
        kernel_size=args.kernel_size,
        channel_mult=args.channels_mult,
        latent_dim=args.latent_dim,
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
                filename=f"vae-{args.board_representation}-ld={args.latent_dim}-ks={args.kernel_size}-cm={args.channels_mult}-{{step}}",
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
