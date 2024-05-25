from chessml import script
from chessml.models.lightning.conv_vae import ConvVAE
from chessml.models.lightning.policy_model import VectorPolicyModel
from chessml.data.boards.board_representation import FullPosition
from chessml.data.policy.policy_representation import PolicyAsVector
from chessml.data.games.games_from_pgn import GamesFromPGN
from chessml.data.policy.policies_from_games import PoliciesFromGames
from pathlib import Path
from chessml.data.games.game import GameResult
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

script.add_argument("-bs", dest="batch_size", type=int, default=512)


@script
def main(args, config):
    board_representation = FullPosition()
    policy_representation = PolicyAsVector()

    ae = ConvVAE.load_from_checkpoint(
        "/var/www/chessml/checkpoints/vae-full_position-ld=32-ks=2-cm=1.7-step=400.ckpt"
    )
    ae.eval()

    model = VectorPolicyModel(
        input_shape=board_representation.shape,
        output_dim=policy_representation.shape[0],
        encoder=ae.encoder,
        encoder_features_mult=0.1,
    )

    def transform(x):
        board, move, result = x

        value = (
            1
            if result == GameResult.WHITE_WON
            else (-1 if result == GameResult.BLACK_WON else 0)
        ) * (1 if board.turn else -1)

        return (
            board_representation(board),
            policy_representation([move]),
            policy_representation(map(str, board.legal_moves)),
            value,
        )

    train_dataloader = DataLoader(
        PoliciesFromGames(
            games=GamesFromPGN(
                paths=map(
                    lambda name: Path(f"./datasets/players/{name}.pgn"),
                    config.dataset.sections.players.names,
                ),
            ),
            transforms=[transform],
            shuffle_buffer=args.batch_size * 10,
            shuffle_seed=42,
        ),
        batch_size=args.batch_size,
    )

    val_size = args.batch_size * 100

    val_dataloader = DataLoader(
        PoliciesFromGames(
            games=GamesFromPGN(path=Path("./datasets/players/Capablanca.pgn"),),
            transforms=[transform],
            shuffle_buffer=val_size,
            shuffle_seed=42,
            limit=val_size,
        ),
        batch_size=args.batch_size,
    )

    trainer = Trainer(
        max_epochs=100,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=TensorBoardLogger(config.logs.tensorboard_path),
        callbacks=[
            ModelCheckpoint(
                dirpath=config.checkpoints.path,
                every_n_train_steps=50,
                filename=f"policy_model_1-{{step}}",
                save_top_k=3,
                monitor="train_loss",
            ),
        ],
        val_check_interval=200,
        check_val_every_n_epoch=None,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )
