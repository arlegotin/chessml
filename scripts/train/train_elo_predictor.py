from chessml import script, config
from pathlib import Path
from chessml.data.games.games_from_pgn import GamesFromPGN
from chessml.data.boards.boards_from_fen import ConsecutiveBoardsFromFEN
import logging
import subprocess
from chessml.models.lightning.elo_predictor_model import EloPredictor
from chessml.data.boards.board_representation import OnlyPieces
from chessml.train.standard_training import standard_training
import torch


script.add_argument("-bs", dest="batch_size", type=int, default=256)
script.add_argument("-vb", dest="val_batches", type=int, default=256)
script.add_argument("-vi", dest="val_interval", type=int, default=256)


@script
def main(args):
    board_representation = OnlyPieces()

    def make_dataset(**kwargs):
        def transform(x):
            current_board, next_board, elo = x
            return (
                board_representation(current_board),
                board_representation(next_board),
                torch.tensor(elo, dtype=torch.float),
            )

        return ConsecutiveBoardsFromFEN(
            path=Path("./datasets/consecutive_fens.txt"),
            transforms=[transform],
            **kwargs,
        )

    model = EloPredictor(input_shape=board_representation.shape)

    standard_training(
        model=model,
        make_dataset=make_dataset,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        val_interval=args.val_interval,
        checkpoint_name=f"ep-2-{{step}}",
    )
