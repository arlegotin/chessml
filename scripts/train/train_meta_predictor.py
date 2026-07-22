from chessml import script
from chessml.models.lightning.meta_predictor_model import MetaPredictor
from chessml.data.boards.boards_from_fen import BoardsFromFEN
from chessml.data.boards.board_representation import OnlyPieces
from pathlib import Path
from chess import Board, WHITE, BLACK
from chessml.train.standard_training import standard_training
import torch
import numpy as np

script.add_argument(
    "-p", dest="path_to_fens", type=str, default="./datasets/unique_fens.txt"
)
script.add_argument("-bs", dest="batch_size", type=int, default=1024)
script.add_argument("-vb", dest="val_batches", type=int, default=256)
script.add_argument("-vi", dest="val_interval", type=int, default=32 * 8)
script.add_argument("-ss", dest="shuffle_seed", type=int, default=68)


@script
def train(args):
    path_to_fens = Path(args.path_to_fens).resolve()
    board_representation = OnlyPieces()
    counter = 0

    def transform(board: Board):
        white_kingside_castling = int(board.has_kingside_castling_rights(WHITE))
        white_queenside_castling = int(board.has_queenside_castling_rights(WHITE))
        black_kingside_castling = int(board.has_kingside_castling_rights(BLACK))
        black_queenside_castling = int(board.has_queenside_castling_rights(BLACK))

        white_turn = int(board.turn == WHITE)

        board_tensor = board_representation(board)

        nonlocal counter
        flipped = counter % 2
        counter += 1

        if flipped == 1:
            board_tensor = np.flip(board_tensor, axis=(1, 2)).copy()

        return (
            board_tensor,
            np.array(
                [
                    white_kingside_castling,
                    white_queenside_castling,
                    black_kingside_castling,
                    black_queenside_castling,
                    white_turn,
                    flipped,
                ]
            ).astype(np.float32),
        )

    def make_dataset(**kwargs):
        return BoardsFromFEN(
            path=path_to_fens,
            transforms=[transform],
            **kwargs,
        )

    model = MetaPredictor(
        input_shape=board_representation.shape,
        path_to_fens=str(path_to_fens),
    )

    standard_training(
        model=model,
        make_dataset=make_dataset,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        val_interval=args.val_interval,
        checkpoint_name=f"bm-6-bs={args.batch_size}-{{step}}",
        shuffle=False,
        checkpoint_monitor="val_loss",
    )
