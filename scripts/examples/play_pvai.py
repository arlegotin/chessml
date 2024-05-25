from chessml import script
from chessml.play.round import Round
from chessml.play.player import TerminalPlayer, AIPlayer
from chess import Board
from chessml.models.lightning.policy_model import VectorPolicyModel
from chessml.data.boards.board_representation import FullPosition
from chessml.data.policy.policy_representation import PolicyAsVector


@script
def main(args, config):
    model = VectorPolicyModel.load_from_checkpoint(
        "./checkpoints/policy_model_1-step=29300.ckpt"
    )

    model.eval()

    round = Round(
        white_player=AIPlayer(
            model=model,
            board_representation=FullPosition(),
            policy_representation=PolicyAsVector(),
            value=1,
        ),
        black_player=AIPlayer(
            model=model,
            board_representation=FullPosition(),
            policy_representation=PolicyAsVector(),
            value=1,
        ),
        verbose=True,
    )

    round.start(Board())
