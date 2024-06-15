from chessml import script
from chessml.play.round import Round
from chessml.play.player import TerminalPlayer
from chess import Board


@script
def main(args, config):
    round = Round(
        white_player=TerminalPlayer(), black_player=TerminalPlayer(), verbose=True
    )

    round.start(Board())
