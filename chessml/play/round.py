from chessml.play.player import Player
from chess import Board, InvalidMoveError, IllegalMoveError, AmbiguousMoveError
import logging


class Round:
    def __init__(
        self, white_player: Player, black_player: Player, verbose: bool = False
    ):
        self.white_player = white_player
        self.black_player = black_player
        self.verbose = verbose

    def start(self, board: Board):
        while not board.is_game_over():
            player = self.white_player if board.turn else self.black_player

            if self.verbose:
                print(
                    f"\n--------- move #{board.fullmove_number}, {'white' if board.turn else 'black'} to move ---------\n"
                )
                print(str(board))

            for move in player.get_moves(board):
                if self.verbose:
                    print(f"\n{'white' if board.turn else 'black'} moves {move}")

                try:
                    board.push_san(move)
                except (InvalidMoveError, IllegalMoveError, AmbiguousMoveError) as e:
                    logging.error(str(e))
                    continue

                break

        outcome = board.outcome()

        if not outcome:
            raise RuntimeError("game over without outcome")

        if self.verbose:
            print(f"\n--------- final position ---------\n")
            print(str(board))

        logging.info(f"game over: {str(outcome.result())}, {str(outcome.termination)}")
