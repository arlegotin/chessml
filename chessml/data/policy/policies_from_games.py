from chess import Board
from torch_exid import ExtendedIterableDataset
from chessml.data.games.game import Game, GameResult
from typing import Iterable, Iterator, Tuple


class PoliciesFromGames(ExtendedIterableDataset):
    """
    Iterates over games and yields boards with the next move and game result
    """

    def __init__(self, games: Iterable[Game], *args, **kwargs):
        super().__init__(transforms_required=True, *args, **kwargs)
        self.games = games

    def generator(self) -> Iterator[Tuple[Board, str, GameResult]]:
        board = Board()

        for game in self.games:
            board.reset()

            for move in game.moves:
                # Validate move before yield:
                parsed_move = board.parse_san(move)

                yield board, str(parsed_move), game.result
                board.push_san(move)
