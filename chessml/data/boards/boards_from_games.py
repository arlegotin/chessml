from chess import Board
from torch_exid import ExtendedIterableDataset
from chessml.data.games.game import Game
from typing import Iterable, Iterator


class BoardsFromGames(ExtendedIterableDataset):
    """
    Iterates over games and yields boards for every game position
    """

    def __init__(self, games: Iterable[Game], *args, **kwargs):
        super().__init__(transforms_required=True, *args, **kwargs)
        self.games = games

    def generator(self) -> Iterator[Board]:
        board = Board()

        for game in self.games:
            board.reset()

            yield board

            for move in game.moves:
                board.push_san(move)
                yield board
