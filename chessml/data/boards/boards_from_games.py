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


class ConsecutiveBoardsFromGames(BoardsFromGames):
    def generator(self) -> Iterator[tuple[Board, Board, Game, bool]]:
        current_board = Board()

        for game in self.games:
            current_board.reset()
            whites_turn = True

            for move in game.moves:
                current_board, next_board = current_board.copy(), current_board
                next_board.push_san(move)
                yield current_board, next_board, game, whites_turn
                current_board = next_board
                whites_turn = not whites_turn
