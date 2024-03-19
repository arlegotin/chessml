from chess.pgn import read_game
from chess import IllegalMoveError
from pathlib import Path
import logging
from typing import Optional, Iterable, Iterator
from chessml.data.games.game import GameResult, Game
from torch_exid import ExtendedIterableDataset

logger = logging.getLogger(__name__)


class GamesFromPGN(ExtendedIterableDataset):
    def __init__(
        self,
        path: Optional[Path] = None,
        paths: Optional[Iterable[Path]] = [],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert bool(paths) ^ bool(path), "either path or paths must be specified"

        self.paths = paths or [path]

    def generator(self) -> Iterator[Game]:
        for path in self.paths:
            logger.info(f"reading {path}")

            with path.open("r") as file:
                while True:
                    try:
                        game = read_game(file)

                        if game is None:
                            break

                        yield Game(
                            moves=list(map(str, game.mainline_moves())),
                            result=GameResult.from_string(game.headers["Result"]),
                        )

                        # yield {
                        #     "moves": list(map(str, game.mainline_moves())),
                        #     "result": GameResult.from_string(game.headers["Result"]),
                        #     "event": game.headers["Event"],
                        #     "round": game.headers["Round"],
                        #     "white_player": game.headers["White"],
                        #     "black_player": game.headers["Black"],
                        # }

                    except (AssertionError, UnicodeDecodeError, IllegalMoveError) as e:
                        logger.warn(f"skip game: {str(e)}")
                        continue
