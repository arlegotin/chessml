from abc import ABC, abstractmethod
from chess import Board
from lightning import LightningModule
from chessml.data.boards.board_representation import BoardRepresentation
from chessml.data.policy.policy_representation import PolicyRepresentation
import numpy as np
import torch
from typing import Iterable


class Player(ABC):
    def __init__(self, name: str = "unknown"):
        self.name = name

    @abstractmethod
    def get_moves(self, board: Board, *args, **kwargs) -> Iterable[str]:
        """
        For given position should return list of moves starting from most desired
        """


class TerminalPlayer(Player):
    def get_moves(self, board: Board) -> Iterable[str]:
        print(f"\npossible moves:\n{', '.join(map(str, board.legal_moves))}")
        return [input("\nyour move: ")]


class AIPlayer(Player):
    def __init__(
        self,
        model: LightningModule,
        board_representation: BoardRepresentation,
        policy_representation: PolicyRepresentation,
        value: float,
    ):
        self.model = model
        self.board_representation = board_representation
        self.policy_representation = policy_representation
        self.value = value

    def get_moves(self, board: Board) -> Iterable[str]:
        board_tensor = torch.from_numpy(
            np.expand_dims(self.board_representation(board), 0)
        )
        value_tensor = torch.from_numpy(np.array([self.value]))

        with torch.no_grad():
            move_tensor = self.model.forward((board_tensor, value_tensor))

        moves_mask = self.policy_representation(map(str, board.legal_moves))

        return self.policy_representation.to_string(move_tensor[0] + moves_mask * 10000)
