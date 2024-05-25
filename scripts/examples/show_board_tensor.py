from chessml import script
from chessml.data.boards.board_representation import OnlyPieces
from chess import Board


@script
def main(args, config):
    board_representation = OnlyPieces()
    board = Board()

    first_tensor = board_representation(board)

    print(f"{first_tensor.shape=}")
    print(board_representation(first_tensor), "\n")

    board.push_san("e2e4")

    second_tensor = board_representation(board)

    print(f"{second_tensor.shape=}")
    print(board_representation(second_tensor))
