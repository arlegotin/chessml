import re
import numpy as np
import torch
from chess import Board
from pathlib import Path


def count_lines_in_file(path: Path) -> int:
    n = 0

    with path.open("r") as file:
        for _ in file:
            n += 1

    return n


def starts_with_number(string: str) -> bool:
    """
    Return True if the given string starts with a number, False otherwise
    """
    pattern = r"^\d"
    match = re.match(pattern, string)
    return bool(match)


def numpy_to_batched_tensor(ndarray: np.ndarray) -> torch.Tensor:
    """
    Converts numpy array into torch tensor with batch dimension
    """
    return torch.from_numpy(np.expand_dims(ndarray, 0))


def display_board(board: Board) -> str:
    fen = board.fen()
    placement = fen.split(" ")[0]

    board_str = " abcdefg "

    pieces_str = ""
    for s in placement:
        if s == "r":
            pieces_str += "♜"
        elif s == "n":
            pieces_str += "♞"
        elif s == "b":
            pieces_str += "♝"
        elif s == "q":
            pieces_str += "♛"
        elif s == "k":
            pieces_str += "♚"
        elif s == "p":
            pieces_str += "♟︎"
        elif s == "R":
            pieces_str += "♖"
        elif s == "N":
            pieces_str += "♘"
        elif s == "B":
            pieces_str += "♗"
        elif s == "Q":
            pieces_str += "♕"
        elif s == "K":
            pieces_str += "♔"
        elif s == "P":
            pieces_str += "♙"
        elif s == "/":
            pieces_str += "\n"
        else:
            pieces_str += "." * int(s)

    board_str += " abcdefg "

    return pieces_str

    # indices_array = np.argmax(array, axis=0)

    # class_symbols = {v: k for k, v in PIECE_CLASSES.items()}
    # class_symbols[0] = "."

    # lines = []
    # for row in indices_array.T:
    #     lines.append(" ".join(map(lambda cls: class_symbols[cls], row)))

    # return "\n".join(lines)
