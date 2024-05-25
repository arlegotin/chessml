from chessml.data.utils.file_lines_dataset import FileLinesDataset
import numpy as np


class ValuesFromFile(FileLinesDataset):
    def transform_line(self, line: str):
        if line == "-":
            return None

        e_type, value, _ = line.split(" ")

        if e_type == "mate":
            return None

        return np.float32(float(value) / 100)
