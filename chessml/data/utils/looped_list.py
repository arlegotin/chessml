from random import Random
from typing import Optional


class LoopedList:
    def __init__(self, data: list, shuffle_seed: Optional[int] = None):
        self.data = data[:]

        if shuffle_seed:
            Random(shuffle_seed).shuffle(self.data)

    def __getitem__(self, index):
        if not self.data:
            raise IndexError("list index out of range")

        return self.data[index % len(self.data)]
