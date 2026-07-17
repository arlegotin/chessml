from random import Random
from typing import Any, Callable, Iterator

from torch.utils.data import IterableDataset


class ExtendedIterableDataset(IterableDataset):
    def __init__(
        self,
        shuffle_buffer: int = 1,
        shuffle_seed: int | None = None,
        offset: int = 0,
        limit: int = -1,
        transforms: list[Callable[[Any], Any]] | None = None,
        transforms_required: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if transforms_required and not transforms:
            raise ValueError("ExtendedIterableDataset requires transforms")

        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = (
            shuffle_seed
            if shuffle_seed is not None
            else Random().randint(0, 999331)
        )
        self.buffer: list[Any] = []
        self.offset = offset
        self.limit = limit
        self.counter = 0
        self._skip_next = False
        self.transforms = [] if transforms is None else transforms

    def skip_next(self) -> None:
        self._skip_next = True

    def generator(self) -> Iterator[Any]:
        raise NotImplementedError("please implement generator method")

    @property
    def _limit_allows_one_more(self) -> bool:
        return self.limit < 0 or self.counter < self.limit + self.offset

    def _items_with_conditions(self) -> Iterator[Any]:
        source = self.generator()

        while True:
            try:
                if not self._limit_allows_one_more:
                    raise StopIteration

                item = next(source)

                for transform in self.transforms:
                    item = transform(item)

                if self._skip_next:
                    self._skip_next = False
                    continue

                self.counter += 1

                if self.counter - 1 < self.offset:
                    continue

                yield item
            except StopIteration:
                break

    def _flush_buffer(self) -> Iterator[Any]:
        Random(self.shuffle_seed).shuffle(self.buffer)

        for item in self.buffer:
            yield item

        self.buffer = []

    def _iterate(self) -> Iterator[Any]:
        if self.shuffle_buffer > 1:
            for item in self._items_with_conditions():
                self.buffer.append(item)

                if len(self.buffer) >= self.shuffle_buffer:
                    yield from self._flush_buffer()

            if self.buffer:
                yield from self._flush_buffer()
        else:
            yield from self._items_with_conditions()

        self.counter = 0
        self.buffer = []
        self._skip_next = False

    def __iter__(self) -> Iterator[Any]:
        return self._iterate()
