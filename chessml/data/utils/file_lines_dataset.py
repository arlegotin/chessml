from pathlib import Path
from typing import Any, Optional, Iterator
import logging
from torch_exid import ExtendedIterableDataset

logger = logging.getLogger(__name__)


class FileLinesDataset(ExtendedIterableDataset):
    """
    Reads file line by line.
    Uses parse_file method to convert line into dataset item
    """

    def __init__(self, path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path

    def transform_line(self, line: str) -> Any:
        """
        Transforms one line.
        Should raise AssertionError if line is invalid
        """
        return line

    def validate_line(self, line: str) -> None:
        """
        Should raise AssertionError if line is invalid
        """

    def generator(self) -> Iterator[Any]:
        logger.info(f"reading {self.path}")

        with self.path.open("r") as file:
            for line in file:
                line = line.rstrip("\n")

                try:
                    self.validate_line(line)
                    line = self.transform_line(line)
                except AssertionError as e:
                    logger.warn(f"skip line: {str(e)}")
                    continue

                yield line
