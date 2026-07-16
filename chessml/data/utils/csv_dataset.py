from pathlib import Path
import logging
import csv
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    """
    Reads file line by line.
    Uses parse_file method to convert line into dataset item
    """

    def __init__(self, path: Path, skip_header: bool = True, limit: int = None, offset: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.skip_header = skip_header
        self.limit = limit
        self.offset = offset
        self._data = []
        self._load_data()

    def _load_data(self):
        """Load data from CSV file into memory, applying limit and offset"""
        with open(self.path, 'r', newline='') as f:
            reader = csv.reader(f)
            if self.skip_header:
                next(reader, None)  # Skip header
            
            # Skip offset rows
            for _ in range(self.offset):
                next(reader, None)
            
            # Read up to limit rows if specified
            if self.limit is not None:
                self._data = [tuple(next(reader)) for _ in range(self.limit) if reader]
            else:
                self._data = [tuple(row) for row in reader]

    def __len__(self):
        """Return the number of rows in the dataset after applying limit and offset"""
        return len(self._data)

    def __getitem__(self, idx):
        """Return the row at the given index as a tuple"""
        return self._data[idx]

    
