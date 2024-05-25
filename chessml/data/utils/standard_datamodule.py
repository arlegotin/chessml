from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split


class StandardDatamodule(LightningDataModule):
    def __init__(
        self, dataset: Dataset, batch_size: int, val_part: float, num_workers: int
    ):
        super().__init__()

        self.train_dataset, self.val_dataset = random_split(
            dataset, [1 - val_part, val_part]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
