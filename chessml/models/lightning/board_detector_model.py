import torch
from lightning import LightningModule
import torch.nn.functional as F
from typing import Type


class BoardDetector(LightningModule):
    def __init__(
        self, base_model_class: Type[torch.nn.Module], base_model_kwargs: dict
    ):
        super().__init__()
        self.model = base_model_class(**base_model_kwargs)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        images, coords = batch
        outputs = self(images)
        loss = F.mse_loss(outputs, coords)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, coords = batch
        outputs = self(images)
        loss = F.mse_loss(outputs, coords)

        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
