from lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import logging
from chessml.models.torch.conv_layers import make_conv_layers
import numpy as np

logger = logging.getLogger(__name__)


class MetaPredictor(LightningModule):
    def __init__(self, input_shape: tuple, path_to_fens: str | None = None):
        super().__init__()
        self.save_hyperparameters()

        downsample_layers, downsample_meta = make_conv_layers(
            input_shape=input_shape,
            kernel_size=2,
            calc_next_channels=lambda c, i: c * 1.5,
        )

        layers = []
        for conv_layer, layer_meta in zip(downsample_layers, downsample_meta):
            layers.append(conv_layer)

            if not layer_meta["is_last"]:
                layers.append(nn.BatchNorm2d(layer_meta["out_shape"][0]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Flatten())

        """
        - 4 for castling rights (KQkq)
        - 1 for whose move
        - 1 for board flipped
        """
        output_features = 6

        f1 = downsample_meta[-1]["out_shape"][0]
        f2 = int(math.sqrt(f1 * output_features))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(in_features=f1, out_features=f2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=f2, out_features=output_features))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def calc_losses(self, x, batch_idx):
        inputs, targets = x
        outputs = self(inputs)

        return F.binary_cross_entropy(outputs, targets)

    def training_step(self, batch, batch_idx):
        loss = self.calc_losses(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calc_losses(batch, batch_idx)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    def predict(self, board: np.ndarray) -> np.ndarray:
        tensor_image = torch.tensor(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self(tensor_image).squeeze()

        return (logits > 0.5).bool().tolist()
