from lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import logging
from chessml.models.torch.conv_layers import make_conv_layers
from torchvision.ops import sigmoid_focal_loss
import numpy as np

logger = logging.getLogger(__name__)


class EloPredictor(LightningModule):
    def __init__(
        self,
        input_shape: tuple,
        encoder_kernel_size: int = 2,
        encoder_channels_mult: int = 1.5,
        head_layers_num: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        downsample_layers, downsample_meta = make_conv_layers(
            input_shape=input_shape,
            kernel_size=encoder_kernel_size,
            calc_next_channels=lambda c, i: c * encoder_channels_mult,
        )
        print("downsample_meta:")
        print(downsample_meta)
        encoder_layers = []
        for conv_layer, layer_meta in zip(downsample_layers, downsample_meta):
            encoder_layers.append(conv_layer)

            if not layer_meta["is_last"]:
                encoder_layers.append(nn.BatchNorm2d(layer_meta["out_shape"][0]))
                encoder_layers.append(nn.ReLU())
            else:
                encoder_layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*encoder_layers)

        head_input_features = downsample_meta[-1]["out_shape"][0] * 2
        head_output_features = 2
        head_mult = (head_output_features / head_input_features) ** (
            1 / (head_layers_num)
        )
        head_layers = []

        for i in range(head_layers_num):
            last = i == head_layers_num - 1
            input_features = int(head_input_features * (head_mult ** i))
            output_features = (
                head_output_features
                if last
                else int(head_input_features * (head_mult ** (i + 1)))
            )
            print(f"head #{i}: {input_features} -> {output_features}")

            head_layers.append(nn.ReLU())

            if not last:
                head_layers.append(nn.Dropout(0.2))

            head_layers.append(
                nn.Linear(in_features=input_features, out_features=output_features)
            )

        self.head = nn.Sequential(*head_layers)

    def sample(self, x, training: bool = False):
        position_before, position_after, _ = x

        encoded_before = self.encoder(position_before)
        encoded_after = self.encoder(position_after)

        concatenated = torch.cat([encoded_before, encoded_after], dim=1)

        output = self.head(concatenated)

        mean = output[:, 0]
        logvar = output[:, 1]

        sigma = torch.exp(logvar * 0.5) if training else 0.0
        standard_normal = torch.randn_like(logvar)

        # Sample normal:
        sampled = mean + sigma * standard_normal

        # Sample log-normal:
        # sampled = torch.exp(mean + sigma * standard_normal)

        # Luckily, both normal and log-normal distribution have the same KL-divergence
        # when Q has mean 0 and variance 1
        kl_divergence = 0.5 * torch.mean(mean ** 2 + torch.exp(logvar) - logvar - 1)

        return sampled, kl_divergence

    def calc_losses(self, x, batch_idx, training: bool):
        _, _, true_elo = x
        sampled, _ = self.sample(x, training=training)

        pred_elo = sampled
        loss = F.huber_loss(pred_elo, true_elo, reduction="mean")

        return {"huber": loss}

        # base_elo = 1500
        # dev_elo = 600

        # reconstruction_loss = F.huber_loss(sampled, (true_elo - base_elo) / dev_elo, reduction='mean')

        # return {
        #     "huber": F.huber_loss(sampled * dev_elo + base_elo, true_elo, reduction='mean'),
        #     "kl": kl_divergence,
        #     "loss": reconstruction_loss + kl_divergence,
        # }

    def training_step(self, batch, batch_idx):
        losses = self.calc_losses(batch, batch_idx, training=True)
        self.log("train_loss", losses["huber"])
        return losses["huber"]

    def validation_step(self, batch, batch_idx):
        losses = self.calc_losses(batch, batch_idx, training=False)
        self.log("val_loss", losses["huber"])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    # def predict(self, board: np.ndarray) -> np.ndarray:
    #     tensor_image = torch.tensor(board).unsqueeze(0).to(self.device)

    #     with torch.no_grad():
    #         logits = self(tensor_image).squeeze()

    #     return (logits > 0.5).bool().tolist()
