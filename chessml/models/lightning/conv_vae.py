from lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import logging
from chessml.models.torch.vae_bottleneck import VAEBottleneck
from chessml.models.torch.conv_layers import make_conv_layers, mirror_conv_layers
from chessml.models.torch.custom_sequential import CustomSequential

logger = logging.getLogger(__name__)


class ConvVAE(LightningModule):
    def __init__(
        self,
        input_shape: tuple,
        kernel_size: int,
        channel_mult: float,
        latent_dim: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        logger.info("assembling autoencoder")

        # Encoder
        logger.info("assembling encoder")

        downsample_layers, downsample_meta = make_conv_layers(
            input_shape=input_shape,
            kernel_size=kernel_size,
            calc_next_channels=lambda c, i: c * channel_mult,
        )

        pre_latent_shape = downsample_meta[-1]["out_shape"]
        pre_latent_dim = int(math.prod(pre_latent_shape))

        self.encoder = CustomSequential()
        for conv_layer, layer_meta in zip(downsample_layers, downsample_meta):
            self.encoder.add_module(layer_meta["name"], conv_layer)

            self.encoder.add_module(
                f"norm #{layer_meta['index']}",
                nn.BatchNorm2d(layer_meta["out_shape"][0]),
            )

            self.encoder.add_module(f"activation #{layer_meta['index']}", nn.ELU())

            if layer_meta["is_last"]:
                self.encoder.add_module(
                    f"output flatten: {pre_latent_shape} -> ({pre_latent_dim})",
                    nn.Flatten(),
                )

        # Bottleneck
        logger.info(
            f"bottleneck: ({pre_latent_dim}) -> Gaussian({latent_dim}) -> ({pre_latent_dim})"
        )
        self.bottleneck = VAEBottleneck(
            external_dim=pre_latent_dim, latent_dim=latent_dim,
        )

        # Decoder
        logger.info("assembling decoder")
        upsample_layers, upsample_meta = mirror_conv_layers(downsample_meta)

        self.decoder = CustomSequential()
        for conv_layer, layer_meta in zip(upsample_layers, upsample_meta):
            if layer_meta["is_first"]:
                self.decoder.add_module(
                    f"input unflatten: ({pre_latent_dim}) -> {pre_latent_shape}",
                    nn.Unflatten(1, pre_latent_shape),
                )

            self.decoder.add_module(layer_meta["name"], conv_layer)

            self.decoder.add_module(
                f"norm #{layer_meta['index']}",
                nn.BatchNorm2d(layer_meta["out_shape"][0]),
            )

            if not layer_meta["is_last"]:
                self.decoder.add_module(f"activation #{layer_meta['index']}", nn.ELU())

    def forward(self, x):
        return self.bottleneck(self.encoder(x))

    def validation_step(self, x, batch_idx):
        pre_latent = self.encoder(x)
        latent = self.bottleneck.inference_sample(pre_latent)
        decoded = self.decoder(latent)

        cross_entropy = F.cross_entropy(decoded, x)

        self.log_dict(
            {"val_cross_entropy": cross_entropy,}
        )

    def training_step(self, x, batch_idx):
        pre_latent = self.encoder(x)
        sampled, kl_divergence = self.bottleneck.train_sample(pre_latent)
        decoded = self.decoder(sampled)

        cross_entropy = F.cross_entropy(decoded, x)
        loss = cross_entropy + kl_divergence

        # if torch.isnan(loss):
        #     print("loss NaN")
        #     print(x)
        #     print(pre_latent)
        #     print(sampled)
        #     print(kl_divergence)
        #     print(decoded)
        #     quit()

        self.log_dict(
            {
                "train_loss": loss,
                "train_cross_entropy": cross_entropy,
                "train_kl_divergence": kl_divergence,
            }
        )

        return loss

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters())
        return torch.optim.AdamW(self.parameters())
