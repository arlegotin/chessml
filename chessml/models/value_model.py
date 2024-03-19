from lightning import LightningModule
import torch.nn as nn
from chessml.models.custom_sequential import CustomSequential
import torch
import torch.nn.functional as F
from math import sqrt


class ValueModel(LightningModule):
    def __init__(
        self,
        input_shape: tuple,
        encoder: CustomSequential,
        encoder_features_mult: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        for param in encoder.parameters():
            param.requires_grad = False

        self.encoder = encoder

        self.post_feature_layers = nn.ModuleList([])

        flatten_dim = 0
        current_shape = input_shape

        for feature_layer in encoder.layers:
            if not self.feature_condition(feature_layer):
                continue

            post_feature_layer = nn.Conv2d(
                feature_layer.out_channels,
                int(feature_layer.out_channels * encoder_features_mult),
                1,
            )

            self.post_feature_layers.append(post_feature_layer)

            current_shape = (
                feature_layer.out_channels,
                current_shape[1] + 1 - feature_layer.kernel_size[0],
                current_shape[2] + 1 - feature_layer.kernel_size[1],
            )

            flatten_dim += (
                current_shape[1] * current_shape[2] * post_feature_layer.out_channels
            )

        hidden_dim = 500

        self.hiddens = nn.ModuleList(
            [
                nn.Linear(flatten_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )

        self.value_output = nn.Linear(hidden_dim, 1)

    def forward(self, board):
        with torch.no_grad():
            features = self.encoder.extract_features(board, self.feature_condition)

        post_features = []
        for feature, post_feature_layer in zip(features, self.post_feature_layers):
            post_feature = post_feature_layer(feature)
            post_feature = torch.flatten(post_feature, start_dim=1)
            post_features.append(post_feature)

        post_features = torch.cat(post_features, dim=1)

        for h in self.hiddens:
            post_features = h(F.relu(post_features))

        return self.value_output(F.relu(post_features))

    def common_step(self, x, batch_idx):
        board, value = x
        predicted_value = self.forward(board)

        loss = F.l1_loss(predicted_value, value)

        return loss

    def training_step(self, x, batch_idx):
        loss = self.common_step(x, batch_idx)

        self.log_dict(
            {
                "train_loss": loss,
            }
        )

        return loss

    def validation_step(self, x, batch_idx):
        loss = self.common_step(x, batch_idx)

        self.log_dict(
            {
                "val_loss": loss,
            }
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    @staticmethod
    def feature_condition(layer: nn.Module):
        return isinstance(layer, nn.Conv2d)
