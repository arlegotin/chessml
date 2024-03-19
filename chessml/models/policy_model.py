from lightning import LightningModule
import torch.nn as nn
from chessml.models.custom_sequential import CustomSequential
import math
import torch
import torch.nn.functional as F


class PolicyModel(LightningModule):
    pass


class VectorPolicyModel(PolicyModel):
    def __init__(
        self,
        input_shape: tuple,
        output_dim: int,
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

        # add dimension for value
        flatten_dim += 1

        hidden_output_dim = int(math.sqrt(flatten_dim * output_dim))

        self.hidden = nn.Linear(flatten_dim, hidden_output_dim)
        self.output = nn.Linear(hidden_output_dim, output_dim)

    def forward(self, x):
        board, value = x

        with torch.no_grad():
            features = self.encoder.extract_features(board, self.feature_condition)

        post_features = [value.unsqueeze(-1)]
        for feature, post_feature_layer in zip(features, self.post_feature_layers):
            post_feature = post_feature_layer(feature)
            post_feature = torch.flatten(post_feature, start_dim=1)
            post_features.append(post_feature)

        post_features = torch.cat(post_features, dim=1)

        hidden = self.hidden(F.relu(post_features))

        return self.output(F.relu(hidden))

    def training_step(self, x, batch_idx):
        board, moves, moves_mask, value = x
        predicted_moves = self.forward((board, value)) - 10000 * (1 - moves_mask)
        cross_entropy = F.cross_entropy(predicted_moves, moves)

        self.log_dict(
            {
                "train_loss": cross_entropy,
            }
        )

        return cross_entropy

    def validation_step(self, x, batch_idx):
        board, moves, _, value = x
        predicted_moves = self.forward((board, value))
        cross_entropy = F.cross_entropy(predicted_moves, moves)

        self.log_dict(
            {
                "val_loss": cross_entropy,
            }
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    @staticmethod
    def feature_condition(layer: nn.Module):
        return isinstance(layer, nn.Conv2d)
