import timm
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import cv2
import torch.nn.functional as F
import math
from chessml.models.torch.conv_layers import make_conv_layers
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.ops import FeaturePyramidNetwork
import logging

logger = logging.getLogger(__name__)


class Backboned(nn.Module):
    def __init__(
        self,
        backbone_model: str,
        pretrained: bool = True,
        features_only: bool = True,
        num_classes: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        logger.info(f"creating Backboned: {backbone_model}")

        self.backbone = timm.create_model(
            backbone_model,
            features_only=features_only,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=True, no_aug=True
        )

    def forward(self, x):
        return self.backbone(x)

    def preprocess_image(self, img):
        return self.transforms(img)

    @property
    def data_config(self):
        return timm.data.resolve_model_data_config(self.backbone)

    @property
    def features_info(self):
        s = self.data_config["input_size"]

        features = self.backbone(torch.randn(1, 3, s[1], s[2]))

        return [(f.shape[1], f.shape[2], f.shape[3]) for f in features]


class BackbonedFPN(Backboned):
    def __init__(
        self,
        output_features: int,
        fpn_channels: int = 256,
        head_layers: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        logger.info(
            f"creating BackbonedFPN: fpn_channels={fpn_channels}, head_layers={head_layers}, output_features={output_features}"
        )

        self.fpn = FeaturePyramidNetwork(
            self.backbone.feature_info.channels(), fpn_channels
        )

        layers = []
        for i in range(head_layers):
            last = i == head_layers - 1

            in_channels = int(fpn_channels * 2 ** (-i))
            out_channels = (
                output_features if last else int(fpn_channels * 2 ** (-i - 1))
            )
            logger.info(f"head conv #{i + 1}: {in_channels} -> {out_channels}")

            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )

            if not last:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        features = self.backbone(x)

        feature_dict = {str(i): feature for i, feature in enumerate(features)}

        fpn_features = self.fpn(feature_dict)

        top_fpn_feature = fpn_features[str(len(fpn_features) - 1)]
        predictions = self.head(top_fpn_feature)

        predictions = nn.functional.adaptive_avg_pool2d(predictions, (1, 1))
        predictions = predictions.view(x.size(0), -1)

        return predictions


class MobileViTV2FPN(BackbonedFPN):
    def __init__(self, **kwargs):
        super().__init__(backbone_model="mobilevitv2_200.cvnets_in1k", **kwargs)


class EfficientNetV2Classifier(Backboned):
    def __init__(self, output_features, **kwargs):
        super().__init__(
            backbone_model="efficientnetv2_rw_s.ra2_in1k",
            features_only=False,
            num_classes=output_features,
            **kwargs,
        )
