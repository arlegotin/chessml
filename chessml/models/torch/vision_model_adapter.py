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

class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    p is learnable, eps guards against zero.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.pow(1.0 / self.p)


class ImprovedBackboned(Backboned):
    def __init__(
        self,
        num_classes: int,
        head_hidden_dim: int = 512,
        dropout: float = 0.5,
        features_only: bool = True,
        *args,
        **kwargs,
    ):
        # force features_only=True so backbone returns feature maps
        super().__init__(
            features_only=True,
            num_classes=0, 
            *args,
            **kwargs,
        )
        logger.info(f"creating ImprovedBackboned head: hid={head_hidden_dim} drop={dropout}")

        # last feature-map channels
        in_ch = self.backbone.feature_info.channels()[-1]

        # GeM pooling
        self.gem = GeM()

        # small “neck” to condition features
        self.neck = nn.Sequential(
            nn.Flatten(),                          # (B, C)
            nn.Linear(in_ch, head_hidden_dim, bias=False),
            nn.BatchNorm1d(head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # final classifier
        self.classifier = nn.Linear(head_hidden_dim, num_classes)

    def forward(self, x):
        # backbone → list of feature‐maps
        feats = self.backbone(x)
        # take highest‐level feature
        x = feats[-1]
        # pool → (B, C, 1, 1)
        x = self.gem(x)
        # neck → (B, head_hidden_dim)
        x = self.neck(x)
        # logits
        return self.classifier(x)

class MobileNetV3LargeClassifier(ImprovedBackboned):
    def __init__(self, output_features, **kwargs):
        super().__init__(
            backbone_model="mobilenetv3_large_100.ra_in1k",
            features_only=False,
            num_classes=output_features,
            **kwargs,
        )

class MobileViTSClassifier(ImprovedBackboned):
    def __init__(self, output_features, **kwargs):
        super().__init__(
            backbone_model="mobilevit_s.cvnets_in1k",
            features_only=False,
            num_classes=output_features,
            **kwargs,
        )

class EfficientNetB3Classifier(ImprovedBackboned):
    def __init__(self, output_features, **kwargs):
        super().__init__(
            backbone_model="efficientnet_b3.ra2_in1k",
            features_only=False,
            num_classes=output_features,
            **kwargs,
        )