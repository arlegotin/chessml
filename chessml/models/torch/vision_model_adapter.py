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
        **kwargs,
    ):
        super().__init__(**kwargs)

        logger.info(f"creating Backboned: {backbone_model}")

        self.backbone = timm.create_model(backbone_model, features_only=True, pretrained=True)

        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = timm.data.create_transform(**data_config, is_training=True, no_aug=True)
        logger.info(f"backbone transforms: {self.transforms}")

    def preprocess_image(self, img):
        return self.transforms(img)


class BackbonedFPN(Backboned):
    def __init__(
        self,
        output_features: int,
        fpn_channels: int = 256,
        head_layers: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        logger.info(f"creating BackbonedFPN: fpn_channels={fpn_channels}, head_layers={head_layers}, output_features={output_features}")

        self.fpn = FeaturePyramidNetwork(
            self.backbone.feature_info.channels(),
            fpn_channels,
        )

        layers = []
        for i in range(head_layers):
            last = i == head_layers - 1

            in_channels = int(fpn_channels * 2**(-i))
            out_channels = output_features if last else int(fpn_channels * 2**(-i-1))
            logger.info(f"head conv #{i + 1}: {in_channels} -> {out_channels}")

            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
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

class BackbonedClassifier(Backboned):
    def __init__(
        self,
        output_features: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        l1 = self.backbone.feature_info.channels()[-1]
        l2 = int(math.sqrt(l1 * output_features))
        l3 = output_features

        logger.info(f"creating BackbonedClassifier: {l1} -> {l2} -> {l3}")

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
        )

    def forward(self, x):
        features = self.backbone(x)

        return self.head(features[-1])

class MobileViTV2FPN(BackbonedFPN):
    def __init__(self, **kwargs):
        super().__init__(
            backbone_model="mobilevitv2_200.cvnets_in1k",
            **kwargs,
        )

class MobileNetV3S50Classifier(BackbonedClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            backbone_model="mobilenetv3_small_050.lamb_in1k",
            **kwargs,
        )

"""
Candidates:
- resnest14d.gluon_in1k 
- efficientnetv2_rw_t.ra2_in1k 
- mobilenetv3_small_050.lamb_in1k
- mobilenetv3_small_075.lamb_in1k
- mobilenetv3_small_100.lamb_in1k
- mobilenetv3_rw.rmsp_in1k
- regnety_032.ra_in1k
"""
