import timm
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import cv2

SIZES = {
    "efficientnet_b0": 224,
    "efficientnet_b6": 528,
    "resnet50": 224,
}


class VisionModelAdapter(nn.Module):
    def __init__(self, model_name: str, output_features: int):
        super().__init__()

        self.base_model = timm.create_model(model_name, pretrained=True)
        self.image_size = SIZES[model_name]

        if "efficientnet" in model_name:
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(in_features, output_features)
        elif "resnet" in model_name:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, output_features)
        else:
            raise ValueError(f"Unknown model name {model_name}")

    def forward(self, x):
        return self.base_model(x)

    def preprocess_image(self, img):
        cv2_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array (H x W x C) to a tensor of shape (C x H x W)
        tensor_image = torch.from_numpy(cv2_image_rgb).permute(2, 0, 1).float() / 255.0

        # Resize image using torchvision's functional API
        tensor_image = TF.resize(
            tensor_image, (self.image_size, self.image_size)
        )  # Example size, adjust as needed

        # Normalize the image tensor
        tensor_image = TF.normalize(
            tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return tensor_image
