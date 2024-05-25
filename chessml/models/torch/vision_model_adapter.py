import timm
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import cv2
import torch.nn.functional as F

TRANSFORMS = {
    "efficientnet_b0": {
        "size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "efficientnet_b5": {
        "size": 448,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "efficientnetv2_rw_t.ra2_in1k": {
        "size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "mobilevit_s.cvnets_in1k": {
        "size": 256,
        "mean": [0.0, 0.0, 0.0],
        "std": [0.1, 0.1, 0.1],
    },
}

# class KolmogorovArnoldLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(KolmogorovArnoldLayer, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         # Define layers for single-variable transformations
#         self.transform_layers = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(input_dim)])
#         # Define output layer
#         self.output_layer = nn.Linear(input_dim * hidden_dim, output_dim)

#     def forward(self, x):
#         # Apply single-variable transformations
#         transformed = []
#         for i in range(self.input_dim):
#             transformed.append(F.relu(self.transform_layers[i](x[:, i].unsqueeze(1))))
        
#         # Concatenate all transformations
#         concatenated = torch.cat(transformed, dim=1)
        
#         # Apply output layer
#         output = self.output_layer(concatenated)
        
#         return output

# class HRNetAdapter(nn.Module):
#     def __init__(self, output_features: int):
#         super().__init__()

#         self.size = 224

#         self.backbone = timm.create_model("hrnet_w18_small_v2.gluon_in1k", pretrained=True)
#         in_features = self.backbone.classifier.in_features
#         # self.backbone.classifier = nn.Linear(in_features, output_features)
#         self.backbone.classifier = KolmogorovArnoldLayer(input_dim=in_features, output_dim=output_features, hidden_dim=128)

#     def forward(self, x):
#         return self.backbone(x)

#     def preprocess_image(self, img):
#         cv2_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Resize image using cv2
#         resized_image = cv2.resize(cv2_image_rgb, (self.size, self.size))
        
#         # Convert the NumPy array (H x W x C) to a tensor of shape (C x H x W)
#         tensor_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0

#         # Normalize the image tensor
#         tensor_image = TF.normalize(
#             tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
#         )

#         return tensor_image

class MobileViTAdapter(nn.Module):
    def __init__(self, output_features: int):
        super().__init__()

        self.size = 256
        
        self.pretrained_mobilevit = timm.create_model("mobilevit_s.cvnets_in1k", pretrained=True)

        in_features = self.pretrained_mobilevit.head.fc.in_features
        self.pretrained_mobilevit.head.fc = nn.Linear(in_features, output_features)

    def forward(self, x):
        return self.pretrained_mobilevit(x)

    def preprocess_image(self, img):
        cv2_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        resized_image = cv2.resize(cv2_image_rgb, (self.size, self.size))
        
        tensor_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0

        tensor_image = TF.normalize(
            tensor_image, mean=[0.0, 0.0, 0.0], std=[0.1, 0.1, 0.1],
        )

        return tensor_image

class EfficientNetV2Adapter(nn.Module):
    def __init__(self, output_features: int):
        super().__init__()

        self.size = 224
        
        self.backbone = timm.create_model("efficientnetv2_rw_t.ra2_in1k", pretrained=True)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, output_features)

    def forward(self, x):
        return self.backbone(x)

    def preprocess_image(self, img):
        cv2_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        resized_image = cv2.resize(cv2_image_rgb, (self.size, self.size))
        
        tensor_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0

        tensor_image = TF.normalize(
            tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )

        return tensor_image


class VisionModelAdapter(nn.Module):
    def __init__(self, model_name: str, output_features: int):
        super().__init__()

        self.base_model = timm.create_model(model_name, pretrained=True, features_only=True)

        # data_config = timm.data.resolve_model_data_config(self.base_model)
        # print(data_config)
        # print(self.base_model)
        o = self.base_model(torch.randn(1, 3, 224, 224))
        for x in o:
            print(x.shape)
        quit()

        self.transform_config = TRANSFORMS[model_name]

        if "efficientnet" in model_name:
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(in_features, output_features)
        elif "resnet" in model_name:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, output_features)
        elif "mobilevit" in model_name:
            in_features = self.base_model.head.fc.in_features
            self.base_model.head.fc = nn.Linear(in_features, output_features)
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
            tensor_image, (self.transform_config["size"], self.transform_config["size"])
        )  # Example size, adjust as needed

        # Normalize the image tensor
        tensor_image = TF.normalize(
            tensor_image, mean=self.transform_config["mean"], std=self.transform_config["std"]
        )

        return tensor_image
