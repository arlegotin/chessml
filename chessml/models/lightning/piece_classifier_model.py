import torch
import torch.nn.functional as F
from lightning import LightningModule
from typing import Type, Optional
import numpy as np
from chessml.data.assets import PIECE_CLASSES_NUMBER, PIECE_WEIGHTS
from chessml.data.images.picture import Picture
from sklearn.metrics import matthews_corrcoef


def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    # Convert class indices to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

    # Compute the log softmax
    logpt = F.log_softmax(inputs, dim=1)
    logpt = logpt * targets_one_hot
    logpt = logpt.sum(dim=1)

    # Compute pt and the focal loss component
    pt = torch.exp(logpt)
    F_loss = -((1 - pt) ** gamma) * logpt

    # Apply the alpha balancing factor
    if alpha is not None:
        alpha_t = alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)
        alpha_t = alpha_t.sum(dim=1)
        F_loss = alpha_t * F_loss

    return F_loss.mean()


class PieceClassifier(LightningModule):
    def __init__(
        self, base_model_class: Type[torch.nn.Module], base_model_kwargs: dict = {}
    ):
        super().__init__()
        self.model = base_model_class(
            output_features=PIECE_CLASSES_NUMBER, **base_model_kwargs
        )
        self.loss_weight = torch.tensor(PIECE_WEIGHTS, dtype=torch.float)

    def forward(self, x):
        return self.model(x)

    def calc_losses(self, batch, batch_idx, label_smoothing: float):
        images, labels = batch
        logits = self(images)
        # ce = F.cross_entropy(logits, labels, weight=self.loss_weight.to(self.device), label_smoothing=label_smoothing)
        ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        preds = torch.argmax(logits, dim=1)
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        mcc = matthews_corrcoef(labels_np, preds_np)

        return {
            # "focal": focal,
            "ce": ce,
            "mcc": mcc,
        }

    def training_step(self, batch, batch_idx):
        losses = self.calc_losses(batch, batch_idx, label_smoothing=0.0)
        self.log("train_loss", losses["ce"])
        return losses["ce"]

    def validation_step(self, batch, batch_idx):
        losses = self.calc_losses(batch, batch_idx, label_smoothing=0.0)
        self.log("val_loss", losses["ce"])
        self.log("val_mcc", losses["mcc"])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4 / 2)

    def classify_piece(self, img: Picture) -> int:
        tensor_image = self.model.preprocess_image(img.pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self(tensor_image)

        return torch.argmax(logits, dim=1).item()

    def classify_pieces(self, imgs: list[Picture]) -> list[int]:
        tensor_image = torch.cat(
            [self.model.preprocess_image(img.pil).unsqueeze(0) for img in imgs], dim=0
        ).to(self.device)

        with torch.no_grad():
            logits = self(tensor_image)

        return torch.argmax(logits, dim=1).tolist()
