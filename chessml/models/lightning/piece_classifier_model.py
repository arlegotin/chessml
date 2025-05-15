import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Type, Optional
import numpy as np
from chessml.data.assets import PIECE_CLASSES_NUMBER, PIECE_WEIGHTS
from chessml.data.images.picture import Picture
from sklearn.metrics import matthews_corrcoef
from torch.optim.lr_scheduler import OneCycleLR

class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute weighted CE per-sample
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none"
        )
        pt = torch.exp(-ce)               # p_t = exp(-CE)
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

class PieceClassifier(LightningModule):
    def __init__(
        self,
        base_model_class: Type[torch.nn.Module],
        base_model_kwargs: dict = {},
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_lr: Optional[float] = None,
        pct_start: float = 0.3,
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
    ):
        super().__init__()
        # save all hparams for checkpointing / sweeping
        self.save_hyperparameters()

        # backbone → 13 logits
        self.model = base_model_class(
            output_features=PIECE_CLASSES_NUMBER,
            **base_model_kwargs
        )

        # register class‐imbalance weights
        cw = torch.tensor(PIECE_WEIGHTS, dtype=torch.float)
        self.register_buffer("class_weight", cw)

        # focal‐loss module
        self.focal_criterion = WeightedFocalLoss(
            weight=cw,
            gamma=self.hparams.focal_gamma,
            reduction="mean"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def calc_losses(self, images, labels):
        logits = self(images)

        # weighted CE
        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weight,
            label_smoothing=self.hparams.label_smoothing
        )

        # weighted focal
        focal_loss = self.focal_criterion(logits, labels)

        # combine
        loss = ce_loss + self.hparams.focal_weight * focal_loss

        # compute Matthews CC
        preds = torch.argmax(logits, dim=1)
        mcc = matthews_corrcoef(labels.cpu().numpy(), preds.cpu().numpy())
        
        # compute accuracy
        accuracy = (preds == labels).float().mean()

        return loss, ce_loss, focal_loss, mcc, accuracy

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss, ce, focal, mcc, accuracy = self.calc_losses(images, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/CE",   ce,   prog_bar=False)
        self.log("train/Focal", focal, prog_bar=False)
        self.log("train/mcc",  mcc,  prog_bar=True)
        self.log("train/accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        loss, ce, focal, mcc, accuracy = self.calc_losses(images, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/CE",   ce,   prog_bar=False)
        self.log("val/Focal", focal, prog_bar=False)
        self.log("val/mcc",  mcc,  prog_bar=True)
        self.log("val/accuracy", accuracy, prog_bar=True)

    def configure_optimizers(self):
        lr       = self.hparams.lr
        wd       = self.hparams.weight_decay
        max_lr   = self.hparams.max_lr or lr * 10
        pct      = self.hparams.pct_start

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=wd
        )

        # one-cycle over total training steps
        steps = self.trainer.estimated_stepping_batches
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=steps,
            pct_start=pct,
            anneal_strategy="cos"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def classify_piece(self, img: Picture) -> int:
        tensor_image = self.model.preprocess_image(img.pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self(tensor_image)
        return logits.argmax(dim=1).item()

    def classify_pieces(self, imgs: list[Picture]) -> list[int]:
        tensor_image = torch.cat(
            [self.model.preprocess_image(img.pil).unsqueeze(0) for img in imgs],
            dim=0
        ).to(self.device)
        with torch.no_grad():
            logits = self(tensor_image)
        return logits.argmax(dim=1).tolist()