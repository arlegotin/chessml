import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Type, Optional
import numpy as np
from chessml.data.images.picture import Picture
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

class SquareClassifier(LightningModule):
    def __init__(
        self,
        base_model_class: Type[torch.nn.Module],
        base_model_kwargs: dict = {},
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_lr: Optional[float] = None,
        pct_start: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        # save all hparams for checkpointing / sweeping
        self.save_hyperparameters()

        self.model = base_model_class(
            output_features=1,
            **base_model_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def calc_losses(self, images, labels):
        logits = self(images)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            label_smoothing=self.hparams.label_smoothing
        )

        # combine
        loss = bce_loss

        # compute Matthews CC
        preds = torch.sigmoid(logits)
        
        # compute accuracy
        accuracy = (preds == labels).float().mean()

        return loss, bce_loss, accuracy

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss, bce, accuracy = self.calc_losses(images, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/bce",   bce,   prog_bar=False)
        self.log("train/accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        loss, bce, accuracy = self.calc_losses(images, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/bce",   bce,   prog_bar=False)
        self.log("val/accuracy", accuracy, prog_bar=True)
        
        # Get predictions for confusion matrix
        logits = self(images)
        preds = torch.sigmoid(logits)
        
        # Store predictions and labels for epoch end
        if not hasattr(self, 'val_preds'):
            self.val_preds = []
            self.val_labels = []
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())

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

    def classify_square(self, img: Picture) -> int:
        tensor_image = self.model.preprocess_image(img.pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self(tensor_image)
        return torch.sigmoid(logits.squeeze() > 0.5).long()

    def classify_squares(self, imgs: list[Picture]) -> list[int]:
        tensor_image = torch.cat(
            [self.model.preprocess_image(img.pil).unsqueeze(0) for img in imgs],
            dim=0
        ).to(self.device)
        with torch.no_grad():
            logits = self(tensor_image)
        return torch.sigmoid(logits.squeeze() > 0.5).tolist()