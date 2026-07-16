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
from chessml.data.assets import EMPTY_SQUARE_CHANCE
from torchmetrics.classification import (
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
    BinaryAveragePrecision
)

class SquareClassifier(LightningModule):
    def __init__(
        self,
        base_model_class: Type[torch.nn.Module],
        base_model_kwargs: dict = {},
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_lr: Optional[float] = None,
        pct_start: float = 0.3,
    ):
        super().__init__()
        # save all hparams for checkpointing / sweeping
        self.save_hyperparameters()

        self.model = base_model_class(
            output_features=1,
            **base_model_kwargs
        )

        self.val_mcc   = BinaryMatthewsCorrCoef()
        self.val_auroc = BinaryAUROC()                 # ROC AUC
        self.val_aupr  = BinaryAveragePrecision()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def calc_losses(self, images, labels):
        logits = self(images)

        smooth = 0.05
        soft_labels = labels * (1 - smooth) + 0.5 * smooth

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            soft_labels.unsqueeze(-1),
            # pos_weight=torch.tensor([EMPTY_SQUARE_CHANCE / (1.0 - EMPTY_SQUARE_CHANCE)], device=logits.device),
        )

        # combine
        loss = bce_loss

        return loss, bce_loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss, bce = self.calc_losses(images, labels)
        # self.log("train/loss", loss, prog_bar=True)
        self.log("train/bce",   bce,   prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        loss, bce = self.calc_losses(images, labels)
        # self.log("val/loss", loss, prog_bar=True)
        self.log("val/bce",   bce,   prog_bar=False)
        logits = self(images).squeeze(-1)
        probs  = torch.sigmoid(logits)

        preds = (probs > 0.5).int()
        # Convert labels to int for metrics that require integer targets
        int_labels = labels.int()
        
        self.val_mcc.update(preds,  int_labels)
        self.val_auroc.update(probs, labels)  # AUROC can handle float labels
        self.val_aupr.update(probs,  int_labels)  # Average Precision requires int labels

    def on_validation_epoch_end(self):
        self.log("val/mcc",   self.val_mcc.compute(),   prog_bar=True)
        self.log("val/rocAUC",self.val_auroc.compute(), prog_bar=True)
        self.log("val/prAUC", self.val_aupr.compute(),  prog_bar=False)
        # Reset for the next epoch
        self.val_mcc.reset(); self.val_auroc.reset(); self.val_aupr.reset()

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

    def classify_squares(self, imgs: list[Picture]) -> list[int]:
        tensor_image = torch.cat(
            [self.model.preprocess_image(img.pil).unsqueeze(0) for img in imgs],
            dim=0
        ).to(self.device)
        with torch.no_grad():
            logits = self(tensor_image)
        probs = torch.sigmoid(logits)
        # preds = (probs > (1 - EMPTY_SQUARE_CHANCE)).int().squeeze(-1)
        preds = (probs > 0.5).int().squeeze(-1)
        return preds.tolist()