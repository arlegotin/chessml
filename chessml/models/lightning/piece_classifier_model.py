import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Type, Optional
import numpy as np
from chessml.data.constants import PIECE_CLASSES,INVERTED_PIECE_CLASSES, PIECE_CLASSES_NUMBER, PIECE_WEIGHTS, PIECE_SYMBOLS
from chessml.data.images.picture import Picture
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from ortools.sat.python import cp_model


class _ConstrainedArgmaxTimeoutError(TimeoutError):
    pass


class _ConstrainedArgmaxRuntimeError(RuntimeError):
    pass


class PieceDecodingError(RuntimeError):
    pass


def constrained_argmax(logits: np.ndarray,
                       time_limit: int = 10,
                       int_scale: int = 1_000) -> np.ndarray:
    """
    Args
    ----
    logits     : numpy array shape (N, 12)
    time_limit : seconds OR‑Tools may spend (optional)
    int_scale  : CP‑SAT needs *integer* objective coefficients –­
                 multiply your floats and round.

    Returns
    -------
    best_labels: optimal feasible labels, shape (N,), with values 0‑11

    Raises
    ------
    TimeoutError: if optimality is not proven within ``time_limit``
    RuntimeError: if the constraint model is infeasible or invalid
    """
    N, C = logits.shape
    assert C == 12,  "Expecting exactly 12 columns"

    model = cp_model.CpModel()

    # Decision vars x[i,j] – 1 iff sample i is assigned to class j
    x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in range(N) for j in range(C)}

    # --- Row constraint: every sample gets exactly one label
    for i in range(N):
        model.Add(sum(x[i, j] for j in range(C)) == 1)

    # --- Chess constraints --------------------------------------------------
    # Up‑to‑eight pawns per side
    model.Add(sum(x[i, PIECE_CLASSES["p"]] for i in range(N)) <= 8)  # black pawns
    model.Add(sum(x[i, PIECE_CLASSES["P"]] for i in range(N)) <= 8)  # white pawns

    # Exactly one king per side
    model.Add(sum(x[i, PIECE_CLASSES["k"]] for i in range(N)) == 1)  # black king
    model.Add(sum(x[i, PIECE_CLASSES["K"]] for i in range(N)) == 1)  # white king

    # Constrain pieces based on pawn promotions
    for color, pieces_info in [("black", {"pawn": "p", "pieces": [("r", 2), ("n", 2), ("b", 2), ("q", 1)]}), 
                               ("white", {"pawn": "P", "pieces": [("R", 2), ("N", 2), ("B", 2), ("Q", 1)]})]:
        pawn_piece = pieces_info["pawn"]
        pieces_with_limits = pieces_info["pieces"]
        
        # A missing pawn can fund exactly one extra rook, knight, bishop, or queen.
        pawn_count = sum(x[i, PIECE_CLASSES[pawn_piece]] for i in range(N))
        promotion_counts = []
        for piece, starting_count in pieces_with_limits:
            piece_count = sum(x[i, PIECE_CLASSES[piece]] for i in range(N))
            promotion_count = model.NewIntVar(0, N, f"{color}_{piece}_promotions")
            model.AddMaxEquality(
                promotion_count,
                [piece_count - starting_count, 0],
            )
            promotion_counts.append(promotion_count)

        model.Add(sum(promotion_counts) + pawn_count <= 8)

    # ------------------------------------------------------------------------
    # Objective: maximise the sum of chosen logits
    int_logits = np.rint(logits * int_scale).astype(np.int64)
    raw_labels = logits.argmax(axis=1)
    # Preserve raw labels only as a tie-break; one rounded-logit point still wins.
    primary_multiplier = N + 1
    model.Maximize(
        sum(
            (
                int(int_logits[i, j]) * primary_multiplier
                + int(j == raw_labels[i])
            )
            * x[i, j]
            for i in range(N)
            for j in range(C)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE:
        raise _ConstrainedArgmaxTimeoutError(
            "CP-SAT stopped before proving the solution optimal"
        )
    if status == cp_model.UNKNOWN:
        raise _ConstrainedArgmaxTimeoutError(
            "CP-SAT timed out before finding a solution"
        )
    if status == cp_model.INFEASIBLE:
        raise _ConstrainedArgmaxRuntimeError(
            "CP-SAT constraint model is infeasible"
        )
    if status == cp_model.MODEL_INVALID:
        raise _ConstrainedArgmaxRuntimeError(
            "CP-SAT constraint model is invalid"
        )
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT returned unexpected status: {status}")

    return np.array([next(j for j in range(C) if solver.Value(x[i, j])) for i in range(N)])

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
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.0,
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
        
        # Get predictions for confusion matrix
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        
        # Store predictions and labels for epoch end
        if not hasattr(self, 'val_preds'):
            self.val_preds = []
            self.val_labels = []
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        # Create confusion matrix
        labels = np.arange(PIECE_CLASSES_NUMBER)
        cm = confusion_matrix(self.val_labels, self.val_preds, labels=labels)

        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()

        class_names = [PIECE_SYMBOLS[INVERTED_PIECE_CLASSES[i]] for i in labels]
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Log to tensorboard
        self.logger.experiment.add_figure('confusion_matrix', plt.gcf(), self.current_epoch)
        plt.close()
        
        # Clear stored predictions and labels
        self.val_preds = []
        self.val_labels = []

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

    def classify_pieces(self, imgs: list[Picture]) -> list[int]:
        tensor_image = torch.cat(
            [self.model.preprocess_image(img.pil).unsqueeze(0) for img in imgs],
            dim=0
        ).to(self.device)
        with torch.no_grad():
            logits = self(tensor_image)

        logits = logits.cpu().numpy()
        try:
            return constrained_argmax(logits)
        except (
            _ConstrainedArgmaxTimeoutError,
            _ConstrainedArgmaxRuntimeError,
        ) as error:
            raise PieceDecodingError(str(error)) from error
