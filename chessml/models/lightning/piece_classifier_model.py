import torch
import torch.nn.functional as F
from lightning import LightningModule
from typing import Type, Optional
import numpy as np
from chessml.const import PIECE_CLASSES_NUMBER

class PieceClassifier(LightningModule):
    def __init__(self, base_model_class: Type[torch.nn.Module], base_model_kwargs: dict = {}):
        super().__init__()
        # Initialize the base model with the assumption that it outputs logits directly for 13 classes.
        self.model = base_model_class(output_features=PIECE_CLASSES_NUMBER, **base_model_kwargs)

    def forward(self, x):
        # Directly return the output from the base model, which should be logits for 13 classes.
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        # Use cross-entropy loss for classification.
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        # Setup the optimizer. Adam is typically a good choice.
        return torch.optim.AdamW(self.model.parameters())
    
    def classify_piece(self, img: np.ndarray) -> int:
        tensor_image = self.model.preprocess_image(img).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            logits = self(tensor_image).squeeze()

        return torch.argmax(logits, dim=0).item()
