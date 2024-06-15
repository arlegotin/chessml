import torch
import torch.nn.functional as F
from lightning import LightningModule
from typing import Type, Optional
import numpy as np
import cv2
from PIL import ImageDraw, Image
from chessml.data.images.picture import Picture
from chessml.data.assets import BOARD_SIZE


class BoardDetector(LightningModule):
    def __init__(
        self, base_model_class: Type[torch.nn.Module], base_model_kwargs: dict = {}
    ):
        super().__init__()
        self.model = base_model_class(output_features=8, **base_model_kwargs)

    def forward(self, x):
        return 2 * torch.sigmoid(self.model(x)) - 0.5

    def calc_losses(self, batch):
        images, gt_coords = batch
        pred_coords = self(images)

        # Reshape gt_coords to (B, 4, 2) for easier manipulation
        # reshaped_gt_coords = gt_coords.view(-1, 4, 2)

        # Create visibility_mask tensor
        # visibility_mask = (reshaped_gt_coords >= 0.0) & (reshaped_gt_coords <= 1.0)
        # visibility_mask = visibility_mask.all(dim=2).unsqueeze(2).repeat(1, 1, 2).view_as(gt_coords)

        # Apply the mask to the coordinates
        # masked_gt_coords = gt_coords * visibility_mask
        # masked_pred_coords = pred_coords * visibility_mask

        # Calculate coordinate loss only for visible points
        coords_loss = F.huber_loss(pred_coords, gt_coords)

        return {"coords_loss": coords_loss}

    def training_step(self, batch, batch_idx):
        losses = self.calc_losses(batch)
        self.log("train_loss", losses["coords_loss"])
        return losses["coords_loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.calc_losses(batch)
        self.log("val_loss", losses["coords_loss"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        return optimizer

    def predict_coords(self, img: Picture) -> np.ndarray:
        tensor_image = self.model.preprocess_image(img.pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self(tensor_image).squeeze().cpu().numpy()

    def mark_board_on_image(self, original_image: Picture) -> Picture:
        coords = self.predict_coords(original_image)

        image = original_image.pil
        w, h = image.size

        tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = coords

        # rectangle = [
        #     [tl_x * w, tl_y * h],
        #     [tr_x * w, tr_y * h],
        #     [br_x * w, br_y * h],
        #     [bl_x * w, bl_y * h],
        # ]

        draw = ImageDraw.Draw(image)

        # for i in range(len(rectangle)):
        #     x1, y1 = rectangle[i]
        #     x2, y2 = rectangle[(i + 1) % len(rectangle)]

        #     draw.line((int(x1), int(y1), int(x2), int(y2)), fill=(0, 255, 0), width=3)

        top = zip(
            np.linspace(tl_x, tr_x, BOARD_SIZE + 1),
            np.linspace(tl_y, tr_y, BOARD_SIZE + 1),
        )

        bottom = zip(
            np.linspace(bl_x, br_x, BOARD_SIZE + 1),
            np.linspace(bl_y, br_y, BOARD_SIZE + 1),
        )

        left = zip(
            np.linspace(tl_x, bl_x, BOARD_SIZE + 1),
            np.linspace(tl_y, bl_y, BOARD_SIZE + 1),
        )

        right = zip(
            np.linspace(tr_x, br_x, BOARD_SIZE + 1),
            np.linspace(tr_y, br_y, BOARD_SIZE + 1),
        )

        for first, second in [(top, bottom), (left, right)]:
            for p1, p2 in zip(first, second):
                x1, y1 = p1
                x2, y2 = p2

                draw.line(
                    (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                    fill=(0, 255, 0),
                    width=3,
                )

        return Picture(image)

    def extract_board_image(self, original_image: Picture) -> Picture:
        coords = self.predict_coords(original_image)

        w, h = original_image.pil.size

        tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = coords

        pts1 = np.float32(
            [
                [tl_x * w, tl_y * h],
                [tr_x * w, tr_y * h],
                [br_x * w, br_y * h],
                [bl_x * w, bl_y * h],
            ]
        )

        width_a = np.sqrt(((br_x - bl_x) ** 2 + (br_y - bl_y) ** 2)) * w
        width_b = np.sqrt(((tr_x - tl_x) ** 2 + (tr_y - tl_y) ** 2)) * w
        maxWidth = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr_x - br_x) ** 2 + (tr_y - br_y) ** 2)) * h
        height_b = np.sqrt(((tl_x - bl_x) ** 2 + (tl_y - bl_y) ** 2)) * h
        maxHeight = max(int(height_a), int(height_b))

        pts2 = np.float32(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ]
        )

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        extracted_image = cv2.warpPerspective(
            original_image.cv2, matrix, (maxWidth, maxHeight)
        )

        return Picture(extracted_image)
