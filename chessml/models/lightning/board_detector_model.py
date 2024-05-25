import torch
import torch.nn.functional as F
from lightning import LightningModule
from typing import Type, Optional
import numpy as np
import cv2


class BoardDetector(LightningModule):
    def __init__(
        self,
        base_model_class: Type[torch.nn.Module],
        base_model_kwargs: dict = {},
    ):
        super().__init__()
        self.model = base_model_class(output_features=12, **base_model_kwargs)

    def forward(self, x):
        # return self.model(x)
        # return 3 * torch.sigmoid(self.model(x)) - 1
        # return torch.sigmoid(self.model(x))
        output = self.model(x)
        coords = output[:, :8]
        visibility = output[:, 8:]

        coords = 1.5 * torch.sigmoid(coords) - 0.25

        visibility = torch.sigmoid(visibility)

        return torch.cat((coords, visibility), dim=1)

    def calc_losses(self, batch):
        images, gt = batch
        pred = self(images)

        # Extract coordinates and visibility
        gt_coords = gt[:, :8]  # Shape: (batch_size, 8)
        pred_coords = pred[:, :8]  # Shape: (batch_size, 8)
        gt_visibility = gt[:, 8:]  # Shape: (batch_size, 4)
        pred_visibility = pred[:, 8:]  # Shape: (batch_size, 4)

        # Create a mask for visible points
        visibility_mask = gt_visibility.repeat(1, 2)  # Shape: (batch_size, 8)

        # Apply the mask to the coordinates
        masked_gt_coords = gt_coords * visibility_mask
        masked_pred_coords = pred_coords * visibility_mask

        # Calculate coordinate loss only for visible points
        coords_loss = F.huber_loss(masked_pred_coords, masked_gt_coords)

        # Calculate visibility loss
        visibility_loss = F.binary_cross_entropy(pred_visibility, gt_visibility)

        # Dynamic weights for losses
        coords_weight = 1.0 / (coords_loss.item() + 1e-6)
        visibility_weight = 1.0 / (visibility_loss.item() + 1e-6)

        total_loss = (coords_loss * coords_weight + visibility_loss * visibility_weight) / (coords_weight + visibility_weight)
        
        return {
            "coords_loss": coords_loss,
            "visibility_loss": visibility_loss,
            "coords_weight": coords_weight,
            "visibility_weight": visibility_weight,
            "total_loss": total_loss,
        } 

    def training_step(self, batch, batch_idx):
        losses = self.calc_losses(batch)
        self.log("train_coords_loss", losses["coords_loss"])
        self.log("train_visibility_loss", losses["visibility_loss"])
        self.log("train_loss", losses["total_loss"])
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.calc_losses(batch)
        self.log("val_coords_loss", losses["coords_loss"])
        self.log("val_visibility_loss", losses["visibility_loss"])
        self.log("val_loss", losses["total_loss"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        return optimizer
    
    def preprocess_input(self, img: np.ndarray, coords: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> tuple[np.ndarray, torch.Tensor]:
        (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = coords
        original_height, original_width = img.shape[:2]

        tl_x_rel = tl_x / original_width
        tl_y_rel = tl_y / original_height
        tl_visible = float(0 <= tl_x_rel <= 1 and 0 <= tl_y_rel <= 1)

        tr_x_rel = tr_x / original_width
        tr_y_rel = tr_y / original_height
        tr_visible = float(0 <= tr_x_rel <= 1 and 0 <= tr_y_rel <= 1)

        br_x_rel = br_x / original_width
        br_y_rel = br_y / original_height
        br_visible = float(0 <= br_x_rel <= 1 and 0 <= br_y_rel <= 1)

        bl_x_rel = bl_x / original_width
        bl_y_rel = bl_y / original_height
        bl_visible = float(0 <= bl_x_rel <= 1 and 0 <= bl_y_rel <= 1)

        relative_coords = torch.tensor(
            [
                # top-left:
                tl_x_rel,
                tl_y_rel,
                # top-right:
                tr_x_rel,
                tr_y_rel,
                # bottom-right:
                br_x_rel,
                br_y_rel,
                # bottom-left:
                bl_x_rel,
                bl_y_rel,
                # visibility:
                tl_visible,
                tr_visible,
                br_visible,
                bl_visible,
            ],
            dtype=torch.float32,
        )
        return self.model.preprocess_image(img), relative_coords.flatten()

    def predict_coords_and_visibility(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor_image = self.model.preprocess_image(img).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            output = self(tensor_image).squeeze().cpu().numpy()

        return output[:8], output[8:]

    def mark_board_on_image(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        coords, visibility = self.predict_coords_and_visibility(img)

        h, w = img.shape[:2]

        tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = coords
        tl_visible, tr_visible, br_visible, bl_visible = visibility

        points = [
            [tl_x * w, tl_y * h, tl_visible],
            [tr_x * w, tr_y * h, tr_visible],
            [br_x * w, br_y * h, br_visible],
            [bl_x * w, bl_y * h, bl_visible],
        ]

        color_green = (0, 255, 0)
        color_red = (0, 0, 255)

        for i in range(len(points)):
            x1, y1, visible1 = points[i]
            x2, y2, visible2 = points[(i + 1) % len(points)]
            
            # Determine the color of the line based on visibility
            if visible1 > 0.5 and visible2 > 0.5:
                color = color_green
            else:
                color = color_red
            
            # Draw the line between points
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        return img, np.all(visibility > 0.5)

    def extract_board_image(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        coords, visibility = self.predict_coords_and_visibility(img)

        h, w = img.shape[:2]
    
        tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = coords
        
        pts1 = np.float32([
            [tl_x * w, tl_y * h],
            [tr_x * w, tr_y * h],
            [br_x * w, br_y * h],
            [bl_x * w, bl_y * h]
        ])

        width_a = np.sqrt(((br_x - bl_x) ** 2 + (br_y - bl_y) ** 2)) * w
        width_b = np.sqrt(((tr_x - tl_x) ** 2 + (tr_y - tl_y) ** 2)) * w
        maxWidth = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr_x - br_x) ** 2 + (tr_y - br_y) ** 2)) * h
        height_b = np.sqrt(((tl_x - bl_x) ** 2 + (tl_y - bl_y) ** 2)) * h
        maxHeight = max(int(height_a), int(height_b))
        
        pts2 = np.float32([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        extracted_img = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
        
        return extracted_img, np.all(visibility > 0.5)
        
    
