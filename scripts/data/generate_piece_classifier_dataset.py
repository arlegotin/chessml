from chessml import script, config
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.models.torch.vision_model_adapter import EfficientNetV2Classifier
from pathlib import Path
import logging
import os
import torch
import csv
from tqdm import tqdm
from chessml.data.images.pieces_images import AugmentedPiecesImages, PiecesImages3x3
from chessml.data.assets import BOARD_COLORS, PIECE_SETS, PIECE_CLASSES

logger = logging.getLogger(__name__)

"""
next:
- no weight (pc-33-bs=64-step=15360 nice but empty squares are recognized as pieces sometimes)
- label smoothing 0.1
- RAdam
"""

script.add_argument("-l", dest="limit", type=int, default=2**18)


@script
def train(args):

    dataset_dir = Path(config.dataset.path_to_big) / "piece_classifier"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / "meta.csv"

    dataset = AugmentedPiecesImages(
        piece_images_3x3=PiecesImages3x3(
            piece_sets=PIECE_SETS,
            board_colors=BOARD_COLORS,
            square_size=64,
        ),
        limit=args.limit,
    )

    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path', 'piece_name'])  # Write header
        
        for idx, (picture, piece_name) in enumerate(tqdm(dataset, desc="Generating dataset", total=args.limit)):
            # Save image with sequential numbering
            image_path = images_dir / f"{idx}.png"
            picture.pil.save(image_path)
            
            # Write to CSV
            csv_writer.writerow([str(image_path), piece_name])
