# ♟️ ChessML

ChessML is a Python package containing a collection of modules and scripts for advanced chess analysis.

https://github.com/arlegotin/chessml/assets/1470560/3f465bbc-a1ee-454e-a0ab-d64e64577457

This toolkit provides a variety of features, including board detection, piece recognition, and position encoding/decoding using Variational Autoencoders (VAEs), among others.

ChessML is built on top of [PyTorch](https://pytorch.org/) and [Lightning](https://lightning.ai/docs/pytorch/stable/). It also offers access to pretrained models and datasets.

## 📚 Table of contents
- [Getting started](#)
  - [Installation](#)
  - [Configuration](#)
- [Models](#)
  - [Pretrained models](#)
  - [Training & inference](#)
    - [BoardDetector](#)
    - [PieceClassifier](#)
- [Datasets & assets](#)
  - [Pregenerated datasets & assets](#)
  - [Generating datasets](#)
  - [Dynamic datasets](#)
    - [FileLinesDataset](#)
    - [BoardsFromFEN](#)
    - [BoardsImagesFromFENs](#)
    - [CompositeBoardsImages](#)
    - [CompositePiecesImages](#)
- [Contribution](#)
- [Acknowledgements](#)

## 🚀 Getting started

### Installation

ChessML uses [Poetry](https://python-poetry.org/) for managing dependencies. Ensure you have Python version 3.11 or higher.

To set up the project, run the following command:
```bash
poetry install
```

Activate the Poetry environment with:
```bash
poetry shell
```

As a sanity check run a test script, which will print out `./config` content:
```bash
python scripts/sanity_check.py
```

>Tip: All script entry points are located in the `./scripts` directory. Use `-h` for guidance on how to use these scripts

Now, you're all set to go!

### Configuration

Configuration is managed through `./config.yaml`, where you can define your hardware specifications, paths to datasets, logging settings, and more.

By default, the configuration is set for a computer equipped with a single GPU and running `Ubuntu 20.04.2 LTS`.

You don’t need to make any changes unless you are using a different OS or hardware setup, or you modify the project's file structure.

## ⚗️ Models

### Pretrained models

Download and unzip them in the `./checkpoints` directory to use:

| Class | Description | Size (unzipped) | Download |
| - | - | - | - |
| BoardDetector on base of MobileViT | Processes an image to predict the corners of the chessboard | 59.7MB | *Coming soon* |
| PieceClassifier on base of EfficientNetV2-B0 | Analyzes an image to identify which chess piece it depicts, including empty squares | 59.7MB | *Coming soon* |

### Training & inference

>Tip: All training scripts are optimized for the `Quadro RTX 8000`. You can modify hyperparameters via CLI arguments.

>Tip: Monitor metrics using TensorBoard by running the command `tensorboard --logdir=logs/tensorboard/lightning_logs`.

>Tip: If you're using `IterableDatasets`, please ignore the PyTorch warning suggesting to increase `num_workers`.

#### BoardDetector

`BoardDetector` is a `LightningModule` that predicts the coordinates of chessboard corners from any image. It utilizes a pretrained model, such as MobileViT, as its backbone and outputs 12 values: eight for the relative coordinates of four 2D points, and four flags indicating whether each point is visible in the image.

During training, it utilizes the `CompositeBoardsImages` dataset. To begin training, run the following script:
```bash
python scripts/train_board_detector.py 
```

To inference pretrained or newly-trained model:
```python
from chessml.models.torch.vision_model_adapter import MobileViTAdapter
from chessml.models.lightning.board_detector_model import BoardDetector
import cv2

model = BoardDetector.load_from_checkpoint(
    "./checkpoints/xxx.ckpt",
    base_model_class=MobileViTAdapter,
)

model.eval()

image = cv2.imread("./path/to/image.jpeg")

# For vanilla output:
coords, visibility = model.predict_coords_and_visibility(image)

# For an unskewed board image (returns None if no board is found):
board_image = model.extract_board_image(image)

# Marks the board on the original image if found:
image_with_marked_board = model.mark_board_on_image(image)
```

#### PieceClassifier

`PieceClassifier` is a `LightningModule` that predicts the chess piece from an image. It uses a pretrained model, such as MobileViT, as its backbone and outputs an index corresponding to the piece class in `PIECE_CLASSES`.

During training, it utilizes the `CompositePiecesImages` dataset. To begin training, run the following script:
```bash
python scripts/train_piece_classifier.py 
```

To inference pretrained or newly-trained model:
```python
from chessml.models.torch.vision_model_adapter import MobileViTAdapter
from chessml.models.lightning.piece_classifier_model import PieceClassifier
from chessml.const import INVERTED_PIECE_CLASSES
import cv2

model = PieceClassifier.load_from_checkpoint(
    "./checkpoints/xxx.ckpt",
    base_model_class=MobileViTAdapter,
)

model.eval()

image = cv2.imread("./path/to/image.jpeg")

class_index = model.classify_piece(image)

# Will be one of the following:
# P, N, B, Q, K, p, n, b, q, k, or None for an empty square
piece_name = INVERTED_PIECE_CLASSES[class_index]
```

## 📦 Datasets & assets

### Pregenerated datasets & assets

> Tip: you can train models using only the "Unique FENs" file. For more information, see the "Dynamic datasets" section below.

Download and unzip them into the `./datasets` or `./assets` directory for use:

| Name | Description | Format | Size (unzipped) | Download |
| - | - | - | - | - |
| Unique FENs (dataset) | A list of 22M+ unique valid FENs used to produce other datasets | Zipped TXT | 1.1GB | [Google Drive](https://drive.google.com/file/d/1uDeD9lupAi7daJm6K5YAao7WYvRdH0Ld/view?usp=drive_link) |
| Unaltered chessboard images (dataset) | Unaltered chessboard images showcasing various piece sets and board themes, complete with corresponding metadata for generating augmented images. | ZIP containing 512x512 JPEGs and TXTs | - | *Coming soon* |
| Augmented chessboard images (dataset) | Images of chessboards with various piece sets and themes, distorted and embedded into diverse backgrounds with assorted degradations. Includes corresponding metadata for training the BoardDetector. | ZIP containing 512x512 JPEGs and TXTs | - | *Coming soon* |
| Augmented pieces (dataset) | Augmented pieces images from various sets and square colors, distorted and with assorted degradations, accompanied by corresponding metadata used to train the PiecesClassifier. | ZIP containing 128x128 JPEGs and TXTs | - | *Coming soon* |
| Piece sets (asset) | Open-source piece sets used to generate chessboard and pieces datasets | ZIP containing SVGs | - | *Coming soon* |
| Backgrounds (asset) | Open-source images used to generate augmented chessboard images | ZIP containing JPEGs | - | *Coming soon* |

### Generating datasets

Begin by downloading PGN files, which will serve as the source for all other datasets:
```bash
python scripts/data/download_pgns.py
```

Next, use the downloaded PGNs to generate a file containing unique FENs:
```bash
python scripts/data/export_unique_fens.py
```

For now, you are good to go with using dynamic datasets (refer to the section below).

Scripts for generating additional datasets will be available soon.

### Dynamic datasets

The datasets used to train the `BoardDetector`, `PieceClassifier`, and other models are based on [IterableDatasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset). 

These generate data – either images or board representations – during runtime using only FENs. 

Although this method is slower than using pre-generated datasets, it allows for the creation of unlimited amounts of data with diverse augmentations from just the original FENs.

## 👷 Contribution

This repository is actively maintained and frequently updated, which can sometimes lead to compatibility issues.

If you encounter any problems or have feature requests, please don’t hesitate to open an issue.

Pull requests are warmly welcomed. To ensure consistency, please format your code using [Black](https://pypi.org/project/black/) before submitting.

## ✨ Acknowledgements

I would like to highlight certain projects that were extremely helpful during development:

- [python-chess](https://github.com/niklasf/python-chess) by [niklasf](https://github.com/niklasf)
- [Fen-To-Board-Image](https://github.com/ReedKrawiec/Fen-To-Board-Image) by [ReedKrawiec](https://github.com/ReedKrawiec)
- [PGN Mentor](https://www.pgnmentor.com/)
- [Lichess piece sets](https://github.com/lichess-org/lila/blob/master/COPYING.md)
