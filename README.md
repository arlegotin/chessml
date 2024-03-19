# ♟️ ChessML

This package is dedicated to exploring chess using [Variational Autoencoders (VAEs)](https://en.wikipedia.org/wiki/Variational_autoencoder).

The project aims to develop models for exploring the game's latent space, predicting the next best moves, and evaluating positions.

Please note that this work is still in progress, and not all components are fully functional yet. Contributions are warmly welcomed.

Pretrained models will be released soon.

## 🗺️ Roadmap

- ✅ Scripts for downloading and preparing datasets
- ✅ Train a VAE to encode chess positions
- ⏳ Train a model for position evaluation (utilizing the VAE encoder)
- ⏳ Train a model for predicting the next best move (utilizing the VAE encoder)

## 📦 Installation

Ensure you have [Poetry](https://python-poetry.org/) installed, then install the project dependencies:
```bash
poetry install
```

If you encounter the error libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference, resolve it by ([source](https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no/75095447#75095447)):
```bash
pip uninstall nvidia_cublas_cu11
```

Finally, activate the Poetry shell:
```bash
poetry shell
````
and you're set to go!

## 💾 Preparing datasets
### Download datasets with games
```bash
kaggle datasets download -d milesh1/35-million-chess-games
```

### Download additional PGNs with player and tournament information:
```bash
python scripts/data/download_pgns.py
```

### Preprocess PGNs
```bash
python scripts/data/export_unique_fens.py
```

## 🤖 Training
### Train the basic board autoencoder
```bash
python scripts/train_board_autoencoder.py
```
### Train the model for predicting moves
Please note: This script is still under development!
```bash
python scripts/train_policy_model.py
```

### Train the model for predicting position evaluation
Please note: This script is still under development!
```bash
python scripts/train_value_model.py
```

## 📚 Related papers
- https://arxiv.org/pdf/2211.07700.pdf
- https://arxiv.org/pdf/1906.06034.pdf
- https://arxiv.org/pdf/2205.14539.pdf