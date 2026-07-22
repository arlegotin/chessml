import subprocess
import sys

from chessml import __version__


CORE_MODULES = (
    "chessml.data.boards.board_representation",
    "chessml.models.lightning.board_detector_model",
    "chessml.models.lightning.square_classifier_model",
    "chessml.models.lightning.piece_classifier_model",
    "chessml.models.utils.board_recognition_helper",
)

SCRIPT_MODULES = (
    "scripts.data.calc_pieces_weights",
    "scripts.train.train_board_detector",
    "scripts.train.train_square_classifier",
    "scripts.train.train_piece_classifier",
    "scripts.train.train_meta_predictor",
    "scripts.validate.validate_board_recognition",
)


def test_version():
    assert __version__ == "0.1.1"


def test_constants_only_imports_do_not_require_raw_assets(tmp_path):
    missing_assets = tmp_path / "missing-assets"
    child_code = f"""
import importlib
import sys

import chessml

chessml.config.assets.path = sys.argv[1]
chessml.Script.__call__ = lambda self, function: function

for module in {CORE_MODULES!r}:
    importlib.import_module(module)

for module in {SCRIPT_MODULES!r}:
    chessml.script = chessml.Script()
    importlib.import_module(module)
"""

    result = subprocess.run(
        [sys.executable, "-c", child_code, str(missing_assets)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_piece_set_inventory_ignores_non_directories(tmp_path):
    assets = tmp_path / "assets"
    (assets / "piece_png" / "valid-set").mkdir(parents=True)
    (assets / "piece_png" / ".DS_Store").write_text("")
    (assets / "bg" / "512").mkdir(parents=True)
    child_code = """
import sys
from pathlib import Path

import chessml

assets = Path(sys.argv[1])
chessml.config.assets.path = str(assets)

from chessml.data.assets import PIECE_SETS

assert PIECE_SETS == [assets / "piece_png" / "valid-set"], PIECE_SETS
"""

    result = subprocess.run(
        [sys.executable, "-c", child_code, str(assets)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
