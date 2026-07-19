from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chessml.data.constants import PIECE_CLASSES
from chessml.data.images.picture import Picture
from chessml.models.lightning import piece_classifier_model
from chessml.models.lightning.piece_classifier_model import (
    PieceClassifier,
    PieceDecodingError,
    constrained_argmax,
)


def logits_for(piece_symbols):
    logits = np.full((len(piece_symbols), 12), -100.0)
    for row, symbol in enumerate(piece_symbols):
        logits[row, PIECE_CLASSES[symbol]] = 100.0
    return logits


def test_constrained_argmax_counts_only_positive_promotions():
    requested = (
        ["p"] * 7
        + ["r"]
        + ["n"] * 3
        + ["b"] * 2
        + ["q"] * 2
        + ["k", "K"]
    )

    logits = logits_for(requested)
    mistaken_queen = requested.index("q")
    logits[mistaken_queen, PIECE_CLASSES["r"]] = 99.0

    labels = constrained_argmax(logits)

    expected = requested.copy()
    expected[mistaken_queen] = "r"
    np.testing.assert_array_equal(
        labels,
        [PIECE_CLASSES[symbol] for symbol in expected],
    )


def test_constrained_argmax_preserves_a_legal_promotion():
    requested = (
        ["p"] * 7
        + ["r"] * 2
        + ["n"] * 2
        + ["b"] * 2
        + ["q"] * 2
        + ["k", "K"]
    )

    labels = constrained_argmax(logits_for(requested))

    np.testing.assert_array_equal(
        labels,
        [PIECE_CLASSES[symbol] for symbol in requested],
    )


def test_constrained_argmax_rounds_objective_coefficients():
    logits = np.full((2, 12), -10.0)
    for row, (black_king, white_king) in enumerate(
        ((1.49, 0.99), (1.99, 1.48))
    ):
        logits[row, PIECE_CLASSES["k"]] = black_king
        logits[row, PIECE_CLASSES["K"]] = white_king

    labels = constrained_argmax(logits, int_scale=1)

    np.testing.assert_array_equal(
        labels,
        [PIECE_CLASSES["K"], PIECE_CLASSES["k"]],
    )


def test_constrained_argmax_preserves_raw_labels_when_rounded_scores_tie():
    logits = np.full((2, 12), -10.0)
    logits[0, PIECE_CLASSES["k"]] = 0.0004
    logits[0, PIECE_CLASSES["K"]] = 0.0
    logits[1, PIECE_CLASSES["k"]] = 0.0
    logits[1, PIECE_CLASSES["K"]] = 0.0004

    labels = constrained_argmax(logits)

    np.testing.assert_array_equal(
        labels,
        [PIECE_CLASSES["k"], PIECE_CLASSES["K"]],
    )


def test_constrained_argmax_rejects_an_unproven_feasible_solution(monkeypatch):
    original_solver = piece_classifier_model.cp_model.CpSolver

    class FeasibleSolver:
        def __init__(self):
            self.solver = original_solver()
            self.parameters = self.solver.parameters

        def Solve(self, model):
            assert self.solver.Solve(model) == piece_classifier_model.cp_model.OPTIMAL
            return piece_classifier_model.cp_model.FEASIBLE

        def Value(self, variable):
            return self.solver.Value(variable)

    monkeypatch.setattr(piece_classifier_model.cp_model, "CpSolver", FeasibleSolver)

    with pytest.raises(TimeoutError, match="proving"):
        constrained_argmax(logits_for(["k", "K"]))


def test_constrained_argmax_reports_timeout_separately_from_infeasibility(
    monkeypatch,
):
    class UnknownSolver:
        def __init__(self):
            self.parameters = SimpleNamespace()

        def Solve(self, model):
            return piece_classifier_model.cp_model.UNKNOWN

    monkeypatch.setattr(piece_classifier_model.cp_model, "CpSolver", UnknownSolver)

    with pytest.raises(TimeoutError, match="timed out"):
        constrained_argmax(logits_for(["k", "K"]))


def test_constrained_argmax_reports_infeasibility():
    with pytest.raises(RuntimeError, match="infeasible"):
        constrained_argmax(np.zeros((1, 12)))


class ClassifierModelStub(torch.nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.output_features = output_features

    def preprocess_image(self, _image):
        return torch.zeros((3, 8, 8), dtype=torch.float32)

    def forward(self, images):
        return torch.zeros(
            (len(images), self.output_features),
            dtype=torch.float32,
        )


class FailingClassifierModelStub(ClassifierModelStub):
    def forward(self, images):
        raise RuntimeError("device execution failed")


class FailingTensorConversionClassifierModelStub(ClassifierModelStub):
    class Output:
        def cpu(self):
            raise RuntimeError("tensor conversion failed")

    def forward(self, images):
        return self.Output()


def test_constrained_argmax_reports_an_invalid_model(monkeypatch):
    class InvalidSolver:
        def __init__(self):
            self.parameters = SimpleNamespace()

        def Solve(self, model):
            return piece_classifier_model.cp_model.MODEL_INVALID

    monkeypatch.setattr(piece_classifier_model.cp_model, "CpSolver", InvalidSolver)

    with pytest.raises(RuntimeError, match="invalid"):
        constrained_argmax(logits_for(["k", "K"]))


@pytest.mark.parametrize(
    ("status", "cause_type"),
    [
        (piece_classifier_model.cp_model.FEASIBLE, TimeoutError),
        (piece_classifier_model.cp_model.UNKNOWN, TimeoutError),
        (piece_classifier_model.cp_model.INFEASIBLE, RuntimeError),
        (piece_classifier_model.cp_model.MODEL_INVALID, RuntimeError),
    ],
)
def test_classify_pieces_translates_known_solver_failures(
    monkeypatch,
    status,
    cause_type,
):
    class Solver:
        def __init__(self):
            self.parameters = SimpleNamespace()

        def Solve(self, model):
            return status

    monkeypatch.setattr(
        piece_classifier_model.cp_model,
        "CpSolver",
        Solver,
    )
    classifier = PieceClassifier(base_model_class=ClassifierModelStub)

    with pytest.raises(PieceDecodingError) as raised:
        classifier.classify_pieces(
            [Picture(np.zeros((8, 8, 3), dtype=np.uint8))]
        )

    assert isinstance(raised.value.__cause__, cause_type)


@pytest.mark.parametrize(
    "decoder_error",
    [
        RuntimeError("unexpected decoder failure"),
        TimeoutError("unexpected decoder timeout"),
    ],
)
def test_classify_pieces_does_not_translate_unexpected_decoder_errors(
    monkeypatch,
    decoder_error,
):
    classifier = PieceClassifier(base_model_class=ClassifierModelStub)

    def fail_decoding(_logits):
        raise decoder_error

    monkeypatch.setattr(
        piece_classifier_model,
        "constrained_argmax",
        fail_decoding,
    )

    with pytest.raises(type(decoder_error)) as raised:
        classifier.classify_pieces(
            [Picture(np.zeros((8, 8, 3), dtype=np.uint8))]
        )

    assert raised.value is decoder_error


def test_classify_pieces_does_not_translate_model_execution_errors():
    classifier = PieceClassifier(base_model_class=FailingClassifierModelStub)

    with pytest.raises(RuntimeError, match="device execution failed") as raised:
        classifier.classify_pieces(
            [Picture(np.zeros((8, 8, 3), dtype=np.uint8))]
        )

    assert type(raised.value) is RuntimeError


def test_classify_pieces_does_not_translate_tensor_conversion_errors():
    classifier = PieceClassifier(
        base_model_class=FailingTensorConversionClassifierModelStub
    )

    with pytest.raises(RuntimeError, match="tensor conversion failed") as raised:
        classifier.classify_pieces(
            [Picture(np.zeros((8, 8, 3), dtype=np.uint8))]
        )

    assert type(raised.value) is RuntimeError
