from types import SimpleNamespace

import numpy as np
import pytest

from chessml.data.assets import PIECE_CLASSES
from chessml.models.lightning import piece_classifier_model
from chessml.models.lightning.piece_classifier_model import constrained_argmax


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
