"""
Tests for the metric functions used in model evaluation and the Model Health tab.

Covers: pinball_loss, PICP, MPIW, and calibration_score.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Metric functions (duplicated from evaluate_models.py / model_trainer.py so
# tests are self-contained and verify the formulas independently)
# ---------------------------------------------------------------------------

def pinball_loss(y_true, y_pred, alpha):
    """Pinball (quantile) loss."""
    error = y_true - y_pred
    return np.mean(np.maximum(alpha * error, (alpha - 1) * error))


def calculate_picp_mpiw(y_true, lower_bound, upper_bound):
    """Prediction Interval Coverage Probability and Mean Prediction Interval Width."""
    covered = ((y_true >= lower_bound) & (y_true <= upper_bound)).astype(int)
    picp = np.mean(covered)
    mpiw = np.mean(upper_bound - lower_bound)
    return picp, mpiw


def calibration_score(y_true, y_pred_quantile):
    """Fraction of actual values falling below the predicted quantile."""
    return np.mean(y_true < y_pred_quantile)


# ---------------------------------------------------------------------------
# Tests — Pinball Loss
# ---------------------------------------------------------------------------

class TestPinballLoss:

    def test_perfect_prediction_is_zero(self):
        """When prediction equals actual, pinball loss should be zero."""
        y = np.array([50.0, 49.9, 50.1])
        assert pinball_loss(y, y, alpha=0.1) == pytest.approx(0.0)
        assert pinball_loss(y, y, alpha=0.9) == pytest.approx(0.0)

    def test_lower_quantile_penalises_over_prediction(self):
        """
        For α=0.1 (lower quantile), over-predicting (pred > actual) should
        produce a larger loss than under-predicting by the same amount.
        """
        y_true = np.array([50.0])
        over = pinball_loss(y_true, np.array([50.1]), alpha=0.1)  # pred > actual
        under = pinball_loss(y_true, np.array([49.9]), alpha=0.1)  # pred < actual
        assert over > under

    def test_upper_quantile_penalises_under_prediction(self):
        """
        For α=0.9 (upper quantile), under-predicting (pred < actual) should
        produce a larger loss than over-predicting by the same amount.
        """
        y_true = np.array([50.0])
        over = pinball_loss(y_true, np.array([50.1]), alpha=0.9)
        under = pinball_loss(y_true, np.array([49.9]), alpha=0.9)
        assert under > over

    def test_non_negative(self):
        """Pinball loss should always be non-negative."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(50, 0.1, 100)
        y_pred = rng.normal(50, 0.1, 100)
        assert pinball_loss(y_true, y_pred, 0.1) >= 0
        assert pinball_loss(y_true, y_pred, 0.9) >= 0


# ---------------------------------------------------------------------------
# Tests — PICP & MPIW
# ---------------------------------------------------------------------------

class TestPICPandMPIW:

    def test_all_covered(self):
        """If all actuals are within the bounds, PICP should be 1.0."""
        y = np.array([50.0, 49.9, 50.1])
        lower = np.array([49.5, 49.5, 49.5])
        upper = np.array([50.5, 50.5, 50.5])
        picp, _ = calculate_picp_mpiw(y, lower, upper)
        assert picp == pytest.approx(1.0)

    def test_none_covered(self):
        """If no actuals are within the bounds, PICP should be 0.0."""
        y = np.array([48.0, 51.0, 48.5])
        lower = np.array([49.5, 49.5, 49.5])
        upper = np.array([50.5, 50.5, 50.5])
        picp, _ = calculate_picp_mpiw(y, lower, upper)
        assert picp == pytest.approx(0.0)

    def test_mpiw_is_band_width(self):
        """MPIW should equal the mean of (upper - lower)."""
        lower = np.array([49.5, 49.6, 49.7])
        upper = np.array([50.5, 50.4, 50.3])
        _, mpiw = calculate_picp_mpiw(np.zeros(3), lower, upper)
        expected = np.mean(upper - lower)
        assert mpiw == pytest.approx(expected)

    def test_picp_partial(self):
        """Partial coverage should return the correct fraction."""
        y = np.array([50.0, 48.0])  # 1 covered, 1 not
        lower = np.array([49.5, 49.5])
        upper = np.array([50.5, 50.5])
        picp, _ = calculate_picp_mpiw(y, lower, upper)
        assert picp == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests — Calibration Score
# ---------------------------------------------------------------------------

class TestCalibrationScore:

    def test_perfect_calibration_lower(self):
        """
        If exactly 10% of actuals fall below the prediction,
        the calibration score should be 0.1.
        """
        # 10 samples, 1 below the prediction
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        pred = np.full(10, 1.5)  # only value 1 is below 1.5
        score = calibration_score(y_true, pred)
        assert score == pytest.approx(0.1)

    def test_all_below(self):
        """All actuals below prediction → score = 1.0."""
        y_true = np.array([1.0, 2.0, 3.0])
        pred = np.full(3, 10.0)
        assert calibration_score(y_true, pred) == pytest.approx(1.0)

    def test_none_below(self):
        """No actuals below prediction → score = 0.0."""
        y_true = np.array([10.0, 20.0, 30.0])
        pred = np.full(3, 1.0)
        assert calibration_score(y_true, pred) == pytest.approx(0.0)
