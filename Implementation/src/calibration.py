"""
Post-hoc quantile recalibration using isotonic regression.

When a quantile model exhibits systematic bias (e.g., the α=0.1 quantile 
captures only 1.8% of observations instead of 10%), isotonic regression can 
remap raw model predictions so that the observed coverage matches the 
nominal quantile level.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fit_calibrator(y_true: np.ndarray, y_pred_quantile: np.ndarray, alpha: float):
    """
    Fit an isotonic regression calibrator on a held-out calibration set.

    The calibrator learns a monotonic mapping from raw model predictions
    to values that are better calibrated w.r.t. the target quantile.

    Parameters
    ----------
    y_true : np.ndarray
        Actual target values on the calibration set.
    y_pred_quantile : np.ndarray
        Raw quantile predictions from the model on the calibration set.
    alpha : float
        The target quantile (e.g., 0.1 for the 10th percentile).

    Returns
    -------
    IsotonicRegression
        Fitted calibrator.
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred_quantile, y_true)

    # Report pre-calibration coverage
    observed = np.mean(y_true < y_pred_quantile)
    logging.info(
        f"Calibrator fit for α={alpha}: "
        f"pre-calibration observed fraction below prediction = {observed:.4f} "
        f"(target = {alpha:.2f})"
    )
    return calibrator


def calibrate_predictions(calibrator, y_pred_quantile: np.ndarray) -> np.ndarray:
    """
    Apply a fitted calibrator to transform raw model predictions.

    Parameters
    ----------
    calibrator : IsotonicRegression
        Fitted calibrator from `fit_calibrator()`.
    y_pred_quantile : np.ndarray
        Raw quantile predictions to recalibrate.

    Returns
    -------
    np.ndarray
        Recalibrated predictions.
    """
    return calibrator.predict(y_pred_quantile)


def save_calibrator(calibrator, path: str):
    """Persist a calibrator to disk."""
    joblib.dump(calibrator, path)
    logging.info(f"Calibrator saved to {path}")


def load_calibrator(path: str):
    """Load a calibrator from disk."""
    calibrator = joblib.load(path)
    logging.info(f"Calibrator loaded from {path}")
    return calibrator
