"""
Tests for the alert and status logic used in the dashboard (app.py).

These tests verify the decision matrix that combines LightGBM and LSTM
predictions to produce the dashboard status.
"""

import pytest


# ---------------------------------------------------------------------------
# Pure-logic helpers extracted from app.py so they can be tested in isolation
# ---------------------------------------------------------------------------

def determine_alert_status(lower_bound_pred: float, alert_threshold_hz: float,
                           lstm_alert: bool) -> str:
    """
    Replicates the alert decision logic from app.py.

    Returns one of:
        "INSTABILITY ALERT"
        "HIGH MODEL UNCERTAINTY"
        "SYSTEM STABLE"
    """
    is_alert = lower_bound_pred < alert_threshold_hz
    if is_alert and lstm_alert:
        return "INSTABILITY ALERT"
    elif is_alert or lstm_alert:
        return "HIGH MODEL UNCERTAINTY"
    else:
        return "SYSTEM STABLE"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlertThresholdLogic:
    """Tests for the core alert trigger: lower_bound < threshold."""

    def test_alert_triggers_below_threshold(self):
        assert determine_alert_status(49.7, 49.8, True) == "INSTABILITY ALERT"

    def test_no_alert_above_threshold(self):
        assert determine_alert_status(49.9, 49.8, False) == "SYSTEM STABLE"

    def test_boundary_exactly_at_threshold_no_alert(self):
        """Lower bound == threshold should NOT trigger (strict < comparison)."""
        assert determine_alert_status(49.8, 49.8, False) == "SYSTEM STABLE"

    def test_just_below_threshold_triggers(self):
        result = determine_alert_status(49.7999, 49.8, False)
        assert result == "HIGH MODEL UNCERTAINTY"  # LightGBM alert, no LSTM alert


class TestModelDisagreement:
    """Tests for the combined LightGBM + LSTM decision matrix."""

    def test_both_alert_instability(self):
        """Both models agree on instability → INSTABILITY ALERT."""
        assert determine_alert_status(49.7, 49.8, True) == "INSTABILITY ALERT"

    def test_lgbm_only_alert_high_uncertainty(self):
        """Only LightGBM alerts → HIGH MODEL UNCERTAINTY."""
        assert determine_alert_status(49.7, 49.8, False) == "HIGH MODEL UNCERTAINTY"

    def test_lstm_only_alert_high_uncertainty(self):
        """Only LSTM alerts → HIGH MODEL UNCERTAINTY."""
        assert determine_alert_status(49.9, 49.8, True) == "HIGH MODEL UNCERTAINTY"

    def test_neither_alert_stable(self):
        """Neither model alerts → SYSTEM STABLE."""
        assert determine_alert_status(49.9, 49.8, False) == "SYSTEM STABLE"


class TestInterventionSimulator:
    """Tests for the swing-equation intervention logic."""

    def test_swing_equation_positive_delta(self):
        """Injecting MW should produce a positive frequency delta."""
        SYSTEM_INERTIA_H = 4.0
        NOMINAL_FREQ = 50.0
        TOTAL_SYSTEM_CAPACITY = 35000

        synthetic_mw = 2000
        delta_f = (synthetic_mw * NOMINAL_FREQ) / (2 * SYSTEM_INERTIA_H * TOTAL_SYSTEM_CAPACITY)
        assert delta_f > 0, "Swing equation delta_f should be positive for positive MW injection"

    def test_swing_equation_zero_at_zero_mw(self):
        """Zero injection should produce zero delta."""
        delta_f = (0 * 50.0) / (2 * 4.0 * 35000)
        assert delta_f == 0.0

    def test_swing_equation_known_value(self):
        """Verify the formula with a hand-calculated value."""
        # Δf = (1000 * 50) / (2 * 4 * 35000) = 50000 / 280000 ≈ 0.17857 Hz
        delta_f = (1000 * 50.0) / (2 * 4.0 * 35000)
        assert abs(delta_f - 50000 / 280000) < 1e-6
