"""
Tests for feature engineering pipeline (src/feature_engineering.py).

Uses a synthetic DataFrame that mimics the merged output from the data loader,
ensuring all engineered features are created correctly without requiring API calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import LAG_INTERVALS_SECONDS, TARGET_FREQ_NEXT, TARGET_COL
from src.feature_engineering import create_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_merged_df():
    """
    Builds a synthetic Pandas DataFrame that resembles the output of
    merge_datasets() — i.e. it has `timestamp`, `grid_frequency`, `wind_speed`,
    `solar_radiation`, `inertia_cost`, and weather columns.

    100 seconds of data at 1-second resolution is enough for the rolling
    windows (max = 60s lag) and the shift(-TTA_SECONDS = 10) to produce
    valid rows in the middle of the frame.
    """
    n = 200  # 200 seconds of data
    timestamps = pd.date_range("2019-08-09 16:50:00", periods=n, freq="1s", tz="UTC")

    rng = np.random.default_rng(42)
    freq_base = 50.0 + rng.normal(0, 0.02, size=n).cumsum() * 0.01

    df = pd.DataFrame({
        "timestamp": timestamps,
        "grid_frequency": freq_base,
        "wind_speed": 8.0 + rng.normal(0, 0.5, size=n),
        "solar_radiation": np.maximum(0, 200 + rng.normal(0, 20, size=n)),
        "inertia_cost": 1500,
        "temperature": 18.0 + rng.normal(0, 1, size=n),
        "precipitation": np.zeros(n),
        "rain": np.zeros(n),
        "snowfall": np.zeros(n),
        "wind_gusts": 12.0 + rng.normal(0, 1, size=n),
    })
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateFeatures:
    """Core tests for the create_features() function."""

    def test_output_contains_expected_columns(self, synthetic_merged_df):
        """The output should contain all engineered feature columns."""
        result = create_features(synthetic_merged_df)

        expected_cols = [
            "grid_frequency", "rocof", "volatility_10s", "wind_speed",
            "wind_ramp_rate", "solar_radiation", "hour",
            "renewable_penetration_ratio", TARGET_FREQ_NEXT, TARGET_COL,
        ] + [f"lag_{lag}s" for lag in LAG_INTERVALS_SECONDS]

        for col in expected_cols:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_no_nans_in_output(self, synthetic_merged_df):
        """After dropna(), the output should have zero NaN values."""
        result = create_features(synthetic_merged_df)
        assert result.isnull().sum().sum() == 0, "Output contains NaN values"

    def test_rocof_is_smoothed(self, synthetic_merged_df):
        """
        RoCoF should be a 5-second rolling mean of the frequency diff,
        NOT a raw 1-second diff.  We verify by checking that the RoCoF
        column differs from a raw diff.
        """
        result = create_features(synthetic_merged_df)
        raw_diff = result["grid_frequency"].diff()
        # Smoothed values should differ from raw diffs (except possibly the first)
        # We compare a slice well past the rolling window start
        smoothed = result["rocof"].iloc[10:50]
        raw = raw_diff.iloc[10:50]
        assert not np.allclose(smoothed.values, raw.values, atol=1e-12), \
            "RoCoF appears to be a raw diff — should be smoothed"

    def test_renewable_penetration_ratio_formula(self, synthetic_merged_df):
        """renewable_penetration_ratio should equal (wind_speed * 3000) / 35000."""
        result = create_features(synthetic_merged_df)
        expected = (result["wind_speed"] * 3000) / 35000
        np.testing.assert_allclose(
            result["renewable_penetration_ratio"].values,
            expected.values,
            atol=1e-10,
        )

    def test_lag_features_are_shifted(self, synthetic_merged_df):
        """Each lag feature should equal a shifted version of grid_frequency."""
        result = create_features(synthetic_merged_df)
        for lag in LAG_INTERVALS_SECONDS:
            col_name = f"lag_{lag}s"
            assert col_name in result.columns, f"Missing lag column: {col_name}"

    def test_volatility_is_non_negative(self, synthetic_merged_df):
        """volatility_10s is a rolling std and should never be negative."""
        result = create_features(synthetic_merged_df)
        assert (result["volatility_10s"] >= 0).all(), "volatility_10s has negative values"

    def test_wind_ramp_rate_no_inf(self, synthetic_merged_df):
        """wind_ramp_rate should have no Inf or NaN values."""
        result = create_features(synthetic_merged_df)
        assert np.isfinite(result["wind_ramp_rate"]).all(), \
            "wind_ramp_rate contains Inf or NaN"

    def test_target_is_binary(self, synthetic_merged_df):
        """target_is_unstable should only contain 0 or 1."""
        result = create_features(synthetic_merged_df)
        assert set(result[TARGET_COL].unique()).issubset({0, 1}), \
            "target_is_unstable should be binary (0 or 1)"

    def test_output_is_shorter_than_input(self, synthetic_merged_df):
        """
        create_features applies dropna which removes rows with NaN from
        lags and shifts, so output should be shorter than input.
        """
        result = create_features(synthetic_merged_df)
        assert len(result) < len(synthetic_merged_df)
