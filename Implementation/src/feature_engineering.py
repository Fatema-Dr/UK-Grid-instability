# src/feature_engineering.py

import logging
import polars as pl
import pandas as pd
from .config import TARGET_COL, TTA_SECONDS, OPSDA_WIDTH, LAG_INTERVALS_SECONDS # Import LAG_INTERVALS_SECONDS
from . import opsda
import numpy as np

logger = logging.getLogger(__name__)

def merge_datasets(df_freq, df_weather, df_inertia):
    """
    Joins frequency, weather, and inertia data using a 'join_asof' strategy.
    """
    logger.info("Merging datasets...")
    df_freq = df_freq.sort("timestamp")
    df_weather = df_weather.sort("timestamp")
    df_inertia = df_inertia.sort("timestamp" if "timestamp" in df_inertia.columns else "timestamp_date")

    # Ensure timezone-aware columns can be joined
    df_weather = df_weather.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", time_zone="UTC"))
    )

    # First, merge frequency and weather data
    df_merged = df_freq.join_asof(
        df_weather,
        on="timestamp",
        strategy="backward"
    )
    
    # Next, merge the result with inertia data
    if "timestamp_date" in df_inertia.columns:
        df_inertia = df_inertia.rename({"timestamp_date": "timestamp"})
        
    df_merged = df_merged.join_asof(
        df_inertia,
        on="timestamp",
        strategy="backward"
    )

    df_merged = df_merged.drop_nulls()
    logger.info(f"Merged Dataset Shape: {df_merged.shape}")
    return df_merged

def calculate_wind_ramp_rate(df):
    """
    Calculates the wind ramp rate using the Swinging Door Algorithm.
    """
    logger.info("Calculating wind ramp rate...")
    # Timestamps need to be converted to Unix time (seconds) for slope calculation
    weather_data = df[["timestamp", "wind_speed"]].drop_duplicates(subset=["timestamp"]).copy()
    weather_data['unix_ts'] = weather_data['timestamp'].astype(np.int64) // 1_000_000_000
    
    # Apply the local Swinging Door algorithm implementation
    data_tuples = list(weather_data[['unix_ts', 'wind_speed']].itertuples(index=False, name=None))
    compressed = opsda.compress(data_tuples, width=OPSDA_WIDTH)
    
    compressed_df = pl.DataFrame(compressed, schema=["unix_ts", "wind_speed"], orient="row")
    compressed_df = compressed_df.with_columns(
        pl.col("unix_ts").cast(pl.Datetime).cast(pl.Datetime("us", "UTC"))
    ).rename({"unix_ts": "timestamp"})

    # Calculate ramp rate (slope between compressed points)
    compressed_df = compressed_df.with_columns(
        ((pl.col("wind_speed").diff()) / (pl.col("timestamp").diff().dt.total_seconds())).alias("wind_ramp_rate")
    ).to_pandas()
    
    # Convert compressed_df timestamp to match the main df's timestamp dtype for merging
    # Ensure it's timezone-aware microsecond precision, matching the input df.
    compressed_df["timestamp"] = pd.to_datetime(compressed_df["timestamp"]).astype('datetime64[us, UTC]')
    
    # Merge ramp rate back into the main dataframe
    df = pd.merge_asof(
        df,
        compressed_df[["timestamp", "wind_ramp_rate"]], # Select columns for merge
        on="timestamp",
        direction="backward"
    )
    return df

def create_features(df):
    """
    Engineers features for the grid stability model.
    """
    logger.info("Engineering features...")
    logger.debug(f"create_features input df shape: {df.shape}")
    
    # First, calculate the wind ramp rate
    df = calculate_wind_ramp_rate(df)
    logger.debug(f"df shape after calculate_wind_ramp_rate: {df.shape}")
    
    # Initialize df_features with the current df
    df_features = df.copy()

    # Fix 1: Causal (backward-only) RoCoF at multiple windows - no future leakage
    df_features["rocof_1s"]  = df_features["grid_frequency"].diff(1).fillna(0)
    df_features["rocof_5s"]  = ((df_features["grid_frequency"] - df_features["grid_frequency"].shift(5)) / 5.0).fillna(0)
    df_features["rocof_10s"] = ((df_features["grid_frequency"] - df_features["grid_frequency"].shift(10)) / 10.0).fillna(0)
    df_features["rocof_30s"] = ((df_features["grid_frequency"] - df_features["grid_frequency"].shift(30)) / 30.0).fillna(0)

    # RoCoF acceleration (second derivative) - detects worsening vs. recovering
    df_features["rocof_accel"] = (df_features["rocof_5s"].diff(5) / 5.0).fillna(0)

    # Smooth only rocof_1s for noise (EWM span=3 reacts faster than SMA)
    df_features["rocof_smooth"] = df_features["rocof_1s"].ewm(span=3, min_periods=1, adjust=False).mean().fillna(0)
    
    # Keep "rocof" column as "rocof_smooth" for backward compatibility with config LGBM_FEATURE_COLS
    df_features["rocof"] = df_features["rocof_smooth"]

    # Fix 7: Wind power approximation (cubic law) and diurnal demand
    df_features["wind_power_proxy"] = np.clip(df_features["wind_speed"]**3 * 3.0, 0, 3000)
    demand_profile = {0:28000, 6:30000, 9:34000, 12:35000, 16:38000, 19:37000, 22:32000}
    def get_demand(h):
        return demand_profile[min(demand_profile.keys(), key=lambda k: abs(k-h))]
    df_features["demand_proxy"] = df_features["timestamp"].dt.hour.map(get_demand)
    df_features["renewable_penetration_ratio"] = (
        df_features["wind_power_proxy"] / df_features["demand_proxy"].replace(0, 35000)
    ).clip(0, 1)

    # Fix 2: Inertia rate-of-change and low-inertia flag
    if "system_inertia_mws" in df_features.columns:
        df_features["inertia_value"] = df_features["system_inertia_mws"] / 1000.0  # Convert MWs to GVAs
    elif "inertia_cost" in df_features.columns:
        df_features["inertia_value"] = df_features["inertia_cost"] / 1000.0
    else:
        df_features["inertia_value"] = 100.0
        
    df_features["inertia_roc"] = df_features["inertia_value"].diff(1800).fillna(0)  # change per 30min
    df_features["low_inertia_flag"] = (df_features["inertia_value"] < 96).astype(np.int8)
    df_features["rocof_inertia_risk"] = df_features["rocof_smooth"].abs() * df_features["renewable_penetration_ratio"]

    df_features["volatility_10s"] = df_features["grid_frequency"].rolling(window=10).std().fillna(0)
    df_features["volatility_30s"] = df_features["grid_frequency"].rolling(window=30).std().fillna(0)
    df_features["volatility_60s"] = df_features["grid_frequency"].rolling(window=60).std().fillna(0)
    df_features["hour"] = df_features["timestamp"].dt.hour
    df_features["minute"] = df_features["timestamp"].dt.minute
    # Widen the instability label window (Danger Window)
    # Catch the approach to instability, not just the exact breach
    freq_next   = df_features["grid_frequency"].shift(-TTA_SECONDS)
    rocof_now   = df_features["rocof_smooth"]
    volatility  = df_features["volatility_10s"]

    df_features["target_freq_next"] = freq_next
    df_features["target_is_unstable"] = (
        (freq_next < 49.85)                          # freq heading low
        | (freq_next > 50.15)                        # freq heading high  
        | ((rocof_now < -0.02) & (freq_next < 49.95)) # fast negative RoCoF near boundary
        | (volatility > 0.03)                        # high turbulence
        | ((df_features["rocof_accel"] < -0.003) & (rocof_now < -0.008))                      # accelerating fall
        | ((rocof_now < -0.01) & (df_features["renewable_penetration_ratio"] > 0.06))  # stressed grid
    ).astype(np.int8)

    # Dynamically generate lag features
    for lag in LAG_INTERVALS_SECONDS:
        df_features[f"lag_{lag}s"] = df_features["grid_frequency"].shift(lag).fillna(50.0)
    
    # Forward fill the ramp rate to cover all seconds
    df_features["wind_ramp_rate"] = df_features["wind_ramp_rate"].replace([np.inf, -np.inf], np.nan).ffill()
    # Backward fill any remaining leading NaNs that ffill couldn't handle
    df_features["wind_ramp_rate"] = df_features["wind_ramp_rate"].bfill()
    # If the column is still all NaNs (e.g., if opsda.compress returned no valid ramp rates), fill with 0
    df_features["wind_ramp_rate"] = df_features["wind_ramp_rate"].fillna(0)
    
    logger.debug(f"df_features shape before dropna(): {df_features.shape}")
    logger.debug(f"df_features null counts before dropna():\n{df_features.isnull().sum()[df_features.isnull().sum() > 0]}")

    df_features = df_features.dropna()
    logger.debug(f"df_features shape after dropna(): {df_features.shape}")
    logger.info(f"Feature Engineering Complete. Shape: {df_features.shape}")
    return df_features
