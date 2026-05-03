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
    df_inertia = df_inertia.sort("timestamp_date")

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
    # The inertia data has a 'timestamp_date' column
    df_merged = df_merged.join_asof(
        df_inertia.rename({"timestamp_date": "timestamp"}),
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

    # Smooth the RoCoF with a 5-second rolling average
    df_features["rocof"] = df_features["grid_frequency"].diff().rolling(window=5, min_periods=1).mean().fillna(0)
    
    # Calculate synthetic Renewable Penetration Ratio as a proxy for physical grid inertia
    # Assuming average daily demand of 35000 MW, and scaling wind_speed by a hypothetical capacity
    # e.g., renewable_penetration_ratio = (wind_speed * 3000 MW wind capacity) / 35000 MW demand
    df_features["renewable_penetration_ratio"] = (df_features["wind_speed"] * 3000) / 35000

    df_features["volatility_10s"] = df_features["grid_frequency"].rolling(window=10).std().fillna(0)
    df_features["volatility_30s"] = df_features["grid_frequency"].rolling(window=30).std().fillna(0)
    df_features["volatility_60s"] = df_features["grid_frequency"].rolling(window=60).std().fillna(0)
    df_features["hour"] = df_features["timestamp"].dt.hour
    df_features["minute"] = df_features["timestamp"].dt.minute
    df_features["target_freq_next"] = df_features["grid_frequency"].shift(-TTA_SECONDS)
    df_features[TARGET_COL] = ((df_features["grid_frequency"].shift(-TTA_SECONDS) > 50.2) |
                               (df_features["grid_frequency"].shift(-TTA_SECONDS) < 49.8)).astype(np.int8)

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
