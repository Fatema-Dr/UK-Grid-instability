#!/usr/bin/env python

import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """
    Installs all required dependencies using pip.
    """
    print("Dependencies should be managed by uv.")
    pass

def main():
    """
    Main function to run the entire data pipeline.
    """
    # Add project root to path to allow importing from src
    project_root = str(Path().resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root added to path: {project_root}")

    # Import all necessary modules after installing dependencies
    from src.config import (
        LGBM_MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH, DEMO_DATA_PATH,
        LGBM_QUANTILE_LOWER_PATH, LGBM_QUANTILE_UPPER_PATH, QUANTILE_ALPHAS,
        WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE
    )
    from src.data_loader import fetch_frequency_data, fetch_weather_data, fetch_inertia_data
    from src.feature_engineering import merge_datasets, create_features
    from src.model_trainer import train_and_evaluate_lgbm_classifier, train_quantile_model, train_lstm_model
    import joblib
    import os
    import polars as pl
    from datetime import datetime, timezone

    # 1. Data Loading
    print("\n--- Starting Data Loading ---")
    df_freq = fetch_frequency_data(WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE)
    df_weather = fetch_weather_data(WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE)
    df_inertia = fetch_inertia_data(WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE)

    # 2. Feature Engineering
    print("\n--- Starting Feature Engineering ---")
    df_merged = merge_datasets(df_freq, df_weather, df_inertia)
    # Convert to pandas BEFORE feature engineering, as required by the legacy code
    df_merged_pd = df_merged.to_pandas()
    df_processed = create_features(df_merged_pd)

    # 3. Model Training: LightGBM Classifier
    print("\n--- Starting LightGBM Classifier Training ---")
    df_processed_pl = pl.from_pandas(df_processed)
    lgbm_classifier, _, _ = train_and_evaluate_lgbm_classifier(df_processed_pl)

    # 4. Model Training: LightGBM Quantile Regressors
    print("\n--- Starting LightGBM Quantile Regressor Training ---")
    lower_model, _, _ = train_quantile_model(df_processed_pl, alpha=QUANTILE_ALPHAS[0])
    upper_model, _, _ = train_quantile_model(df_processed_pl, alpha=QUANTILE_ALPHAS[1])

    # 5. Model Training: LSTM
    print("\n--- Starting LSTM Training ---")
    lstm_model, scaler = train_lstm_model(df_processed_pl)

    # 6. Save Demo Data
    print("\n--- Saving Demo Data for Dashboard ---")
    start_date = datetime(2019, 8, 9).replace(tzinfo=timezone.utc)
    end_date = datetime(2019, 8, 10).replace(tzinfo=timezone.utc)
    df_demo = df_processed_pl.filter(
        pl.col("timestamp").is_between(start_date, end_date)
    )
    df_demo.write_csv(DEMO_DATA_PATH)
    print(f"Demo data saved to {DEMO_DATA_PATH}")

    # 7. Export Models and Data for Dashboard
    print("\n--- Exporting Models and Data ---")
    os.makedirs("notebooks", exist_ok=True)
    joblib.dump(lgbm_classifier, LGBM_MODEL_PATH)
    joblib.dump(lower_model, LGBM_QUANTILE_LOWER_PATH)
    joblib.dump(upper_model, LGBM_QUANTILE_UPPER_PATH)
    lstm_model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("All models saved.")

    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    install_dependencies()
    main()
