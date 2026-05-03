#!/usr/bin/env python

import sys
import subprocess
from pathlib import Path


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
        WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE,
        LGBM_FEATURE_COLS, TARGET_FREQ_NEXT,
        CALIBRATION_START_DATE, CALIBRATION_END_DATE,
        LOWER_CALIBRATOR_PATH, UPPER_CALIBRATOR_PATH
    )
    from src.data_loader import fetch_frequency_data, fetch_weather_data, fetch_inertia_data
    from src.feature_engineering import merge_datasets, create_features
    from src.model_trainer import train_and_evaluate_lgbm_classifier, train_quantile_model, train_lstm_model
    from src.calibration import fit_calibrator, save_calibrator
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

    # 5b. Post-Hoc Quantile Recalibration
    print("\n--- Fitting Quantile Calibrators (Isotonic Regression) ---")
    cal_start = datetime.strptime(CALIBRATION_START_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    cal_end = datetime.strptime(CALIBRATION_END_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    cal_data = df_processed_pl.filter(
        (pl.col("timestamp") >= pl.lit(cal_start)) &
        (pl.col("timestamp") < pl.lit(cal_end))
    ).to_pandas().dropna(subset=[TARGET_FREQ_NEXT])
    
    X_cal = cal_data[LGBM_FEATURE_COLS]
    y_cal = cal_data[TARGET_FREQ_NEXT].values
    
    lower_cal_preds = lower_model.predict(X_cal)
    upper_cal_preds = upper_model.predict(X_cal)
    
    lower_calibrator = fit_calibrator(y_cal, lower_cal_preds, alpha=QUANTILE_ALPHAS[0])
    upper_calibrator = fit_calibrator(y_cal, upper_cal_preds, alpha=QUANTILE_ALPHAS[1])

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
    save_calibrator(lower_calibrator, LOWER_CALIBRATOR_PATH)
    save_calibrator(upper_calibrator, UPPER_CALIBRATOR_PATH)
    print("All models and calibrators saved.")

    # 8. Winter Validation (out-of-season robustness test)
    print("\n--- Running Winter Validation ---")
    from src.config import WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE
    try:
        df_freq_w = fetch_frequency_data(WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE)
        df_weather_w = fetch_weather_data(WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE)
        df_inertia_w = fetch_inertia_data(WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE)

        if df_freq_w.is_empty() or df_weather_w.is_empty() or df_inertia_w.is_empty():
            print("⚠️  Winter data unavailable — skipping validation.")
        else:
            df_merged_w = merge_datasets(df_freq_w, df_weather_w, df_inertia_w)
            df_merged_w_pd = df_merged_w.to_pandas()
            df_winter = create_features(df_merged_w_pd)

            from src.config import TARGET_FREQ_NEXT as TFN_W
            df_winter_eval = df_winter.dropna(subset=[TFN_W])
            if len(df_winter_eval) > 100:
                import numpy as np_w
                y_true_w = df_winter_eval[TFN_W].values
                X_w = df_winter_eval[LGBM_FEATURE_COLS]
                lower_w = lower_model.predict(X_w)
                upper_w = upper_model.predict(X_w)
                covered_w = ((y_true_w >= lower_w) & (y_true_w <= upper_w)).astype(int)
                picp_w = np_w.mean(covered_w)
                mpiw_w = np_w.mean(upper_w - lower_w)
                print(f"  Winter PICP (80% CI): {picp_w:.4f} (target ≥ 0.8)")
                print(f"  Winter MPIW (Hz):     {mpiw_w:.6f}")
                print(f"  Winter samples:       {len(y_true_w):,}")
                if picp_w >= 0.80:
                    print("  ✅ Winter validation passes.")
                else:
                    print(f"  ⚠️  Winter PICP ({picp_w:.4f}) below 80% — model may not generalise well.")
            else:
                print("⚠️  Not enough winter data for validation.")
    except Exception as e:
        print(f"⚠️  Winter validation failed: {e}")

    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
