#!/usr/bin/env python

import sys
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()


def main():
    """
    Main function to run the entire data pipeline.
    """
    # Add project root to path to allow importing from src
    project_root = str(Path().resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    console.print(f"[dim]Project root added to path: {project_root}[/dim]")

    # Import all necessary modules after installing dependencies
    from src.config import (
        EXPORT_DIR,
        LGBM_MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH, DEMO_DATA_PATH,
        LGBM_QUANTILE_LOWER_PATH, LGBM_QUANTILE_UPPER_PATH, QUANTILE_ALPHAS,
        WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE,
        LGBM_FEATURE_COLS, TARGET_FREQ_NEXT,
        CALIBRATION_START_DATE, CALIBRATION_END_DATE,
        LOWER_CALIBRATOR_PATH, UPPER_CALIBRATOR_PATH
    )
    from src.data_loader import fetch_frequency_data, fetch_weather_data, fetch_inertia_data_halfhourly
    from src.feature_engineering import merge_datasets, create_features
    from src.model_trainer import train_and_evaluate_lgbm_classifier, train_quantile_model, train_all_quantile_models, train_lstm_model, train_lstm_quantile_comparator
    from src.calibration import fit_calibrator, save_calibrator
    import joblib
    import os
    import polars as pl
    from datetime import datetime, timezone

    # 1. Data Loading
    console.rule("[bold cyan]Starting Data Loading[/bold cyan]")
    df_freq = fetch_frequency_data(WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE)
    df_weather = fetch_weather_data(WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE)
    df_inertia = fetch_inertia_data_halfhourly(WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE)

    # 2. Feature Engineering
    console.rule("[bold cyan]Starting Feature Engineering[/bold cyan]")
    df_merged = merge_datasets(df_freq, df_weather, df_inertia)
    # Convert to pandas BEFORE feature engineering, as required by the legacy code
    df_merged_pd = df_merged.to_pandas()
    df_processed = create_features(df_merged_pd)

    # 3. Model Training: LightGBM Classifier
    console.rule("[bold cyan]Starting LightGBM Classifier Training[/bold cyan]")
    df_processed_pl = pl.from_pandas(df_processed)
    lgbm_classifier, _, _ = train_and_evaluate_lgbm_classifier(df_processed_pl)

    # 4. Model Training: LightGBM Quantile Regressors
    console.rule("[bold cyan]Starting LightGBM Quantile Regressor Training (All Alphas)[/bold cyan]")
    quantile_models, quantile_results, reliability_diagram = train_all_quantile_models(df_processed_pl)
    # Use 0.10 and 0.90 for lower/upper bands
    lower_model = quantile_models[0.10]
    upper_model = quantile_models[0.90]
    
    console.print("\n[bold green]Reliability Diagram:[/bold green]")
    for alpha, rel in reliability_diagram.items():
        console.print(f"Alpha {alpha:.2f}: Expected {rel['expected']:.2f}, Observed {rel['observed']:.4f}, Deviation {rel['deviation_pp']:.2f}%")

    # 5. Model Training: LSTM
    console.rule("[bold cyan]Starting LSTM Training[/bold cyan]")
    lstm_model, scaler = train_lstm_model(df_processed_pl)
    
    console.rule("[bold cyan]Starting LSTM Quantile Comparator Training[/bold cyan]")
    lstm_quant_model, lstm_scaler, X_test_lstm, y_test_lstm = train_lstm_quantile_comparator(df_processed_pl)
    # Save the LSTM quantile model
    lstm_quant_model.save(f"{EXPORT_DIR}/lstm_quantile_comparator.keras")

    # 5b. Post-Hoc Quantile Recalibration
    console.rule("[bold cyan]Fitting Quantile Calibrators (Isotonic Regression)[/bold cyan]")
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
    
    lower_calibrator = fit_calibrator(y_cal, lower_cal_preds, alpha=0.10)
    upper_calibrator = fit_calibrator(y_cal, upper_cal_preds, alpha=0.90)

    # 6. Save Demo Data
    console.rule("[bold cyan]Saving Demo Data for Dashboard[/bold cyan]")
    start_date = datetime(2019, 8, 9).replace(tzinfo=timezone.utc)
    end_date = datetime(2019, 8, 10).replace(tzinfo=timezone.utc)
    df_demo = df_processed_pl.filter(
        pl.col("timestamp").is_between(start_date, end_date)
    )
    df_demo.write_csv(DEMO_DATA_PATH)
    console.print(f"[dim]Demo data saved to {DEMO_DATA_PATH}[/dim]")

    # 7. Export Models and Data for Dashboard
    console.rule("[bold cyan]Exporting Models and Data[/bold cyan]")
    os.makedirs("notebooks", exist_ok=True)
    joblib.dump(lgbm_classifier, LGBM_MODEL_PATH)
    joblib.dump(lower_model, LGBM_QUANTILE_LOWER_PATH)
    joblib.dump(upper_model, LGBM_QUANTILE_UPPER_PATH)
    joblib.dump(quantile_models, f"{EXPORT_DIR}/lgbm_quantiles_all.pkl")
    lstm_model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    save_calibrator(lower_calibrator, LOWER_CALIBRATOR_PATH)
    save_calibrator(upper_calibrator, UPPER_CALIBRATOR_PATH)
    console.print("[bold green]All models and calibrators saved.[/bold green]")

    # 8. Winter Validation (out-of-season robustness test)
    console.rule("[bold cyan]Running Winter Validation[/bold cyan]")
    from src.config import WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE
    try:
        df_freq_w = fetch_frequency_data(WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE)
        df_weather_w = fetch_weather_data(WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE)
        df_inertia_w = fetch_inertia_data_halfhourly(WINTER_VALIDATION_START_DATE, WINTER_VALIDATION_END_DATE)

        if df_freq_w.is_empty() or df_weather_w.is_empty() or df_inertia_w.is_empty():
            console.print("[yellow]⚠️  Winter data unavailable — skipping validation.[/yellow]")
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
                console.print(f"  Winter PICP (80% CI): {picp_w:.4f} (target ≥ 0.8)")
                console.print(f"  Winter MPIW (Hz):     {mpiw_w:.6f}")
                console.print(f"  Winter samples:       {len(y_true_w):,}")
                if picp_w >= 0.80:
                    console.print("[bold green]  ✅ Winter validation passes.[/bold green]")
                else:
                    console.print(f"[bold yellow]  ⚠️  Winter PICP ({picp_w:.4f}) below 80% — model may not generalise well.[/bold yellow]")
            else:
                console.print("[yellow]⚠️  Not enough winter data for validation.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]⚠️  Winter validation failed: {e}[/bold red]")

    console.rule("[bold green]Pipeline finished successfully![/bold green]")

if __name__ == "__main__":
    main()
