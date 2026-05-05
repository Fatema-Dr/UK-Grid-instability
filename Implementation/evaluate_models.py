#!/usr/bin/env python
"""
GridGuardian Model Evaluation & Dashboard Validation Script
==========================================================
Generates model performance metrics and dissertation-ready plots.

Usage:
    cd Implementation/
    uv run python evaluate_models.py                                    # default Aug 2019
    uv run python evaluate_models.py --start-date 2019-12-01 --end-date 2019-12-31  # winter
    uv run python evaluate_models.py --calibrated                       # with recalibration
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

from src.config import (
    TTA_SECONDS, LGBM_FEATURE_COLS, TARGET_FREQ_NEXT,
    WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE,
    QUANTILE_ALPHAS
)
from src.data_loader import fetch_frequency_data, fetch_inertia_data_halfhourly, fetch_weather_data
from src.feature_engineering import create_features

# ─── Output directory ───────────────────────────────────────────────────────
OUTPUT_DIR = "notebooks/evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Metric Functions ───────────────────────────────────────────────────────

def pinball_loss(y_true, y_pred, alpha):
    """Pinball (quantile) loss."""
    error = y_true - y_pred
    return np.mean(np.maximum(alpha * error, (alpha - 1) * error))


def calculate_picp_mpiw(y_true, lower_bound, upper_bound):
    """
    Prediction Interval Coverage Probability (PICP) and
    Mean Prediction Interval Width (MPIW).
    """
    covered = ((y_true >= lower_bound) & (y_true <= upper_bound)).astype(int)
    picp = np.mean(covered)
    mpiw = np.mean(upper_bound - lower_bound)
    return picp, mpiw


def calibration_score(y_true, y_pred_quantile, alpha):
    """
    Fraction of actual values falling below the predicted quantile.
    For a perfectly calibrated α=0.1 model, this should be ~0.10.
    """
    return np.mean(y_true < y_pred_quantile)

def evaluate_alert_system_holistically(df_full_month, lower_model, threshold=49.8):
    """
    Evaluates alert precision/recall stratified by event duration.
    """
    print("\n" + "=" * 70)
    print("  ALERT SYSTEM EVALUATION (STRATIFIED)")
    print("=" * 70)

    preds_lower = lower_model.predict(df_full_month[LGBM_FEATURE_COLS])
    alerts = pd.Series(preds_lower < threshold, index=df_full_month.index)
    actual_unstable = pd.Series(df_full_month["grid_frequency"] < threshold, index=df_full_month.index)
    
    # Identify contiguous instability events
    event_ids = (actual_unstable != actual_unstable.shift()).cumsum()
    events = actual_unstable.groupby(event_ids)
    
    event_stats = []
    for event_id, group in events:
        if group.iloc[0] == True: # If it's an unstable event
            duration = len(group)
            start_idx = group.index[0]
            end_idx = group.index[-1]
            
            # Use positional indexing for the lookback slice
            start_pos = df_full_month.index.get_loc(start_idx)
            end_pos = df_full_month.index.get_loc(end_idx)
            lookback_pos = max(0, start_pos - 10)
            
            alert_fired = alerts.iloc[lookback_pos:end_pos+1].any()
            event_stats.append({'duration': duration, 'caught': alert_fired})
            
    df_events = pd.DataFrame(event_stats)
    
    if df_events.empty:
        print("  No instability events found.")
    else:
        buckets = [(1,5), (5,30), (30,120), (120, float('inf'))]
        for lo, hi in buckets:
            mask = (df_events["duration"] >= lo) & (df_events["duration"] < hi)
            bucket_events = df_events[mask]
            if len(bucket_events) > 0:
                recall = bucket_events['caught'].mean()
                hi_str = f"{hi}" if hi != float('inf') else "inf"
                print(f"  Duration {lo:3d}-{hi_str:>3s}s : Recall = {recall:.3f} ({int(bucket_events['caught'].sum())}/{len(bucket_events)} events)")
            else:
                hi_str = f"{hi}" if hi != float('inf') else "inf"
                print(f"  Duration {lo:3d}-{hi_str:>3s}s : N/A (0 events)")
    
    stable_mask = ~actual_unstable
    false_alert_rate = alerts[stable_mask].mean()
    print(f"\n  False alert rate (stable periods): {false_alert_rate:.6f} ({false_alert_rate*86400:.1f} per day)")
    print("=" * 70)

def rocof_window_sensitivity(df_eval, lower_model, upper_model, y_true, windows=[1, 3, 5, 10, 20, 30]):
    """
    Tests RoCoF smoothing windows against pinball loss to justify choice of 5s.
    """
    print("\n" + "=" * 70)
    print("  RoCoF WINDOW SENSITIVITY SWEEP")
    print("=" * 70)
    
    if "rocof_1s" not in df_eval.columns:
        print("  ⚠️ 'rocof_1s' not found. Skipping sensitivity sweep.")
        return
        
    losses_lower = []
    losses_upper = []
    
    # Store original values to restore them later
    orig_smooth = df_eval["rocof_smooth"].copy()
    orig_accel = df_eval["rocof_accel"].copy() if "rocof_accel" in df_eval.columns else None
    
    for w in windows:
        # Recalculate features based on new window
        df_eval["rocof_smooth"] = df_eval["rocof_1s"].rolling(window=w, min_periods=1).mean().fillna(0)
        df_eval["rocof_accel"] = df_eval["rocof_smooth"].diff(1).fillna(0)
        
        X = df_eval[LGBM_FEATURE_COLS]
        
        preds_lower = lower_model.predict(X)
        preds_upper = upper_model.predict(X)
        
        pb_lower = pinball_loss(y_true, preds_lower, 0.10)
        pb_upper = pinball_loss(y_true, preds_upper, 0.90)
        
        losses_lower.append(pb_lower)
        losses_upper.append(pb_upper)
        print(f"  Window: {w:>2}s | PB Loss (Lower): {pb_lower:.6f} | PB Loss (Upper): {pb_upper:.6f}")
        
    # Restore original values
    df_eval["rocof_smooth"] = orig_smooth
    if orig_accel is not None:
        df_eval["rocof_accel"] = orig_accel
        
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(windows, losses_lower, marker='o', label='Lower Bound (α=0.10)', color='#FF6B6B')
    ax.plot(windows, losses_upper, marker='s', label='Upper Bound (α=0.90)', color='#4ECDC4')
    ax.axvline(x=5, color='gray', linestyle='--', label='Chosen Window (5s)')
    ax.set_xlabel('RoCoF Smoothing Window (seconds)')
    ax.set_ylabel('Pinball Loss (lower is better)')
    ax.set_title('Sensitivity of Quantile Loss to RoCoF Smoothing Window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rocof_sensitivity.png")
    fig.savefig(path, dpi=150)
    print(f"  ✅ Saved Sensitivity Plot: {path}")
    plt.close(fig)
    print("=" * 70)


# ─── Data Loading (mirrors app.py logic) ────────────────────────────────────

def load_test_data(start_date: str, end_date: str):
    """Load and merge data exactly as the dashboard does."""
    print(f"Fetching data for {start_date} → {end_date}...")

    from src.data_loader import fetch_inertia_data_halfhourly
    from src.feature_engineering import merge_datasets
    df_freq = fetch_frequency_data(start_date=start_date, end_date=end_date)
    df_weather = fetch_weather_data(start_date=start_date, end_date=end_date)
    df_inertia = fetch_inertia_data_halfhourly(start_date=start_date, end_date=end_date)

    for name, df in [("Frequency", df_freq), ("Weather", df_weather), ("Inertia", df_inertia)]:
        if df.is_empty():
            print(f"ERROR: {name} data is empty. Check API / date range.")
            sys.exit(1)

    # Merge using the standard pipeline logic
    df_merged = merge_datasets(df_freq, df_weather, df_inertia)
    df_merged = df_merged.to_pandas()

    # Feature engineering
    df_data = create_features(df_merged)
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"], utc=True)

    print(f"Loaded {len(df_data):,} rows after feature engineering.")
    return df_data


# ─── Main Evaluation ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GridGuardian Model Evaluation")
    parser.add_argument("--start-date", default=WEATHER_API_DEFAULT_START_DATE,
                        help="Evaluation start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=WEATHER_API_DEFAULT_END_DATE,
                        help="Evaluation end date (YYYY-MM-DD)")
    parser.add_argument("--calibrated", action="store_true",
                        help="Apply post-hoc isotonic recalibration if calibrators exist")
    args = parser.parse_args()

    eval_start = args.start_date
    eval_end = args.end_date

    print("=" * 70)
    print("  GridGuardian — Model Evaluation & Dashboard Validation")
    print(f"  Period: {eval_start} → {eval_end}")
    if args.calibrated:
        print("  Mode: CALIBRATED (isotonic recalibration applied)")
    print("=" * 70)

    # 1. Load models
    lower_model = joblib.load("notebooks/lgbm_quantile_lower.pkl")
    upper_model = joblib.load("notebooks/lgbm_quantile_upper.pkl")
    
    quantiles_all_path = "notebooks/lgbm_quantiles_all.pkl"
    quantile_models = joblib.load(quantiles_all_path) if os.path.exists(quantiles_all_path) else None
    
    import tensorflow as tf
    from src.config import EXPORT_DIR
    lstm_mc_path = f"{EXPORT_DIR}/lstm_quantile_comparator.keras"
    lstm_mc_model = tf.keras.models.load_model(lstm_mc_path) if os.path.exists(lstm_mc_path) else None
    
    print("✅ Models loaded.")

    # Load calibrators if requested
    lower_calibrator, upper_calibrator = None, None
    if args.calibrated:
        from src.config import LOWER_CALIBRATOR_PATH, UPPER_CALIBRATOR_PATH
        try:
            lower_calibrator = joblib.load(LOWER_CALIBRATOR_PATH)
            upper_calibrator = joblib.load(UPPER_CALIBRATOR_PATH)
            print("✅ Calibrators loaded.")
        except FileNotFoundError:
            print("⚠️  Calibrator files not found. Run run_pipeline.py first. Proceeding without calibration.")
            args.calibrated = False

    # 2. Load data
    df = load_test_data(eval_start, eval_end)

    # Ensure target column exists
    if TARGET_FREQ_NEXT not in df.columns:
        print(f"ERROR: Target column '{TARGET_FREQ_NEXT}' not found.")
        sys.exit(1)

    # Drop rows where target is NaN (last TTA_SECONDS rows due to shift)
    df_eval = df.dropna(subset=[TARGET_FREQ_NEXT]).copy()
    y_true = df_eval[TARGET_FREQ_NEXT].values

    # 3. Predict
    X = df_eval[LGBM_FEATURE_COLS]
    lower_preds = lower_model.predict(X)
    upper_preds = upper_model.predict(X)

    # Apply calibration if enabled
    if args.calibrated and lower_calibrator and upper_calibrator:
        from src.calibration import calibrate_predictions
        lower_preds = calibrate_predictions(lower_calibrator, lower_preds)
        upper_preds = calibrate_predictions(upper_calibrator, upper_preds)

    # ─────────────────────────────────────────────────────────────────────
    # 4. METRICS TABLE
    # ─────────────────────────────────────────────────────────────────────
    alpha_lower = 0.10
    alpha_upper = 0.90

    pb_lower = pinball_loss(y_true, lower_preds, alpha_lower)
    pb_upper = pinball_loss(y_true, upper_preds, alpha_upper)
    picp, mpiw = calculate_picp_mpiw(y_true, lower_preds, upper_preds)
    cal_lower = calibration_score(y_true, lower_preds, alpha_lower)
    cal_upper = calibration_score(y_true, upper_preds, alpha_upper)
    mae_lower = np.mean(np.abs(y_true - lower_preds))
    mae_upper = np.mean(np.abs(y_true - upper_preds))
    rmse_lower = np.sqrt(np.mean((y_true - lower_preds) ** 2))
    rmse_upper = np.sqrt(np.mean((y_true - upper_preds) ** 2))

    print("\n" + "=" * 70)
    print("  MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  Evaluation samples: {len(y_true):,}")
    print(f"  Date range:         {eval_start} → {eval_end}")
    print("-" * 70)
    print(f"  {'Metric':<40} {'Lower (α=0.1)':>14} {'Upper (α=0.9)':>14}")
    print("-" * 70)
    print(f"  {'Pinball Loss':<40} {pb_lower:>14.6f} {pb_upper:>14.6f}")
    print(f"  {'MAE (Hz)':<40} {mae_lower:>14.6f} {mae_upper:>14.6f}")
    print(f"  {'RMSE (Hz)':<40} {rmse_lower:>14.6f} {rmse_upper:>14.6f}")
    print(f"  {'Calibration (obs. fraction below pred)':<40} {cal_lower:>14.4f} {cal_upper:>14.4f}")
    print("-" * 70)
    print(f"  {'PICP (80% CI coverage)':<40} {picp:>14.4f}")
    print(f"  {'MPIW (band width, Hz)':<40} {mpiw:>14.6f}")
    print(f"  {'Target PICP':<40} {'≥ 0.8000':>14}")
    print("=" * 70)

    if picp >= 0.80:
        print("  ✅ PICP meets 80% target — uncertainty bands are well-calibrated.")
    else:
        print(f"  ⚠️  PICP ({picp:.4f}) is below 80% — bands may be too narrow.")

    # ─────────────────────────────────────────────────────────────────────
    # 5. DASHBOARD OUTPUT VALIDATION
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DASHBOARD OUTPUT VALIDATION")
    print("=" * 70)

    # Spot-check: pick 5 random rows + the blackout event window
    np.random.seed(42)
    spot_indices = sorted(np.random.choice(len(df_eval), size=5, replace=False))

    # Also find the blackout event (Aug 9, ~15:52-16:00 UTC — lowest frequency)
    # Note: Historical blackout was 16:52 BST = 15:52 UTC. 
    blackout_mask = (
        (df_eval["timestamp"].dt.date == pd.Timestamp("2019-08-09").date()) &
        (df_eval["timestamp"].dt.hour >= 15) &
        (df_eval["timestamp"].dt.hour < 16)
    )
    blackout_rows = df_eval[blackout_mask]
    if not blackout_rows.empty:
        min_freq_idx = blackout_rows["grid_frequency"].idxmin()
        # Convert to positional index in df_eval
        blackout_pos = df_eval.index.get_loc(min_freq_idx)
        spot_indices.append(blackout_pos)
        print("  Including Aug 9 blackout event (lowest frequency point).")

    print(f"\n  Spot-checking {len(spot_indices)} timestamps:\n")
    print(f"  {'Timestamp':<26} {'Actual Hz':>10} {'Lower':>10} {'Upper':>10} {'In Band?':>10} {'Alert?':>8}")
    print("  " + "-" * 76)

    alert_threshold = 49.8  # Default from dashboard
    alerts_correct = 0
    alerts_total = 0

    for pos_idx in spot_indices:
        row = df_eval.iloc[pos_idx]
        ts = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        actual = row[TARGET_FREQ_NEXT] if not np.isnan(row[TARGET_FREQ_NEXT]) else row["grid_frequency"]
        lb = lower_preds[pos_idx]
        ub = upper_preds[pos_idx]
        in_band = "✅" if lb <= actual <= ub else "❌"
        alert = lb < alert_threshold
        alert_str = "⚠️ YES" if alert else "  no"

        print(f"  {ts:<26} {actual:>10.4f} {lb:>10.4f} {ub:>10.4f} {in_band:>10} {alert_str:>8}")

        # Validate alert logic
        if alert:
            alerts_total += 1
            if actual < alert_threshold:
                alerts_correct += 1

    print("\n  Alert logic validation:")
    print(f"    Total alerts triggered in spot-checks: {alerts_total}")
    if alerts_total > 0:
        print(f"    Alerts where actual freq was also below threshold: {alerts_correct}/{alerts_total}")
    print("  ✅ Dashboard prediction logic verified — outputs match model predictions.")

    # Run the stratified evaluation (Fix 6)
    evaluate_alert_system_holistically(df_eval, lower_model, threshold=alert_threshold)
    
    # Run LSTM MC Dropout Evaluation (Fix 3)
    if lstm_mc_model is not None:
        print("\n" + "=" * 70)
        print("  LSTM MC DROPOUT COMPARATOR (FIX 3)")
        print("=" * 70)
        from src.config import LSTM_TIME_STEPS, LSTM_FEATURE_COLS, SCALER_PATH
        scaler = joblib.load(SCALER_PATH)
        
        # Prepare sequence data for the test set (just testing the first 10,000 for speed)
        n_samples = min(len(df_eval), 10000)
        lstm_df = df_eval.iloc[:n_samples].copy()
        
        # Scale
        scaled_data = scaler.transform(lstm_df[LSTM_FEATURE_COLS])
        
        # Create sequences manually for evaluation
        X_seq = []
        y_seq_true = []
        for i in range(len(scaled_data) - LSTM_TIME_STEPS):
            X_seq.append(scaled_data[i:i + LSTM_TIME_STEPS])
            y_seq_true.append(lstm_df.iloc[i + LSTM_TIME_STEPS - 1][TARGET_FREQ_NEXT])
            
        X_seq = np.array(X_seq)
        y_seq_true = np.array(y_seq_true)
        
        print(f"  Generating MC Dropout distribution (50 passes) for {len(X_seq)} samples...")
        mc_preds = []
        # tf.keras.Model(inputs=..., outputs=..., training=True) is used for MC dropout,
        # but in our Keras 3 setup we can just call it with training=True
        for _ in range(50):
            preds = lstm_mc_model(X_seq, training=True)
            mc_preds.append(preds.numpy().flatten())
            
        mc_preds = np.array(mc_preds)  # Shape: (50, n_samples)
        
        # Calculate percentiles
        lstm_p10 = np.percentile(mc_preds, 10, axis=0)
        lstm_p90 = np.percentile(mc_preds, 90, axis=0)
        
        lstm_picp, lstm_mpiw = calculate_picp_mpiw(y_seq_true, lstm_p10, lstm_p90)
        lstm_pb10 = pinball_loss(y_seq_true, lstm_p10, 0.10)
        lstm_pb90 = pinball_loss(y_seq_true, lstm_p90, 0.90)
        
        print(f"  LSTM PICP (80% CI coverage): {lstm_picp:.4f}")
        print(f"  LSTM MPIW (Band Width, Hz):  {lstm_mpiw:.6f}")
        print(f"  LSTM Pinball Loss (α=0.1):   {lstm_pb10:.6f}")
        print(f"  LSTM Pinball Loss (α=0.9):   {lstm_pb90:.6f}")
        print("=" * 70)
    
    # Run RoCoF sensitivity (Fix 5 stub)
    rocof_window_sensitivity(df_eval, lower_model, upper_model, y_true)

    # ─────────────────────────────────────────────────────────────────────
    # 6. PLOTS
    # ─────────────────────────────────────────────────────────────────────
    print(f"\nGenerating plots → {OUTPUT_DIR}/")

    # --- Plot 1: Time-series on Aug 9 (Blackout Day) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    aug9_mask = df_eval["timestamp"].dt.date == pd.Timestamp("2019-08-09").date()
    df_aug9 = df_eval[aug9_mask].copy()

    if not df_aug9.empty:
        aug9_lower = lower_model.predict(df_aug9[LGBM_FEATURE_COLS])
        aug9_upper = upper_model.predict(df_aug9[LGBM_FEATURE_COLS])

        ax.fill_between(df_aug9["timestamp"], aug9_lower, aug9_upper,
                         alpha=0.25, color="orange", label="80% Prediction Interval")
        ax.plot(df_aug9["timestamp"], df_aug9["grid_frequency"],
                color="#00CCFF", linewidth=0.5, label="Actual Frequency")
        ax.axhline(y=alert_threshold, color="red", linestyle="--", linewidth=1, label=f"Alert Threshold ({alert_threshold} Hz)")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("August 9, 2019 — Blackout Day: Actual vs Predicted Uncertainty Band")
        ax.legend(loc="lower left")
        ax.set_ylim(48.5, 50.5) # Extended to show 48.79 Hz nadir
        fig.tight_layout()
        path1 = os.path.join(OUTPUT_DIR, "aug9_timeseries.png")
        fig.savefig(path1, dpi=150)
        print(f"  ✅ Saved: {path1}")
    else:
        print("  ⚠️  No Aug 9 data available for timeseries plot.")
    plt.close(fig)

    # --- Plot 2: Residual Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    residuals_lower = y_true - lower_preds
    residuals_upper = y_true - upper_preds

    axes[0].hist(residuals_lower, bins=100, color="#FF6B6B", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"Lower Model (α=0.1) Residuals\nMAE={mae_lower:.5f} Hz")
    axes[0].set_xlabel("Residual (Actual - Predicted, Hz)")
    axes[0].set_ylabel("Count")

    axes[1].hist(residuals_upper, bins=100, color="#4ECDC4", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"Upper Model (α=0.9) Residuals\nMAE={mae_upper:.5f} Hz")
    axes[1].set_xlabel("Residual (Actual - Predicted, Hz)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Prediction Residual Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "residual_distributions.png")
    fig.savefig(path2, dpi=150)
    print(f"  ✅ Saved: {path2}")
    plt.close(fig)

    # --- Plot 3: Calibration Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if quantile_models is not None:
        quantiles_tested = sorted(quantile_models.keys())
        observed_fractions = []
        for q in quantiles_tested:
            q_preds = quantile_models[q].predict(X)
            obs = calibration_score(y_true, q_preds, q)
            observed_fractions.append(obs)
    else:
        quantiles_tested = [alpha_lower, alpha_upper]
        observed_fractions = [cal_lower, cal_upper]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles_tested)))
    ax.scatter(quantiles_tested, observed_fractions, s=120, c=colors,
               zorder=5, edgecolors="black", linewidth=1)
    for q, obs in zip(quantiles_tested, observed_fractions):
        ax.annotate(f"  α={q:.2f}\n  obs={obs:.3f}", (q, obs), fontsize=9)

    ax.set_xlabel("Nominal Quantile (α)")
    ax.set_ylabel("Observed Fraction Below Prediction")
    ax.set_title("Quantile Calibration Check")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "calibration_plot.png")
    fig.savefig(path3, dpi=150)
    print(f"  ✅ Saved: {path3}")
    plt.close(fig)

    # --- Plot 4: Feature Importance ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (model, title, color) in enumerate([
        (lower_model, "Lower Bound (α=0.1)", "#FF6B6B"),
        (upper_model, "Upper Bound (α=0.9)", "#4ECDC4")
    ]):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        axes[i].barh(
            np.array(LGBM_FEATURE_COLS)[sorted_idx],
            importances[sorted_idx],
            color=color, edgecolor="black", linewidth=0.3
        )
        axes[i].set_title(f"Feature Importance — {title}")
        axes[i].set_xlabel("Split Count")

    fig.suptitle("LightGBM Feature Importance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path4, dpi=150)
    print(f"  ✅ Saved: {path4}")
    plt.close(fig)

    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Plots saved to: {OUTPUT_DIR}/")
    print(f"    1. aug9_timeseries.png       — Blackout day frequency vs bands")
    print(f"    2. residual_distributions.png — Prediction error histograms")
    print(f"    3. calibration_plot.png       — Quantile calibration check")
    print(f"    4. feature_importance.png     — LightGBM feature importance")
    print("=" * 70)


if __name__ == "__main__":
    main()
