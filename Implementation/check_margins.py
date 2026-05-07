import pandas as pd
import numpy as np
import joblib
import os

IMPL_DIR = "/home/yogipatel/Documents/UK-Grid-instability/Implementation"
LOWER_MODEL_PATH = os.path.join(IMPL_DIR, "notebooks", "lgbm_quantile_lower.pkl")
DEMO_DATA_PATH   = os.path.join(IMPL_DIR, "notebooks", "demo_data_aug9.csv")
LGBM_FEATURE_COLS = [
    "grid_frequency", 
    "rocof_1s", "rocof_5s", "rocof_10s", "rocof_30s", "rocof_accel", "rocof_smooth", "rocof",
    "volatility_10s", "volatility_30s", "volatility_60s",
    "wind_speed", "wind_power_proxy", "demand_proxy", "wind_ramp_rate",
    "solar_radiation", 
    "hour",
    "renewable_penetration_ratio",
    "inertia_value", "inertia_roc", "low_inertia_flag", "rocof_inertia_risk",
    "lag_1s", "lag_5s", "lag_60s"
]
TARGET_FREQ_NEXT = "target_freq_next"

lo_m = joblib.load(LOWER_MODEL_PATH)
df = pd.read_csv(DEMO_DATA_PATH, parse_dates=["timestamp"])
df = df.dropna(subset=LGBM_FEATURE_COLS + [TARGET_FREQ_NEXT])

# Focus on the blackout event window (after 15:00 UTC)
df = df[df["timestamp"] > "2019-08-09 15:00:00"]
X = df[LGBM_FEATURE_COLS]
y_true = df[TARGET_FREQ_NEXT].values
lo_pred = lo_m.predict(X)

# Statutory failure: actual frequency drops below 49.8
failure_mask = y_true < 49.8
if failure_mask.any():
    failure_idx = np.where(failure_mask)[0][0]
    failure_time = df["timestamp"].iloc[failure_idx]
    print(f"Statutory failure (Actual < 49.8) at: {failure_time}")
    
    # Check alert thresholds
    for thresh in [49.9, 49.88, 49.85, 49.82, 49.8]:
        alert_mask = lo_pred < thresh
        if alert_mask.any():
            alert_idx = np.where(alert_mask)[0][0]
            alert_time = df["timestamp"].iloc[alert_idx]
            margin = (failure_time - alert_time).total_seconds()
            print(f"Threshold {thresh}: Alert at {alert_time}, Margin = {margin}s")
        else:
            print(f"Threshold {thresh}: No alert triggered.")
else:
    print("No statutory failure found in data.")
