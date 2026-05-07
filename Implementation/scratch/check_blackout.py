import pandas as pd
import joblib
import numpy as np

from src.config import LGBM_FEATURE_COLS, LGBM_MODEL_PATH
from src.feature_engineering import create_features, merge_datasets
from src.data_loader import fetch_frequency_data, fetch_weather_data, fetch_inertia_data_halfhourly

# Fast load for one hour
start = "2019-08-09 15:00:00"
end = "2019-08-09 16:00:00"
df_freq = fetch_frequency_data("2019-08-09", "2019-08-09")
df_weather = fetch_weather_data("2019-08-09", "2019-08-09")
df_inertia = fetch_inertia_data_halfhourly("2019-08-09", "2019-08-09")

df_merged_pl = merge_datasets(df_freq, df_weather, df_inertia)
df_merged = df_merged_pl.to_pandas()
df_data = create_features(df_merged)
df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

mask = (df_data['timestamp'] >= pd.to_datetime("2019-08-09 15:52:20+00:00")) & (df_data['timestamp'] <= pd.to_datetime("2019-08-09 15:53:00+00:00"))
df_sub = df_data[mask]

classifier_model = joblib.load(LGBM_MODEL_PATH)
lower_model = joblib.load("notebooks/lgbm_quantile_lower.pkl")

print(f"{'Time':<20} | {'Freq':<7} | {'RoCoF':<8} | {'Accel':<8} | {'Penet':<6} | {'SigCnt':<6} | {'Emerg':<6} | {'LB':<7}")
for idx, row in df_sub.iterrows():
    rocof_now   = row.get('rocof_smooth', row.get('rocof', 0.0))
    rocof_accel = row.get('rocof_accel', 0.0)
    volatility  = row.get('volatility_10s', 0.0)
    freq_now    = row['grid_frequency']
    ren_pen     = row.get('renewable_penetration_ratio', 0.0)
    
    input_lgbm_cls = pd.DataFrame([row[LGBM_FEATURE_COLS].values], columns=LGBM_FEATURE_COLS)
    classifier_prob = classifier_model.predict_proba(input_lgbm_cls)[0][1]
    
    lb = lower_model.predict(input_lgbm_cls)[0]

    rocof_alert = (rocof_now < -0.015) and (freq_now < 50.05)
    accel_alert = (rocof_accel < -0.005)
    volatility_alert = (volatility > 0.02) and (freq_now < 50.1)
    renewable_stress = (ren_pen > 0.15) and (rocof_now < -0.01)
    freq_boundary = freq_now < 49.95

    signal_count = sum([
        rocof_alert,
        accel_alert,
        volatility_alert,
        renewable_stress,
        freq_boundary,
        classifier_prob > 0.35,
    ])
    
    alert_threshold_hz = 49.8
    emergency_trigger = signal_count >= 3 or freq_now < alert_threshold_hz or lb < alert_threshold_hz
    
    t = row['timestamp'].strftime("%H:%M:%S")
    print(f"{t:<20} | {freq_now:<7.4f} | {rocof_now:<8.5f} | {rocof_accel:<8.5f} | {ren_pen:<6.2f} | {signal_count:<6} | {emergency_trigger} | {lb:<7.4f}")

