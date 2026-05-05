import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.config import (
    LGBM_FEATURE_COLS, LSTM_FEATURE_COLS, LSTM_TIME_STEPS,
    LGBM_QUANTILE_LOWER_PATH, LSTM_MODEL_PATH, SCALER_PATH
)

# Load Models
lower_model = joblib.load(LGBM_QUANTILE_LOWER_PATH)
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load Data
CACHE_DIR = Path("data/processed_cache")
cache_files = list(CACHE_DIR.glob("*.parquet"))
cf = [f for f in cache_files if "2019-08-09" in f.name][0]
df_data = pd.read_parquet(cf)

# Critical window: 15:52:10 to 15:52:40 UTC
mask = (df_data['timestamp'] >= "2019-08-09 15:52:10+00:00") & (df_data['timestamp'] <= "2019-08-09 15:52:40+00:00")
df_crit = df_data[mask].copy()

alert_threshold = 49.8
results = []

print(f"{'Timestamp':<25} | {'Actual Freq':<12} | {'Pred Lower (10s ahead)':<22} | {'LGBM Alert?':<12} | {'LSTM Prob':<10} | {'Final'}")
print("-" * 105)

for i in range(len(df_crit)):
    row = df_crit.iloc[i]
    ts = row['timestamp']
    freq = row['grid_frequency']
    
    input_lgbm = pd.DataFrame([row[LGBM_FEATURE_COLS].values], columns=LGBM_FEATURE_COLS)
    # The dashboard uses lower_bound_pred = lower_bound_raw + swing_delta_f. 
    # Here swing_delta_f is 0.
    lower_bound_pred = lower_model.predict(input_lgbm)[0]
    
    is_alert_lgbm = lower_bound_pred < alert_threshold or freq < alert_threshold
    
    # LSTM Probability
    global_idx = df_crit.index[i]
    w_start = max(0, global_idx - LSTM_TIME_STEPS + 1)
    lstm_input_df = df_data.iloc[w_start : global_idx + 1]
    lstm_prob = 0.0
    if len(lstm_input_df) == LSTM_TIME_STEPS:
        lstm_input_scaled = scaler.transform(lstm_input_df[LSTM_FEATURE_COLS])
        lstm_prob = float(lstm_model.predict(np.array([lstm_input_scaled]), verbose=0)[0][0])
    
    lstm_alert = lstm_prob > 0.5
    final = "RED" if (is_alert_lgbm and lstm_alert) else ("YELLOW" if (is_alert_lgbm or lstm_alert) else "GREEN")
    
    print(f"{ts.strftime('%H:%M:%S UTC'):<25} | {freq:<12.4f} | {lower_bound_pred:<22.4f} | {'YES' if is_alert_lgbm else 'no':<12} | {lstm_prob:<10.4f} | {final}")
