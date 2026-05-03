import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import joblib
from src.config import LGBM_FEATURE_COLS

model = joblib.load('notebooks/lgbm_quantile_lower.pkl')
df = pd.read_csv('notebooks/demo_data_aug9.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
window = df[(df.timestamp >= '2019-08-09 15:52:00') & (df.timestamp <= '2019-08-09 15:53:00')]
preds = model.predict(window[LGBM_FEATURE_COLS])
window = window.copy()
window['pred_lower'] = preds
alerts = window[window['pred_lower'] < 49.8]
if not alerts.empty:
    first_alert = alerts.iloc[0]
    print(f"Alert Triggered at: {first_alert['timestamp']} with Pred Lower: {first_alert['pred_lower']}")
else:
    print("No alert triggered in window")
