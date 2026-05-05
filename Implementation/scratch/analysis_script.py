import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap

# Add the project root to the python path
sys.path.append("/home/fatema/University/Dissertation/Implementation")

from src.data_loader import fetch_frequency_data, fetch_inertia_data, fetch_weather_data
from src.feature_engineering import create_features
from src.config import LGBM_FEATURE_COLS

def load_data(start_date="2019-08-01", end_date="2019-08-31"):
    df_freq = fetch_frequency_data(start_date=start_date, end_date=end_date)
    df_weather = fetch_weather_data(start_date=start_date, end_date=end_date)
    df_inertia = fetch_inertia_data(start_date=start_date, end_date=end_date)
    
    df_freq = df_freq.sort("timestamp")
    df_weather = df_weather.sort("timestamp")
    df_inertia = df_inertia.sort("timestamp_date")

    df_merged = df_freq.join_asof(df_weather, on="timestamp", strategy="backward")
    import polars as pl
    df_inertia = df_inertia.with_columns(
        pl.col("timestamp_date").cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias("timestamp")
    )
    df_merged = df_merged.join_asof(
        df_inertia.select(["timestamp", "inertia_cost"]),
        on="timestamp",
        strategy="backward"
    )
    df_merged = df_merged.drop_nulls().to_pandas()

    df_data = create_features(df_merged)
    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"])
    return df_data

def investigate_fnr(df, lower_preds, threshold=49.8):
    print("\n--- Investigating High False Negative Rate (Issue 2) ---")
    
    df['is_unstable'] = df['grid_frequency'] < threshold
    df['pred_unstable'] = lower_preds < threshold
    
    # Identify events
    df['event_id'] = (df['is_unstable'] != df['is_unstable'].shift()).cumsum()
    events = df[df['is_unstable']].groupby('event_id')
    
    total_events = 0
    missed_events = 0
    event_durations = []
    
    for name, group in events:
        duration = len(group)
        event_durations.append(duration)
        total_events += 1
        
        # Did we predict unstable during this event or 10 seconds prior?
        start_idx = group.index[0]
        end_idx = group.index[-1]
        
        # Lookback 10 seconds before the event
        lookback_start = max(0, start_idx - 10)
        
        predicted_unstable = df.loc[lookback_start:end_idx, 'pred_unstable'].any()
        if not predicted_unstable:
            missed_events += 1
            
    print(f"Total Instability Events: {total_events}")
    print(f"Missed Events: {missed_events} (FNR: {missed_events/total_events*100:.1f}%)")
    
    # Analyze by duration
    df_events = pd.DataFrame({'duration': event_durations})
    print("Event Duration Distribution:")
    print(f"  < 5s: {len(df_events[df_events['duration'] < 5])} events")
    print(f"  5-15s: {len(df_events[(df_events['duration'] >= 5) & (df_events['duration'] < 15)])} events")
    print(f"  > 15s: {len(df_events[df_events['duration'] >= 15])} events")
    
def sanity_check_feature_importance(model, df):
    print("\n--- Sanity Check Feature Importance (Issue 3) ---")
    importances = model.feature_importances_
    
    # Global Importance
    total = np.sum(importances)
    for col, imp in zip(LGBM_FEATURE_COLS, importances):
        if col == 'rocof':
            print(f"Global Importance - RoCoF: {imp/total*100:.2f}%")
        elif col == 'grid_frequency':
            print(f"Global Importance - Grid Frequency: {imp/total*100:.2f}%")
            
    # Local Importance (Blackout Event)
    blackout_mask = (
        (df["timestamp"].dt.date == pd.Timestamp("2019-08-09").date()) &
        (df["timestamp"].dt.hour == 16) &
        (df["timestamp"].dt.minute >= 50) &
        (df["timestamp"].dt.minute <= 55)
    )
    blackout_data = df[blackout_mask][LGBM_FEATURE_COLS]
    
    if not blackout_data.empty:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(blackout_data)
        
        # Average absolute SHAP values for the blackout period
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_total = np.sum(mean_abs_shap)
        
        for i, col in enumerate(LGBM_FEATURE_COLS):
            if col == 'rocof':
                print(f"Local Importance (Blackout) - RoCoF SHAP Magnitude: {mean_abs_shap[i]:.4f} ({mean_abs_shap[i]/shap_total*100:.2f}%)")
            elif col == 'grid_frequency':
                print(f"Local Importance (Blackout) - Grid Frequency SHAP Magnitude: {mean_abs_shap[i]:.4f} ({mean_abs_shap[i]/shap_total*100:.2f}%)")
    else:
        print("No blackout data found.")

def investigate_predictive_capability(df, lower_preds, threshold=49.8):
    print("\n--- Investigating Predictive Capability (Issue 1) ---")
    # Blackout event
    blackout_mask = (
        (df["timestamp"].dt.date == pd.Timestamp("2019-08-09").date()) &
        (df["timestamp"].dt.hour == 16) &
        (df["timestamp"].dt.minute >= 52) &
        (df["timestamp"].dt.minute <= 53)
    )
    b_df = df[blackout_mask].copy()
    
    if not b_df.empty:
        b_df['pred_lb'] = lower_preds[blackout_mask]
        
        breach_row = b_df[b_df['grid_frequency'] < threshold].iloc[0]
        breach_time = breach_row['timestamp']
        
        alert_row = b_df[b_df['pred_lb'] < threshold].iloc[0]
        alert_time = alert_row['timestamp']
        
        nadir_row = b_df.loc[b_df['grid_frequency'].idxmin()]
        nadir_time = nadir_row['timestamp']
        
        print(f"Initial Breach Time (Freq < 49.8): {breach_time}")
        print(f"Alert Trigger Time (Pred < 49.8):  {alert_time}")
        print(f"Nadir Time (Lowest Freq):          {nadir_time}")
        
        lead_time = (nadir_time - alert_time).total_seconds()
        react_time = (alert_time - breach_time).total_seconds()
        print(f"Lead time to nadir: {lead_time} seconds")
        print(f"Reaction time to breach: {react_time} seconds")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print("Loading model...")
    lower_model = joblib.load("/home/fatema/University/Dissertation/Implementation/notebooks/lgbm_quantile_lower.pkl")
    
    print("Generating predictions...")
    X = df[LGBM_FEATURE_COLS]
    lower_preds = lower_model.predict(X)
    
    investigate_fnr(df, lower_preds)
    sanity_check_feature_importance(lower_model, df)
    investigate_predictive_capability(df, lower_preds)
