import pandas as pd
df = pd.read_csv('/home/fatema/University/Dissertation/Implementation/notebooks/demo_data_aug9.csv')
blackout_mask = (df['timestamp'] >= '2019-08-09 16:50:00') & (df['timestamp'] <= '2019-08-09 17:00:00')
blackout_df = df[blackout_mask]
if not blackout_df.empty:
    min_freq_row = blackout_df.loc[blackout_df['grid_frequency'].idxmin()]
    print(f"Min Freq: {min_freq_row['grid_frequency']} at {min_freq_row['timestamp']}")
    # Find alert time (predicted lower < 49.8)
    # Since I don't have predictions here, I'll just look for actual crossing
    alert_mask = blackout_df['grid_frequency'] < 49.8
    if alert_mask.any():
        actual_breach = blackout_df[alert_mask].iloc[0]
        print(f"Actual Breach (49.8): {actual_breach['grid_frequency']} at {actual_breach['timestamp']}")
else:
    print("Blackout window not found in CSV")
