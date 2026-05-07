import pandas as pd
from src.data_loader import fetch_frequency_data, fetch_weather_data, fetch_inertia_data_halfhourly
from src.feature_engineering import create_features, merge_datasets

df_freq = fetch_frequency_data("2019-08-01", "2019-08-10")
df_weather = fetch_weather_data("2019-08-01", "2019-08-10")
df_inertia = fetch_inertia_data_halfhourly("2019-08-01", "2019-08-10")

df_merged_pl = merge_datasets(df_freq, df_weather, df_inertia)
df_merged = df_merged_pl.to_pandas()
df_data = create_features(df_merged)
df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

mask = (df_data['timestamp'].dt.day == 9) & (df_data['timestamp'].dt.hour == 15) & (df_data['timestamp'].dt.minute == 52) & (df_data['timestamp'].dt.second >= 25) & (df_data['timestamp'].dt.second <= 36)
df_sub = df_data[mask].sort_values('timestamp')

print(f"{'Time':<20} | {'Freq':<7} | {'RoCoF':<8} | {'RoCoF_5s':<8} | {'Accel':<8} | {'Volat':<8}")
for idx, row in df_sub.iterrows():
    t = row['timestamp'].strftime("%H:%M:%S")
    f = row['grid_frequency']
    r = row.get('rocof_smooth', row.get('rocof', 0.0))
    r5 = row.get('rocof_5s', 0.0)
    a = row.get('rocof_accel', 0.0)
    v = row.get('volatility_10s', 0.0)
    print(f"{t:<20} | {f:<7.4f} | {r:<8.5f} | {r5:<8.5f} | {a:<8.5f} | {v:<8.5f}")

