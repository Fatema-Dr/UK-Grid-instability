import pandas as pd
from src.data_loader import fetch_frequency_data
df_freq = fetch_frequency_data("2019-08-01", "2019-08-10")

# Convert polars to pandas if it is polars
if hasattr(df_freq, "to_pandas"):
    df_freq = df_freq.to_pandas()

mask = (df_freq['timestamp'].dt.day == 9) & (df_freq['timestamp'].dt.hour == 15) & (df_freq['timestamp'].dt.minute == 52)
df_sub = df_freq[mask].sort_values('timestamp')

for idx, row in df_sub.iterrows():
    t = row['timestamp'].strftime("%H:%M:%S")
    f = row['grid_frequency']
    print(f"{t:<20} | {f:<7.4f}")
