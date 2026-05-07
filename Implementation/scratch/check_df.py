import pandas as pd
from pathlib import Path
import os

cache_dir = Path("data/processed_cache")
parquet_files = list(cache_dir.glob("*.parquet"))
if not parquet_files:
    print("No parquet files found")
else:
    # Get the latest parquet file
    latest_file = max(parquet_files, key=os.path.getmtime)
    df = pd.read_parquet(latest_file)
    
    # Filter for 15:52:00 to 15:53:00
    mask = (df['timestamp'].dt.hour == 15) & (df['timestamp'].dt.minute == 52)
    df_sub = df[mask]
    
    print(f"{'Time':<20} | {'Freq':<7} | {'RoCoF':<8} | {'RoCoF5s':<8} | {'Accel':<8}")
    for idx, row in df_sub.iterrows():
        t = row['timestamp'].strftime("%H:%M:%S")
        f = row['grid_frequency']
        r = row.get('rocof_smooth', row.get('rocof', 0.0))
        r5 = row.get('rocof_5s', 0.0)
        a = row.get('rocof_accel', 0.0)
        print(f"{t:<20} | {f:<7.4f} | {r:<8.5f} | {r5:<8.5f} | {a:<8.5f}")
        
        if f < 49.8:
            break
