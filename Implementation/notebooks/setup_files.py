import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

print("🔄 Generating project files locally...")

# -----------------------------------------------------------------------------
# 1. CREATE DEMO DATA (August 9th Simulation)
# -----------------------------------------------------------------------------
print("   Creating dummy data for August 9...")
# Create 24 hours of data at 1-second resolution (86400 rows)
dates = pd.date_range(start='2019-08-09 00:00:00', periods=86400, freq='S')
df = pd.DataFrame({'timestamp': dates})

# Simulate Frequency (Normal noise around 50Hz)
np.random.seed(42)
df['grid_frequency'] = 50.0 + np.random.normal(0, 0.05, size=len(df))

# Simulate the Blackout Drop at 16:52 (Index ~60700)
# We make it drop to 49.0 Hz
blackout_start = 60720
df.loc[blackout_start:blackout_start+300, 'grid_frequency'] = np.linspace(49.8, 48.8, 301)

# Feature Engineering
df['rocof'] = df['grid_frequency'].diff().fillna(0)
df['volatility_10s'] = df['grid_frequency'].rolling(10).std().fillna(0)
df['lag_1s'] = df['grid_frequency'].shift(1).fillna(50.0)
df['lag_5s'] = df['grid_frequency'].shift(5).fillna(50.0)
df['lag_60s'] = df['grid_frequency'].shift(60).fillna(50.0)
df['wind_speed'] = 10.0 + np.random.normal(0, 1, size=len(df)) # Dummy weather
df['solar_radiation'] = 0.0
df['hour'] = df['timestamp'].dt.hour

# Target (0 = Stable, 1 = Unstable)
df['target'] = 0
df.loc[df['grid_frequency'] < 49.5, 'target'] = 1

# Save CSV
df.to_csv("demo_data_aug9.csv", index=False)
print("   ✅ demo_data_aug9.csv created.")

# -----------------------------------------------------------------------------
# 2. TRAIN & SAVE LIGHTGBM MODEL
# -----------------------------------------------------------------------------
print("   Training LightGBM model...")
features_lgbm = ["grid_frequency", "rocof", "volatility_10s", 
                 "lag_1s", "lag_5s", "lag_60s", 
                 "wind_speed", "solar_radiation", "hour"]
X = df[features_lgbm]
y = df['target']

lgbm = lgb.LGBMClassifier(n_estimators=10, random_state=42) # Mini model
lgbm.fit(X, y)
joblib.dump(lgbm, "lightgbm_model.pkl")
print("   ✅ lightgbm_model.pkl saved.")

# -----------------------------------------------------------------------------
# 3. TRAIN & SAVE LSTM MODEL + SCALER
# -----------------------------------------------------------------------------
print("   Training LSTM model...")
features_lstm = ["grid_frequency", "rocof", "volatility_10s", "wind_speed"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features_lstm])

# Save Scaler
joblib.dump(scaler, "scaler.pkl")
print("   ✅ scaler.pkl saved.")

# Reshape for LSTM [Samples, TimeSteps, Features]
# We just create a tiny batch to initialize the model architecture
X_lstm = X_scaled[:1000].reshape(1000, 1, 4) 
# Note: In the app we resize input to (1, 30, 4), so model must accept variable length or we fix it here.
# To match the App's expectation of (None, 30, 4), we build the model explicitly.

model = Sequential([
    LSTM(10, input_shape=(None, 4)), # None allows any sequence length (like 30)
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
# No real training needed for the demo file structure, but we fit once to initialize weights
model.fit(X_lstm, y[:1000], epochs=1, verbose=0)

model.save("lstm_model.keras")
print("   ✅ lstm_model.keras saved.")

print("\n🎉 ALL FILES READY! You can now run 'python -m streamlit run app.py'")