# Configuration file for the Grid Stability Alert System

# 1. FILE PATHS & API CONFIGURATION
# -----------------------------------------------------------------------------
# Path to the raw 1-second frequency data (old CSV path - no longer used directly)
DATA_DIR = "data"
# FREQUENCY_DATA_FILE = f"{DATA_DIR}/f-2019-aug/f 2019 8.csv"
# INERTIA_DATA_FILE = f"{DATA_DIR}/inertia_costs19.csv"

# NESO CKAN API Configuration
NESO_API_BASE_URL = "https://api.neso.energy/api/3/action"
# Placeholder resource IDs - these would need to be dynamically retrieved or
# updated with actual UUIDs from the NESO Data Portal for the relevant periods.
NESO_FREQUENCY_RESOURCE_ID_2019_08 = "819a0821-cc6d-4909-a1ea-7dba5cab0c33" # Correct UUID for Aug 2019
NESO_INERTIA_RESOURCE_ID_2019 = "620491fa-ae1b-45b3-baa0-6e87c2d574cf" # Correct UUID for 2019

# Directory to save exported models and data for the dashboard
EXPORT_DIR = "notebooks" # Models are saved alongside the notebook
LGBM_MODEL_PATH = f"{EXPORT_DIR}/lightgbm_classifier.pkl"
LGBM_QUANTILE_LOWER_PATH = f"{EXPORT_DIR}/lgbm_quantile_lower.pkl"
LGBM_QUANTILE_UPPER_PATH = f"{EXPORT_DIR}/lgbm_quantile_upper.pkl"
LSTM_MODEL_PATH = f"{EXPORT_DIR}/lstm_model.keras"
SCALER_PATH = f"{EXPORT_DIR}/scaler.pkl"
DEMO_DATA_PATH = f"{EXPORT_DIR}/demo_data_aug9.csv"


# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
# Time to Alert (in seconds) for the prediction target
TTA_SECONDS = 10

# OpSDA width parameter for wind ramp rate calculation
OPSDA_WIDTH = 0.5 # Default value, can be optimized

# Lag intervals for frequency features (in seconds)
LAG_INTERVALS_SECONDS = [1, 5, 60]

# List of features to be used for the LightGBM model
LGBM_FEATURE_COLS = [
    "grid_frequency", 
    "rocof", 
    "volatility_10s",
    "wind_speed",
    "wind_ramp_rate",
    "solar_radiation", 
    "hour",
    "inertia_cost"
] + [f"lag_{lag}s" for lag in LAG_INTERVALS_SECONDS]

# List of features for the LSTM model
LSTM_FEATURE_COLS = [
    "grid_frequency", 
    "rocof", 
    "volatility_10s", 
    "wind_speed",
    "wind_ramp_rate",
    "inertia_cost"
] + [f"lag_{lag}s" for lag in LAG_INTERVALS_SECONDS]

# Target column for classification
TARGET_COL = "target_is_unstable"
# Target column for regression
TARGET_FREQ_NEXT = "target_freq_next"


# 3. MODEL TRAINING
# -----------------------------------------------------------------------------
# Quantiles for the uncertainty bands
QUANTILE_ALPHAS = [0.1, 0.9]

# Date to split the training and testing data
SPLIT_DATE = "2019-08-09 00:00:00"
END_TEST_DATE = "2019-08-10 00:00:00"

# LSTM training parameters
LSTM_TIME_STEPS = 30
LSTM_EPOCHS = 5
LSTM_BATCH_SIZE = 64
LSTM_VALIDATION_SPLIT = 0.1

# LightGBM parameters
LGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# 4. WEATHER API CONFIGURATION
# -----------------------------------------------------------------------------
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_API_LATITUDE = 54.0 # Approx center of UK
WEATHER_API_LONGITUDE = -2.0
WEATHER_API_TIMEZONE = "Europe/London"
WEATHER_API_HOURLY_VARS = ["temperature_2m", "precipitation", "rain", "snowfall", "wind_speed_10m", "wind_gusts_10m", "direct_radiation"]
WEATHER_API_DEFAULT_START_DATE = "2019-08-01"
WEATHER_API_DEFAULT_END_DATE = "2019-08-31"

