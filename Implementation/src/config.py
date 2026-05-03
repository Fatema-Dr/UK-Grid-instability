# Configuration file for the Grid Stability Alert System

# 1. FILE PATHS & API CONFIGURATION
# -----------------------------------------------------------------------------
# Path to the raw 1-second frequency data (old CSV path - no longer used directly)
DATA_DIR = "data"
# FREQUENCY_DATA_FILE = f"{DATA_DIR}/f-2019-aug/f 2019 8.csv"
# INERTIA_DATA_FILE = f"{DATA_DIR}/inertia_costs19.csv"

# NESO CKAN API Configuration
NESO_API_BASE_URL = "https://api.neso.energy/api/3/action"

# Monthly frequency data resource IDs (2019) from the NESO Data Portal
NESO_FREQUENCY_RESOURCE_MAP = {
    "2019-01": "2a653f90-7948-4203-a49e-8763733debb2",
    "2019-02": "3343dbfb-58ac-478a-8aed-618a35b38475",
    "2019-03": "b4fc11ec-2f9b-465e-8974-37cc289f2aaa",
    "2019-04": "5a511d6f-0cc4-4054-bb45-8ad3b81051ad",
    "2019-05": "84a85749-18e1-4b6b-bb97-73888ccacfe0",
    "2019-06": "f967b00a-36b4-4979-920e-77fdb6be8a9c",
    "2019-07": "da48b1fe-8e54-48fc-87b9-f6b0362422e2",
    "2019-08": "819a0821-cc6d-4909-a1ea-7dba5cab0c33",
    "2019-09": "9ffafdfb-cf42-46b3-802f-a6d9a45794aa",
    "2019-10": "65b4f284-4963-46c5-ae78-cab57fe5372f",
    "2019-11": "3d1a42c0-5637-4702-b9c3-76a7c5d8f062",
    "2019-12": "f0933bdd-1b0e-4dd3-aa7f-5498df1ba5b9",
}
# Legacy alias for backward compatibility
NESO_FREQUENCY_RESOURCE_ID_2019_08 = NESO_FREQUENCY_RESOURCE_MAP["2019-08"]

# Inertia data resource IDs
NESO_INERTIA_RESOURCE_ID_2019 = "620491fa-ae1b-45b3-baa0-6e87c2d574cf"  # Daily costs
NESO_INERTIA_HALFHOURLY_RESOURCE_ID = "2f2dbaa1-3047-4e48-85f2-ec24e669678f"  # Half-hourly system inertia 2019-2020

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
    "volatility_30s",
    "volatility_60s",
    "wind_speed",
    "wind_ramp_rate",
    "solar_radiation", 
    "hour",
    "renewable_penetration_ratio"
] + [f"lag_{lag}s" for lag in LAG_INTERVALS_SECONDS]

# List of features for the LSTM model
LSTM_FEATURE_COLS = [
    "grid_frequency", 
    "rocof", 
    "volatility_10s",
    "volatility_30s",
    "volatility_60s",
    "wind_speed",
    "wind_ramp_rate",
    "solar_radiation",
    "hour",
    "renewable_penetration_ratio"
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

# 5. CALIBRATION & VALIDATION
# -----------------------------------------------------------------------------
# Calibration split: a held-out subset BEFORE the test set, used for
# post-hoc quantile recalibration via isotonic regression.
CALIBRATION_START_DATE = "2019-08-07 00:00:00"
CALIBRATION_END_DATE = "2019-08-09 00:00:00"   # same as SPLIT_DATE

# Winter validation dates (for out-of-season robustness testing)
WINTER_VALIDATION_START_DATE = "2019-12-01"
WINTER_VALIDATION_END_DATE = "2019-12-31"

# Calibrator model paths
LOWER_CALIBRATOR_PATH = f"{EXPORT_DIR}/lower_calibrator.pkl"
UPPER_CALIBRATOR_PATH = f"{EXPORT_DIR}/upper_calibrator.pkl"
