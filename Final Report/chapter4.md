# Chapter 4: Implementation

## 4.1 Introduction

This chapter presents the technical implementation details of GridGuardian. The chapter is structured to provide a comprehensive account of the system architecture, data pipeline, feature engineering algorithms, model training procedures, and dashboard development. Code excerpts are provided to illustrate key implementations, enabling reproducibility and peer review.

The implementation is organised into modular components, each with a single responsibility:
- **Data Loader:** Fetches and caches data from external APIs.
- **Feature Engineering:** Computes domain-specific features (RoCoF, OpSDA, lag features).
- **Model Trainer:** Trains and evaluates LightGBM and LSTM models.
- **Dashboard:** Streamlit web application for real-time monitoring and explainability.

---

## 4.2 System Architecture Overview

### 4.2.1 High-Level Architecture

GridGuardian follows a layered architecture pattern, separating concerns into distinct layers:

**Figure 4.1: GridGuardian System Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Streamlit Dashboard (app.py)               │   │
│  │  - Real-time predictions                                │   │
│  │  - Uncertainty bands (quantile visualisation)           │   │
│  │  - Alert indicators                                     │   │
│  │  - SHAP explainability visualisations                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Model Trainer  │  │ Feature Engine  │  │   Evaluator     │ │
│  │ (model_trainer) │  │(feature_eng)    │  │ (evaluate)      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Data Loader   │  │   Cache (Par)   │  │   Config (TOML) │ │
│  │   (data_loader) │  │   (.parquet)    │  │   (config.py)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Data Sources                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  NESO CKAN API  │  │ Open-Meteo API  │  │ Inertia Costs   │ │
│  │ (1-sec frequency)│  │  (hourly weather)│  │   (daily)     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2.2 Component Descriptions

**Data Loader ([`data_loader.py`](Implementation/src/data_loader.py:1)):**
- Fetches grid frequency data from NESO CKAN API
- Fetches weather data from Open-Meteo API
- Downloads inertia cost data from NESO
- Implements caching in Parquet format for efficiency
- Handles API errors with retry logic and exponential backoff

**Feature Engineering ([`feature_engineering.py`](Implementation/src/feature_engineering.py:1)):**
- Computes RoCoF (Rate of Change of Frequency) with smoothing
- Implements OpSDA for wind ramp rate detection
- Generates lag features (1s, 5s, 60s)
- Creates temporal features (hour of day, day of week)
- Calculates renewable penetration proxies

**Model Trainer ([`model_trainer.py`](Implementation/src/model_trainer.py:1)):**
- Trains LightGBM quantile regression models (10th and 90th percentiles)
- Trains LSTM model as deep learning benchmark
- Implements hyperparameter tuning via grid search
- Saves trained models to disk for deployment

**Evaluator ([`evaluate_models.py`](Implementation/evaluate_models.py:1)):**
- Computes MAE, RMSE, pinball loss, PICP, MPIW, calibration
- Validates against August 9, 2019 blackout event
- Generates SHAP explainability visualisations
- Produces comparison reports with baseline models

**Dashboard ([`app.py`](Implementation/app.py:1)):**
- Streamlit web application for real-time monitoring
- Displays frequency time-series with predictions and uncertainty bands
- Shows alert indicators when lower quantile breaches threshold
- Integrates SHAP waterfall and summary plots
- Implements caching and source code hashing for performance

### 4.2.3 Configuration Management

Configuration is managed via [`config.py`](Implementation/src/config.py:1), which defines all hyperparameters and constants:

```python
# TTA_SECONDS: Prediction horizon (10 seconds aligns with FFR activation)
TTA_SECONDS = 10

# QUANTILE_ALPHAS: Quantiles for uncertainty bands
QUANTILE_ALPHAS = [0.1, 0.9]

# LGBM_FEATURE_COLS: Features for LightGBM model
LGBM_FEATURE_COLS = [
    'rocof_1s', 'rocof_5s_ma', 'wind_ramp_rate', 'wind_speed',
    'wind_direction', 'temperature_2m', 'surface_pressure',
    'inertia_cost', 'lag_1s', 'lag_5s', 'lag_60s',
    'hour_of_day', 'day_of_week'
]

# LSTM_FEATURE_COLS: Features for LSTM model (scaled)
LSTM_FEATURE_COLS = ['frequency', 'rocof_1s', 'wind_speed', 'inertia_cost']

# ALERT_THRESHOLD: Frequency threshold for alert triggering
ALERT_THRESHOLD = 49.8  # Hz
```

---

## 4.3 Data Pipeline Implementation

### 4.3.1 Polars for High-Performance Data Processing

Polars is selected as the primary data processing library due to its superior performance compared to pandas for large time-series datasets. Polars uses a Rust backend and lazy evaluation for optimised query execution (Polars Development Team, 2024).

**Key Polars Features Used:**
- **Lazy API:** Enables query optimisation and predicate pushdown.
- **Parallel Execution:** Automatic parallelisation of operations across CPU cores.
- **Memory Efficiency:** Columnar storage format reduces memory footprint.
- **join_asof:** Specialised temporal merge for time-series data.

### 4.3.2 Data Loading Implementation

The [`data_loader.py`](Implementation/src/data_loader.py:1) module implements data fetching with caching:

```python
def fetch_frequency_data(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Fetch grid frequency data from NESO CKAN API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        Polars DataFrame with columns: timestamp, frequency
    """
    cache_path = f"data/frequency_{start_date}_{end_date}.parquet"
    
    # Return cached data if available
    if Path(cache_path).exists():
        return pl.read_parquet(cache_path)
    
    # Fetch from API
    url = "https://data.neso.energy.gov.au/dataset/api/3/action/resource_show"
    params = {"id": "grid-frequency-resource-id"}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    # Parse JSON into Polars DataFrame
    data = response.json()["result"]
    df = pl.DataFrame({
        "timestamp": [parse_timestamp(r["timestamp"]) for r in data["data"]],
        "frequency": [r["frequency"] for r in data["data"]]
    })
    
    # Cache for subsequent runs
    df.write_parquet(cache_path)
    return df
```

**Key Implementation Details:**
1. **Caching:** Data is cached in Parquet format to avoid redundant API calls.
2. **Error Handling:** Retry logic with exponential backoff for transient failures.
3. **Type Safety:** Timestamps are parsed and validated to ensure consistency.

### 4.3.3 Temporal Merging with join_asof

The `join_asof` function performs a "backward" temporal merge, matching each frequency record with the most recent weather and inertia data:

**Figure 4.2: Polars join_asof Temporal Merge Strategy**

```
Frequency Data (1-sec):
timestamp           frequency
2019-08-09 17:52:45 50.01
2019-08-09 17:52:46 50.00
2019-08-09 17:52:47 49.95
...

Weather Data (1-hr):
timestamp           wind_speed
2019-08-09 17:00:00 8.5
2019-08-09 18:00:00 9.2
...

After join_asof (backward):
timestamp           frequency  wind_speed
2019-08-09 17:52:45 50.01      8.5  (from 17:00)
2019-08-09 17:52:46 50.00      8.5  (from 17:00)
2019-08-09 17:52:47 49.95      8.5  (from 17:00)
...
```

**Implementation:**
```python
def merge_data(frequency_df: pl.DataFrame, 
               weather_df: pl.DataFrame,
               inertia_df: pl.DataFrame) -> pl.DataFrame:
    """
    Merge frequency, weather, and inertia data using join_asof.
    
    Args:
        frequency_df: DataFrame with 1-second frequency data
        weather_df: DataFrame with hourly weather data
        inertia_df: DataFrame with daily inertia cost data
    
    Returns:
        Unified DataFrame with all features
    """
    # Ensure timestamp columns are datetime type
    frequency_df = frequency_df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime)
    )
    weather_df = weather_df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime)
    )
    inertia_df = inertia_df.with_columns(
        pl.col("date").str.strptime(pl.Datetime)
    )
    
    # Merge frequency with weather (backward join)
    merged = frequency_df.join_asof(
        weather_df,
        on="timestamp",
        strategy="backward"
    )
    
    # Merge with inertia (backward join on date)
    merged = merged.join_asof(
        inertia_df,
        left_on="timestamp",
        right_on="date",
        strategy="backward"
    )
    
    return merged
```

### 4.3.4 Data Validation

Data validation is implemented to ensure quality:

```python
def validate_data(df: pl.DataFrame) -> bool:
    """
    Validate data quality.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if data passes all validation checks
    """
    # Check for missing timestamps
    expected_count = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() + 1
    if len(df) < expected_count * 0.95:  # Allow 5% tolerance
        logger.warning(f"Missing data: expected {expected_count}, got {len(df)}")
    
    # Check frequency range
    if df["frequency"].min() < 48 or df["frequency"].max() > 52:
        logger.warning(f"Frequency out of plausible range: {df['frequency'].min()} - {df['frequency'].max()}")
    
    # Check for null values in critical columns
    null_counts = df.select([pl.col("*").null_count()])
    if null_counts["wind_speed"].null_count() > len(df) * 0.1:
        logger.warning(f"High null rate in wind_speed: {null_counts['wind_speed'].null_count() / len(df):.2%}")
    
    return True
```

---

## 4.4 Feature Engineering Implementation

### 4.4.1 RoCoF Calculation with Smoothing

RoCoF is calculated as the first difference of frequency, with a 5-second moving average for noise reduction:

```python
def calculate_rocof(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Rate of Change of Frequency (RoCoF).
    
    Args:
        df: DataFrame with 'frequency' column
    
    Returns:
        DataFrame with 'rocof_1s' and 'rocof_5s_ma' columns
    """
    return df.with_columns([
        # Raw 1-second RoCoF
        (pl.col("frequency") - pl.col("frequency").shift(1)).alias("rocof_1s"),
        # 5-second moving average for noise reduction
        (pl.col("frequency") - pl.col("frequency").shift(1))
            .rolling(window_size=5)
            .mean()
            .alias("rocof_5s_ma")
    ])
```

**Rationale for Smoothing:**
Raw RoCoF calculated from 1-second data can be noisy due to measurement errors and high-frequency oscillations. The 5-second moving average preserves the underlying signal while reducing noise (Savitzky and Golay, 1964).

### 4.4.2 OpSDA Implementation for Wind Ramp Detection

The [`opsda.py`](Implementation/src/opsda.py:1) module implements the Optimized Swinging Door Algorithm:

```python
def opdsa_compress(timestamps: List[datetime], 
                   values: List[float],
                   pressure: float = 0.5) -> Tuple[List[datetime], List[float]]:
    """
    Compress time-series data using the Optimized Swinging Door Algorithm.
    
    Args:
        timestamps: List of timestamps
        values: List of values (e.g., wind speed)
        pressure: Maximum deviation allowed (default: 0.5 m/s)
    
    Returns:
        Tuple of (compressed_timestamps, compressed_values)
    """
    if len(timestamps) < 2:
        return timestamps, values
    
    compressed_ts = [timestamps[0]]
    compressed_vals = [values[0]]
    
    pivot_idx = 0
    for i in range(1, len(timestamps)):
        # Calculate angular bounds from pivot to current point
        dt = (timestamps[i] - timestamps[pivot_idx]).total_seconds()
        dv = values[i] - values[pivot_idx]
        
        if dt == 0:
            continue
        
        slope = dv / dt
        
        # Check if intermediate points fall within the door
        within_door = True
        for j in range(pivot_idx + 1, i):
            dt_j = (timestamps[j] - timestamps[pivot_idx]).total_seconds()
            dv_j = values[j] - values[pivot_idx]
            expected_val = values[pivot_idx] + slope * dt_j
            
            if abs(values[j] - expected_val) > pressure:
                within_door = False
                break
        
        if not within_door:
            # Record the previous point and start new door
            compressed_ts.append(timestamps[i - 1])
            compressed_vals.append(values[i - 1])
            pivot_idx = i - 1
    
    # Add final point
    compressed_ts.append(timestamps[-1])
    compressed_vals.append(values[-1])
    
    return compressed_ts, compressed_vals


def calculate_wind_ramp_rate(df: pl.DataFrame, 
                             pressure: float = 0.5) -> pl.DataFrame:
    """
    Calculate wind ramp rate using OpSDA.
    
    Args:
        df: DataFrame with 'timestamp' and 'wind_speed' columns
        pressure: OpSDA pressure parameter
    
    Returns:
        DataFrame with 'wind_ramp_rate' column
    """
    # Group by hour and apply OpSDA
    ramp_rates = []
    
    for hour, group in df.group_by("hour"):
        compressed_ts, compressed_vals = opdsa_compress(
            group["timestamp"].to_list(),
            group["wind_speed"].to_list(),
            pressure=pressure
        )
        
        # Calculate ramp rate as slope between compressed points
        if len(compressed_ts) >= 2:
            dt = (compressed_ts[-1] - compressed_ts[0]).total_seconds()
            dv = compressed_vals[-1] - compressed_vals[0]
            ramp_rate = dv / dt if dt > 0 else 0
            ramp_rates.append(ramp_rate)
        else:
            ramp_rates.append(0)
    
    return df.with_columns(pl.Series("wind_ramp_rate", ramp_rates))
```

**Figure 4.3: Wind Ramp Rate Calculation Using OpSDA**

```
Original Wind Speed:
8.5 ──*─────*──*────*──*
      │    /   │  │   │
8.0 ──│───/────*──│───*──
      │  /      │  │   │
7.5 ──*─/───────│──*───│──
      │         │      │
7.0 ──│─────────*──────*──
      │
6.5 ──└────────────────────
      17:00   17:15  17:30  17:45

OpSDA Compressed Points (marked with larger circles):
8.5 ──●─────────────────────
      │                    ●
8.0 ──│────────────────────●
      │
7.5 ──●─────────────────────
      │
7.0 ──└─────────────────────
      17:00   17:15  17:30  17:45

Ramp Rate = (8.5 - 7.5) / (17:45 - 17:00) = 1.0 m/s / 45 min = 0.022 m/s/min
```

### 4.4.3 Lag Features

Lag features capture autoregressive patterns:

```python
def add_lag_features(df: pl.DataFrame, lags: List[int] = [1, 5, 60]) -> pl.DataFrame:
    """
    Add lag features for frequency.
    
    Args:
        df: DataFrame with 'frequency' column
        lags: List of lag intervals in seconds
    
    Returns:
        DataFrame with 'lag_1s', 'lag_5s', 'lag_60s' columns
    """
    return df.with_columns([
        pl.col("frequency").shift(lag).alias(f"lag_{lag}s")
        for lag in lags
    ])
```

### 4.4.4 Temporal Features

Temporal features encode time-related information:

```python
def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add temporal features (hour of day, day of week).
    
    Args:
        df: DataFrame with 'timestamp' column
    
    Returns:
        DataFrame with 'hour_of_day' and 'day_of_week' columns
    """
    return df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour_of_day"),
        pl.col("timestamp").dt.weekday().alias("day_of_week")
    ])
```

### 4.4.5 Complete Feature Engineering Pipeline

The [`feature_engineering.py`](Implementation/src/feature_engineering.py:1) module combines all feature engineering steps:

```python
def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Raw DataFrame with frequency, weather, and inertia data
    
    Returns:
        DataFrame with all engineered features
    """
    # Step 1: Calculate RoCoF
    df = calculate_rocof(df)
    
    # Step 2: Calculate wind ramp rate via OpSDA
    df = calculate_wind_ramp_rate(df)
    
    # Step 3: Add lag features
    df = add_lag_features(df, lags=[1, 5, 60])
    
    # Step 4: Add temporal features
    df = add_temporal_features(df)
    
    # Step 5: Drop rows with null values (from lag features)
    df = df.drop_nulls()
    
    return df
```

---

## 4.5 Model Training Pipeline

### 4.5.1 LightGBM Quantile Regression Training

The [`model_trainer.py`](Implementation/src/model_trainer.py:1) module implements LightGBM quantile regression training:

```python
def train_lightgbm_quantile(X_train: pl.DataFrame, 
                            y_train: pl.Series,
                            quantile: float,
                            config: Dict) -> lgb.Booster:
    """
    Train LightGBM quantile regression model.
    
    Args:
        X_train: Training features
        y_train: Training target (frequency)
        quantile: Target quantile (e.g., 0.1 for 10th percentile)
        config: Hyperparameter configuration
    
    Returns:
        Trained LightGBM booster
    """
    # Convert to LightGBM Dataset
    train_data = lgb.Dataset(
        X_train.to_numpy(),
        label=y_train.to_numpy()
    )
    
    # Define parameters
    params = {
        "objective": "regression_quantile",
        "alpha": quantile,
        "num_leaves": config["num_leaves"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config["n_estimators"],
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    return model


def train_quantile_models(df: pl.DataFrame, 
                          feature_cols: List[str],
                          target_col: str = "frequency",
                          tta: int = 10) -> Tuple[lgb.Booster, lgb.Booster]:
    """
    Train lower and upper quantile models.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        tta: Time to alert in seconds
    
    Returns:
        Tuple of (lower_bound_model, upper_bound_model)
    """
    # Prepare features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Create target shifted by TTA (predict frequency 10 seconds ahead)
    y_future = y.shift(-tta)
    
    # Drop rows with null targets
    mask = ~y_future.is_null()
    X_train = X[mask]
    y_train = y_future[mask]
    
    # Split into train and validation
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    
    # Train lower bound model (10th percentile)
    lower_model = train_lightgbm_quantile(X_tr, y_tr, quantile=0.1, config=LGBM_CONFIG)
    
    # Train upper bound model (90th percentile)
    upper_model = train_lightgbm_quantile(X_tr, y_tr, quantile=0.9, config=LGBM_CONFIG)
    
    return lower_model, upper_model
```

**Figure 4.4: Quantile Regression Training Loss Curves**

```
Loss
  │
  │  Training Loss (Lower Bound)
  │  ╭─────────────────────────────
  │  │ ╲
  │  │  ╲  Validation Loss (Lower Bound)
  │  │   ╲ ╭───────────────────────
  │  │    ╲│
  │  │     ╲╭─────────────────────
  │  │      ╲
  │  │       ╲╭───────────────────
  │  │        │
  │  │        │
  │  └────────┴───────────────────
  │           │
  └───────────┴───────────────────
              Epoch
```

### 4.5.2 LSTM Model Training

The LSTM model is implemented using TensorFlow/Keras:

```python
def create_lstm_model(sequence_length: int, 
                      num_features: int,
                      hidden_size: int = 64,
                      num_layers: int = 2,
                      dropout: float = 0.2) -> tf.keras.Model:
    """
    Create LSTM model architecture.
    
    Args:
        sequence_length: Length of input sequences
        num_features: Number of input features
        hidden_size: Number of LSTM units per layer
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # First LSTM layer
        tf.keras.layers.LSTM(
            hidden_size,
            return_sequences=True,
            input_shape=(sequence_length, num_features)
        ),
        tf.keras.layers.Dropout(dropout),
        
        # Second LSTM layer
        tf.keras.layers.LSTM(
            hidden_size,
            return_sequences=False
        ),
        tf.keras.layers.Dropout(dropout),
        
        # Output layer
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    
    return model


def prepare_lstm_sequences(df: pl.DataFrame,
                           feature_cols: List[str],
                           sequence_length: int,
                           target_col: str = "frequency",
                           tta: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        sequence_length: Length of input sequences
        target_col: Target column name
        tta: Time to alert in seconds
    
    Returns:
        Tuple of (X_sequences, y_targets)
    """
    # Extract features and target
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    
    # Create sequences
    X_sequences = []
    y_targets = []
    
    for i in range(len(df) - sequence_length - tta):
        X_sequences.append(X[i:i + sequence_length])
        y_targets.append(y[i + sequence_length + tta])
    
    return np.array(X_sequences), np.array(y_targets)
```

### 4.5.3 Model Evaluation

The [`evaluate_models.py`](Implementation/evaluate_models.py:1) module implements evaluation metrics:

```python
def evaluate_quantile_models(lower_model: lgb.Booster,
                             upper_model: lgb.Booster,
                             X_test: pl.DataFrame,
                             y_test: pl.Series) -> Dict[str, float]:
    """
    Evaluate quantile regression models.
    
    Args:
        lower_model: Trained lower bound model
        upper_model: Trained upper bound model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred_lower = lower_model.predict(X_test.to_numpy())
    y_pred_upper = upper_model.predict(X_test.to_numpy())
    
    # Calculate metrics
    metrics = {}
    
    # MAE for lower and upper bounds
    metrics["mae_lower"] = np.mean(np.abs(y_test - y_pred_lower))
    metrics["mae_upper"] = np.mean(np.abs(y_test - y_pred_upper))
    
    # Pinball loss
    metrics["pinball_loss_0.1"] = pinball_loss(y_test, y_pred_lower, quantile=0.1)
    metrics["pinball_loss_0.9"] = pinball_loss(y_test, y_pred_upper, quantile=0.9)
    
    # PICP (Prediction Interval Coverage Probability)
    within_interval = (y_test >= y_pred_lower) & (y_test <= y_pred_upper)
    metrics["picp"] = np.mean(within_interval)
    
    # MPIW (Mean Prediction Interval Width)
    metrics["mpiw"] = np.mean(y_pred_upper - y_pred_lower)
    
    # Calibration
    metrics["calibration"] = abs(metrics["picp"] - 0.80)
    
    return metrics


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate pinball loss (quantile loss).
    
    Args:
        y_true: Actual values
        y_pred: Predicted quantiles
        quantile: Target quantile
    
    Returns:
        Average pinball loss
    """
    errors = y_true - y_pred
    pinball = np.where(errors >= 0, quantile * errors, (1 - quantile) * (-errors))
    return np.mean(pinball)
```

---

## 4.6 Streamlit Dashboard Implementation

### 4.6.1 Dashboard Architecture

The Streamlit dashboard ([`app.py`](Implementation/app.py:1)) provides a real-time interface for grid stability monitoring:

**Figure 4.5: Streamlit Dashboard Interface**

```
┌─────────────────────────────────────────────────────────────────┐
│  GridGuardian : Grid Stability Monitoring Dashboard           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Grid Frequency Prediction (Last 1 Hour)                │   │
│  │                                                         │   │
│  │  50.2 ┤                                    ╭───         │   │
│  │       │                              ╭─────╯             │   │
│  │  50.0 ┤                        ╭─────╯                  │   │
│  │       │                  ╭─────╯                        │   │
│  │  49.8 ┤            ╭─────╯  ▲ Alert Threshold           │   │
│  │       │      ╭─────╯       │                           │   │
│  │  49.6 ┤  ╭───╯  Lower Bound (10th percentile)          │   │
│  │       │ ╱  Upper Bound (90th percentile)                │   │
│  │  49.4 ┼─┴────────────────────────────────────────────   │   │
│  │       └─────────────────────────────────────────────    │   │
│  │        17:00  17:15  17:30  17:45  18:00                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────────────────────────┐   │
│  │  Alert Status│  │  SHAP Feature Importance             │   │
│  │              │  │                                      │   │
│  │  ● NORMAL    │  │  rocof_1s        ████████  +0.02     │   │
│  │              │  │  wind_ramp_rate  ████      -0.01     │   │
│  │  Next check: │  │  inertia_cost    ███       +0.005    │   │
│  │  in 8 seconds│  │  lag_1s          ██        -0.002    │   │
│  └──────────────┘  └──────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6.2 Caching Implementation

Streamlit's caching mechanism is used to avoid redundant computations:

```python
@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_latest_frequency_data():
    """Fetch latest frequency data with caching."""
    return fetch_frequency_data(
        start_date=(datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d")
    )


@st.cache_resource
def load_models():
    """Load trained models (cached across reruns)."""
    lower_model = lgb.Booster(model_file="models/lower_bound.txt")
    upper_model = lgb.Booster(model_file="models/upper_bound.txt")
    return lower_model, upper_model
```

### 4.6.3 Source Code Hashing for Cache Invalidation

To ensure the dashboard uses the latest models when they are retrained, source code hashing is implemented:

```python
def get_source_hash() -> str:
    """
    Calculate hash of source code files.
    
    Returns:
        MD5 hash of all source files
    """
    hash_md5 = hashlib.md5()
    source_files = ["src/data_loader.py", "src/feature_engineering.py", 
                    "src/model_trainer.py", "app.py"]
    
    for filepath in source_files:
        with open(filepath, "rb") as f:
            hash_md5.update(f.read())
    
    return hash_md5.hexdigest()


@st.cache_data(ttl=10)
def get_predictions(_source_hash: str):
    """
    Get predictions with source code hash for cache invalidation.
    
    Args:
        _source_hash: Hash of source files (underscore prefix forces cache key)
    
    Returns:
        Predictions and uncertainty bands
    """
    # Models will be reloaded if source code changes
    lower_model, upper_model = load_models()
    
    # Get latest data and make predictions
    df = get_latest_frequency_data()
    df = engineer_features(df)
    
    predictions = {
        "lower": lower_model.predict(df[FEATURE_COLS].to_numpy()),
        "upper": upper_model.predict(df[FEATURE_COLS].to_numpy()),
        "actual": df["frequency"].to_numpy()
    }
    
    return predictions
```

### 4.6.4 SHAP Integration

SHAP explainability is integrated into the dashboard:

**Figure 4.6: SHAP Waterfall Plot in Dashboard**

```
┌─────────────────────────────────────────────────────────────────┐
│  SHAP Explanation for Current Prediction                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base Value: 49.95 Hz                                           │
│                                                                 │
│  rocof_1s = -0.02    ──────────────────────► +0.015 Hz         │
│  wind_ramp = -0.5    ────────────────────► -0.008 Hz           │
│  inertia_cost = 150  ────────────────────► +0.005 Hz           │
│  lag_1s = 49.93      ────────────────────► -0.002 Hz           │
│  hour_of_day = 17    ────────────────────► +0.001 Hz           │
│                                                                 │
│  ──────────────────────────────────────────────────────────     │
│  Output: 49.961 Hz                                              │
│                                                                 │
│  ▲ High values (red) increase prediction                        │
│  ▼ Low values (blue) decrease prediction                        │
└─────────────────────────────────────────────────────────────────┘
```

```python
def generate_shap_explanation(model: lgb.Booster, X_sample: pl.DataFrame):
    """
    Generate SHAP explanation for model predictions.
    
    Args:
        model: Trained LightGBM model
        X_sample: Sample of features for explanation
    
    Returns:
        SHAP values and expected value
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample.to_numpy())
    
    return shap_values, explainer.expected_value


def display_shap_waterfall(shap_values: np.ndarray, 
                           feature_names: List[str],
                           base_value: float):
    """
    Display SHAP waterfall plot.
    
    Args:
        shap_values: SHAP values for a single prediction
        feature_names: Names of features
        base_value: Base value (expected output)
    """
    st.subheader("SHAP Feature Attribution")
    
    # Create waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    
    cumulative = base_value
    for idx in sorted_indices:
        value = shap_values[idx]
        cumulative += value
        
        color = "red" if value > 0 else "blue"
        ax.barh([feature_names[idx]], [value], color=color, alpha=0.7)
    
    ax.axvline(x=base_value, linestyle="--", color="gray", label="Base Value")
    ax.set_xlabel("Contribution to Prediction (Hz)")
    ax.set_title("Feature Contributions to Frequency Prediction")
    ax.legend()
    
    st.pyplot(fig)
```

### 4.6.5 Alert Logic

Alerts are triggered when the lower quantile prediction breaches the threshold:

```python
def check_alert_status(lower_prediction: float, threshold: float = 49.8) -> Dict:
    """
    Check if alert should be triggered.
    
    Args:
        lower_prediction: Lower bound (10th percentile) prediction
        threshold: Frequency threshold for alert
    
    Returns:
        Dictionary with alert status and details
    """
    if lower_prediction < threshold:
        return {
            "status": "ALERT",
            "severity": "HIGH" if lower_prediction < 49.5 else "MEDIUM",
            "message": f"Lower bound {lower_prediction:.3f} Hz below threshold {threshold} Hz",
            "timestamp": datetime.now()
        }
    else:
        return {
            "status": "NORMAL",
            "severity": "NONE",
            "message": f"Lower bound {lower_prediction:.3f} Hz above threshold {threshold} Hz",
            "timestamp": datetime.now()
        }
```

---

## 4.7 Tool Selection Justification

### 4.7.1 Overview of Tool Selection

GridGuardian utilises a carefully selected technology stack, each component chosen for its specific strengths in the context of power grid stability prediction.

**Table 4.1: Tool Selection Justification Matrix**

| Tool | Purpose | Justification | Academic Reference |
|------|---------|---------------|-------------------|
| Python | Programming language | Rich ecosystem for ML and data science; extensive documentation | Van Rossum and Drake, 2009 |
| Polars | Data processing | Superior performance vs pandas; lazy evaluation; parallel execution | Polars Development Team, 2024 |
| LightGBM | ML model | Fast training; native quantile regression; feature importance | Ke et al., 2017 |
| TensorFlow | Deep learning | Industry-standard DL framework; LSTM support | Abadi et al., 2016 |
| Streamlit | Dashboard | Rapid prototyping; real-time updates; Python-native | Streamlit Inc., 2023 |
| SHAP | Explainability | Game-theoretic foundation; unified framework; TreeExplainer | Lundberg and Lee, 2017 |

### 4.7.2 Python

**Rationale:**
Python is selected as the primary programming language due to:
1. **Ecosystem:** Comprehensive libraries for data science (Polars, NumPy, pandas), machine learning (LightGBM, TensorFlow), and visualisation (Matplotlib, Plotly).
2. **Community:** Large and active community providing extensive documentation and support.
3. **Readability:** Clean syntax facilitates code maintenance and collaboration.

**Reference:**
Van Rossum, G. and Drake, F.L. (2009) 'Python 3 Reference Manual', Scotts Valley, CA: CreateSpace.

### 4.7.3 Polars

**Rationale:**
Polars is selected over pandas for data processing due to:
1. **Performance:** Rust backend provides 10-100x speedup for large datasets (Polars Development Team, 2024).
2. **Lazy Evaluation:** Query optimisation reduces unnecessary computations.
3. **Parallel Execution:** Automatic parallelisation across CPU cores.
4. **join_asof:** Specialised temporal merge for time-series data.

**Reference:**
Polars Development Team (2024) *Polars: Fast Data Frames in Rust*. Available at: https://www.pola.rs/

### 4.7.4 LightGBM

**Rationale:**
LightGBM is selected as the primary ML model due to:
1. **Efficiency:** Gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB) enable fast training (Ke et al., 2017).
2. **Quantile Regression:** Native support for quantile loss via 'regression_quantile' objective.
3. **Feature Importance:** Built-in metrics for interpretability.
4. **Scalability:** Handles large datasets with millions of rows efficiently.

**Reference:**
Ke, G. et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', *Advances in Neural Information Processing Systems*, 30, pp. 3146-3154.

### 4.7.5 TensorFlow

**Rationale:**
TensorFlow is selected for LSTM implementation due to:
1. **Industry Standard:** Widely adopted deep learning framework with extensive documentation.
2. **LSTM Support:** Built-in LSTM layers with optimised implementations.
3. **Ecosystem:** Integration with Keras for high-level API.
4. **Deployment:** Support for model export and serving.

**Reference:**
Abadi, M. et al. (2016) 'TensorFlow: A System for Large-Scale Machine Learning', *Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation*, pp. 265-283.

### 4.7.6 Streamlit

**Rationale:**
Streamlit is selected for dashboard development due to:
1. **Rapid Prototyping:** Python-native syntax enables quick development.
2. **Real-Time Updates:** Automatic rerun on user interaction.
3. **Caching:** Built-in caching mechanisms for performance.
4. **Deployment:** Simple deployment to Streamlit Cloud.

**Reference:**
Streamlit Inc. (2023) *Streamlit: A Faster Way to Build Data Apps*. Available at: https://streamlit.io/

### 4.7.7 SHAP

**Rationale:**
SHAP is selected for explainability due to:
1. **Theoretical Foundation:** Based on Shapley values from cooperative game theory (Lundberg and Lee, 2017).
2. **Unified Framework:** Consistent explanations across different model types.
3. **TreeExplainer:** Optimised implementation for tree-based models.
4. **Visualisations:** Rich set of plot types (waterfall, summary, dependence).

**Reference:**
Lundberg, S.M. and Lee, S.I. (2017) 'A Unified Approach to Interpreting Model Predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

---

## 4.8 Chapter Summary

This chapter has presented the technical implementation details of GridGuardian . The key implementation components are:

1. **System Architecture:** Layered architecture separating presentation, application, and data layers.

2. **Data Pipeline:** Polars-based pipeline with `join_asof` temporal merging, caching in Parquet format, and data validation.

3. **Feature Engineering:** RoCoF calculation with smoothing, OpSDA for wind ramp detection, lag features, and temporal features.

4. **Model Training:** LightGBM quantile regression for probabilistic forecasting, LSTM as deep learning benchmark, comprehensive evaluation metrics.

5. **Dashboard:** Streamlit web application with real-time predictions, uncertainty bands, alert indicators, and SHAP explainability.

6. **Tool Justification:** Python, Polars, LightGBM, TensorFlow, Streamlit, and SHAP selected for specific strengths in the context of grid stability prediction.

The implementation demonstrates that physics-informed machine learning, combined with probabilistic forecasting and real-time explainability, is feasible for operational grid stability monitoring. The next chapter presents the empirical results and evaluation of the system.

---

