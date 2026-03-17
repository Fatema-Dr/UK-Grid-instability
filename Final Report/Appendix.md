# Appendix A: Code Listings

## A.1 Optimized Swinging Door Algorithm (OpSDA)

The following code listing shows the core implementation of the Swinging Door Algorithm used for wind ramp rate detection. This algorithm compresses time-series data while preserving significant trends and ramp events.

```python
# src/opsda.py
# Implementation of the Swinging Door Algorithm
# Adapted from a public implementation.

def compress(data, width):
    """
    Compresses a list of (timestamp, value) tuples using the Swinging Door Algorithm.
    
    Args:
        data: List of (timestamp, value) tuples
        width: Tolerance parameter controlling compression aggressiveness
    
    Returns:
        List of compressed (timestamp, value) tuples
    """
    if not data:
        return []

    compressed_data = [data[0]]
    start_point_index = 0
    
    for i in range(1, len(data)):
        current_point = data[i]
        pivot_point = data[start_point_index]
        
        # Form a "door" from the pivot point to the current point
        upper_bound_slope = (pivot_point[1] + width - current_point[1]) / (pivot_point[0] - current_point[1]) if pivot_point[0] != current_point[0] else float('inf')
        lower_bound_slope = (pivot_point[1] - width - current_point[1]) / (pivot_point[0] - current_point[0]) if pivot_point[0] != current_point[0] else float('-inf')

        # Check all intermediate points
        for j in range(start_point_index + 1, i):
            intermediate_point = data[j]
            slope = (pivot_point[1] - intermediate_point[1]) / (pivot_point[0] - intermediate_point[0]) if pivot_point[0] != intermediate_point[0] else float('inf')
            
            if slope > upper_bound_slope or slope < lower_bound_slope:
                # Point is outside the door, so we record the previous point and start a new door
                compressed_data.append(data[i-1])
                start_point_index = i - 1
                break
    
    # Add the last point
    compressed_data.append(data[-1])
    
    return compressed_data
```

**Listing A.1:** Swinging Door Algorithm implementation for time-series compression and ramp detection.

---

## A.2 RoCoF Calculation

The Rate of Change of Frequency (RoCoF) is a critical feature for grid stability prediction. The following code shows its calculation with rolling average smoothing.

```python
# src/feature_engineering.py
# RoCoF calculation with smoothing

def calculate_rocof(df):
    """
    Calculates the Rate of Change of Frequency (RoCoF).
    
    RoCoF is calculated as the derivative of frequency with respect to time,
    smoothed with a 5-second rolling average to reduce noise.
    
    Args:
        df: DataFrame with 'grid_frequency' column
    
    Returns:
        DataFrame with added 'rocof' column
    """
    # Calculate derivative (difference) of frequency
    df["rocof"] = df["grid_frequency"].diff()
    
    # Apply 5-second rolling average for smoothing
    df["rocof"] = df["rocof"].rolling(window=5, min_periods=1).mean().fillna(0)
    
    return df
```

**Listing A.2:** RoCoF calculation with Savitzky-Golay-like smoothing.

---

## A.3 Wind Ramp Rate Calculation

The wind ramp rate is calculated using the OpSDA algorithm to detect significant wind speed changes that may impact grid stability.

```python
# src/feature_engineering.py
# Wind ramp rate calculation using OpSDA

def calculate_wind_ramp_rate(df):
    """
    Calculates the wind ramp rate using the Swinging Door Algorithm.
    
    Args:
        df: DataFrame with 'timestamp' and 'wind_speed' columns
    
    Returns:
        DataFrame with added 'wind_ramp_rate' column
    """
    # Convert timestamps to Unix time for slope calculation
    weather_data = df[["timestamp", "wind_speed"]].drop_duplicates(subset=["timestamp"]).copy()
    weather_data['unix_ts'] = weather_data['timestamp'].astype(np.int64) // 1_000_000_000
    
    # Apply the Swinging Door algorithm
    data_tuples = list(weather_data[['unix_ts', 'wind_speed']].itertuples(index=False, name=None))
    compressed = opsda.compress(data_tuples, width=OPSDA_WIDTH)
    
    compressed_df = pl.DataFrame(compressed, schema=["unix_ts", "wind_speed"], orient="row")
    compressed_df = compressed_df.with_columns(
        pl.col("unix_ts").cast(pl.Datetime).cast(pl.Datetime("us", "UTC"))
    ).rename({"unix_ts": "timestamp"})

    # Calculate ramp rate (slope between compressed points)
    compressed_df = compressed_df.with_columns(
        ((pl.col("wind_speed").diff()) / (pl.col("timestamp").diff().dt.total_seconds())).alias("wind_ramp_rate")
    )
    
    # Merge ramp rate back into the main dataframe
    df = pd.merge_asof(
        df,
        compressed_df[["timestamp", "wind_ramp_rate"]],
        on="timestamp",
        direction="backward"
    )
    return df
```

**Listing A.3:** Wind ramp rate calculation using OpSDA compression.

---

## A.4 LightGBM Quantile Regression Training

The following code shows the training of LightGBM models for quantile regression, providing uncertainty bands for frequency predictions.

```python
# src/model_trainer.py
# LightGBM quantile regression training

def train_quantile_regression(df, feature_cols, quantile_alphas=[0.1, 0.9]):
    """
    Trains LightGBM models for quantile regression.
    
    Args:
        df: Training DataFrame
        feature_cols: List of feature column names
        quantile_alphas: List of quantiles to predict (e.g., [0.1, 0.9] for 80% prediction interval)
    
    Returns:
        Dictionary of trained LGBMRegressor models keyed by quantile
    """
    X = df[feature_cols]
    y = df["target_freq_next"]
    
    models = {}
    for alpha in quantile_alphas:
        model = lgbm.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model.fit(X, y)
        models[alpha] = model
    
    return models
```

**Listing A.4:** LightGBM quantile regression training for uncertainty quantification.

---

## A.5 SHAP Explainability Integration

The following code shows how SHAP values are computed and integrated into the Streamlit dashboard for real-time explainability.

```python
# app.py
# SHAP explainability integration

def compute_shap_values(model, X_sample):
    """
    Computes SHAP values for model explanations.
    
    Args:
        model: Trained LightGBM model
        X_sample: Sample of feature data for explanation
    
    Returns:
        SHAP values array
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values

def display_shap_summary(shap_values, feature_names):
    """
    Displays SHAP summary plot in Streamlit.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
    """
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, plot_size="auto")
```

**Listing A.5:** SHAP explainability integration for real-time model interpretation.

---

## A.6 Configuration Parameters

The configuration file defines key parameters for the GridGuardian v2 system.

```python
# src/config.py
# Configuration for GridGuardian v2

# Time to Alert (in seconds) for the prediction target
TTA_SECONDS = 10

# OpSDA width parameter for wind ramp rate calculation
OPSDA_WIDTH = 0.5

# Lag intervals for frequency features (in seconds)
LAG_INTERVALS_SECONDS = [1, 5, 60]

# Quantiles for the uncertainty bands
QUANTILE_ALPHAS = [0.1, 0.9]

# Date to split the training and testing data
SPLIT_DATE = "2019-08-09 00:00:00"
END_TEST_DATE = "2019-08-10 00:00:00"

# LightGBM feature columns
LGBM_FEATURE_COLS = [
    "grid_frequency", 
    "rocof", 
    "volatility_10s",
    "wind_speed",
    "wind_ramp_rate",
    "solar_radiation", 
    "hour",
    "renewable_penetration_ratio"
] + [f"lag_{lag}s" for lag in LAG_INTERVALS_SECONDS]
```

**Listing A.6:** Configuration parameters for GridGuardian v2.

---

**Appendix A Word Count: 2,200**

---

*End of Appendix A*

---

# Appendix B: Additional Figures and Tables

## B.1 System Architecture Diagram

![System Architecture](../system_architecture.png)

**Figure B.1:** GridGuardian v2 system architecture showing data flow from NESO API and Open-Meteo API through the Polars data pipeline, model training, and Streamlit dashboard deployment.

---

## B.2 Data Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   NESO API      │     │  Open-Meteo API │     │  OpSDA Grid     │
│  (Frequency)    │     │   (Weather)     │     │   (Inertia)     │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Polars Data Pipeline  │
                    │  - join_asof merging    │
                    │  - Feature engineering  │
                    │  - Target generation    │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  LightGBM         │  │  LSTM             │  │  SARIMAX          │
│  Quantile         │  │  (TensorFlow)     │  │  (Statsmodels)    │
│  Regression       │  │                   │  │                   │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Streamlit Dashboard   │
                    │  - Real-time predictions│
                    │  - SHAP explanations    │
                    │  - Alert visualization  │
                    └─────────────────────────┘
```

**Figure B.2:** Data flow diagram showing the complete pipeline from data sources to dashboard visualization.

---

## B.3 Model Performance Comparison Table

| Model | MAE (Hz) | RMSE (Hz) | R² Score | Training Time (s) |
|-------|----------|-----------|----------|-------------------|
| LightGBM (Lower) | 0.033 | 0.045 | 0.942 | 12.3 |
| LightGBM (Upper) | 0.018 | 0.028 | 0.968 | 11.8 |
| LSTM | 0.045 | 0.062 | 0.891 | 145.2 |
| SARIMAX | 0.062 | 0.089 | 0.756 | 8.7 |

**Table B.1:** Model performance comparison on test set (August 9-10, 2019).

---

## B.4 Feature Importance Table (SHAP)

| Feature | Mean | Std | Min | Max | SHAP Importance |
|---------|------|-----|-----|-----|-----------------|
| grid_frequency | 49.98 | 0.12 | 49.45 | 50.15 | 0.42 |
| rocof | 0.001 | 0.015 | -0.12 | 0.09 | 0.28 |
| wind_speed | 8.5 | 4.2 | 0.5 | 22.1 | 0.12 |
| wind_ramp_rate | 0.02 | 0.08 | -0.45 | 0.38 | 0.08 |
| solar_radiation | 245 | 312 | 0 | 890 | 0.05 |
| renewable_penetration_ratio | 0.73 | 0.36 | 0.04 | 1.89 | 0.03 |
| volatility_10s | 0.008 | 0.006 | 0.001 | 0.045 | 0.02 |

**Table B.2:** Feature statistics and SHAP-based importance ranking.

---

## B.5 Uncertainty Quantification Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| PICP (Prediction Interval Coverage Probability) | 77.5% | 80% | Slightly Below |
| MPIW (Mean Prediction Interval Width) | 0.051 Hz | < 0.1 Hz | Pass |
| Calibration Error | 0.025 | < 0.05 | Pass |
| Pinball Loss (α=0.1) | 0.012 | - | - |
| Pinball Loss (α=0.9) | 0.008 | - | - |

**Table B.3:** Uncertainty quantification metrics for quantile regression predictions.

---

## B.6 Early Warning Performance

| Metric | Value | Description |
|--------|-------|-------------|
| Time to Nadir (Alert Trigger) | 3 seconds | Lead time before frequency nadir |
| Recall | 94% | True positive rate for instability detection |
| Precision | 79% | Positive predictive value |
| F1 Score | 0.86 | Harmonic mean of precision and recall |
| False Positive Rate | 21% | Rate of incorrect alerts |

**Table B.4:** Early warning system performance on August 9, 2019 blackout event.

---

## B.7 Ablation Study Results

| Feature Set | MAE (Hz) | % Increase vs Full |
|-------------|----------|-------------------|
| Full Feature Set | 0.033 | Baseline |
| Without RoCoF | 0.067 | +103% |
| Without OpSDA Ramp Rate | 0.041 | +24% |
| Without Lag Features | 0.048 | +45% |
| Without Renewable Penetration | 0.045 | +36% |
| Only Raw Frequency | 0.089 | +169% |

**Table B.5:** Ablation study showing impact of individual feature groups on prediction accuracy.

---

**Appendix B Word Count: 1,200**

---

*End of Appendix B*

---

# Appendix C: Data Dictionary

## C.1 Data Sources

### C.1.1 NESO National Frequency Data

| Field | Type | Description | Unit | Resolution |
|-------|------|-------------|------|------------|
| timestamp | DateTime | Timestamp of frequency measurement | UTC datetime | 1 second |
| grid_frequency | Float | Measured grid frequency | Hz | - |

**Source:** National Energy System Operator (NESO) CKAN API  
**API Endpoint:** `https://api.neso.energy/api/3/action`  
**Resource ID:** `819a0821-cc6d-4909-a1ea-7dba5cab0c33` (August 2019)

---

### C.1.2 Open-Meteo Weather Data

| Field | Type | Description | Unit | Resolution |
|-------|------|-------------|------|------------|
| timestamp | DateTime | Timestamp of weather measurement | UTC datetime | 1 hour |
| wind_speed | Float | Wind speed at 10m height | m/s | - |
| solar_radiation | Float | Global horizontal irradiance | W/m² | - |
| temperature | Float | Air temperature at 2m | °C | - |

**Source:** Open-Meteo API  
**API Endpoint:** `https://api.open-meteo.com/v1/forecast`  
**Location:** Great Britain (54.0, -2.0)

---

### C.1.3 NESO Inertia Data

| Field | Type | Description | Unit | Resolution |
|-------|------|-------------|------|------------|
| timestamp_date | Date | Date of inertia measurement | Date | Daily |
| inertia_cost | Float | System inertia cost signal | £/MW | - |
| inertia_level | Float | Estimated system inertia level | MWs | - |

**Source:** National Energy System Operator (NESO) CKAN API  
**Resource ID:** `620491fa-ae1b-45b3-baa0-6e87c2d574cf` (2019)

---

## C.2 Engineered Features

### C.2.1 Primary Features

| Feature | Type | Description | Calculation |
|---------|------|-------------|-------------|
| rocof | Float | Rate of Change of Frequency | `diff(grid_frequency).rolling(5).mean()` |
| wind_ramp_rate | Float | Wind speed ramp rate (OpSDA) | `slope(compressed_wind_data)` |
| renewable_penetration_ratio | Float | Synthetic renewable proxy | `(wind_speed * 3000) / 35000` |
| volatility_10s | Float | 10-second frequency volatility | `std(grid_frequency, window=10)` |
| hour | Integer | Hour of day (0-23) | `extract_hour(timestamp)` |

---

### C.2.2 Lag Features

| Feature | Type | Description | Interval |
|---------|------|-------------|----------|
| lag_1s | Float | Frequency 1 second ago | 1 second |
| lag_5s | Float | Frequency 5 seconds ago | 5 seconds |
| lag_60s | Float | Frequency 60 seconds ago | 60 seconds |

---

### C.2.3 Target Variables

| Target | Type | Description | Calculation |
|--------|------|-------------|-------------|
| target_is_unstable | Boolean | Grid instability flag | `freq_next < 49.8` |
| target_freq_next | Float | Frequency at T+10s | `frequency at timestamp + 10s` |

---

## C.3 Model Outputs

### C.3.1 LightGBM Quantile Outputs

| Output | Type | Description | Quantile |
|--------|------|-------------|----------|
| freq_lower | Float | Lower bound prediction | α = 0.1 |
| freq_point | Float | Point prediction | α = 0.5 |
| freq_upper | Float | Upper bound prediction | α = 0.9 |

---

### C.3.2 SHAP Explanations

| Output | Type | Description |
|--------|------|-------------|
| shap_values | Float array | SHAP values for each feature |
| base_value | Float | Model output expectation |
| feature_names | String array | Feature names for explanation |

---

## C.4 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| TTA_SECONDS | 10 | Time-to-Alert prediction horizon (seconds) |
| OPSDA_WIDTH | 0.5 | OpSDA tolerance parameter |
| LAG_INTERVALS_SECONDS | [1, 5, 60] | Lag feature intervals (seconds) |
| QUANTILE_ALPHAS | [0.1, 0.9] | Quantile levels for uncertainty bands |
| SPLIT_DATE | "2019-08-09 00:00:00" | Training/test split date |
| END_TEST_DATE | "2019-08-10 00:00:00" | End date for test period |

---

## C.5 Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| MAE | `mean(|y - ŷ|)` | Mean Absolute Error |
| RMSE | `sqrt(mean((y - ŷ)²))` | Root Mean Square Error |
| R² | `1 - SS_res/SS_tot` | Coefficient of Determination |
| PICP | `mean(y ∈ [ŷ_lower, ŷ_upper])` | Prediction Interval Coverage |
| MPIW | `mean(ŷ_upper - ŷ_lower)` | Mean Prediction Interval Width |
| Pinball Loss | `ρ_α(y - ŷ)` | Quantile regression loss |
| F1 Score | `2 * (P * R) / (P + R)` | Harmonic mean of precision and recall |

---

**Appendix C Word Count: 1,400**

---

*End of Appendix C*

---

# Appendix D: Glossary

| Term | Definition |
|------|------------|
| **RoCoF** | Rate of Change of Frequency - a measure of how quickly grid frequency changes, measured in Hz/s |
| **OpSDA** | Optimized Swinging Door Algorithm - a time-series compression algorithm for detecting ramp events |
| **SHAP** | SHapley Additive exPlanations - a game-theoretic approach to explaining machine learning model outputs |
| **PICP** | Prediction Interval Coverage Probability - the proportion of true values falling within predicted uncertainty bands |
| **MPIW** | Mean Prediction Interval Width - the average width of prediction intervals, measuring uncertainty band tightness |
| **FFR** | Fast Frequency Response - a grid service providing rapid frequency support from inverter-based resources |
| **ESR** | Enhanced Frequency Response - a grid service for maintaining frequency within operational limits |
| **LSTM** | Long Short-Term Memory - a type of recurrent neural network for sequence modelling |
| **LightGBM** | Light Gradient Boosting Machine - a gradient boosting framework using tree-based learning |
| **SARIMAX** | Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors |
| **Inertia** | The resistance of rotating masses to changes in speed, providing natural frequency stability |
| **Nadir** | The lowest point reached by grid frequency following a disturbance |
| **NESO** | National Energy System Operator - the operator of Great Britain's electricity transmission system |
| **CKAN** | Comprehensive Knowledge Archive Network - an open-source data portal platform used by NESO |

---
