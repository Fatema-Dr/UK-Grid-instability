#let formal_box(title, content) = {
  block(
    width: 100%,
    stroke: 0.5pt + black,
    inset: 15pt,
    radius: 0pt,
    fill: none,
    [
      #set align(left)
      #text(weight: "bold", size: 1.2em)[#title]
      #v(2pt)
      #line(length: 100%, stroke: 0.5pt + gray)
      #v(8pt)
      #content
    ]
  )
}

= Appendices

== Appendix A: Initial Project Proposal

#formal_box("Initial Project Specification (BSc 2025-26)", [
  #grid(
    columns: (1fr, 2fr),
    row-gutter: 12pt,
    [*Student Number:*], [2604383],
    [*Proposed Title:*], [Forecasting UK Frequency Grid Instability: A Data-Driven Alert System],
    [*Proposed Aim:*], [Predict UK grid instability from weather and frequency data for proactive renewable energy alert system.],
  )

  #v(15pt)
  *Proposed Objectives:* \
  By the end of this project, I will be able:
  1. To research and select a robust proxy variable (e.g., System Frequency deviation > ±0.2 Hz) for grid instability using National Grid ESO data.
  2. To build a demo of data pipeline to integrate, clean, and feature-engineer National Grid ESO and HadUK-Grid datasets using polars.
  3. To benchmark three forecasting models (SARIMAX, LightGBM, LSTM) using RMSE, MAE, and training time.
  4. To incorporate probabilistic forecasting (quantile regression) to support uncertainty-aware alerts.
  5. To develop a proof-of-concept decision-support system integrating a rule-based alert engine and Streamlit dashboard.
  6. To validate the alert system on historical data, including the August 9, 2019 blackout event.
  7. Critically evaluate the system’s performance using SHAP values and assess deployment viability.

  #v(10pt)
  *Proposed Rationale:* \
  The UK's transition to renewable energy introduces significant grid volatility. Currently, grid operators use costly, high-carbon reactive measures to prevent blackouts. This project addresses this gap by developing a proactive solution that forecasts instability before it happens, allowing for planned, cost-effective, and low-carbon interventions.

  #v(10pt)
  *Facilities Required:* \
  Python (Polars, Pandas, Scikit-learn, LightGBM, TensorFlow), Streamlit, HadUK-Grid, National Grid ESO API.
  #v(15pt)
  #line(length: 100%, stroke: 0.5pt + gray)
  #v(10pt)
])

*Project Reflection & Outcomes:* \
The project evolved from the initial proposal's focus on SARIMAX and LightGBM point forecasting to a *Physics-Informed LightGBM Quantile Regression* model, with an LSTM Monte Carlo Dropout model serving as a probabilistic baseline. SARIMAX was found to be insufficient for capturing the non-linear, high-frequency transients observed during grid disturbances. The final implementation incorporates physics-informed features (Swing Equation logic, OpSDA wind compression) and a multi-signal alert fusion engine to provide a robust dashboard solution.

#v(10pt)
*Achieved Technical Deliverables:*
#table(
  columns: (1fr, 2fr),
  align: (left, left),
  stroke: 0.5pt + gray,
  table.header([*Original Objective*], [*Final Implementation Outcome*]),
  [1. Select robust proxy], [Established instability threshold at 49.85 Hz with physics-informed triggers (RoCoF & Acceleration).],
  [2. Data pipeline], [Engineered a high-performance Polars ETL pipeline merging 1s frequency, HadUK weather, and inertia data.],
  [3. Benchmark models], [LightGBM and LSTM benchmarked; Result: LightGBM quantile regression achieved sub-second inference, outperforming the deeper LSTM baseline.],
  [4. Probabilistic alerts], [Implemented Quantile Regression (α=0.1 to 0.9) with post-hoc Isotonic Calibration for reliable uncertainty.],
  [5. Streamlit Dashboard], [Developed real-time "Control Room" interface with probabilistic bounds and SHAP-based risk driver analysis.],
  [6. Blackout Validation], [Validated against August 9, 2019 event; provided high-fidelity reactive identification and post-nadir forensic confirmation.],
  [7. Critical Evaluation], [Conducted SHAP global/local importance analysis and out-of-season robustness testing on Winter data.],
)

#pagebreak()

== Appendix B: Ethical Approval

This project utilizes publicly available open datasets from National Grid ESO and Open-Meteo, classifying it as primary quantitative analysis. As the study does not involve human subjects or the collection of sensitive personal data, it qualifies for the internal ethical approval process for non-human research.

The following documentation confirms the internal ethical clearance obtained for the study:

#figure(
  image("../figures/appendix-C-1.jpeg", width: 80%),
  caption: [Ethical Approval Form (Page 1)],
)

#figure(
  image("../figures/appendix-C-2.jpeg", width: 80%),
  caption: [Ethical Approval Form (Page 2)],
)

#figure(
  image("../figures/appendix-C-3.jpeg", width: 80%),
  caption: [Ethical Approval Form (Page 3)], 
)

#figure(
  image("../figures/appendix-C-4.jpeg", width: 80%),
  caption: [Ethical Approval Form (Page 4)],
)

#figure(   
  image("../figures/appendix-C-5.jpeg", width: 80%),
  caption: [Ethical Approval Form (Page 5)], 
)

#pagebreak()

== Appendix C: Dashboard Outputs

This appendix provides a visual walkthrough of the GridGuardian "Control Room" dashboard. The interface translates complex, high-resolution machine learning outputs—such as LightGBM quantile regression bounds, multi-signal physics heuristics, and SHAP-based feature importance—into a user-friendly, real-time diagnostic tool for grid operators.

The Control Room dashboard is structured to provide intuitive, real-time oversight of grid stability. Key features include:
- *Time Navigation (Sidebar):* Allows operators to jump to specific dates, scrub through time with 1s/5s precision, or use autoplay to seamlessly replay historical transients like the August 2019 blackout.
- *Top KPI Panel:* Provides an immediate summary of the system state, displaying real-time frequency, predicted bounds, and an aggregated status banner (e.g., Stable, Uncertainty, Alert) alongside the Time to Alert.
- *Real-Time Monitor (Main Chart):* Visualizes the raw telemetry against the LightGBM predicted quantile bounds (uncertainty ribbon), immediately highlighting when the grid is approaching physical thresholds.
- *Risk Drivers (XAI Chart):* Translates complex model features into an interpretable SHAP waterfall chart, instantly isolating whether inertia, wind ramps, or RoCoF is driving the current instability risk.

=== C.1: Normal Operational State (Pre-Fault)

#figure(
  image("../figures/pre-alert.png", width: 95%),
  caption: [Dashboard displaying stable grid conditions prior to the fault event.],
)

*System State:* Stable (Frequency near 50.0 Hz). \
*Insight:* The uncertainty bounds are narrow and the risk drivers show minimal physical stress, indicating healthy grid inertia and nominal volatility.

=== C.2: High Volatility & Model Uncertainty

#figure(
  image("../figures/model-uncertainity.png", width: 95%),
  caption: [The system detecting early-warning turbulence, triggering a "High Model Uncertainty" state.],
)

*System State:* High Model Uncertainty (Pre-Alert Buffer). \
*Insight:* As turbulence increases, the LightGBM confidence intervals dynamically widen. This alerts operators to elevated volatility before the absolute frequency breaches critical safety limits.

=== C.3: Critical Instability Alert (August 9, 2019 Nadir)

#figure(
  image("../figures/alert.png", width: 95%),
  caption: [Dashboard at the point of catastrophic failure, triggering an "Instability Alert" based on converged physics signals.],
)

*System State:* Instability Alert (Emergency Trigger). \
*Insight:* The multi-signal fusion engine converges as frequency drops and RoCoF plummets. The SHAP chart provides immediate diagnostic clarity, isolating extreme negative RoCoF as the dominant physical force driving the collapse.

=== C.4: Post-Fault Recovery Monitoring

#figure(
  image("../figures/post-alert.png", width: 95%),
  caption: [The recovery phase, demonstrating the model tracking the post-nadir stabilization.],
)

*System State:* Post-Fault Recovery. \
*Insight:* The uncertainty bounds expand to capture massive variance during the unprecedented transient phase, while the LSTM probability timeline (bottom chart) tracks the grid's gradual stabilization.

=== C.5: Model Health & Calibration Diagnostics

#figure(
  image("../figures/model-health-tab.png", width: 95%),
  caption: [The "Model Health" tab providing continuous transparency into the ML engine's performance.],
)

*System State:* Diagnostic Oversight. \
*Insight:* Operators can monitor rigorous real-time statistics (Pinball Loss, PICP Coverage, Confusion Matrix). This ensures the ML engine remains a transparent, trusted advisory tool rather than an opaque oracle.

#pagebreak()

== Appendix D: Code Snippets

This appendix presents the core implementation logic for the GridGuardian system, providing a technical reference for the feature engineering, probabilistic modeling, and real-time dashboard engine.

=== D.1: Physics-Informed Feature Engineering

The following snippets detail the physics-informed feature engineering pipeline, including the Optimised Swinging Door Algorithm (OpSDA) implementation and the derivation of causal multi-window RoCoF and wind ramp rates.

*Optimised Swinging Door Algorithm (Implementation: `opsda.py`)*

```python
def compress(data, width):
    """
    Compresses a list of (timestamp, value) tuples using the Swinging Door Algorithm.
    """
    if not data:
        return []

    compressed_data = [data[0]]
    start_point_index = 0
    
    for i in range(1, len(data)):
        current_point = data[i]
        pivot_point = data[start_point_index]
        
        # Form a "door" from the pivot point to the current point
        upper_bound_slope = (pivot_point[1] + width - current_point[1]) / (pivot_point[0] - current_point[0]) if pivot_point[0] != current_point[0] else float('inf')
        lower_bound_slope = (pivot_point[1] - width - current_point[1]) / (pivot_point[0] - current_point[0]) if pivot_point[0] != current_point[0] else float('-inf')

        # Check all intermediate points
        for j in range(start_point_index + 1, i):
            intermediate_point = data[j]
            slope = (pivot_point[1] - intermediate_point[1]) / (pivot_point[0] - intermediate_point[0]) if pivot_point[0] != intermediate_point[0] else float('inf')
            
            if slope > upper_bound_slope or slope < lower_bound_slope:
                compressed_data.append(data[i-1])
                start_point_index = i - 1
                break
    
    compressed_data.append(data[-1])
    return compressed_data
```

*Wind Ramp Rate and Causal RoCoF (Selection from: `feature_engineering.py`)*

```python
def calculate_wind_ramp_rate(df):
    weather_data = df[["timestamp", "wind_speed"]].drop_duplicates(subset=["timestamp"]).copy()
    weather_data['unix_ts'] = weather_data['timestamp'].astype(np.int64) // 1_000_000_000
    
    data_tuples = list(weather_data[['unix_ts', 'wind_speed']].itertuples(index=False, name=None))
    compressed = opsda.compress(data_tuples, width=OPSDA_WIDTH)
    
    compressed_df = pl.DataFrame(compressed, schema=["unix_ts", "wind_speed"], orient="row")
    # Calculate ramp rate (slope between compressed points)
    compressed_df = compressed_df.with_columns(
        ((pl.col("wind_speed").diff()) / (pl.col("timestamp").diff().dt.total_seconds())).alias("wind_ramp_rate")
    )
    return df.merge_asof(compressed_df, on="timestamp", direction="backward")

def create_features(df):
    # Causal (backward-only) RoCoF at multiple windows
    df["rocof_1s"]  = df["grid_frequency"].diff(1).fillna(0)
    df["rocof_5s"]  = ((df["grid_frequency"] - df["grid_frequency"].shift(5)) / 5.0).fillna(0)
    df["rocof_10s"] = ((df["grid_frequency"] - df["grid_frequency"].shift(10)) / 10.0).fillna(0)
    df["rocof_30s"] = ((df["grid_frequency"] - df["grid_frequency"].shift(30)) / 30.0).fillna(0)

    # RoCoF acceleration (second derivative) - detects worsening vs. recovering
    df["rocof_accel"] = df["rocof_5s"].diff(5).fillna(0)

    # Smooth only rocof_1s for noise (backward window only - causal)
    df["rocof_smooth"] = df["rocof_1s"].rolling(window=5, min_periods=1).mean().fillna(0)
    
    # Physics-informed risk signals
    df["wind_power_proxy"] = np.clip(df["wind_speed"]**3 * 3.0, 0, 3000)
    demand_profile = {0:28000, 6:30000, 9:34000, 12:35000, 16:38000, 19:37000, 22:32000}
    def get_demand(h):
        return demand_profile[min(demand_profile.keys(), key=lambda k: abs(k-h))]
    df["demand_proxy"] = df["timestamp"].dt.hour.map(get_demand)
    df["renewable_penetration_ratio"] = (
        df["wind_power_proxy"] / df["demand_proxy"].replace(0, 35000)
    ).clip(0, 1)
    df["rocof_inertia_risk"] = df["rocof_smooth"].abs() * df["renewable_penetration_ratio"]
    return df
```

=== D.2: Probabilistic Model Logic

This section presents the probabilistic model logic, showcasing the quantile regression setup with weighted sampling to prioritize tail-risk events and the post-hoc isotonic recalibration routine.

*Quantile Regression and Weighted Sampling (Selection from: `model_trainer.py`)*

```python
def train_quantile_model(df, alpha):
    params = LGBM_QUANTILE_PARAMS.copy()
    params['objective'] = 'quantile'
    params['alpha'] = alpha
    
    model = lgb.LGBMRegressor(**params)
    
    # Upweight samples where target frequency is low (approaching instability)
    sample_weights = np.where(y_train < 49.95, 10.0,   
                     np.where(y_train < 50.00, 3.0,     
                     1.0))                               
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate Pinball Loss on test set
    y_pred_quantile = model.predict(X_test)
    loss = pinball_loss(y_test, y_pred_quantile, alpha)
    print(f"Pinball Loss (alpha={alpha}): {loss:.4f}")
    return model, X_test, y_test
```

*LSTM MC Dropout Comparator (Selection from: `model_trainer.py`)*

```python
def train_lstm_quantile_comparator(df_processed, n_mc_samples=25):
    inputs = tf.keras.Input(shape=(LSTM_TIME_STEPS, train_scaled.shape[1]))
    x = tf.keras.layers.LSTM(50, return_sequences=False)(inputs)
    # Dropout kept ON during inference (training=True) for MC Dropout
    x = tf.keras.layers.Dropout(0.2)(x, training=True)
    outputs = tf.keras.layers.Dense(1)(x)  # regression, not classification
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mae')
    
    # Stochastic forward passes
    tiled_inputs = tf.repeat(batch[0], n_mc_samples, axis=0)
    tiled_preds = model(tiled_inputs, training=True)
    
    batch_samples = tf.reshape(tiled_preds, (batch_size, n_mc_samples, 1))
    batch_mean = tf.reduce_mean(batch_samples, axis=1)
    return model, scaler, batch_mean.numpy()
```

*Post-hoc Quantile Recalibration (`calibration.py`)*

```python
def fit_calibrator(y_true, y_pred_quantile, alpha):
    """Post-hoc recalibration using isotonic regression."""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred_quantile, y_true)
    return calibrator

def calibrate_predictions(calibrator, y_pred_quantile):
    return calibrator.predict(y_pred_quantile)
```

=== D.3: Dashboard Engine & Multi-Signal Alert Logic

The dashboard engine logic illustrates the high-performance model monitoring and the multi-physics alert fusion engine that powers the real-time Control Room interface.

*Cache Design and Alert Fusion (Selection from: `app.py`)*

```python
# Persists heavy model objects across user sessions
@st.cache_resource
def load_models(model_stamp: tuple):
    lower_model = joblib.load(LGBM_QUANTILE_LOWER_PATH)
    upper_model = joblib.load(LGBM_QUANTILE_UPPER_PATH)
    classifier_model = joblib.load(LGBM_MODEL_PATH)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    return lower_model, upper_model, classifier_model, lstm_model

# Variable Extraction from Current State
rocof_now   = current_row.get('rocof_smooth', current_row.get('rocof', 0.0))
rocof_accel = current_row.get('rocof_accel', 0.0)
volatility  = current_row.get('volatility_10s', 0.0)
freq_now    = current_row['grid_frequency']
ren_pen     = current_row.get('renewable_penetration_ratio', 0.0)

# Physics-Informed Multi-Signal Alert Fusion
rocof_alert = (rocof_now < -0.015) and (freq_now < 50.05)
accel_alert = (rocof_accel < -0.005)
volatility_alert = (volatility > 0.02) and (freq_now < 50.1)
renewable_stress = (ren_pen > 0.15) and (rocof_now < -0.01)
freq_boundary = freq_now < 49.95

# Score-based fusion (convergence of signals)
signal_count = sum([
    rocof_alert, accel_alert, volatility_alert,
    renewable_stress, freq_boundary,
    classifier_prob > 0.35, lstm_prob > 0.35,
])

emergency_trigger = (signal_count >= 3) or (freq_now < alert_threshold_hz) or (lower_bound_pred < alert_threshold_hz)

# Intervention Simulator (Swing Equation Implementation)
# Δf = (ΔP × f₀) / (2 × H × S_base)
if synthetic_inertia_mw > 0:
    swing_delta_f = (synthetic_inertia_mw * NOMINAL_FREQ) / (2 * SYSTEM_INERTIA_H * TOTAL_SYSTEM_CAPACITY)
    lower_bound_pred = lower_bound_raw + swing_delta_f
```

=== D.4: Global Configuration

The global configuration file defines the system thresholds, physics constraints, and model hyperparameters used throughout the GridGuardian pipeline.

*System Thresholds and Feature Set (`config.py`)*

```python
TTA_SECONDS = 10
OPSDA_WIDTH = 0.5 
LAG_INTERVALS_SECONDS = [1, 5, 60]

LGBM_FEATURE_COLS = [
    "grid_frequency", 
    "rocof_1s", "rocof_5s", "rocof_10s", "rocof_30s", "rocof_accel", "rocof_smooth", "rocof",
    "volatility_10s", "volatility_30s", "volatility_60s",
    "wind_speed", "wind_power_proxy", "demand_proxy", "wind_ramp_rate",
    "solar_radiation", "hour", "renewable_penetration_ratio",
    "inertia_value", "inertia_roc", "low_inertia_flag", "rocof_inertia_risk"
] + [f"lag_{lag}s" for lag in LAG_INTERVALS_SECONDS]

LGBM_QUANTILE_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.03,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}
```