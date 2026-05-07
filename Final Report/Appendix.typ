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
])

#pagebreak()

== Appendix B: Final Project Proposal & Scope Evolution

#formal_box("Final Project Specification (GridGuardian)", [
  #grid(
    columns: (1fr, 2fr),
    row-gutter: 12pt,
    [*Final Title:*], [GridGuardian: Proactive Grid Instability Forecasting using Physics-Informed Machine Learning],
    [*Final Aim:*], [Develop a robust, real-time alert system using a hybrid ensemble (LightGBM + LSTM) and physics-informed features to provide proactive foresight for UK grid frequency stability.],
  )

  #v(15pt)
  *Evolution of Methodology:* \
  The project evolved from the initial proposal's focus on SARIMAX to a **Hybrid LGBM-LSTM Ensemble**. SARIMAX was found to be insufficient for capturing the non-linear, high-frequency transients observed during grid disturbances (e.g., the August 2019 blackout). The final implementation incorporates physics-informed features (Swing Equation logic, OpSDA wind compression) and a multi-signal alert fusion engine to provide a more robust "Command Deck" solution.

  #v(10pt)
  *Achieved Technical Deliverables:*
  #table(
    columns: (1fr, 2fr),
    align: (left, left),
    stroke: 0.5pt + gray,
    table.header([*Original Objective*], [*Final Implementation Outcome*]),
    [1. Select robust proxy], [Established instability threshold at 49.85 Hz with physics-informed triggers (RoCoF & Acceleration).],
    [2. Data pipeline], [Engineered a high-performance Polars ETL pipeline merging 1s frequency, HadUK weather, and inertia data.],
    [3. Benchmark models], [SARIMAX, LightGBM, and LSTM benchmarked; Result: Hybrid LGBM-LSTM selected for superior precision.],
    [4. Probabilistic alerts], [Implemented Quantile Regression (α=0.1 to 0.9) with post-hoc Isotonic Calibration for reliable uncertainty.],
    [5. Streamlit Dashboard], [Developed "Command Deck" v3 with split-cache design and SHAP-based real-time risk driver analysis.],
    [6. Blackout Validation], [Validated against August 9, 2019 event; demonstrated proactive lead-time warning of >10 seconds.],
    [7. Critical Evaluation], [Conducted SHAP global/local importance analysis and out-of-season robustness testing on Winter data.],
  )

  #v(10pt)
  *Final System Rationale:* \
  By moving beyond simple forecasting to a physics-aware "Command Deck," the GridGuardian system demonstrates that data-driven foresight can significantly reduce reliance on carbon-intensive reactive measures, directly supporting the UK's transition to a high-renewables grid.
])

#pagebreak()

== Appendix C: Ethical Approval

This project utilizes publicly available open datasets from National Grid ESO and Open-Meteo, classifying it as primary quantitative analysis. As the study does not involve human subjects or the collection of sensitive personal data, it qualifies for the internal ethical approval process for non-human research.

The following documentation confirms the internal ethical clearance obtained for the study:

#figure(
  image("WhatsApp Image 2026-05-06 at 15.21.45.jpeg", width: 80%),
  caption: [Internal screening confirming the use of public datasets and absence of human data collection.],
)

#figure(
  image("WhatsApp Image 2026-05-06 at 15.21.46.jpeg", width: 80%),
  caption: [Registration of the research topic and supervisor oversight for the data science project.],
)

#figure(
  image("WhatsApp Image 2026-05-06 at 15.21.46 (1).jpeg", width: 80%),
  caption: [Declaration of compliance with field research and participant debriefing standards.], 
)

#figure(
  image("WhatsApp Image 2026-05-06 at 15.21.46 (2).jpeg", width: 80%),
  caption: [Confirmation of UEL data storage policies and ethical data usage protocols.],
)

#figure(
  image("WhatsApp Image 2026-05-06 at 15.21.46 (3).jpeg", width: 80%),
  caption: [Final submission and confirmation of the internal ethical approval process.],
)

#pagebreak()

== Appendix D: Dashboard Outputs

*Note:* This section is reserved for high-resolution exports and annotated screenshots of the GridGuardian dashboard during various simulated and historical instability events (e.g., August 9, 2019 blackout, winter peak demand scenarios).

#pagebreak()

== Appendix E: Code Snippets

This appendix presents the core implementation logic for the GridGuardian system, providing a technical reference for the feature engineering, probabilistic modeling, and real-time dashboard engine.

=== E.1: Physics-Informed Feature Engineering

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
    # Causal RoCoF and Acceleration
    df["rocof_1s"]  = df["grid_frequency"].diff(1).fillna(0)
    df["rocof_5s"]  = ((df["grid_frequency"] - df["grid_frequency"].shift(5)) / 5.0).fillna(0)
    df["rocof_accel"] = (df["rocof_5s"].diff(5) / 5.0).fillna(0)
    
    # Physics-informed risk signals
    df["wind_power_proxy"] = np.clip(df["wind_speed"]**3 * 3.0, 0, 3000)
    df["renewable_penetration_ratio"] = df["wind_power_proxy"] / df["demand_proxy"].replace(0, 35000)
    df["rocof_inertia_risk"] = df["rocof_smooth"].abs() * df["renewable_penetration_ratio"]
    return df
```

=== E.2: Probabilistic Model Logic

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
    return model
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

=== E.3: Dashboard Engine & Multi-Signal Alert Logic

The dashboard engine logic illustrates the split-cache architecture for high-performance monitoring and the multi-physics alert fusion engine that powers the real-time Command Deck.

*Cache Design and Alert Fusion (Selection from: `app.py`)*

```python
# Split-Cache Architecture
@st.cache_resource
def load_models(model_stamp):
    """Persists heavy model objects across user sessions."""
    lower_model = joblib.load(LGBM_QUANTILE_LOWER_PATH)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    return lower_model, lstm_model

# Variable Extraction from Current State
freq_now    = current_row['grid_frequency']
rocof_now   = current_row.get('rocof_smooth', 0.0)
rocof_accel = current_row.get('rocof_accel', 0.0)
volatility  = current_row.get('volatility_10s', 0.0)
ren_pen     = current_row.get('renewable_penetration_ratio', 0.0)

# Physics-Informed Multi-Signal Alert Fusion
rocof_alert = (rocof_now < -0.012) and (freq_now < 50.05)
accel_alert = (rocof_accel < -0.005)
volatility_alert = (volatility > 0.018) and (freq_now < 50.1)

# Score-based fusion (convergence of signals)
signal_count = sum([
    rocof_alert, accel_alert, volatility_alert,
    renewable_stress, freq_boundary,
    classifier_prob > 0.30, lstm_prob > 0.30, quantile_signal,
])

emergency_trigger = signal_count >= 3 or freq_now < alert_threshold_hz or quantile_signal

# Intervention Simulator (Swing Equation Implementation)
# Δf = (ΔP × f₀) / (2 × H × S_base)
if synthetic_inertia_mw > 0:
    swing_delta_f = (synthetic_inertia_mw * NOMINAL_FREQ) / (2 * SYSTEM_INERTIA_H * TOTAL_SYSTEM_CAPACITY)
    lower_bound_pred = lower_bound_raw + swing_delta_f
```

=== E.4: Global Configuration

The global configuration file defines the system thresholds, physics constraints, and model hyperparameters used throughout the GridGuardian pipeline.

*System Thresholds and Feature Set (`config.py`)*

```python
TTA_SECONDS = 10
OPSDA_WIDTH = 0.5 
LAG_INTERVALS_SECONDS = [1, 5, 60]

LGBM_FEATURE_COLS = [
    "grid_frequency", "rocof_smooth", "rocof_accel",
    "wind_speed", "wind_ramp_rate", "renewable_penetration_ratio",
    "inertia_value", "inertia_roc", "low_inertia_flag"
] + [f"lag_{lag}s" for lag in LAG_INTERVALS_SECONDS]

LGBM_QUANTILE_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.03,
    'max_depth': 6,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}
```
