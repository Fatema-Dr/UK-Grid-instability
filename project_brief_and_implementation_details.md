This project is a sophisticated and well-structured data science application named 'GridGuardian,' designed to proactively monitor and predict instability in the UK power grid. It addresses the 'Inertia Crisis' caused by the shift to renewable energy sources.

**Project Purpose & Domain:**
The main goal is to provide a real-time alert system for grid operators. It moves beyond simple point forecasts by using quantile regression to predict an uncertainty band for grid frequency 10 seconds into the future. An alert is triggered if the predicted lower bound (10th percentile) breaches a critical threshold (e.g., 49.8 Hz). The project's domain is energy infrastructure, specifically power grid stability and time-series forecasting.

**Technology Stack:**
- **Language:** Python
- **Core Libraries:** Polars (for high-performance data manipulation), LightGBM (for quantile regression), TensorFlow/Keras (for the LSTM model), and Scikit-learn.
- **APIs & Data:** It ingests data from the NESO CKAN API (for grid frequency and inertia costs) and the Open-Meteo API (for weather data).
- **Dashboard:** Streamlit is used to create an interactive web application.
- **Explainability:** SHAP is integrated to provide model-agnostic explanations for the predictions, a key feature for operational trust.

**Project Structure & Data Flow:**
1.  **`Implementation/app.py`:** The main entry point for the interactive dashboard.
2.  **Data Ingestion (`src/data_loader.py`):** On user request (selecting a date range in the app), this module fetches frequency, inertia, and weather data from external APIs. The requests are cached for efficiency.
3.  **Feature Engineering (`src/feature_engineering.py`):** The raw data is processed to create predictive features. This includes merging multi-resolution time-series data and calculating domain-specific metrics like the `wind_ramp_rate` using the Optimized Swinging Door Algorithm (implemented in `src/opsda.py`).
4.  **Prediction (`app.py`):** The Streamlit app loads pre-trained LightGBM quantile models from the `notebooks/` directory (`lgbm_quantile_lower.pkl`, `lgbm_quantile_upper.pkl`).
5.  **Alerting & Visualization:** For each time step, the app predicts the upper and lower frequency bounds. If the lower bound is below a user-set threshold, it displays a prominent alert. The results are visualized using Plotly charts, showing the frequency, the predicted uncertainty band, and the alert threshold.
6.  **Explainability:** SHAP values are calculated for the lower-bound prediction and displayed as a bar chart, showing which features are contributing the most to the instability risk.
7.  **Model Training (`src/model_trainer.py` & `run_pipeline.py`):** The models used by the app were trained using a separate pipeline. The `run_pipeline.py` script orchestrates the process of loading data, engineering features, training LightGBM and LSTM models, and saving the artifacts to the `notebooks/` directory. This script appears slightly outdated compared to the live app's data loading mechanism but accurately reflects the training workflow.

**Key Machine Learning Aspects:**
- **LightGBM Quantile Regression:** This is the core of the alert system. Two models are trained to predict the 10th and 90th percentiles of the future grid frequency, providing a probabilistic forecast rather than a single point estimate. This is ideal for risk assessment.
- **LSTM Model:** An LSTM model is also trained (`lstm_model.keras`), likely as a benchmark or for capturing deeper temporal dependencies, but it is not used in the final Streamlit application.
- **SHAP (SHapley Additive exPlanations):** The use of SHAP is a standout feature, providing critical transparency and interpretability for the model's predictions, which is essential for a system intended for operational use.

----
GridGuardian v2: UK Frequency Alert System — Features Breakdown
Based on your screenshot and the 

app.py
 code, here's what your dashboard does:

🎛️ Sidebar Controls (Left Panel)
Data Selection — Date range pickers (Start Date / End Date) that fetch live data from APIs:

Grid frequency data (from a frequency API)
Weather data (via Open-Meteo)
Inertia cost data
All three datasets are merged, feature-engineered, and cached via @st.cache_data.

Simulation Settings:

Simulation Time slider — Lets you scrub through the loaded time range to explore different moments (like a time-travel playback)
Go to time (HH:MM:SS UTC) — Jump directly to a specific timestamp
Time to Alert (seconds ahead) — Configures how far ahead (5–60 sec) the prediction window looks (currently set to 10s)
Alerting Threshold — A slider (49.5 – 49.9 Hz) that sets the instability threshold. If the predicted lower bound drops below this, an alert fires.

📊 Top KPI Row (5 Metrics)
Metric	What it shows
Grid Frequency	Current actual frequency (49.977 Hz) with rate-of-change (RoCoF) as delta
Predicted Lower Bound	LightGBM quantile regression lower bound (49.945 Hz)
Predicted Upper Bound	LightGBM quantile regression upper bound (49.996 Hz)
System Status	✅ SYSTEM STABLE or ⚠️ INSTABILITY ALERT (based on whether predicted lower bound < threshold)
Time to Alert	Shows the TTA countdown when alert is active, "N/A" when stable
📈 Main Chart — Real-Time Frequency Monitor with Uncertainty Bands
Cyan line — Actual grid frequency over a 300-second rolling window
Orange shaded band — Uncertainty band from the upper/lower quantile regression models
Red dashed line — The alert threshold you set in the sidebar
Red dot — Current time position marker
Uses Plotly with a dark theme for the "control room" aesthetic
📊 Right Panel — Risk Drivers for Instability (SHAP)
This is the Explainable AI (XAI) component using SHAP TreeExplainer:

A horizontal bar chart showing which features are pushing the frequency prediction up (blue/positive SHAP) or down (red/negative SHAP)
In your screenshot, grid_frequency and wind_ramp_rate have the largest negative SHAP values → these are the biggest risk drivers pushing the predicted frequency lower
Features like rocof, volatility_10s, inertia_cost, lag_1s, wind_speed, etc. show their individual contributions
This gives operators actionable insight into why the system might be approaching instability, not just that it is.

🧠 Under the Hood
Two LightGBM quantile regression models (lgbm_quantile_lower.pkl and lgbm_quantile_upper.pkl) predict the lower and upper frequency bounds
Feature engineering via create_features() creates lag features, RoCoF, volatility, and temporal features (hour, etc.)
Data pipeline: frequency + weather + inertia → merge via join_asof → feature engineering → prediction
Dark "control room" CSS styling for the professional monitoring look
In essence, this is a real-time grid stability monitoring dashboard with ML-powered prediction and explainability — designed to give power grid operators early warning of frequency instability events.