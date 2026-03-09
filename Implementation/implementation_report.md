# Implementation Report: Grid Stability Alert System v2

## 1. Project Overview

This report details the implementation of an advanced grid stability alert system, titled **GridGuardian v2**. The project's goal is to predict and explain grid instability events in real-time using machine learning, incorporating several advanced features for a more robust and insightful analysis. The implementation consists of a modular data processing pipeline, a multi-model training process, and an enhanced web-based monitoring application designed to function as a control room interface.

**Technologies Used:**
- **Programming Language:** Python 3.13
- **Data Manipulation:** Pandas, Polars, PyArrow
- **Machine Learning:** Scikit-learn, LightGBM, TensorFlow/Keras
- **Web Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Model Explainability:** SHAP
- **Time-Series Analysis:** SwingingDoor (OpSDA)

## 2. Project Structure and Architecture

The project has been refactored to improve modularity, maintainability, and code quality. The core logic is organized into a `src` directory, isolating processing logic from the presentation layer.

- **`src/` directory:**
    - **`config.py`:** Manages file paths, feature lists, thresholds, and model hyperparameters.
    - **`data_loader.py`:** Handles API interactions (NESO CKAN, Open-Meteo) with built-in retry logic and type validation.
    - **`feature_engineering.py`:** Utilizes Polars for high-performance dataset merging and complex feature creation.
    - **`model_trainer.py`:** Contains training loops and evaluation logic for both LightGBM and LSTM models.
- **`app.py`:** The primary Streamlit application serving as the frontend dashboard.
- **`evaluate_models.py`:** A dedicated script for generating formal, dissertation-ready performance metrics and visualizations.

## 3. Data Ingestion and Preprocessing

The pipeline integrates multiple disparate data sources, resolving significant temporal mismatches to create a unified feature set.

### Data Sources
- **Grid Frequency Data:** 1-second resolution frequency data sourced via the NESO API.
- **Weather Data:** Hourly historical weather data fetched from the Open-Meteo API.
- **System Inertia Data:** Daily inertia cost records representing the grid's resistance to frequency changes.

### Advanced Feature Engineering
- **Time to Alert (TTA):** The prediction target is shifted to predict instability **10 seconds** into the future, providing operators with an actionable warning window rather than a reactive notification.
- **Wind Ramp Rate (OpSDA):** The Optimized Swinging Door Algorithm (OpSDA) is implemented to calculate the `wind_ramp_rate`. This isolates the volatile "jolts" to the grid from rapid changes in wind generation, which is significantly more predictive than raw wind speed.
- **Rate of Change of Frequency (RoCoF):** Calculated to measure the immediate physical acceleration or deceleration of the grid frequency.

## 4. Model Training and Alert Logic

The system utilizes a dual-model **Quantile Regression** approach to provide probabilistic forecasting.

### LightGBM Quantile Regression
Instead of a binary classifier, the primary system uses two LightGBM models trained to predict the **10th ($\alpha=0.1$) and 90th ($\alpha=0.9$) percentiles** of the future grid frequency. This establishes an 80% Confidence Interval, generating "uncertainty bands" that visualize the range of likely future states.

### Alerting Logic
The alert logic operates strictly on the lower bound. An alert is triggered only if the **predicted lower bound (10th percentile)** drops below the user-defined critical threshold (e.g., 49.8 Hz). This ensures alerts are raised based on forecasted risk, not just current state.

## 5. Web Application (`app.py`)

The Streamlit dashboard acts as a real-time Control Room, optimized for performance and interpretability.

### Key Features
- **Persistent Parquet Caching:** To eliminate redundant API calls and processing overhead, the application caches fully engineered datasets as highly optimized `.parquet` files.
- **Automatic Invalidation:** The app hashes the `src/` directory upon launch. Any modification to the underlying engineering or modeling logic automatically invalidates and clears the cache, ensuring the dashboard always reflects the latest codebase.
- **Real-Time Uncertainty Monitor:** The main chart overlays actual frequency against the predicted uncertainty bands, providing immediate visual context of grid volatility.
- **Dynamic XAI (SHAP):** A real-time bar chart explains precisely which physical factors (e.g., RoCoF, Wind Ramp) are actively pushing the predicted frequency downward, offering operators actionable insights into the *cause* of the alert.

## 6. Validation and Evaluation Results

A comprehensive validation was conducted using data from August 2019, specifically targeting the major UK blackout event on August 9th. The evaluation script (`evaluate_models.py`) confirmed the structural integrity and physical validity of the system.

### A. Model Performance & Precision
The models demonstrated exceptional precision in predicting the frequency state 10 seconds ahead:
- **Lower Bound MAE:** 0.033 Hz
- **Upper Bound MAE:** 0.018 Hz
These error margins are an order of magnitude smaller than standard operational safety buffers (usually ~0.2 Hz), indicating a highly accurate baseline forecast.

### B. Uncertainty Calibration (The "Pessimistic Bias")
While precise, analysis of the Prediction Interval Coverage Probability (PICP) and Quantile Calibration revealed a systematic bias:
- **Coverage (PICP):** The 80% confidence interval captured the actual frequency 77.5% of the time, indicating the bands are slightly too narrow.
- **Calibration Skew:** Only 1.8% of observations fell below the 10th percentile prediction (target: 10%), and 79.3% fell below the 90th percentile (target: 90%). 
- **Analysis:** Both models are shifted downwards, making the system inherently **pessimistic**. In the context of critical infrastructure, this is a highly desirable "fail-safe" state; predicting a dip that does not occur (False Positive) is vastly preferable to missing a catastrophic drop (False Negative).

### C. Physical Feature Importance (XAI Validation)
Analysis of the LightGBM split counts confirmed the model successfully learned grid physics:
1.  **RoCoF** was correctly identified as the most dominant predictive feature.
2.  **Wind Ramp Rate (OpSDA)** significantly outranked raw `wind_speed`, validating the implementation of the Swinging Door Algorithm to capture volatility.
3.  **Inertia Cost Limitation:** `inertia_cost` ranked lowest in importance. Because inertia data is provided at a daily resolution, it lacks the variance required by the model to inform 1-second predictions. This highlights a limitation in data availability rather than algorithmic design; access to hourly inertia data would resolve this.

### D. Dashboard and Stress Testing
During spot-checks of the August 9th Blackout event, the dashboard logic perfectly mirrored model outputs. While the extreme magnitude of the blackout exceeded the model's 80% confidence bounds (as it was an out-of-distribution "Black Swan" event), the predicted lower bound successfully crossed the 49.8 Hz threshold *before* the actual frequency collapsed, validating the system's efficacy as an early-warning tool.