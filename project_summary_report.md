# Project Summary Report: Grid Stability Alert System v2

## 1. Project Overview and Context (Literature Review Summary)

The project addresses the "Inertia Crisis" in the Great Britain (GB) power grid, a critical consequence of the global energy transition from synchronous (fossil-fuel) to asynchronous (renewable) generation. This shift reduces system inertia, making the grid more susceptible to rapid frequency deviations and severe events like the 9 August 2019 UK blackout. The core problem is to move from reactive to proactive grid management by predicting and explaining grid instability events in real-time.

Key findings from the literature review highlight:
*   The increasing penetration of Variable Renewable Energy Sources (VRES) introduces variability and impacts grid stability.
*   Low-inertia power systems exhibit heightened sensitivity to disturbances, leading to high Rates of Change of Frequency (RoCoF) and substantial frequency deviations.
*   The operational limits for UK grid frequency are 49.8 Hz to 50.2 Hz; deviations outside this range are considered instability events.
*   Proxy variables like RoCoF, Frequency Nadir, and Market-Based Inertia Costs are crucial for quantifying grid instability.
*   Data-driven forecasting methodologies, particularly Machine Learning (ML) and Deep Learning (DL), are necessary due to the complexity of modern power systems. LightGBM shows promise for balancing accuracy and computational efficiency, while LSTMs excel in capturing temporal dependencies.
*   A critical research gap exists in integrating physics-informed feature engineering with market-based proxies, benchmarking models for the UK context, and providing probabilistic, explainable alerts for operators.

## 2. Methodology

The project employs a **physics-informed data science** framework, integrating domain-specific knowledge into the modeling process.

### Research Design and Strategy
*   **Proactive Stance:** The methodology focuses on forecasting instability *before* it manifests, aiming to predict the probability of frequency breaching critical operational thresholds (49.8 Hz to 50.2 Hz) with a **Time to Alert (TTA)** of 10 seconds (as defined in `src/config.py`).
*   **Stochastic Modelling and Uncertainty Quantification:** A **probabilistic approach** using **Quantile Regression** (specifically, LightGBM models predicting the 10th and 90th percentiles, configured via `QUANTILE_ALPHAS` in `src/config.py`) is used to provide operators with an uncertainty band, capturing "tail risks" beyond point forecasts.
*   **Hybrid Modelling Framework:**
    1.  **Gradient Boosting Machines (LightGBM):** Chosen for performance on large tabular datasets, low inference latency, and suitability for classification and quantile regression.
    2.  **Deep Learning (LSTM):** Utilized to capture long-range temporal dependencies and sequential patterns in frequency and weather data.
    3.  **Financial and Physical Proxies:** Features like `inertia_cost` (`data/inertia_costs19.csv`) are integrated as proxies for grid conditions.

### Software Development Life Cycle (SDLC)
A hybrid SDLC combining **CRISP-DM** (for iterative, research-intensive phases like data understanding, preparation, and modeling) and **Evolutionary Prototyping** (for the Streamlit dashboard UI/UX refinement) was adopted.

### Operationalization of Variables
Key variables include:
*   **Dependent (Target):** System Frequency Deviation (continuous) for regression; Instability Event (binary: > +/- 0.2 Hz deviation) for classification.
*   **Independent (Predictors):** VRE Generation, Grid Load, Inertia Proxy.
*   **Derived (Engineered):**
    *   **Ramp Rate:** Calculated using the **Optimized Swinging Door Algorithm (OpSDA)** to detect sudden power swings in wind data (`src/opsda.py`, `src/feature_engineering.py`).
    *   **Lagged Features:** Historical values (e.g., `lag_1s`, `lag_5s`, `lag_60s`) capture autoregressive properties.

### Analytical Pipeline
1.  **Data Pre-processing:** Multi-resolution synchronization (using Polars `join_asof`), outlier handling (Winsorization), and **Min-Max scaling** (for LSTM) are applied.
2.  **Preliminary Analysis:** Descriptive and correlational analyses are performed before modeling.
3.  **Predictive Modelling:** Benchmarking against SARIMAX (statistical baseline), LightGBM (ensemble), and LSTM (deep learning).

### Tools and Technologies
*   **Polars:** Used for high-performance, multi-threaded data manipulation (`src/data_loader.py`, `src/feature_engineering.py`).
*   **LightGBM:** Primary modeling tool, optimized for large datasets and quantile regression.
*   **TensorFlow/Keras:** Used for LSTM model implementation.
*   **SHAP (SHapley Additive exPlanations):** Integrated for model interpretability and explaining alert drivers (`app.py`).
*   **Open-Meteo API:** Used for fetching weather data (`src/data_loader.py`).
*   **Streamlit:** For the interactive web dashboard (`app.py`).

## 3. Implementation Details

The project's implementation is structured for modularity and maintainability, centered around a `src/` directory and a Streamlit dashboard.

### Project Structure and Refactoring
The project is organized into:
*   `src/`: Contains core logic modules:
    *   `config.py`: Centralized management of file paths, feature lists (`LGBM_FEATURE_COLS`, `LSTM_FEATURE_COLS`), and model parameters (e.g., `TTA_SECONDS`, `QUANTILE_ALPHAS`, `SPLIT_DATE`).
    *   `data_loader.py`: Handles loading grid frequency data (`data/f-2019-aug/f 2019 8.csv`), fetching weather data from Open-Meteo API, and loading inertia cost data (`data/inertia_costs19.csv`).
    *   `feature_engineering.py`: Merges diverse datasets (`merge_datasets`), calculates `wind_ramp_rate` using the Optimized Swinging Door Algorithm (`calculate_wind_ramp_rate` utilizing `src/opsda.py`), and creates other features like RoCoF, volatility, lag features, and time embeddings.
    *   `model_trainer.py`: Encapsulates the training and evaluation logic for LightGBM classifier, LightGBM quantile regressors, and LSTM models, including handling class imbalance with `scale_pos_weight` and Min-Max scaling for LSTM.
    *   `opsda.py`: Implements the core logic for the Swinging Door Algorithm.
*   `notebooks/`: Stores saved model artifacts (`lgbm_quantile_lower.pkl`, `lgbm_quantile_upper.pkl`, `lstm_model.keras`, `scaler.pkl`) and the main Jupyter notebook (`data_ingestion.ipynb`) which serves as a high-level driver for the pipeline.
*   `app.py`: The Streamlit web application.
*   `run_pipeline.py`: A command-line script to install dependencies and execute the entire data pipeline (`setup_files.py` seems to be a previous version or alternative for generating dummy data).

### Data Ingestion and Preprocessing
The pipeline combines 1-second resolution grid frequency data, hourly weather data, and daily inertia cost data. Advanced feature engineering is performed to create `inertia_cost`, `wind_ramp_rate` (using OpSDA), and `TTA_SECONDS` (10-second ahead prediction target for instability).

### Model Training and Evaluation
*   **LightGBM Quantile Regression:** Two LightGBM models are trained to predict the 10th and 90th percentiles of future grid frequency, providing uncertainty bands.
*   **LightGBM Classifier & LSTM:** While present in the codebase, the Streamlit app primarily leverages the quantile regression approach for alerts. The LSTM model is trained to capture temporal dependencies.
*   **Alerting Logic:** An alert is triggered if the predicted lower bound (10th percentile) of the frequency drops below a user-defined threshold (e.g., 49.8 Hz).

### Web Application (`app.py`)
The Streamlit dashboard (`GridGuardian v2: UK Frequency Alert System`) provides an interactive interface for real-time monitoring and interpretation:
*   **Visualization:** Displays real-time frequency with shaded uncertainty bands (from quantile regression models).
*   **Alerts:** Shows "INSTABILITY ALERT" and "Time to Alert" (10 seconds) when the lower bound prediction breaches the threshold.
*   **Explainability (SHAP):** Presents a "Risk Drivers for Instability" chart, utilizing SHAP values to explain which features are contributing to the predicted lower bound, enhancing transparency.
*   **Controls:** Allows users to simulate different time points and adjust the alert threshold.
*   **KPIs:** Displays key metrics like current grid frequency, predicted bounds, and inertia cost.

Overall, the project delivers a robust, transparent, and proactive system for forecasting and mitigating grid instability in the UK's evolving low-carbon energy landscape.
