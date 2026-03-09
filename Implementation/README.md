# GridGuardian v2: UK Frequency Alert System

GridGuardian v2 is an advanced machine learning-based monitoring system designed to forecast and explain electrical grid instability in the UK. By leveraging high-resolution frequency data, weather patterns, and system inertia metrics, it provides a probabilistic "Time to Alert" warning system for grid operators.

## 🚀 Key Features

*   **Probabilistic Forecasting:** Uses **LightGBM Quantile Regression** to predict the 10th and 90th percentiles of future grid frequency, generating dynamic uncertainty bands.
*   **Physics-Informed Engineering:** 
    *   **Wind Ramp Rate (OpSDA):** Implements the *Optimized Swinging Door Algorithm* to capture volatile changes in wind generation.
    *   **RoCoF Analysis:** Real-time calculation of the *Rate of Change of Frequency*.
*   **Explainable AI (XAI):** Integrated **SHAP** visualizations to explain the physical drivers behind every instability alert.
*   **Performance Optimized:** Features a persistent Parquet-based caching system with automatic source-code hashing to ensure fast dashboard loading and logic integrity.

## 🛠 Project Structure

```text
Implementation/
├── app.py                # Streamlit Control Room Dashboard
├── evaluate_models.py    # Dissertation Validation & Metrics Suite
├── run_pipeline.py       # End-to-end data/model pipeline CLI
├── src/                  # Core logic (Loader, Features, Training)
├── data/                 # Raw data and processed cache
├── notebooks/            # Model artifacts (.pkl) and evaluation plots
└── implementation_report.md  # Detailed technical documentation
```

## 🚦 Getting Started

### 1. Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) installed for fast environment management.

### 2. Setup Environment
```bash
uv sync
```

### 3. Run the Dashboard
```bash
uv run streamlit run app.py
```

### 4. Validate Model Performance
To generate performance metrics and validation plots (PICP, Pinball Loss, Calibration):
```bash
uv run python evaluate_models.py
```

## 📊 Validation Summary (August 2019)
The system was validated against the **August 9, 2019 UK Blackout**.
*   **Precision:** Lower Bound MAE of **0.033 Hz**.
*   **Safety:** The system maintains a "pessimistic bias," ensuring early warnings for potential instability events.
*   **Reliability:** Successfully triggered alerts before the 2019 blackout collapse, providing critical lead time.

## 📜 Documentation
For a deep dive into the methodology, feature engineering (OpSDA), and model calibration results, see the [Implementation Report](implementation_report.md).
