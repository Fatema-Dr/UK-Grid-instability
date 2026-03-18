# ⚡ GridGuardian v2 — Implementation

> The engineering core: data pipeline, ML models, real-time dashboard, and automated tests.

---

## 🚀 Key Features

- **Probabilistic Forecasting** — LightGBM Quantile Regression (α=0.1 & α=0.9) predicts uncertainty bands 10s ahead
- **Physics-Informed Features** — Smoothed RoCoF, OpSDA wind ramp rate, renewable penetration ratio
- **Explainable AI** — Real-time SHAP visualisations for instability risk drivers
- **Intervention Simulator** — Swing-equation-based "What-If" tool: `Δf = (ΔP × f₀) / (2 × H × S_base)`
- **LSTM Residual Monitor** — Flags "High Model Uncertainty" when LSTM disagrees with LightGBM
- **Dynamic Model Health** — Live Pinball Loss, PICP, MPIW, and Calibration Score per selected day
- **Post-Hoc Recalibration** — Isotonic regression via `src/calibration.py` corrects systematic quantile bias
- **Out-of-Season Validation** — CLI args for evaluating on any month (Dec 2019 winter test, etc.)
- **Sub-Daily Inertia** — Fetches half-hourly system inertia from NESO, interpolates to 1-second
- **Smart Caching** — Parquet cache with source-code hash invalidation for fast dashboard reloads

## 🛠 Project Structure

```text
Implementation/
├── app.py                     # Streamlit "Control Room" dashboard
├── evaluate_models.py         # Model evaluation & validation CLI
├── run_pipeline.py            # End-to-end data fetch → train → export pipeline
├── pyproject.toml             # Dependencies (managed by uv)
├── src/
│   ├── config.py              # Constants, API keys, feature lists, model paths
│   ├── data_loader.py         # NESO CKAN + Open-Meteo data ingestion & validation
│   ├── feature_engineering.py # RoCoF, OpSDA, lags, targets, renewable proxy
│   ├── model_trainer.py       # LightGBM classifier/quantile + LSTM training
│   ├── calibration.py         # Isotonic regression for quantile recalibration
│   └── opsda.py               # Optimized Swinging Door Algorithm
├── tests/                     # Automated test suite (31 tests)
│   ├── test_feature_engineering.py   # Feature output, RoCoF smoothing, NaN checks
│   ├── test_alert_logic.py           # Alert thresholds, model disagreement, swing eq
│   └── test_metrics.py               # Pinball loss, PICP, MPIW, calibration score
├── data/                      # Cached Parquet files (auto-generated)
├── notebooks/                 # Trained model artifacts (.pkl, .keras)
└── implementation_report.md   # Technical documentation
```

## 🚦 Getting Started

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run the Full Pipeline

Fetches data from NESO + Open-Meteo APIs, engineers features, trains all models, fits calibrators, and exports artifacts:

```bash
uv run python run_pipeline.py
```

### 3. Launch the Dashboard

```bash
uv run streamlit run app.py
```

The dashboard provides:
- **Control Room** tab — real-time frequency monitor with uncertainty bands, SHAP risk drivers, alert status
- **Model Health** tab — dynamically computed Pinball Loss, PICP, MPIW, and calibration scores

### 4. Evaluate Models

```bash
# Default: August 2019 (includes blackout day stress test)
uv run python evaluate_models.py

# Winter robustness test (December 2019)
uv run python evaluate_models.py --start-date 2019-12-01 --end-date 2019-12-31

# With isotonic recalibration applied
uv run python evaluate_models.py --calibrated

# Both combined
uv run python evaluate_models.py --start-date 2019-12-01 --end-date 2019-12-31 --calibrated
```

### 5. Run Tests

```bash
uv run pytest tests/ -v    # 31 tests, ~0.6s
```

## 📊 Validation Summary

| Metric | Lower Bound (α=0.1) | Upper Bound (α=0.9) |
|---|---|---|
| **Pinball Loss** | 0.0142 | 0.0168 |
| **MAE** | 0.033 Hz | 0.018 Hz |
| **PICP (80% CI)** | ~77.5–81.5% | — |

- Errors are an order of magnitude below operational safety margins (~0.2 Hz)
- Maintains a "pessimistic bias" — alerts are raised conservatively (fail-safe for critical infrastructure)
- Successfully flagged the **August 9, 2019 UK blackout** before frequency collapsed to 48.8 Hz

## 🔧 Data Sources

| Source | Resolution | Data |
|---|---|---|
| [NESO CKAN API](https://api.neso.energy) | 1-second | Grid frequency |
| [NESO CKAN API](https://api.neso.energy) | Daily / Half-hourly | Inertia cost / System inertia |
| [Open-Meteo Archive](https://archive-api.open-meteo.com) | Hourly → interpolated to 1s | Wind speed, solar radiation, temperature |

## 📜 Documentation

For the full methodology, feature engineering details, and calibration analysis, see the [Implementation Report](implementation_report.md).
