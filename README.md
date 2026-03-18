# ⚡ GridGuardian v2 — UK Power Grid Stability Alert System

> A real-time machine learning system that predicts and explains electrical grid instability in the UK's low-inertia, high-renewable energy landscape.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 The Problem

As the UK transitions from synchronous thermal generation to non-synchronous renewables (wind, solar), the grid's physical **inertia** is declining — making frequency collapses faster and harder to stop. Traditional monitoring systems react *after* a threshold is breached. GridGuardian predicts breaches **10 seconds ahead** using probabilistic forecasting.

## 🧠 How It Works

```
1-sec Frequency Data ─┐
                       ├─→ Feature Engineering ─→ LightGBM Quantile ─→ Uncertainty Bands
Hourly Weather Data ──┤    (RoCoF, OpSDA,         Regression            (10th / 90th
                       │    Renewable Proxy)        (α=0.1 & α=0.9)       percentile)
Daily Inertia Data ───┘                                │
                                                       ├─→ SHAP Explainer ─→ Risk Drivers
                                                       └─→ LSTM Residual Monitor ─→ Disagreement Alerts
```

## 🚀 Key Features

| Feature | Description |
|---|---|
| **Probabilistic Forecasting** | LightGBM Quantile Regression predicts 10th/90th percentile frequency bounds 10s ahead |
| **Physics-Informed Features** | Smoothed RoCoF, OpSDA wind ramp rate, renewable penetration ratio as inertia proxy |
| **Explainable AI (XAI)** | Real-time SHAP feature attribution for every prediction |
| **Intervention Simulator** | Swing-equation-based "What-If" tool for simulating synthetic inertia injection |
| **LSTM Residual Monitor** | Flags "High Model Uncertainty" when LSTM disagrees with LightGBM |
| **Dynamic Model Health** | Live calibration metrics (Pinball Loss, PICP, MPIW) per selected day |
| **Post-Hoc Recalibration** | Isotonic regression corrects systematic quantile bias |
| **Out-of-Season Validation** | CLI args for evaluating on any month (e.g., winter robustness testing) |
| **Sub-Daily Inertia** | Half-hourly system inertia data, interpolated to 1-second resolution |
| **Performance Caching** | Parquet cache with source-code hash invalidation |

## 📂 Repository Structure

```text
├── Implementation/              # Full engineering pipeline + dashboard
│   ├── app.py                   # Streamlit "Control Room" dashboard
│   ├── evaluate_models.py       # Evaluation & validation CLI
│   ├── run_pipeline.py          # End-to-end training pipeline
│   ├── src/
│   │   ├── config.py            # All constants, paths, API config
│   │   ├── data_loader.py       # NESO + Open-Meteo data ingestion
│   │   ├── feature_engineering.py  # RoCoF, OpSDA, lags, targets
│   │   ├── model_trainer.py     # LightGBM + LSTM training
│   │   ├── calibration.py       # Isotonic regression recalibration
│   │   └── opsda.py             # Swinging Door Algorithm
│   └── tests/                   # 31 automated pytest tests
│       ├── test_feature_engineering.py
│       ├── test_alert_logic.py
│       └── test_metrics.py
├── Report/                      # Literature review & methodology
├── Final Report/                # Dissertation chapters
├── project_brief_and_implementation_details.md
├── validation_checks.md
└── supervisor_project_pitch.md
```

## 🚦 Quick Start

### Prerequisites
- Python 3.13
- [uv](https://github.com/astral-sh/uv) package manager

### Setup & Run

```bash
cd Implementation/
uv sync                          # Install dependencies
uv run python run_pipeline.py    # Train models (fetches data from APIs)
uv run streamlit run app.py      # Launch dashboard
```

### Evaluate Models

```bash
# Default: August 2019 (includes blackout day)
uv run python evaluate_models.py

# Winter robustness test
uv run python evaluate_models.py --start-date 2019-12-01 --end-date 2019-12-31

# With post-hoc recalibration
uv run python evaluate_models.py --calibrated
```

### Run Tests

```bash
uv run pytest tests/ -v          # 31 tests
```

## 📊 Validation Highlights

| Metric | Lower Bound (α=0.1) | Upper Bound (α=0.9) |
|---|---|---|
| **Pinball Loss** | 0.0142 | 0.0168 |
| **MAE** | 0.033 Hz | 0.018 Hz |
| **PICP (80% CI)** | ~77.5–81.5% | — |

- Error margins are an order of magnitude smaller than operational safety buffers (~0.2 Hz)
- Successfully triggered alerts before the **August 9, 2019 UK blackout** collapse
- Maintains a "pessimistic bias" — a desirable fail-safe for critical infrastructure

## �️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Data Processing | Polars (join_asof for multi-resolution merge) |
| ML Models | LightGBM (quantile regression), TensorFlow/Keras (LSTM) |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| Data Sources | NESO CKAN API, Open-Meteo API |
| Package Manager | uv |
| Testing | pytest |

## 📜 Documentation

| Document | Description |
|---|---|
| [Project Brief](project_brief_and_implementation_details.md) | Comprehensive technical reference |
| [Validation Checks](validation_checks.md) | Model testing against the 2019 blackout |
| [Implementation Report](Implementation/implementation_report.md) | Detailed methodology & results |
| [Supervisor Pitch](supervisor_project_pitch.md) | Project summary with architecture diagram |
