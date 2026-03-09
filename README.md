# GridGuardian: UK Grid Stability Alert System

This repository contains the complete dissertation project for predicting and explaining UK electrical grid instability in a low-inertia, high-renewable energy landscape.

## 📂 Project Overview

The project is divided into two primary sections:

### 1. [Implementation/](./Implementation/)
This folder contains the full engineering pipeline, machine learning models, and the real-time monitoring dashboard.
*   **Dashboard:** Built with Streamlit for a "Control Room" experience.
*   **Models:** LightGBM Quantile Regression for probabilistic frequency forecasting.
*   **XAI:** SHAP-based explainability for risk drivers.
*   **Key Algorithms:** Optimized Swinging Door Algorithm (OpSDA) for wind ramp detection.

### 2. [Report/](./Report/)
This folder contains the academic documentation and research supporting the project.
*   **Literature Review:** Analysis of the "Inertia Crisis" and modern forecasting paradigms (LSTM, LightGBM, SARIMAX).
*   **Methodology:** Detailed breakdown of the data-driven approach and validation strategies.

## 📝 Key Reports

- [UK Power Grid Stability Report](UK_Power_Grid_Stability_Report.md): High-level analysis of grid dynamics.
- [Project Summary Report](project_summary_report.md): Overview of project goals and achievements.
- [Validation Checks](validation_checks.md): Summary of model testing against the August 9, 2019 blackout.

## 🚦 Getting Started

To run the code, navigate to the `Implementation/` directory and follow the instructions in its specific [README](./Implementation/README.md).

```bash
cd Implementation/
uv sync
uv run streamlit run app.py
```
