# Implementation Report: Grid Stability Alert System v2

## 1. Project Overview

This report details the implementation of an advanced grid stability alert system. The project's goal is to predict and explain grid instability events in real-time using machine learning, incorporating several advanced features for a more robust and insightful analysis. The implementation consists of a modular data processing pipeline, a multi-model training process, and an enhanced web-based monitoring application.

**Technologies Used:**
- **Programming Language:** Python
- **Data Manipulation:** Pandas, Polars
- **Machine Learning:** Scikit-learn, LightGBM, TensorFlow/Keras
- **Web Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Model Explainability:** SHAP
- **Time-Series Analysis:** SwingingDoor

## 2. Project Structure and Refactoring

The project has been refactored to improve modularity, maintainability, and code quality. The core logic is now organized into a `src` directory, and the main notebook has been streamlined.

- **`src/` directory:**
    - **`config.py`:** A configuration file to manage file paths, feature lists, and model parameters.
    - **`data_loader.py`:** Contains functions for loading frequency, weather, and inertia data.
    - **`feature_engineering.py`:** Contains functions for merging datasets and creating advanced features.
    - **`model_trainer.py`:** Contains functions for training the classifier and quantile regression models.
- **`data_ingestion.ipynb`:** The main Jupyter notebook now serves as a high-level driver for the entire pipeline.
- **`app.py`:** The Streamlit application, located in the project root.

## 3. Data Ingestion and Preprocessing

The pipeline, defined by functions in `src/data_loader.py` and `src/feature_engineering.py`, now incorporates multiple data sources and advanced feature engineering.

### Data Sources
- **Grid Frequency Data:** 1-second resolution frequency data.
- **Weather Data:** Fetched from the Open-Meteo API.
- **System Inertia Data:** Daily inertia cost data for 2019, provided by the user.

### Advanced Feature Engineering
- **Inertia Cost:** The daily inertia cost has been merged with the high-resolution data to be used as a feature.
- **Time to Alert (TTA):** The prediction target has been shifted to predict instability **10 seconds** into the future, providing an actionable warning window.
- **Wind Ramp Rate (OpSDA):** The Optimized Swinging Door Algorithm has been implemented to calculate the `wind_ramp_rate`. This feature more accurately captures the "jolt" to the grid from rapid changes in wind generation, replacing the simple `wind_speed` feature.

## 4. Model Training and Evaluation

The model training logic, encapsulated in `src/model_trainer.py`, has been upgraded to include quantile regression.

### Models
- **LightGBM Quantile Regression:** The primary model is now a set of two LightGBM models trained to predict the **10th and 90th percentiles** of the future grid frequency. This allows the system to generate "uncertainty bands".
- **LightGBM Classifier & LSTM:** The original classifier and LSTM models are still available in the codebase but the application now focuses on the superior quantile regression approach.

### Alerting Logic
The alert logic has been updated. An alert is now triggered if the **predicted lower bound (10th percentile)** of the grid frequency drops below the user-defined threshold (e.g., 49.8 Hz).

## 5. Web Application (`app.py`)

The Streamlit dashboard has been significantly enhanced to visualize the outputs of the new models.

### New Features
- **Uncertainty Bands:** The main frequency monitor now displays a shaded "uncertainty band" representing the range between the predicted lower and upper bounds of the future frequency.
- **Time to Alert (TTA) Display:** When an alert is triggered, a "Time to Alert" metric appears, showing the 10-second warning window.
- **Enhanced SHAP Explanations:** The risk factor chart now explains what features are pushing the **predicted lower bound** down, giving a direct insight into the drivers of instability risk.
- **Inertia Cost Display:** The dashboard now includes a Key Performance Indicator (KPI) for the daily inertia cost.

## 6. Other Files

- **`data/`:** Contains the raw grid frequency and inertia cost data.
- **`notebooks/`:** This directory now primarily holds the saved model files and the main driver notebook.
- **`run_pipeline.py` (previously `setup_files.py`):** This script has been refactored to run the entire data pipeline from the command line, providing an alternative to the Jupyter notebook.
