# Chapter 3: Methodology

## 3.1 Introduction

This chapter describes the methodology employed in the development and evaluation of GridGuardian. The chapter is structured to provide a comprehensive account of the research design, software development lifecycle, data sources, feature engineering strategy, model selection rationale, evaluation metrics, and ethical considerations. The methodological choices described herein are justified against the research questions formulated in Chapter 1 and the literature reviewed in Chapter 2.

The primary objective of this chapter is to enable reproducibility. A researcher with access to the same data sources and codebase should be able to replicate the results reported in this dissertation. To this end, specific algorithms, hyperparameters, and implementation details are documented with sufficient precision.

---

## 3.2 Research Design Philosophy

### 3.2.1 Physics-Informed Data Science Framework

This research adopts a "physics-informed data science" approach, which integrates domain knowledge from power systems engineering with machine learning methodologies. This philosophy recognises that purely data-driven models may fail to capture fundamental physical constraints and may generalise poorly to out-of-distribution scenarios (Raissi et al., 2019).

The physics-informed framework is implemented through three mechanisms:

1. **Domain-Specific Feature Engineering:** Features are designed to encode physical relationships, such as RoCoF (Rate of Change of Frequency) and wind ramp rates detected via the Optimized Swinging Door Algorithm (OpSDA).

2. **Physically Constrained Prediction Horizon:** The 10-second Time to Alert (TTA) is aligned with the activation time of Firm Frequency Response (FFR) services, ensuring that predictions are operationally actionable (National Grid ESO, 2020a).

3. **Safety-Oriented Quantile Selection:** The 10th and 90th percentiles are selected to provide conservative (worst-case) bounds for risk assessment, prioritising safety over precision.

### 3.2.2 Quantitative Research Approach

This research employs a quantitative approach, focusing on empirical measurement of model performance using statistical metrics. The quantitative approach is appropriate because:

- The research questions concern prediction accuracy and model comparison, which are inherently quantitative.
- High-resolution time-series data (1-second frequency, hourly weather) enables rigorous statistical analysis.
- The August 9, 2019 blackout provides a well-documented case study for validation.

### 3.2.3 Iterative Development Process

The research follows an iterative development process, with multiple cycles of:
1. **Hypothesis Formulation:** Based on literature review and domain understanding.
2. **Implementation:** Coding the proposed approach.
3. **Evaluation:** Measuring performance against defined metrics.
4. **Refinement:** Adjusting the approach based on evaluation results.

This iterative process is consistent with the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which is described in detail in Section 3.3.

---

## 3.3 Software Development Life Cycle

### 3.3.1 CRISP-DM Framework

The Cross-Industry Standard Process for Data Mining (CRISP-DM) provides the overarching framework for this research (Shearer, 2000). CRISP-DM consists of six phases:

**Figure 3.1: CRISP-DM Methodology Adapted for GridGuardian**

| CRISP-DM Phase | GridGuardian Implementation |
|----------------|-------------------------------|
| Business Understanding | Define grid stability monitoring requirements; identify August 2019 blackout as validation case |
| Data Understanding | Explore NESO API, Open-Meteo API, inertia cost data; assess data quality and resolution |
| Data Preparation | Build Polars data pipeline; implement join_asof temporal merging; handle missing values |
| Modelling | Train LightGBM quantile regression; implement LSTM benchmark; tune hyperparameters |
| Evaluation | Compute MAE, PICP, MPIW, pinball loss; validate against August 2019 blackout |
| Deployment | Develop Streamlit dashboard; implement caching and SHAP explainability |

### 3.3.2 Evolutionary Prototyping

In addition to CRISP-DM, this research employs evolutionary prototyping for the dashboard component. Evolutionary prototyping involves building an initial prototype and iteratively refining it based on feedback and testing (Sommerville, 2016).

The dashboard evolution proceeded as follows:

1. **Prototype 1:** Basic Streamlit interface displaying frequency time-series and predictions.
2. **Prototype 2:** Added uncertainty bands (quantile predictions) and alert indicators.
3. **Prototype 3:** Integrated SHAP explainability visualisations.
4. **Prototype 4:** Implemented caching and performance optimisation.
5. **Final Version:** Added source code hashing for cache invalidation and comprehensive error handling.

### 3.3.3 Version Control and Reproducibility

All code is managed using Git version control, with commits documenting the evolution of the implementation. Key milestones are tagged for reproducibility:

- `v1.0`: Initial data pipeline and baseline model
- `v1.5`: OpSDA integration for wind ramp detection
- `v2.0`: SHAP explainability integration
- `v2.1`: Final version with caching and error handling

The use of `pyproject.toml` and `uv.lock` ensures that dependencies are pinned to specific versions, enabling exact reproduction of the experimental environment.

---

## 3.4 Data Sources and Acquisition

### 3.4.1 Overview of Data Sources

GridGuardian integrates data from three primary sources:

**Table 3.1: Data Sources and Characteristics**

| Data Source | Type | Resolution | Time Range | Access Method |
|-------------|------|------------|------------|---------------|
| NESO CKAN API | Grid frequency | 1-second | Aug 2019 | HTTP REST API |
| Open-Meteo API | Weather data | 1-hour | Aug 2019 | HTTP REST API |
| NESO Inertia Costs | Financial data | Daily | Aug 2019 | CSV download |

### 3.4.2 NESO CKAN API: Grid Frequency Data

The National Energy System Operator (NESO) provides open access to grid frequency data through their CKAN (Comprehensive Knowledge Archive Network) API. The API endpoint used is:

```
https://data.neso.energy.gov.au/dataset/api/3/action/package_show?id=grid-frequency
```

**Data Characteristics:**
- **Format:** JSON
- **Resolution:** 1-second
- **Fields:** timestamp, frequency (Hz)
- **Quality:** High; validated by NESO operational systems

**Access Implementation:**
The [`data_loader.py`](Implementation/src/data_loader.py:1) module implements a `fetch_frequency_data()` function that:
1. Sends HTTP GET requests to the NESO API
2. Parses JSON responses into Polars DataFrames
3. Implements retry logic with exponential backoff for reliability
4. Caches results in Parquet format for subsequent runs

### 3.4.3 Open-Meteo API: Weather Data

The Open-Meteo API provides historical weather data for the Hornsea offshore wind farm location (54.13°N, 0.33°E). The API endpoint used is:

```
https://archive-api.open-meteo.com/v1/archive?latitude=54.13&longitude=-0.33&start_date=2019-08-01&end_date=2019-08-31&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,surface_pressure
```

**Data Characteristics:**
- **Format:** JSON
- **Resolution:** 1-hour
- **Fields:** wind_speed_10m (m/s), wind_direction_10m (degrees), temperature_2m (°C), surface_pressure (hPa)
- **Quality:** Good; reanalysis data with validation

**Access Implementation:**
The `fetch_weather_data()` function in [`data_loader.py`](Implementation/src/data_loader.py:45) implements:
1. API request with location and time range parameters
2. JSON parsing and column renaming
3. Temporal alignment with frequency data via `join_asof`

### 3.4.4 Inertia Cost Data

NESO publishes daily inertia cost data as part of their Stability Pathfinder programme. This data is downloaded as CSV from:

```
https://data.neso.energy.gov.au/dataset/stability-pathfinder
```

**Data Characteristics:**
- **Format:** CSV
- **Resolution:** Daily
- **Fields:** date, inertia_cost (£/MW)
- **Quality:** Good; financial settlement data

**Usage:**
Inertia cost serves as a market-based proxy for inertia scarcity. High inertia costs indicate periods of low system inertia and elevated stability risk.

### 3.4.5 Data Merging Strategy

The three data sources have different temporal resolutions (1-second, 1-hour, daily). To merge them into a unified dataset, the Polars `join_asof` function is used. This function performs a "backward" join, matching each frequency record with the most recent weather and inertia data.

**Figure 3.2: Data Flow Architecture**

```
NESO API (1-sec) ──┐
                   ├──> Polars join_asof ──> Unified DataFrame
Open-Meteo (1-hr) ─┤
                   │
Inertia (daily) ───┘
```

The `join_asof` strategy ensures that:
- Each 1-second frequency record has associated weather and inertia features
- No future information leaks into the training data (backward join)
- Missing weather data is handled gracefully (null values)

### 3.4.6 Data Validation and Quality Assurance

Data quality is ensured through the following validation checks:

1. **Completeness:** Check for missing timestamps in frequency data; interpolate gaps < 10 seconds.
2. **Consistency:** Verify that frequency values are within plausible range (48-52 Hz).
3. **Accuracy:** Cross-check a sample of records against NESO's published data.
4. **Temporal Alignment:** Verify that `join_asof` produces expected results for edge cases.

---

## 3.5 Feature Engineering Strategy

### 3.6.1 Overview of Feature Engineering

Feature engineering is a critical component of the GridGuardian framework. Well-designed features capture domain-specific patterns that raw data may not explicitly contain, improving prediction accuracy and model interpretability (Domingos, 2012).

The feature engineering strategy is guided by three principles:
1. **Physical Relevance:** Features should encode physical relationships relevant to grid stability.
2. **Computational Efficiency:** Features should be computable in real-time with low latency.
3. **Interpretability:** Features should be meaningful to grid operators.

**Table 3.2: Engineered Features and Descriptions**

| Feature | Description | Physical Significance |
|---------|-------------|----------------------|
| `rocof_1s` | Rate of Change of Frequency (1-second difference) | Measures frequency dynamics; high RoCoF indicates instability |
| `rocof_5s_ma` | 5-second moving average of RoCoF | Smoothed RoCoF for noise reduction |
| `wind_ramp_rate` | Wind speed ramp rate via OpSDA | Detects rapid wind changes that may impact generation |
| `wind_speed` | Wind speed at 10m height | Proxy for wind generation output |
| `wind_direction` | Wind direction at 10m height | Indicates wind pattern changes |
| `temperature_2m` | Temperature at 2m height | Affects load (heating/cooling demand) |
| `surface_pressure` | Surface atmospheric pressure | Correlates with weather system stability |
| `inertia_cost` | Daily inertia cost (£/MW) | Market-based proxy for inertia scarcity |
| `lag_1s` | Frequency at t-1 | Autoregressive feature |
| `lag_5s` | Frequency at t-5 | Autoregressive feature |
| `lag_60s` | Frequency at t-60 | Captures longer-term trends |
| `hour_of_day` | Hour (0-23) | Captures daily load patterns |
| `day_of_week` | Day (0-6) | Captures weekly load patterns |

### 3.5.2 Rate of Change of Frequency (RoCoF)

RoCoF is calculated as the first difference of frequency:

$$\text{RoCoF}_t = f_t - f_{t-1}$$

Where $f_t$ is the frequency at time $t$.

**Noise Reduction:**
Raw RoCoF calculated from 1-second data can be noisy due to measurement errors and high-frequency oscillations. A 5-second moving average is applied to reduce noise:

$$\text{RoCoF}_t^{\text{smoothed}} = \frac{1}{5} \sum_{i=0}^{4} (f_{t-i} - f_{t-i-1})$$

This smoothing preserves the underlying signal while reducing high-frequency noise (Savitzky and Golay, 1964).

### 3.5.3 Wind Ramp Detection via OpSDA

The Optimized Swinging Door Algorithm (OpSDA) is used to detect wind ramp events. OpSDA is a time-series compression technique that identifies significant changes while filtering out noise (Kao et al., 2007).

**Algorithm Overview:**

1. **Initialisation:** Start with the first data point as a pivot.
2. **Door Formation:** Form a "door" (angular bounds) from the pivot to the current point.
3. **Point Checking:** Check if intermediate points fall within the door.
4. **Compression:** If a point falls outside, record the previous point and start a new door.

**Ramp Rate Calculation:**
The ramp rate is calculated as the slope between compressed points:

$$\text{Ramp Rate} = \frac{\Delta \text{Wind Speed}}{\Delta \text{Time}}$$

**Implementation:**
The [`opsda.py`](Implementation/src/opsda.py:1) module implements the OpSDA algorithm with configurable parameters:
- `pressure`: Maximum deviation allowed (default: 0.5 m/s)
- `min_points`: Minimum number of points between compressions (default: 5)

**Figure 3.3: OpSDA Compression Principle**

```
Original:  *     *   *  *    *  *   *
           | \  / |  | \/ |  | \/  |
Compressed:*--*---*--*----*--*-----*
           Pivot  Door   Door  Pivot
```

### 3.5.4 Lag Features

Lag features capture the autoregressive nature of frequency dynamics:

$$\text{lag}_k = f_{t-k}$$

Where $k$ is the lag interval (1s, 5s, 60s).

**Rationale:**
Recent frequency history is a strong predictor of future frequency, particularly during stable periods. Lag features enable the model to learn temporal patterns without requiring explicit modelling of seasonality.

### 3.5.5 Temporal Features

Temporal features encode time-related information:
- **Hour of Day:** Load and generation patterns vary by time of day (e.g., higher demand during evening peak).
- **Day of Week:** Weekend patterns differ from weekday patterns (e.g., lower industrial demand on weekends).

These features are encoded as integers (0-23 for hour, 0-6 for day) to preserve ordinal relationships.

### 3.5.6 Renewable Penetration Proxies

Direct measurements of renewable penetration are not available in real-time. Synthetic proxies are constructed:

- **Wind Speed / Load Ratio:** Higher wind speeds relative to load indicate higher wind penetration.
- **Inertia Cost:** Market-based proxy for inertia scarcity; high costs indicate low inertia.

These proxies provide surrogate measures for physical quantities that are difficult to measure directly.

---

## 3.6 Model Selection and Justification

### 3.6.1 Overview of Model Selection

Three model types are considered for grid frequency prediction:
1. **LightGBM Quantile Regression:** Primary model for probabilistic forecasting.
2. **LSTM (Long Short-Term Memory):** Deep learning benchmark for capturing temporal dependencies.
3. **SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Variables):** Statistical baseline.

**Table 3.3: Model Hyperparameters**

| Model | Hyperparameter | Value |
|-------|----------------|-------|
| LightGBM | num_leaves | 31 |
| LightGBM | max_depth | 6 |
| LightGBM | learning_rate | 0.1 |
| LightGBM | n_estimators | 1000 |
| LightGBM | objective | regression_quantile |
| LightGBM | alpha (quantile) | 0.1, 0.9 |
| LSTM | hidden_size | 64 |
| LSTM | num_layers | 2 |
| LSTM | dropout | 0.2 |
| LSTM | sequence_length | 60 |
| SARIMAX | order | (1, 1, 1) |
| SARIMAX | seasonal_order | (1, 1, 1, 24) |

### 3.6.2 LightGBM Quantile Regression

**Rationale for Selection:**
LightGBM is selected as the primary model for the following reasons:
1. **Efficiency:** Fast training and inference, suitable for real-time applications (Ke et al., 2017).
2. **Accuracy:** Competitive or superior performance to deep learning on tabular data.
3. **Quantile Regression:** Native support for quantile loss functions via the 'regression_quantile' objective.
4. **Feature Importance:** Built-in feature importance metrics for interpretability.

**Quantile Regression Implementation:**
Two separate models are trained:
- **Lower Bound Model:** $\tau = 0.1$ (10th percentile)
- **Upper Bound Model:** $\tau = 0.9$ (90th percentile)

The pinball loss function is minimised:

$$L_\tau(y, \hat{y}) = \begin{cases} \tau(y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-\tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

### 3.6.3 LSTM Benchmark

**Rationale for Inclusion:**
LSTM is included as a deep learning benchmark to assess whether capturing long-range temporal dependencies improves prediction accuracy.

**Architecture:**
- **Input Layer:** Sequence of 60 timesteps (1 minute of frequency data)
- **LSTM Layers:** 2 layers with 64 hidden units each
- **Dropout:** 0.2 dropout between LSTM layers for regularisation
- **Output Layer:** Single value (predicted frequency)

**Training:**
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam with learning rate 0.001
- **Batch Size:** 32
- **Epochs:** 100 with early stopping

### 3.6.4 SARIMAX Baseline

**Rationale for Inclusion:**
SARIMAX serves as a statistical baseline to assess whether machine learning approaches provide significant improvement over traditional methods.

**Model Specification:**
- **Order:** (1, 1, 1) - non-seasonal ARIMA
- **Seasonal Order:** (1, 1, 1, 24) - daily seasonality (24 timesteps)
- **Exogenous Variables:** Weather features (wind speed, temperature)

### 3.6.5 Model Comparison Criteria

Models are compared based on:
1. **Prediction Accuracy:** MAE, RMSE
2. **Uncertainty Quantification:** PICP, MPIW, pinball loss
3. **Computational Efficiency:** Training time, inference latency
4. **Interpretability:** Feature importance, explainability

---

## 3.7 Evaluation Metrics

### 3.7.1 Overview of Evaluation Metrics

GridGuardian is evaluated using a comprehensive set of metrics that assess both point prediction accuracy and uncertainty quantification quality.

**Table 3.4: Evaluation Metrics and Formulas**

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE | $\frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$ | Point prediction accuracy |
| RMSE | $\sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$ | Point prediction accuracy (penalises large errors) |
| Pinball Loss | $\frac{1}{n} \sum_{i=1}^n L_\tau(y_i, \hat{y}_i)$ | Quantile prediction accuracy |
| PICP | $\frac{1}{n} \sum_{i=1}^n \mathbb{I}(y_i \in [\hat{y}_{0.1}, \hat{y}_{0.9}])$ | Prediction interval coverage |
| MPIW | $\frac{1}{n} \sum_{i=1}^n (\hat{y}_{0.9} - \hat{y}_{0.1})$ | Prediction interval sharpness |
| Calibration | $|\text{PICP} - 0.80|$ | Quantile calibration quality |

### 3.7.2 Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

MAE is interpretable (in Hz for frequency prediction) and robust to outliers.

### 3.7.3 Pinball Loss

Pinball loss (quantile loss) measures the accuracy of quantile predictions:

$$\text{Pinball Loss}_\tau = \frac{1}{n} \sum_{i=1}^n L_\tau(y_i, \hat{y}_{i,\tau})$$

Lower pinball loss indicates better quantile predictions.

### 3.7.4 Prediction Interval Coverage Probability (PICP)

PICP measures the fraction of actual values that fall within the prediction interval:

$$\text{PICP} = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(y_i \in [\hat{y}_{0.1}, \hat{y}_{0.9}])$$

For an 80% prediction interval ($\tau_1 = 0.1, \tau_2 = 0.9$), a well-calibrated model should have PICP ≈ 0.80.

### 3.7.5 Mean Prediction Interval Width (MPIW)

MPIW measures the average width of prediction intervals:

$$\text{MPIW} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_{0.9} - \hat{y}_{0.1})$$

Narrower intervals are preferred, but only if coverage is maintained.

### 3.7.6 Calibration Score

Calibration measures how well the predicted quantiles match the actual coverage:

$$\text{Calibration} = |\text{PICP} - 0.80|$$

Lower calibration error indicates better-calibrated predictions.

### 3.7.7 Early Warning Metrics

For the August 9, 2019 blackout validation, additional metrics are defined:

- **Time to Alert (TTA):** Time between alert trigger and frequency breach (target: > 10 seconds).
- **True Positive Rate:** Fraction of instability events correctly predicted.
- **False Positive Rate:** Fraction of stable periods incorrectly flagged as unstable.

---

## 3.8 Ethical Considerations

### 3.8.1 Data Privacy and Security

The data used in this research is publicly available through NESO and Open-Meteo APIs. No personal or sensitive data is processed. The research complies with the UK General Data Protection Regulation (UK GDPR) and the Data Protection Act 2018.

### 3.8.2 Algorithmic Bias and Fairness

Machine learning models can exhibit bias if training data is not representative of all scenarios. To mitigate this risk:
- Training data spans the full August 2019 period, including both stable and unstable periods.
- Model performance is evaluated across different time periods to assess generalisation.
- SHAP explainability enables detection of unexpected feature dependencies.

### 3.8.3 Safety and Liability

GridGuardian is a research prototype, not an operational system. The dissertation includes clear disclaimers that:
- Predictions should not be used for operational decision-making without further validation.
- The system has not been certified for safety-critical deployment.
- Grid operators should rely on official National Grid ESO tools for operational decisions.

### 3.8.4 Open Source and Reproducibility

All code is open-source and publicly available, enabling peer review and reproducibility. This aligns with the principles of open science and promotes transparency in AI research.

---

## 3.9 Chapter Summary

This chapter has described the methodology employed in the development and evaluation of GridGuardian. The key methodological choices are:

1. **Physics-Informed Data Science:** Integration of domain knowledge through feature engineering and physically constrained prediction horizons.

2. **CRISP-DM Framework:** Six-phase data mining process guiding the research from business understanding to deployment.

3. **Data Sources:** NESO CKAN API (1-second frequency), Open-Meteo API (hourly weather), and NESO inertia cost data (daily), merged via Polars `join_asof`.

4. **Feature Engineering:** RoCoF, OpSDA-based wind ramp detection, lag features, temporal features, and renewable penetration proxies.

5. **Model Selection:** LightGBM quantile regression (primary), LSTM (deep learning benchmark), SARIMAX (statistical baseline).

6. **Evaluation Metrics:** MAE, RMSE, pinball loss, PICP, MPIW, calibration, and early warning metrics (TTA, true/false positive rates).

7. **Ethical Considerations:** Data privacy, algorithmic bias mitigation, safety disclaimers, and open-source reproducibility.

The next chapter presents the technical implementation details, including system architecture, data pipeline, feature engineering algorithms, model training procedures, and dashboard development.

---


