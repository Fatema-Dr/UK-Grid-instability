# Chapter 3: Methodology

## 3.1 Introduction

This chapter delineates the systematic methodology employed to design, develop, and validate a data-driven alert system for forecasting power grid instability in the United Kingdom. The research is a direct response to the escalating "Inertia Crisis"—a term describing the structural shift in the Great Britain (GB) grid from predictable, synchronous, carbon-intensive power generation to asynchronous, inverter-based renewable energy sources (RES). This transition, while essential for decarbonisation (Fearn, 2025), fundamentally reduces the grid's intrinsic physical resilience to supply-demand mismatches, increasing the risk of severe frequency deviations (Hong et al., 2021).

The core of this research confronts a dual-faceted computing problem. Firstly, ensuring grid stability in this new landscape has become a "Big Data" challenge, necessitating the high-velocity ingestion and processing of time-series data at one-second resolution from multiple heterogeneous data sources. Secondly, for an alert system to be operationally viable, it must deliver predictions with extremely low computational latency, providing a sufficient "Time to Alert" (TTA) for control room operators to take mitigating action before instability materialises. This project establishes a configurable TTA, with a default value of 10 seconds as defined in the system configuration.

The goal of this methodology is therefore to implement and evaluate a **physics-informed data science** framework. This approach moves beyond generic machine learning applications by embedding domain-specific knowledge of power system dynamics—including concepts such as Rate of Change of Frequency (RoCoF), wind ramp events, and inertia proxies—directly into the feature engineering and modelling process. This physics-informed philosophy, which is gaining significant traction in recent power systems research (Shuai and Li, 2025), ensures that the predictive models are guided by physical reality rather than relying solely on statistical pattern recognition.

## 3.2 Research Philosophy and Design

### 3.2.1 Research Philosophy

The methodological approach adopted for this research is predicated on a **positivist, deductive** philosophy. Grid instability is characterised as an observable, measurable, and quantifiable phenomenon governed by established physical laws (electromechanical frequency dynamics) and market mechanisms (supply-demand balance, inertia pricing). This philosophical stance justifies the use of quantitative, empirical methods—specifically, the application of machine learning models trained on historical observational data—to test the hypothesis that instability events can be predicted in advance.

The deductive component of this philosophy is manifest in the formulation of specific, testable hypotheses (presented in Section 3.6) which are then evaluated against empirical data. The research moves from the general theory (that data-driven methods can improve grid stability management) to the specific prediction (that LightGBM with physics-informed features will outperform alternative models for the UK grid).

### 3.2.2 Predictive vs. Reactive Stance

Traditional grid management has historically relied on reactive measures, responding to frequency deviations only after they have breached operational thresholds. This research argues that such a posture is increasingly untenable in a low-inertia environment, where the time available for corrective action is diminishing rapidly. Instead, the methodology adopted here is fundamentally **proactive**, seeking to forecast instability *before* it manifests.

The core objective of the GridGuardian system is not merely to predict the future value of the frequency itself, but rather to predict the *probability* and *range* of the frequency breaching critical operational thresholds (49.8 Hz to 50.2 Hz) within a defined future time window. This probabilistic, forward-looking approach provides operators with actionable intelligence rather than a mere alarm that confirms what they can already observe.

### 3.2.3 Stochastic Modelling and Uncertainty Quantification

Given that the output of wind and solar generation is inherently intermittent and non-deterministic, a purely deterministic forecasting model—one that produces a single "best guess" value—is fundamentally insufficient for operational risk management. Such models may predict an average outcome but inherently fail to capture the "tail risks" associated with rare but catastrophic events like blackouts (Saleem et al., 2024).

To address this critical limitation, this research adopts a **probabilistic approach** centred on **Quantile Regression**. As implemented in the LightGBM modelling framework, this technique does not yield a single point forecast but rather a predictive distribution. Specifically, the system predicts the 10th and 90th percentiles of the future frequency (configured via `QUANTILE_ALPHAS = [0.1, 0.9]` in the project's configuration module). This produces an 80% confidence interval—an "uncertainty band"—that provides operators with a quantitative measure of risk at every prediction step. If the lower bound (10th percentile) of this band drops below the alert threshold, the system triggers an instability alert, flagging not a certainty of failure but a statistically significant *risk* of failure (Dey et al., 2025).

### 3.2.4 Hybrid Modelling Framework

To address the complexity and multi-scale nature of the input data, a hybrid modelling framework is employed that synergises three distinct modelling paradigms:

1.  **Gradient Boosting Machines (LightGBM):** Chosen for its established, exceptional performance on large-scale tabular datasets and its critically low inference latency. LightGBM's leaf-wise growth algorithm and histogram-based binning make it an "operational champion" for the real-time classification and regression tasks at the heart of the alert system (Zhou et al., 2025). It serves as the primary model in the deployed dashboard.

2.  **Deep Learning (LSTM):** Long Short-Term Memory networks are utilised to capture long-range temporal dependencies and sequential patterns within the frequency and weather data that may be missed by tree-based models, which process each observation independently (Pandit, Mu and Astolfi, 2025). The LSTM model serves as a benchmark to assess whether the temporal modelling capabilities of deep learning offer a significant advantage over the explicitly engineered features used by LightGBM.

3.  **Statistical Baseline (SARIMAX):** The SARIMAX model is included as a classical statistical baseline to establish the minimum performance that any advanced model must exceed. Its interpretability provides a useful comparison point for understanding the non-linear relationships captured by the machine learning models.

## 3.3 Software Development Life Cycle (SDLC)

A hybrid Software Development Life Cycle (SDLC) model was adopted, combining the structured, research-oriented nature of the Cross-Industry Standard Process for Data Mining (CRISP-DM) with the flexibility of Evolutionary Prototyping.

### 3.3.1 CRISP-DM

The CRISP-DM framework provided the necessary iterative structure for a research-intensive project. The standard cycle of Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment was not followed linearly. Instead, the process was inherently iterative: insights gained during the "Evaluation" phase—particularly regarding feature importance as revealed by SHAP analysis—prompted repeated loops back to the "Data Preparation" and "Feature Engineering" phases. For example, early SHAP analysis revealed that raw wind speed had limited predictive power for frequency deviations, prompting the development and integration of the physics-informed `wind_ramp_rate` feature using the Swinging Door Algorithm. Similarly, the parameterisation of the OpSDA `width` parameter (initially set to 0.5 in `config.py`) was revisited during iterative evaluation cycles.

### 3.3.2 Evolutionary Prototyping

The final system output is a decision-support dashboard designed for human operators. As the precise requirements for "situational awareness"—how information should be presented, what controls are needed, and what constitutes an optimal alert interface—were not fully known at the outset, an evolutionary prototyping approach was adopted for the development of the Streamlit dashboard. This methodology allowed for the rapid construction of initial UI prototypes, which were then iteratively refined based on the analytical capabilities of the underlying models and an assessment of operator needs. The dashboard evolved through multiple iterations, with enhancements including the addition of dynamic date range selection, configurable Time to Alert settings, and the integration of SHAP-based explanation charts.

## 3.4 Data Collection

The GridGuardian system integrates three distinct data sources, each providing a different perspective on the state of the GB grid. These sources differ substantially in temporal resolution, origin, and format, necessitating sophisticated preprocessing to create a unified dataset.

### 3.4.1 Grid Frequency Data

-   **Source:** National Energy System Operator (NESO) CKAN API
-   **Resolution:** 1-second intervals
-   **Key Variables:** `timestamp` (UTC datetime), `grid_frequency` (Hz)
-   **Period:** August 2019 (encompassing the 9 August blackout event for validation)
-   **Description:** This is the highest-resolution dataset and forms the core of the analytical pipeline. Each record represents a single measurement of the GB system frequency at one-second granularity. The data is fetched programmatically from the NESO API using a custom data loading function (`fetch_frequency_data` in `src/data_loader.py`) that implements retry logic with exponential backoff for network resilience using the `tenacity` library.

### 3.4.2 Weather Data

-   **Source:** Open-Meteo Archive API
-   **Resolution:** Hourly intervals
-   **Key Variables:** `temperature_2m`, `precipitation`, `rain`, `snowfall`, `wind_speed_10m`, `wind_gusts_10m`, `direct_radiation`
-   **Location:** Latitude 54.0°N, Longitude 2.0°W (approximate geographic centre of the UK), as configured in `config.py`
-   **Description:** Hourly historical weather data provides the exogenous environmental variables that drive renewable energy generation. Wind speed is the most critical variable, as it directly determines the output of wind turbines and, through the feature engineering process described in Section 3.5, the `wind_ramp_rate`. Solar radiation (`direct_radiation`) is included as a proxy for solar PV generation. The data is fetched using the `openmeteo-requests` Python library, with API responses cached locally using `requests_cache.CachedSession` to prevent redundant API calls during iterative development.

### 3.4.3 System Inertia Cost Data

-   **Source:** National Energy System Operator (NESO) CKAN API
-   **Resolution:** Daily
-   **Key Variable:** `inertia_cost` (£/day)
-   **Description:** This dataset provides the daily cost incurred by the system operator for procuring inertia services. As discussed in the Literature Review (Section 2.3.3), this cost serves as a financial proxy for the scarcity of physical inertia on the grid. A high inertia cost signals that synchronous generation is scarce and the grid is operationally "fragile," even if no physical disturbance has yet occurred. The data is fetched from the NESO API and parsed using the Polars library, with the `Settlement Date` converted to a UTC datetime at midnight for temporal alignment.

### 3.4.4 Multi-Resolution Data Synchronisation

A fundamental challenge in this project is the synchronisation of datasets with vastly different temporal resolutions: 1-second frequency data, hourly weather data, and daily inertia cost data. This is addressed using an **asof join** strategy, implemented via the Polars library's `join_asof` function.

The `merge_datasets` function in `src/feature_engineering.py` operates as follows:

1.  All datasets are sorted chronologically by their respective timestamp columns.
2.  The 1-second frequency data is joined with the hourly weather data using a `backward` strategy. This effectively associates each second of frequency data with the most recent available hourly weather observation, forward-filling the weather values until the next observation becomes available.
3.  The resulting merged dataset is then joined with the daily inertia cost data using the same `backward` strategy, associating each record with the day's inertia cost.
4.  Any remaining null values (typically occurring at the very beginning of the dataset where no prior weather or inertia observation exists) are removed using `drop_nulls()`.

This approach preserves the full 1-second resolution of the frequency data while enriching each observation with the relevant environmental and market context.

## 3.5 Feature Engineering: The Physics-Informed Distinction

The most significant methodological distinction of this project lies in its physics-informed approach to feature engineering. Rather than feeding raw data directly into a "black box" machine learning model and relying on the model to implicitly discover relevant patterns, the data is explicitly transformed to represent the physical phenomena known to drive grid instability. This section describes the key engineered features.

### 3.5.1 Operationalisation of Variables

To bridge the gap between abstract physical and economic concepts and concrete model inputs, the following table operationalises the variables used in this experiment:

| Variable Type | Variable Name | Data Type | Operationalisation |
| :--- | :--- | :--- | :--- |
| **Dependent (Target)** | System Frequency Deviation (`target_freq_next`) | Ratio | The continuous grid frequency value predicted 10 seconds into the future, used for Regression models |
| | Instability Event (`target_is_unstable`) | Nominal (Binary) | Classified as 1 if `target_freq_next` deviates by more than ±0.2 Hz from 50 Hz, 0 otherwise; used for Classification/Alert models |
| **Independent (Predictors)** | Grid Frequency (`grid_frequency`) | Ratio | Current system frequency in Hz at 1-second resolution |
| | Wind Speed (`wind_speed`) | Ratio | Hourly wind speed at 10m altitude, measured in m/s |
| | Solar Radiation (`direct_radiation`) | Ratio | Direct solar radiation in W/m², proxy for solar PV output |
| | Inertia Cost (`inertia_cost`) | Ratio | Daily system inertia procurement cost in GBP, serving as a financial proxy for grid fragility |
| **Derived (Engineered)** | Rate of Change of Frequency (`rocof`) | Ratio | First difference of `grid_frequency` (df/dt), representing the instantaneous acceleration of the frequency signal |
| | Volatility (`volatility_10s`) | Ratio | Rolling standard deviation of `grid_frequency` over a 10-second window, capturing short-term frequency instability |
| | Wind Ramp Rate (`wind_ramp_rate`) | Ratio | Rate of change in wind speed between significant ramp events, calculated using the Optimised Swinging Door Algorithm (OpSDA) |
| | Lag Features (`lag_1s`, `lag_5s`, `lag_60s`) | Interval | Historical values of `grid_frequency` at t−1, t−5, and t−60 seconds, capturing autoregressive properties |
| | Temporal Features (`hour`, `minute`) | Ordinal | Hour and minute of the observation, capturing time-of-day patterns in demand and generation |

### 3.5.2 Wind Ramp Rate and the Optimised Swinging Door Algorithm (OpSDA)

A critical innovation of this project is the use of the Optimised Swinging Door Algorithm (OpSDA) to quantify wind variability. As established in the Literature Review (Section 2.4.7), sudden *changes* in wind output—ramp events—are far more disruptive to grid stability than the absolute level of wind generation. The OpSDA provides a principled method for identifying these ramp events and quantifying their severity.

The OpSDA is a data compression algorithm that identifies the "significant" changes in a time-series signal, discarding periods of relative stationarity. As implemented in `src/opsda.py`, the `compress` function operates as follows:

1.  It receives a list of `(timestamp, value)` tuples (in this case, Unix timestamps and wind speed values).
2.  Starting from the first data point (the "pivot"), it constructs upper and lower "doors" defined by a configurable `width` parameter (set to 0.5 in `config.py`).
3.  For each subsequent data point, it checks whether all intermediate points between the pivot and the current point remain within the bounds of these doors.
4.  When an intermediate point falls outside the door bounds, the previous point is recorded as a significant inflection point, and a new door is started from that point.
5.  The output is a compressed series of only the significant inflection points.

The `calculate_wind_ramp_rate` function in `src/feature_engineering.py` then calculates the slope (rate of change) between consecutive compressed points to produce the `wind_ramp_rate` feature. This feature is subsequently merged back into the primary 1-second frequency dataset using an asof join and forward-filled, such that the most recent ramp rate persists until the next significant ramp event is detected.

This feature, `wind_ramp_rate`, explicitly represents the physical "jolt" that a sudden change in wind generation imparts to the grid—a technique validated in wind power forecasting research (Pandit, Mu and Astolfi, 2025). As demonstrated in the project's validation results (see Chapter 4), SHAP analysis confirms that `wind_ramp_rate` significantly outranks raw `wind_speed` in predictive importance, validating the physics-informed approach.

### 3.5.3 Rate of Change of Frequency (RoCoF) and Volatility

Two additional derived features capture the immediate physical dynamics of the frequency signal:

-   **RoCoF** is calculated as the first-order difference of the `grid_frequency` column: `grid_frequency[t] - grid_frequency[t-1]`. This approximates df/dt and captures the instantaneous acceleration or deceleration of the system frequency. A large negative RoCoF indicates a rapid frequency decline, often the first measurable sign of a significant power imbalance.

-   **Volatility** is calculated as the rolling standard deviation of `grid_frequency` over a 10-second window (`rolling(window=10).std()`). This feature captures the degree of short-term frequency "nervousness," where a high volatility value indicates that the frequency is fluctuating erratically—often a precursor to a larger disturbance.

Both features provide the model with information about the *dynamics* of the frequency signal (how it is changing) rather than merely its *state* (what it currently is), enabling the prediction of future behaviour.

### 3.5.4 Lag Features and Temporal Embeddings

For tree-based models such as LightGBM, which process each data point independently without inherent temporal awareness, explicit lag features are necessary to encode the recent history of the signal. Three lag intervals are configured in `config.py` via `LAG_INTERVALS_SECONDS = [1, 5, 60]`:

-   `lag_1s` captures very short-term persistence (the frequency one second ago).
-   `lag_5s` captures short-term trends over five seconds.
-   `lag_60s` captures medium-term trends over one minute.

These features, combined with any null values filled with the nominal frequency of 50.0 Hz, allow the model to learn autoregressive patterns in the data.

Temporal embeddings (`hour` and `minute`, extracted from the timestamp) encode time-of-day patterns. These are important because demand profiles and generation mixes follow predictable diurnal cycles—for example, solar generation peaks at midday and demand peaks in the evening—which influence the grid's baseline stability profile.

### 3.5.5 Inertia Proxy

While the direct, real-time measurement of system inertia is a complex engineering problem, a proxy can be derived from the available data. The `inertia_cost` feature, taken directly from the NESO inertia cost data at daily resolution, serves as this proxy. Periods of high inertia procurement costs are indicative of periods when synchronous generation is scarce and the grid operator must actively procure stability services—signalling an inherently more fragile grid state. Although the daily resolution of this data limits its ability to capture intra-day variations, it provides the model with a valuable macro-level indicator of the grid's structural resilience.

## 3.6 Model Architecture and Training

This section describes the architecture, training procedure, and hyperparameter configuration for each of the three modelling paradigms employed in this research.

### 3.6.1 Data Splitting Strategy

A fixed chronological split is used to divide the August 2019 dataset into training and testing partitions:

-   **Training Data:** August 1–8, 2019 (8 days of normal grid operation).
-   **Testing Data:** August 9, 2019 (the blackout day).

This partition is defined by `SPLIT_DATE = "2019-08-09 00:00:00"` and `END_TEST_DATE = "2019-08-10 00:00:00"` in `config.py`. The deliberate selection of the blackout day as the exclusive test set constitutes a rigorous "stress test," evaluating each model's ability to predict and alert on an extreme, previously unseen event. This is a more demanding evaluation than a random split, as it tests the model's ability to generalise to out-of-distribution conditions.

### 3.6.2 LightGBM Classifier

An LightGBM classifier is trained to predict the binary `target_is_unstable` label. Key implementation details include:

-   **Class Imbalance Handling:** Instability events are extremely rare in the training data (the grid is stable for the vast majority of time). To prevent the model from trivially predicting "stable" for all observations, `scale_pos_weight` is calculated as the ratio of negative to positive samples in the training set and passed to the classifier. This effectively increases the penalty for misclassifying the rare "unstable" events.
-   **Hyperparameters:** Configured via `LGBM_PARAMS` in `config.py`: `n_estimators=100`, `learning_rate=0.05`, `max_depth=10`, `random_state=42`, `n_jobs=-1` (utilising all available CPU cores).
-   **Evaluation:** Classification Report, Confusion Matrix, and AUC-ROC.

### 3.6.3 LightGBM Quantile Regressors

The core of the alert system consists of two LightGBM regressors trained using the **quantile regression** objective. This is the primary modelling approach used in the deployed GridGuardian dashboard.

-   **Lower Bound Model (α = 0.1):** Trained with `objective='quantile'` and `alpha=0.1`, this model predicts the 10th percentile of the `target_freq_next` variable. This represents a conservatively low estimate of the frequency 10 seconds into the future.
-   **Upper Bound Model (α = 0.9):** Trained with `objective='quantile'` and `alpha=0.9`, this model predicts the 90th percentile, representing a conservatively high estimate.
-   **Loss Function:** Both models implicitly optimise the **Pinball Loss** function, which penalises predictions that fall on the wrong side of the true value asymmetrically depending on the quantile being estimated. The Pinball Loss is also explicitly calculated on the test set to report model performance.
-   **Alerting Logic:** An instability alert is triggered when the predicted lower bound (10th percentile) drops below a user-configurable threshold (default 49.8 Hz). This design ensures that alerts are raised based on the *risk* of a frequency breach, accounting for uncertainty, rather than requiring a deterministic prediction of failure.

### 3.6.4 LSTM Network

An LSTM model is trained as a deep learning benchmark for the binary classification task:

-   **Data Preprocessing:** Input features are scaled using **Min-Max normalisation** (`sklearn.preprocessing.MinMaxScaler`) to ensure efficient convergence of the gradient-based optimisation. The scaler is fit on the training data only and applied to both training and test data to prevent information leakage.
-   **Sequence Creation:** The scaled data is transformed into overlapping sequences of `LSTM_TIME_STEPS = 30` time steps, creating input tensors of shape `[samples, 30, features]`. To manage computational cost during development, the last 50,000 training samples are used for sequence creation.
-   **Architecture:** A `tf.keras.Sequential` model consisting of:
    -   An LSTM layer with 50 units and `return_sequences=False`
    -   A Dropout layer with rate 0.2 for regularisation
    -   A Dense output layer with 1 unit and `sigmoid` activation for binary classification
-   **Training Configuration:**
    -   Optimiser: Adam
    -   Loss Function: Binary Cross-Entropy
    -   Epochs: 5 (with Early Stopping on validation loss, patience=3, restoring best weights)
    -   Batch Size: 64
    -   Validation Split: 10%
-   **Early Stopping:** An `EarlyStopping` callback monitors `val_loss` and halts training if no improvement is observed for 3 consecutive epochs, restoring the weights from the best epoch. This prevents overfitting and optimises training duration.

## 3.7 Tools and Technologies

The selection of tools was driven by the need for performance, scalability, and reproducibility in a research context.

| Tool | Purpose | Justification |
| :--- | :--- | :--- |
| **Python 3.13** | Programming language | Industry-standard for data science and ML |
| **Polars** | Data manipulation and merging | Multi-threaded, query-optimised architecture delivering significant performance improvements over Pandas for large-scale data processing; empirically validated for superior performance and energy efficiency (Nahrstedt et al., 2024) |
| **LightGBM** | Primary ML model | High speed, efficiency, native support for quantile regression and custom loss functions |
| **TensorFlow / Keras** | LSTM deep learning model | Mature and well-supported deep learning framework |
| **Scikit-learn** | Preprocessing and metrics | MinMaxScaler, classification metrics, standard ML utilities |
| **SHAP** | Model explainability (XAI) | `TreeExplainer` provides theoretically grounded, per-prediction feature importance for LightGBM models (Ucar, 2023) |
| **Streamlit** | Interactive web dashboard | Rapid development of data applications in Python |
| **Plotly** | Interactive visualisation | Dark-themed, interactive charts suitable for "control room" aesthetics |
| **Open-Meteo API** | Weather data acquisition | High-quality historical and forecast weather data, freely accessible |
| **NESO CKAN API** | Grid & inertia data | Official source for UK grid frequency and market data |

## 3.8 Evaluation Strategy

The research component of this project is designed to rigorously test the central hypothesis through a quantitative, experimental design.

### 3.8.1 Hypotheses

To formalise the experimental evaluation, the following hypotheses are defined:

-   **Null Hypothesis (H₀):** There is no significant difference in predictive accuracy (measured by RMSE and Recall) between LightGBM, LSTM, and SARIMAX models for forecasting UK grid frequency instability.
-   **Alternative Hypothesis (H₁):** LightGBM, when augmented with physics-informed ramp-rate features from the OpSDA, will achieve a significantly lower RMSE and higher Recall for instability events compared to LSTM and SARIMAX.

### 3.8.2 Performance Metrics

Model performance is evaluated using the following metrics, each addressing a different dimension of system quality:

**Regression Metrics (for Quantile Models):**
-   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values.
-   **Root Mean Squared Error (RMSE):** Penalises large errors more heavily than MAE, appropriate for stability applications where large prediction errors are disproportionately costly.
-   **Pinball Loss:** Evaluates the calibration of quantile predictions, asymmetrically penalising errors depending on the quantile being estimated.
-   **Prediction Interval Coverage Probability (PICP):** The proportion of actual values falling within the predicted lower–upper bound interval. For an 80% confidence interval (10th to 90th percentile), the target PICP is 80%.
-   **Mean Prediction Interval Width (MPIW):** The average width of the predicted interval, measuring the sharpness of the uncertainty estimate. A narrower interval (lower MPIW) with maintained coverage indicates a more informative model.

**Classification Metrics (for Alert/Classifier Models):**
-   **Precision:** Proportion of triggered alerts that correspond to genuine instability events. Measuring the false alarm rate.
-   **Recall:** Proportion of genuine instability events that are successfully detected. **In this context, Recall is the paramount safety metric**, as a missed detection (False Negative) represents a potentially catastrophic failure.
-   **F1-Score:** Harmonic mean of Precision and Recall, providing a balanced measure.
-   **AUC-ROC:** Area Under the Receiver Operating Characteristic curve, measuring the model's ability to discriminate between stable and unstable states across all classification thresholds.

### 3.8.3 Reliability and Validity

-   **Internal Validity (Explainability):** To ensure the model learns physical laws rather than statistical noise or spurious correlations, **SHAP values** are computed for the LightGBM models. High SHAP values for `wind_ramp_rate` during instability events, and for `rocof` as the dominant overall feature, would confirm that the model is identifying the correct physical drivers, thus ensuring internal validity.

-   **External Validity (Generalisability):** The system's real-world applicability is validated by testing its performance exclusively on the **9 August 2019 blackout** data, which was held out from all training. Success is defined as the model's ability to trigger an alert *before* the critical frequency drop occurred—demonstrating external validity against an unseen, extreme, and historically documented event.

-   **Reliability (Reproducibility):** Model reliability is ensured through fixed random seeds (`random_state=42`), deterministic data splitting, and the explicit documentation of all hyperparameters in `config.py`. The use of cached API responses ensures that repeated runs of the pipeline produce identical results given the same configuration.

## 3.9 Challenges and Limitations

### 3.9.1 Implementation Challenges

Several key implementation challenges were addressed during the development of the GridGuardian system:

-   **Multi-Resolution Data Synchronisation:** The primary challenge was merging data sources with vastly different temporal resolutions (1-second frequency, hourly weather, daily inertia). This was addressed through the `join_asof` strategy described in Section 3.4.4, using Polars for computational efficiency.
-   **Severe Class Imbalance:** Instability events are extremely rare in the training data, creating a heavily imbalanced classification problem. This was addressed through the use of `scale_pos_weight` in LightGBM and by prioritising Recall as the key evaluation metric for classifiers.
-   **Computational Scale:** The 1-second resolution dataset for a full month contains approximately 2.7 million records. The Polars library was specifically selected over Pandas for its multi-threaded processing capabilities, which proved essential for making the feature engineering pipeline tractable.

### 3.9.2 Research Limitations

The following limitations are acknowledged:

-   **Data Resolution Constraints:** The model is limited by the resolution of the available input data. Grid frequency is available at 1-second resolution, but some transient events occur at sub-second timescales. Similarly, the daily resolution of inertia cost data limits its utility as a predictor for second-by-second forecasts. Access to higher-resolution inertia or generation mix data would likely improve model performance.
-   **Weather Forecast Error Propagation:** Any inherent error in the source weather data (or, in a live deployment, weather forecasts) will propagate through the feature engineering pipeline and into the model's predictions. The accuracy of the alert system is therefore bounded by the accuracy of the weather inputs.
-   **Geographic Aggregation:** Weather data is obtained for a single geographic point (the approximate centre of the UK). The UK's wind and solar resources are geographically distributed, and a single-point weather measurement may not capture localised weather events that affect specific generation sites.
-   **Single Test Period:** The validation is conducted on a single test day (9 August 2019). While this represents the most demanding possible stress test, a more comprehensive evaluation would include validation across multiple distinct periods and seasons to assess the model's robustness to varying conditions.

## 3.10 Ethical Considerations

The deployment of Artificial Intelligence in the management of critical national infrastructure raises significant ethical questions. This research addresses these through a principled **"Human-in-the-Loop"** design philosophy.

### 3.10.1 Human-in-the-Loop Design

The GridGuardian system is designed strictly as a **Decision Support Tool**, not as an autonomous controller. The system provides predictions, uncertainty bounds, and feature-level explanations to enhance the situational awareness and decision-making capability of human operators. Critically, it does not make or execute any control actions autonomously. The accountability for safety-critical decisions—such as initiating demand response measures or ordering generator dispatch changes—remains firmly with qualified human operators. This design philosophy ensures that the system augments, rather than replaces, human expertise and judgement in the control room.

### 3.10.2 Transparency and Accountability through XAI

The integration of **Explainable AI (XAI)** via SHAP values is a core ethical feature of the system's design. By providing a clear, per-prediction explanation of the factors driving each forecast, the system allows operators to scrutinise the model's reasoning, identify potential anomalies or unreliable predictions, and make informed decisions about whether to act on an alert. This transparency addresses the "Right to Explanation" principle and is fundamental to building the trust required for the adoption of AI in safety-critical applications (Ucar, 2023; Drewnick et al., 2025).

### 3.10.3 Energy Justice

By providing tools designed to prevent widespread blackouts and grid disruptions, this project contributes to the goal of energy justice. Power disruptions disproportionately affect vulnerable populations—including the elderly, individuals dependent on medical equipment, and those in fuel poverty—who are least equipped to cope with the consequences of power loss. A more stable and resilient grid, enabled by proactive alerting, therefore serves a broader social good.

### 3.10.4 Data Protection

The project exclusively utilises publicly available, anonymised, and aggregated grid data sourced from NESO and weather data from the Open-Meteo API. No personally identifiable information (PII) is collected, processed, or stored at any stage. The research therefore complies with the requirements of the UK Data Protection Act 2018 and the UK GDPR, and presents no direct privacy risks.
