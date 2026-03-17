# GridGuardian v2: Dissertation Report Plan

## Overview
This document outlines the comprehensive plan for writing the GridGuardian v2 dissertation report. The report will be written in Harvard referencing style and will justify all tools and methodologies used.

---

## Dissertation Structure

### Front Matter
- Title Page
- Abstract (250-300 words)
- Acknowledgements
- Table of Contents
- List of Figures
- List of Tables
- List of Abbreviations

### Main Body (7 Chapters)

---

## Chapter 1: Introduction

### 1.1 Background and Context
**Content Source:** [`UK_Power_Grid_Stability_Report.md`](UK_Power_Grid_Stability_Report.md:1), [`supervisor_project_pitch.md`](supervisor_project_pitch.md:1)

**Key Points:**
- UK energy transition from fossil fuels to renewables
- The "Inertia Crisis" - loss of system inertia due to renewable integration
- Grid frequency operational limits (49.8-50.2 Hz)
- August 9, 2019 blackout case study (1.1M customers affected)

**Required Citations:**
- National Grid ESO reports on inertia crisis
- UK Grid Code frequency limits documentation
- Official August 2019 blackout investigation reports

### 1.2 Problem Statement
**Content Source:** [`project_summary_report.md`](project_summary_report.md:1), [`project_brief_and_implementation_details.md`](project_brief_and_implementation_details.md:1)

**Key Points:**
- Reactive vs proactive grid management
- Limitations of current monitoring systems
- Need for probabilistic forecasting with explainability

### 1.3 Research Objectives
**Content Source:** [`supervisor_project_pitch.md`](supervisor_project_pitch.md:86)

**Objectives:**
1. Develop a physics-informed ML framework for grid stability prediction
2. Implement quantile regression for probabilistic uncertainty bands
3. Integrate explainable AI (SHAP) for operational trust
4. Validate against the August 2019 blackout event
5. Create a real-time monitoring dashboard for grid operators

### 1.4 Research Contributions
**Content Source:** [`project_detailed_analysis_and_improvements.md`](project_detailed_analysis_and_improvements.md:1)

**Contributions:**
- Novel integration of OpSDA algorithm for wind ramp detection
- Probabilistic risk framework using quantile regression
- Real-time SHAP explainability in control room dashboard
- High-performance Polars data pipeline for 1-second resolution data

### 1.5 Dissertation Structure
Outline of remaining chapters

---

## Chapter 2: Literature Review

### 2.1 The Inertia Crisis: Technical Background
**Content Source:** [`UK_Power_Grid_Stability_Report.md`](UK_Power_Grid_Stability_Report.md:10)

**Key Topics:**
- Definition of system inertia in power grids
- Impact of inverter-based resources (IBR) on grid stability
- RoCoF (Rate of Change of Frequency) and frequency nadir
- Relationship between inertia and frequency stability

**Required Academic References:**
- Papers on low-inertia power systems
- IEEE/Elsevier journals on grid stability
- NGESO Stability Pathfinder reports

### 2.2 UK Grid Stability Challenges
**Content Source:** [`UK_Power_Grid_Stability_Report.md`](UK_Power_Grid_Stability_Report.md:11)

**Key Topics:**
- Historical power outages (2003, 2008, 2019)
- August 9, 2019 blackout detailed analysis
- System strength and voltage management issues
- Infrastructure limitations

**Required Citations:**
- Ofgem investigation reports
- National Grid ESO incident reports
- Academic analysis of 2019 blackout

### 2.3 Machine Learning Approaches for Power Systems
**Content Source:** [`project_summary_report.md`](project_summary_report.md:12), [`implementation_report.md`](Implementation/implementation_report.md:44)

**Key Topics:**
- Time-series forecasting methods (SARIMAX, ARIMA)
- Gradient Boosting Machines (LightGBM, XGBoost)
- Deep Learning approaches (LSTM, GRU, Transformers)
- Comparative analysis for power system applications

**Required Academic References:**
- LightGBM original paper (Ke et al., 2017)
- LSTM for time-series forecasting papers
- Comparative ML studies in power systems

### 2.4 Quantile Regression for Uncertainty Quantification
**Content Source:** [`project_summary_report.md`](project_summary_report.md:21), [`implementation_report.md`](Implementation/implementation_report.md:44)

**Key Topics:**
- Principles of quantile regression
- Pinball loss function
- Prediction Interval Coverage Probability (PICP)
- Applications in risk assessment

**Required Academic References:**
- Koenker's quantile regression foundational work
- Recent papers on quantile regression in energy forecasting

### 2.5 Explainable AI in Critical Infrastructure
**Content Source:** [`project_brief_and_implementation_details.md`](project_brief_and_implementation_details.md:11), [`supervisor_project_pitch.md`](supervisor_project_pitch.md:87)

**Key Topics:**
- SHAP (SHapley Additive exPlanations) theory
- TreeExplainer for tree-based models
- Importance of interpretability in operational settings
- Trust and adoption of AI systems

**Required Academic References:**
- Lundberg & Lee SHAP paper (2017)
- XAI in critical infrastructure papers
- Human-AI interaction studies

### 2.6 Feature Engineering for Power Systems
**Content Source:** [`project_summary_report.md`](project_summary_report.md:35), [`implementation_report.md`](Implementation/implementation_report.md:39)

**Key Topics:**
- OpSDA (Optimized Swinging Door Algorithm) for ramp detection
- RoCoF calculation methods
- Lag features and autoregressive modeling
- Renewable penetration proxies

**Required Academic References:**
- Swinging Door Algorithm original papers
- Power system feature engineering studies

### 2.7 Research Gap Identification
**Content Source:** [`project_summary_report.md`](project_summary_report.md:13), [`project_detailed_analysis_and_improvements.md`](project_detailed_analysis_and_improvements.md:1)

**Identified Gaps:**
1. Lack of integrated physics-informed ML frameworks for UK grid
2. Limited probabilistic forecasting with explainability
3. Need for real-time operational dashboards
4. Granularity mismatch in available inertia data

---

## Chapter 3: Methodology

### 3.1 Research Design
**Content Source:** [`project_summary_report.md`](project_summary_report.md:17)

**Key Topics:**
- Physics-informed data science framework
- Design science research methodology
- Proactive vs reactive monitoring paradigm

### 3.2 Software Development Life Cycle
**Content Source:** [`project_summary_report.md`](project_summary_report.md:28)

**Key Topics:**
- CRISP-DM methodology adaptation
- Evolutionary Prototyping for dashboard UI/UX
- Iterative development approach

### 3.3 Data Sources
**Content Source:** [`Implementation/src/data_loader.py`](Implementation/src/data_loader.py:1), [`Implementation/data/apis.md`](Implementation/data/apis.md:1)

**Data Sources:**
1. **NESO CKAN API**: 1-second grid frequency data
2. **Open-Meteo API**: Hourly weather data (wind speed, solar radiation, temperature)
3. **Inertia Cost Data**: Daily market-based inertia proxies

**Data Validation:**
- Type checking and casting
- Range validation
- Missing data handling

### 3.4 Feature Engineering Strategy
**Content Source:** [`Implementation/src/feature_engineering.py`](Implementation/src/feature_engineering.py:1), [`project_summary_report.md`](project_summary_report.md:35)

**Features:**
1. **Wind Ramp Rate** (OpSDA): Detects sudden changes in wind generation
2. **RoCoF**: 5-second smoothed rate of change of frequency
3. **Renewable Penetration Ratio**: Synthetic proxy for grid inertia
4. **Volatility Metrics**: 10-second rolling standard deviation
5. **Lag Features**: 1s, 5s, 60s historical frequency values
6. **Temporal Features**: Hour, minute embeddings

### 3.5 Model Selection Justification
**Content Source:** [`project_summary_report.md`](project_summary_report.md:23), [`Implementation/src/model_trainer.py`](Implementation/src/model_trainer.py:1)

**Models:**
1. **LightGBM Quantile Regression** (Primary)
   - Justification: Efficiency, accuracy, native quantile support
   - Alpha values: 0.1 (lower bound), 0.9 (upper bound)

2. **LSTM** (Secondary/Benchmark)
   - Justification: Temporal dependency modeling
   - Architecture: 30 time steps, 5 epochs

3. **LightGBM Classifier** (Baseline)
   - Justification: Binary instability classification baseline

### 3.6 Evaluation Metrics
**Content Source:** [`Implementation/evaluate_models.py`](Implementation/evaluate_models.py:36), [`implementation_report.md`](Implementation/implementation_report.md:63)

**Metrics:**
1. **Mean Absolute Error (MAE)**: Prediction accuracy
2. **Pinball Loss**: Quantile regression accuracy
3. **PICP**: Prediction Interval Coverage Probability
4. **MPIW**: Mean Prediction Interval Width
5. **Calibration Score**: Fraction below predicted quantile
6. **AUC-ROC**: Classification performance

### 3.7 Alert Logic
**Content Source:** [`Implementation/src/config.py`](Implementation/src/config.py:30), [`project_summary_report.md`](project_summary_report.md:73)

**Logic:**
- Time to Alert (TTA): 10 seconds
- Threshold: 49.8 Hz (user-adjustable 49.5-49.9 Hz)
- Trigger: Predicted lower bound < threshold

---

## Chapter 4: Implementation

### 4.1 System Architecture Overview
**Content Source:** [`supervisor_project_pitch.md`](supervisor_project_pitch.md:28), [`system_architecture.mmd`](system_architecture.mmd:1)

**Architecture Diagram:** (Include Mermaid diagram from supervisor_project_pitch.md)

**Components:**
- Data Ingestion Layer
- Physics-Informed Feature Engineering
- Predictive Layer (Quantile Regression)
- Control Room Dashboard

### 4.2 Data Pipeline Implementation
**Content Source:** [`Implementation/src/data_loader.py`](Implementation/src/data_loader.py:1), [`Implementation/src/feature_engineering.py`](Implementation/src/feature_engineering.py:9)

**Key Implementation Details:**
- Polars for high-performance data manipulation
- `join_asof` for temporal merging
- Retry logic with tenacity for API calls
- Request caching with requests_cache

**Code Excerpt:** [`merge_datasets()`](Implementation/src/feature_engineering.py:9)

### 4.3 Feature Engineering Implementation
**Content Source:** [`Implementation/src/opsda.py`](Implementation/src/opsda.py:1), [`Implementation/src/feature_engineering.py`](Implementation/src/feature_engineering.py:42)

**Key Algorithms:**
- OpSDA compress function for wind ramp detection
- 5-second rolling average for RoCoF smoothing
- Renewable penetration ratio calculation

**Code Excerpt:** [`calculate_wind_ramp_rate()`](Implementation/src/feature_engineering.py:42), [`compress()`](Implementation/src/opsda.py:5)

### 4.4 Model Training Pipeline
**Content Source:** [`Implementation/src/model_trainer.py`](Implementation/src/model_trainer.py:1), [`Implementation/run_pipeline.py`](Implementation/run_pipeline.py:1)

**Implementation Details:**
- Train/test split: Before Aug 9, 2019 / Aug 9-10, 2019
- Class imbalance handling with `scale_pos_weight`
- MinMax scaling for LSTM
- Model serialization with joblib and Keras

**Code Excerpt:** [`train_quantile_model()`](Implementation/src/model_trainer.py:96), [`train_and_evaluate_lgbm_classifier()`](Implementation/src/model_trainer.py:45)

### 4.5 Streamlit Dashboard Implementation
**Content Source:** [`Implementation/app.py`](Implementation/app.py:1), [`project_brief_and_implementation_details.md`](project_brief_and_implementation_details.md:28)

**Key Features:**
- Persistent Parquet caching with source code hashing
- Real-time SHAP calculation
- Interactive time simulation slider
- Dark mode "control room" aesthetic

**Code Excerpt:** [`get_src_hash()`](Implementation/app.py:23), [`get_current_row_safely()`](Implementation/app.py:45)

### 4.6 Tool Justification
**Content Source:** [`Implementation/pyproject.toml`](Implementation/pyproject.toml:1), [`Implementation/requirements.txt`](Implementation/requirements.txt:1)

| Tool | Purpose | Justification |
|------|---------|---------------|
| Python 3.13 | Core language | Rich ML ecosystem, readability |
| Polars | Data manipulation | 10x faster than Pandas, lazy evaluation |
| LightGBM | Gradient boosting | Native quantile regression, efficiency |
| TensorFlow/Keras | Deep learning | LSTM implementation, GPU support |
| Streamlit | Dashboard | Rapid prototyping, interactive widgets |
| SHAP | Explainability | Model-agnostic, theoretical foundations |
| Plotly | Visualization | Interactive charts, dark theme support |
| Open-Meteo | Weather data | Free API, historical archive |
| NESO CKAN API | Grid data | Official source, 1-second resolution |
| uv | Package management | Fast dependency resolution |

---

## Chapter 5: Results & Evaluation

### 5.1 Model Performance Metrics
**Content Source:** [`implementation_report.md`](Implementation/implementation_report.md:63), [`Implementation/evaluate_models.py`](Implementation/evaluate_models.py:1)

**Results:**
- Lower Bound MAE: 0.033 Hz
- Upper Bound MAE: 0.018 Hz
- PICP: 77.5% (target: 80%)
- Calibration: 1.8% below 10th percentile (target: 10%)

### 5.2 August 9, 2019 Blackout Validation
**Content Source:** [`implementation_report.md`](Implementation/implementation_report.md:84), [`project_detailed_analysis_and_improvements.md`](project_detailed_analysis_and_improvements.md:23)

**Validation Results:**
- Alert triggered before frequency collapse
- Pessimistic bias provides safety margin
- Model successfully learned grid physics

### 5.3 Feature Importance Analysis
**Content Source:** [`implementation_report.md`](Implementation/implementation_report.md:78), [`project_brief_and_implementation_details.md`](project_brief_and_implementation_details.md:62)

**SHAP Results:**
1. RoCoF: Most dominant feature
2. Wind Ramp Rate (OpSDA): Higher importance than raw wind speed
3. Grid Frequency: Strong predictor
4. Inertia Cost: Lowest (due to daily granularity)

### 5.4 Dashboard Validation
**Content Source:** [`project_brief_and_implementation_details.md`](project_brief_and_implementation_details.md:28)

**Validation:**
- Real-time performance with caching
- SHAP calculation latency
- User interface usability

### 5.5 Comparison with Baseline Approaches
**Content Source:** [`project_summary_report.md`](project_summary_report.md:41)

**Comparison:**
- LightGBM vs LSTM performance
- Quantile regression vs binary classification
- Physics-informed features vs raw data

---

## Chapter 6: Discussion

### 6.1 Critical Evaluation of Strengths
**Content Source:** [`project_detailed_analysis_and_improvements.md`](project_detailed_analysis_and_improvements.md:7)

**Strengths:**
- Probabilistic risk framework
- Domain-specific feature engineering (OpSDA)
- High-performance Polars pipeline
- SHAP integration for transparency

### 6.2 Limitations
**Content Source:** [`project_detailed_analysis_and_improvements.md`](project_detailed_analysis_and_improvements.md:16)

**Limitations:**
1. **Granularity Gap**: Daily inertia costs vs 1-second frequency
2. **Evaluation Bias**: Single month (August 2019) testing
3. **Noisy RoCoF**: Simple diff calculation
4. **LSTM Non-Deployment**: Not integrated in dashboard
5. **Seasonal Bias**: Model may not generalize to other months

### 6.3 Future Work Recommendations
**Content Source:** [`project_detailed_analysis_and_improvements.md`](project_detailed_analysis_and_improvements.md:40), [`supervisor_project_pitch.md`](supervisor_project_pitch.md:91)

**Recommendations:**
1. **Intervention Simulator**: "What-if" analysis for grid operators
2. **Dynamic Renewable Penetration Proxy**: Replace daily inertia costs
3. **Calibration Improvements**: Reliability diagrams, quantile calibration
4. **Smoothed RoCoF**: Savitzky-Golay filter or linear regression
5. **LSTM Integration**: Ensemble with LightGBM or uncertainty indicator
6. **Out-of-Season Testing**: Validate on different months

### 6.4 Practical Implications for Grid Operators
**Content Source:** [`supervisor_project_pitch.md`](supervisor_project_pitch.md:17)

**Implications:**
- 10-second warning enables automated battery response
- Explainability builds operational trust
- Probabilistic bands support risk-based decision making

---

## Chapter 7: Conclusion

### 7.1 Summary of Key Findings
**Content Source:** All existing reports

**Findings:**
- Quantile regression provides actionable uncertainty bands
- OpSDA successfully captures wind ramp events
- SHAP explainability is feasible in real-time
- Pessimistic bias is desirable for safety-critical systems

### 7.2 Research Contributions Restated
- Physics-informed ML framework for UK grid
- Real-time explainable alert system
- High-performance data pipeline

### 7.3 Impact for UK Grid Stability
- Supports NESO's zero-carbon 2025 target
- Enables proactive vs reactive management
- Demonstrates AI trustworthiness for critical infrastructure

### 7.4 Final Recommendations
- Deploy with intervention simulator
- Integrate with actual grid control systems
- Continuous calibration with live data

---

## References

### Academic Papers (Required)
1. Ke, G., et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', *Advances in Neural Information Processing Systems*.
2. Lundberg, S.M. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', *Advances in Neural Information Processing Systems*.
3. Hochreiter, S. and Schmidhuber, J. (1997) 'Long Short-Term Memory', *Neural Computation*.
4. Koenker, R. and Hallock, K.F. (2001) 'Quantile Regression', *Journal of Economic Perspectives*.

### Grid Stability References (Required)
1. National Grid ESO (2019) 'August 9th 2019 Power Outage Report'.
2. National Grid ESO (2022) 'Stability Pathfinder Phase 2 Report'.
3. Ofgem (2020) 'Investigation into the 9 August 2019 Power Outage'.
4. UK Grid Code documents on frequency limits.

### Technical Documentation
1. Polars Documentation
2. LightGBM Documentation
3. Streamlit Documentation
4. SHAP Documentation
5. Open-Meteo API Documentation

---

## Appendices

### Appendix A: Code Listings
- [`opsda.py`](Implementation/src/opsda.py:1) - Complete OpSDA implementation
- [`create_features()`](Implementation/src/feature_engineering.py:78) - Feature engineering function
- [`train_quantile_model()`](Implementation/src/model_trainer.py:96) - Quantile regression training

### Appendix B: Additional Figures
- System architecture diagram
- SHAP summary plots
- Quantile calibration plots
- August 9, 2019 event timeline

### Appendix C: Data Dictionary
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | UTC timestamp (1-second resolution) |
| grid_frequency | float | Grid frequency in Hz |
| rocof | float | Rate of Change of Frequency (Hz/s) |
| wind_speed | float | Wind speed (m/s) |
| wind_ramp_rate | float | OpSDA-calculated ramp rate |
| inertia_cost | float | Daily inertia cost (£/MW) |
| renewable_penetration_ratio | float | Synthetic inertia proxy |
| target_is_unstable | int | Binary instability target |
| target_freq_next | float | Frequency TTA seconds ahead |

---

## Writing Schedule

| Week | Task |
|------|------|
| 1 | Chapters 1-2 (Introduction, Literature Review) |
| 2 | Chapters 3-4 (Methodology, Implementation) |
| 3 | Chapters 5-6 (Results, Discussion) |
| 4 | Chapter 7 (Conclusion), References, Appendices |
| 5 | Review, proofreading, formatting |

---

## Next Steps

1. **Create the "Final Report" folder**
2. **Gather academic references** for literature review
3. **Start writing Chapter 1: Introduction**
4. **Proceed chapter by chapter**
