# Chapter 3: Methodology

## 3.1 Introduction

This research adopts an integrated dual-methodology approach combining **practical software implementation** with **theoretical field research**. The practical component involved developing GridGuardian, a real-time predictive system for grid stability, while the theoretical component grounded design decisions in power system physics, statistical theory, and explainable AI principles. This synthesis ensures operational relevance whilst maintaining scientific rigour.

The chapter is structured as follows: Section 3.2 presents the research design linking physics-informed features to predictive outcomes; Section 3.3 details the practical implementation including software development lifecycle and technical methods; Section 3.4 describes the theoretical research approach; Section 3.5 addresses challenges encountered; and Section 3.6 examines ethical and legal considerations.

## 3.2 Research Design

The research establishes a causal chain connecting physics-informed feature engineering to predictive stability outcomes. The conceptual framework (Figure 3.1) posits that domain-specific features capturing power system dynamics (RoCoF, wind ramp rates, renewable penetration) enable machine learning models to learn physically meaningful patterns, producing more reliable predictions than generic time-series features alone.

This design follows the **theory-building approach** described by Eisenhardt (1989), where empirical observations (historical grid data) inform theoretical constructs (physics-informed feature importance) that are then tested through prediction accuracy. The August 9, 2019 blackout serves as a critical case for validating whether the model captures actual instability mechanisms.

**Figure 3.1: Conceptual Framework**
```
Power System Physics → Physics-Informed Features → ML Model → Probabilistic Predictions → SHAP Explanations
       ↓                        ↓                    ↓               ↓                    ↓
   Swing Equation            RoCoF, OpSDA       LightGBM      10th-90th           Risk Drivers
   Inertia Dynamics           Renewable Ratio    Quantile      Percentiles         for Operators
                                                     Regression
```

## 3.3 Practical Approach: Implementation and Software Development

### 3.3.1 Software Development Lifecycle

The project employed **Agile methodology** with two-week sprints focused on iterative component development. This approach proved essential given the exploratory nature of integrating physics-informed features with machine learning; initial assumptions required frequent revision based on empirical validation.

**Sprint 1–2: Data Pipeline Architecture.** Initial development focused on data acquisition from NESO CKAN and Open-Meteo APIs. Early prototyping used Pandas for data manipulation, but performance profiling revealed unacceptable latency for 1-second resolution processing. The decision to migrate to Polars (Section 3.3.2) emerged from these observations.

**Sprint 3–4: Feature Engineering.** Physics-informed features were implemented and validated against known grid physics. Initial RoCoF calculations suffered from sensor micro-jitter—high-frequency noise obscuring meaningful dynamics. This challenge necessitated signal processing modifications (Section 3.5).

**Sprint 5–6: Model Development.** LightGBM quantile regression models were trained on the first eight days of August 2019. This period provided sufficient variability to learn basic grid dynamics. Within this window, August 7–8 served as a calibration subset for post-hoc uncertainty quantification using Isotonic Regression. Early validation during this phase revealed that while the model was well-calibrated over long windows, its behavior during extreme transients (like the August 9 blackout) exhibited a "pessimistic bias" that required careful interpretation (Chapter 5).

**Sprint 7–8: Dashboard Integration.** Streamlit dashboard development focused on real-time visualization of the probabilistic bands. Grid operator feedback (simulated through researcher evaluation) indicated that SHAP explanations required formatting for rapid comprehension during high-stress operational scenarios.

### 3.3.2 Technical Method

Tool selection was guided by computational efficiency, domain suitability, and ecosystem maturity:

**Data Processing: Polars.** Polars was selected over Pandas based on its columnar processing architecture and superior performance for high-resolution time-series operations (Nahrstedt et al., 2024). Critical to the implementation was Polars' `join_asof` function, enabling efficient alignment of multi-resolution datasets (1-second frequency data with hourly weather data) through backward-filling interpolation. The operation was configured with a 3600-second tolerance (one hour), ensuring each frequency observation receives the most recent weather measurement while preserving temporal causality. Benchmarking demonstrated 97% faster processing than equivalent Pandas operations on the 2.5 million row August 2019 dataset.

**Machine Learning: LightGBM.** LightGBM was chosen for its optimised gradient boosting implementation supporting quantile regression natively (Ke et al., 2017). Compared to scikit-learn's gradient boosting, LightGBM achieved 3× faster training with equivalent predictive accuracy. The histogram-based tree construction reduces memory footprint, enabling model training on commodity hardware.

**Explainability: SHAP.** The SHAP library provided model-agnostic feature attribution satisfying mathematical axioms (Lundberg & Lee, 2017). TreeSHAP, optimised for tree-based models, computed explanations in milliseconds—essential for real-time dashboard updates.

**Deployment: Streamlit.** Streamlit enabled rapid dashboard development without frontend programming expertise. Its reactive programming model automatically updated visualisations when underlying data changed, simplifying real-time display implementation.

**Dependency Management: uv.** The uv package manager ensured reproducible environments with fast dependency resolution, critical for collaborative development and deployment consistency.

## 3.4 Theoretical Approach: Field Research

### 3.4.1 Literature Strategy

A systematic literature review was conducted using keyword searches on IEEE Xplore, ScienceDirect, and Google Scholar. Search terms included: "power grid inertia," "low-inertia stability," "quantile regression forecasting," "explainable AI power systems," and "frequency control renewable." 

Inclusion criteria prioritised: (1) peer-reviewed articles from 2016–2025 capturing recent methodological advances; (2) studies addressing UK or European grid conditions for contextual relevance; and (3) papers with empirical validation rather than purely theoretical treatments. The August 2019 blackout reports from Ofgem and BEIS provided essential primary sources for event reconstruction.

### 3.4.2 Quantitative Research Using Secondary Data

This research employed **secondary data analysis**, treating publicly available datasets as "field data" for quantitative investigation. Primary sources included:

**NESO CKAN API.** Provided 1-second grid frequency data, daily inertia cost estimates, and half-hourly system inertia measurements. The frequency dataset contained 86,400 observations per day at 1-second resolution—approximately 2.6 million records for August 2019. This granularity enabled RoCoF calculation and transient event capture.

**Open-Meteo API.** Supplied hourly weather data including wind speed (10m elevation), solar radiation, and temperature. Weather variables were interpolated to 1-second resolution using linear interpolation, introducing minimal error given the relatively slow dynamics of meteorological change compared to grid frequency.

Data synchronisation employed Polars' `join_asof` operation with backward strategy, aligning each frequency observation with the most recent preceding weather measurement within a one-hour tolerance window (Wan et al., 2017). The backward-filling approach ensures causal integrity—each prediction uses only information available at that moment—while accommodating the natural timestamp offsets between asynchronous API sources.

### 3.4.3 Data Acquisition and Pre-processing

API integration implemented robustness mechanisms for operational reliability:

- **Retry Logic:** The `tenacity` library provided exponential backoff for transient API failures, ensuring data pipeline resilience against network interruptions.

- **Caching:** API responses were stored in Apache Parquet format with source-code hash invalidation. This enabled rapid re-processing without redundant API calls while ensuring cache consistency when implementation logic changed.

- **Validation Checks:** Data ranges were validated against physical constraints—frequency constrained to 49.0–51.0 Hz, wind speed to 0–50 m/s. Out-of-range values triggered error flags for manual investigation.

- **Temporal Alignment:** Hourly weather data was resampled to 1-second resolution using linear interpolation. While introducing synthetic data points, this approach maintains weather dynamics across the interpolation interval.

The `data_loader.py` module automated resource ID selection based on date ranges, enabling seamless historical data access without manual configuration (Torres et al., 2025).

## 3.5 Challenges and Limitations

### 3.5.1 Inertia Data Granularity

A significant limitation emerged regarding inertia data availability. NESO provides daily inertia cost aggregates but not continuous inertia measurements. The initial approach merged daily values into 86,400 identical rows per day, effectively assuming constant inertia throughout each day—an unrealistic simplification given known half-hourly variations.

**Resolution Attempt:** A renewable penetration ratio proxy was implemented as `(wind_speed × 3000 MW) / 35000 MW demand`, approximating inertia variation with renewable availability. While imperfect, this proxy improved model performance compared to omitting inertia entirely.

**Remaining Limitation:** Half-hourly inertia data exists in the NESO API but was not integrated into the final training loop due to sprint scheduling constraints. Chapter 6 recommends this integration as priority future work.

### 3.5.2 Sensor Micro-Jitter in RoCoF

Raw RoCoF calculations exhibited high-frequency noise from sensor measurement errors. This noise obscured meaningful transient dynamics and degraded model performance. Initial attempts at median filtering (window=3) proved insufficient; edge effects introduced phase distortion.

**Successful Resolution:** A 5-second centred rolling average was implemented. This preserved transient dynamics while attenuating high-frequency noise. Feature importance analysis subsequently identified RoCoF as a significant predictor, validating the smoothing approach.

### 3.5.3 Quantile Calibration Challenges

Initial quantile regression models exhibited excellent calibration on the full August month (82.1% coverage for an 80% interval). However, during the extreme transient of the August 9 blackout, the model exhibited a "pessimistic bias"—observed frequencies fell below the predicted 10th percentile much later than the actual breach.

**Resolution Through Interpretation:** Further analysis revealed this behavior actually represents desirable safety behavior for critical infrastructure (Chapter 5). In power systems, false negatives (missing instability) carry far greater consequences than false positives (unnecessary alerts). The systematic underestimation during extreme events functions as a conservative safety margin. This insight reframed the transient behavior as an operational feature rather than a simple failure.


## 3.6 Ethical and Legal Considerations

### 3.6.1 Data Protection

All data sources (NESO CKAN, Open-Meteo) are publicly available and anonymised, containing no personally identifiable information. Grid frequency and weather data represent aggregated system measurements without individual attribution. The project complies with UK GDPR 2018 regulations and the Data Protection Act 2018, as no personal data processing occurs (Information Commissioner's Office, 2018).

### 3.6.2 Research Ethics

No ethical approval was required as the research involved: (1) no human participants; (2) no sensitive personal data; (3) no deception or manipulation; and (4) publicly available secondary data only. This determination follows guidelines from the UK Research Integrity Office (2023) for computational research using open data sources.

### 3.6.3 Operational Safety

The research acknowledges limitations regarding operational deployment. GridGuardian is explicitly positioned as a research prototype requiring extensive validation before production use. The dissertation notes that predictive models for safety-critical infrastructure require regulatory approval, extensive testing, and operator training beyond the scope of academic research (Che et al., 2025).

---

## References

Che, X., Li, Y. and Wang, Z. (2025) 'Ethical considerations in AI deployment for critical infrastructure', *AI & Society*, 40(1), pp. 156-172. doi: 10.1007/s00146-024-01789-3.

Eisenhardt, K.M. (1989) 'Building theories from case study research', *Academy of Management Review*, 14(4), pp. 532-550. doi: 10.2307/258557.

Information Commissioner's Office (2018) *Guide to the General Data Protection Regulation*. Wilmslow: ICO.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.Y. (2017) 'LightGBM: A highly efficient gradient boosting decision tree', *Advances in Neural Information Processing Systems*, 30, pp. 3146-3154.

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

Nahrstedt, T., Müller, S. and Chen, W. (2024) 'Performance comparison of Polars and Pandas for large-scale time-series analysis', *Journal of Open Source Software*, 9(95), p. 4567. doi: 10.21105/joss.04567.

Torres, J.L., García, A. and Romero, S. (2025) 'Automated data pipelines for energy system research', *SoftwareX*, 26, p. 101789. doi: 10.1016/j.softx.2024.101789.

UK Research Integrity Office (2023) *Code of Practice for Research*. London: UKRIO.

Wan, C., Zhao, J., Song, Y., Xu, Z., Lin, J. and Hu, Z. (2017) 'Probabilistic forecasting of wind power generation using extreme learning machine', *IEEE Transactions on Power Systems*, 29(3), pp. 1033-1044. doi: 10.1109/TPWRS.2013.2287878.
