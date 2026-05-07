= Methodology

== Introduction

This chapter details the methodology implemented to engineer the GridGuardian alerting system. The research follows an empirical software engineering approach, translating established power system physics into a functional machine learning prototype. The methodology is constrained entirely to the techniques, architectures, and data processing routines actively deployed in the project codebase. The complete implementation, including all data ingestion scripts, feature engineering pipelines, trained model artefacts, and dashboard outputs, is publicly available at: https://github.com/Fatema-Dr/UK-Grid-instability. This codebase serves as the definitive evidence of the project’s technical execution, verifying that GridGuardian is a functional engineering prototype rather than a conceptual model.

#figure(
  image("../figures/figure_3_1_system_architecture.png", width: 90%),
  caption: [High-level system architecture of the GridGuardian predictive prototype, detailing asynchronous data fusion, feature engineering, and the dual-model inference path.]
)

== Software Development Life Cycle (SDLC)

An Agile methodology managed the project lifecycle to accommodate the iterative nature of machine learning development. The project was structured into three distinct two-week sprints:

#figure(
  table(
    columns: (1.5fr, 3fr, 2fr),
    align: left,
    [*Sprint Phase*], [*Key Deliverables*], [*Risk Mitigation*],
    [Sprint 1: Data Eng (Weeks 1-2)], [NESO/Open-Meteo APIs, Polars `join_asof` pipeline], [Mitigated memory limits via Rust-based Polars over Pandas],
    [Sprint 2: ML Prototyping (Weeks 3-4)], [LightGBM baseline, quantile calibration, LSTM dual-architecture], [Mitigated structural overfitting via strict L1/L2 regularization],
    [Sprint 3: System Integration (Weeks 5-6)], [SHAP interpretability module, final threshold logic, empirical validation], [Mitigated inference latency via histogram binning in LightGBM]
  ),
  caption: [Agile Sprint Timeline and Project Management Risk Register.]
)

Version control (Git) was maintained throughout. To prevent look-ahead bias during data aggregation, continuous unit testing verified the chronological integrity of all temporal joins.

== Data Acquisition and Pre-processing

The system processes multi-resolution data from the National Energy System Operator (NESO) and the Open-Meteo API.

=== Asynchronous Data Alignment and Leakage Prevention

The core of the data fusion pipeline is the Rust-based Polars `join_asof` algorithm. This implementation achieves an alignment latency of less than 15ms for the entire August dataset, a critical technical requirement for sub-second operations. By enforcing a strict backward-filling causal constraint ($t_w <= t_f$), the system mathematically prevents the 'forward-looking' data leakage that often invalidates time-series forecasting projects. This alignment is not a conceptual design but is fully realized in the project’s `src/data_ingestion.py` module.

=== Hardware Portability and Latency Benchmarking

The system was developed on a Linux-based architecture (Ubuntu 22.04 LTS) to ensure maximum compatibility with grid-scale edge hardware. LightGBM @ke2017lightgbm was selected as the primary inference engine due to its histogram-based binning and leaf-wise (best-first) growth strategy, which allows for deeper, more asymmetrical trees than traditional depth-first algorithms. 

To ensure the ablation study results are operationally relevant, all inference latency benchmarks were conducted strictly on CPU-only hardware (AMD Ryzen 9 5950X), mirroring the power-constrained environment of grid substations where dedicated GPU resources are typically unavailable. The resulting 0.20s inference latency confirms the system’s computational viability for 1Hz telemetry processing, providing a reproducible CPU-only benchmark that approximates the computational constraint of power-limited edge deployments, though formal validation on certified substation hardware remains outside the scope of this prototype.

== Feature Engineering

Feature engineering translates raw telemetry into physically meaningful variables representing grid momentum and stress.

=== The Swinging Door Algorithm (OpSDA) Compression

To computationally extract the structural wind ramp rate, the system applies the Swinging Door Algorithm (OpSDA) @bristol1990swinging. By converting raw, high-frequency wind velocity into discrete OpSDA ramp rates, the feature engineering pipeline isolates sustained physical momentum shifts from irrelevant localized turbulence, directly providing the LightGBM engine with a clean derivative of meteorological stress.

=== Target Variable and Evaluative Horizon

A critical distinction must be made between the model's objective function and the system's operational alert output. The LightGBM architecture performs *continuous quantile regression* to probabilistically forecast the 10th and 90th percentile frequency boundaries targeting a 10-second temporal horizon ($t+10$). 

This specific window was selected to test whether a predictive alert could comfortably exceed the 1.0-second mechanical deployment constraint of Enhanced Frequency Response (EFR) batteries. It must be noted that this $t+10$ horizon is a *design target*; the realized predictive lead-time during a cascading fault is a function of the model’s realized sensitivity and the non-linear dynamics of the specific event, and is not a guaranteed constant of the architecture. A future state is flagged with a binary `target_is_unstable` alert if the model's predicted lower quantile breaches any of the following physical failure bounds:

1.  *Statutory Breach:* The predicted 10th percentile frequency drops below $49.85$ Hz.
2.  *Momentum Failure:* The predicted 10th percentile frequency drops below $49.95$ Hz while the current smoothed RoCoF is worse than $-0.02$ Hz/s.

== Machine Learning Architecture

=== Temporal Hold-Out Validation and Calibration Splits

Standard $k$-fold cross-validation is invalid for time-series data. The evaluation utilizes a strict chronological three-way split to ensure out-of-sample integrity:

1. *Training Set*: August 1–6, 2019 (~518,400 records).
2. *Calibration Set*: August 7–8, 2019 (Used for post-hoc isotonic recalibration).
3. *Testing Set*: August 9, 2019 (Including the discontinuous blackout event).

While the formal training boundary is set at August 9, the authors acknowledge that the calibration set (August 7-8) was part of the broader model development phase, representing a quantified risk of *calibration leakage* that is addressed during the final evaluation in Chapter 5.

=== Post-hoc Isotonic Recalibration

Initial evaluation demonstrated that the LightGBM quantile outputs exhibited a pessimistic calibration bias. To rectify this, Post-hoc Isotonic Recalibration @kuleshov2018accurate was applied. Isotonic regression fits a monotonically increasing mapping function to ensure that nominal quantile levels correspond to true empirical frequencies. This calibration is a post-processing step that does not modify the underlying LightGBM weights, allowing for a transparent account of the model's 'raw' vs. 'adjusted' performance ceilings.

=== Baseline Architecture: LSTM with Monte Carlo Dropout

To establish a rigorous performance benchmark, a deep recurrent Long Short-Term Memory (LSTM) network was implemented. To ensure a direct comparison with the probabilistic LightGBM engine, the LSTM utilizes Monte Carlo Dropout @gal2016dropout during inference. By maintaining dropout layers in an active state during prediction across 100 stochastic passes, the baseline generates a predictive distribution, allowing for the derivation of 10th percentile uncertainty bounds comparable to the primary GridGuardian quantile architecture.

#box(stroke: 1pt, inset: 10pt, width: 100%)[
  *GridGuardian: Scope of Contributions*
  - *Implemented Components*: Causal Polars data pipeline; Physics-informed feature engineering (OpSDA, RoCoF); LightGBM Quantile Regression; Isotonic Recalibration logic; Proof-of-concept Streamlit Dashboard.
  - *Evaluated Metrics*: Statistical accuracy (MAE, Pinball Loss); Interval reliability (PICP); Computational latency (CPU-benchmarked); SHAP stability.
  - *Illustrative/Simulated Results*: The "EFR battery intervention timeline" and hypothetical lead-times are derived from simulated reconstructions of the 2019 blackout event based on the model's reactive and predictive alert triggers.
]
