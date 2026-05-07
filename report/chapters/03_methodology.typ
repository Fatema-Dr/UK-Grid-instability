= Methodology

== Introduction

This chapter details the methodology implemented to engineer the GridGuardian alerting system. The research follows an empirical software engineering approach, translating established power system physics into a functional machine learning pipeline. The methodology is constrained entirely to the techniques, architectures, and data processing routines actively deployed in the project codebase.

== Software Development Life Cycle (SDLC)

An Agile methodology managed the project lifecycle to accommodate the iterative nature of machine learning development. The project was structured into three distinct sprints:

1.  *Sprint 1: Data Engineering Foundation:* Focused on data ingestion, delivering the API connectors for NESO and Open-Meteo, and implementing the `join_asof` temporal alignment.
2.  *Sprint 2: Machine Learning Prototyping:* Focused on model training, producing the baseline LightGBM and LSTM architectures.
3.  *Sprint 3: System Integration:* Focused on merging the physical heuristic rules with the statistical predictions to formulate the final instability target variable.

Version control (Git) was maintained throughout. To prevent look-ahead bias during data aggregation, continuous unit testing verified the chronological integrity of all temporal joins.

== Data Acquisition and Pre-processing

The system processes multi-resolution data from the National Energy System Operator (NESO) and the Open-Meteo API.

=== Quality Control and Imputation

Live telemetry is subject to sensor anomalies and transmission gaps. Data quality control relies on hard-boundary validation; frequency readings falling outside the physical operational limits of the UK grid ($40.0 < f < 60.0$ Hz) are rejected as sensor errors. 

Following rejection, missing temporal gaps smaller than 5 seconds are resolved using linear interpolation to maintain the continuity required for derivative calculations. Gaps exceeding 5 seconds remain as nulls to prevent the interpolation of artificial stability across undocumented periods.

=== Asynchronous Data Alignment and Leakage Prevention

The GridGuardian system must continuously process and fuse two fundamentally incompatible data structures: a high-frequency grid telemetry stream (sampled at 1 Hz, generating 86,400 records per day) and a sparse meteorological stream (sampled at 1/3600 Hz). In traditional pandas-based architectures, interpolating between these disparate temporal frequencies often results in extreme memory overhead and dangerous "forward-looking" data leakage, where future weather data accidentally informs past frequency predictions.

To guarantee causal integrity and achieve sub-second execution latency, the system abandoned standard dataframes in favour of the Rust-based Polars library. Polars utilizes Apache Arrow memory formatting and lazy evaluation to drastically reduce the IO bottleneck of time-series alignment. The core of the data fusion pipeline is an asynchronous, backward-filling temporal join (`join_asof`).

Let $t_f$ represent the exact microsecond timestamp of a 1-second grid frequency observation, and $T_w$ represent the set of all available hourly weather timestamps. The `join_asof` algorithm performs a binary search to assign a weather vector $W(t_f)$ such that:

$ W(t_f) = W(max {t_w in T_w | t_w <= t_f}) $

By enforcing the condition $t_w <= t_f$, the system mathematically guarantees that each high-frequency grid observation is joined *exclusively* to the most recent preceding weather measurement. The algorithm is strictly configured with a 3600-second maximum tolerance window. If a weather API outage exceeds 60 minutes, the join defaults to null values rather than interpolating stale data, preventing the model from inferring stability based on obsolete meteorological conditions. This architecture ensures the predictive model operates purely on historical reality without forward-looking bias.

== Feature Engineering

Feature engineering translates raw telemetry into physically meaningful variables representing grid momentum and stress.

=== Digital Filtering and RoCoF

Raw frequency measurements ($f_t$) contain high-frequency sensor noise. Discretely calculating the Rate of Change of Frequency (RoCoF) across 1-second intervals produces extreme volatility. A 5-second moving average functions as a discrete digital low-pass filter to isolate structural momentum shifts. The smoothed frequency $tilde(f)_t$ is:

$ tilde(f)_t = 1/5 sum_(k=0)^(4) f_(t-k) $

The filtered RoCoF is then derived via backward finite difference.

=== The Swinging Door Algorithm (OpSDA) Compression

A major engineering hurdle in wind-generation forecasting is distinguishing between transient, harmless wind gusts and sustained meteorological shifts that will fundamentally alter grid momentum. To computationally extract the structural wind ramp rate—the speed at which wind generation increases or decreases—the system applies the Swinging Door Algorithm (OpSDA) @zhao2019machine. 

OpSDA is a lossy temporal compression algorithm originally designed for process historian databases. It reduces raw, noisy time-series data into a sequence of definitive linear segments by discarding any data points that fall within a defined compression corridor ($epsilon$). The algorithm establishes a "door" bounded by an upper and lower slope. As new data arrives, the door swings to accommodate the geometry of the curve. When a data point breaches the maximum open angle of the door, a new structural segment is recorded. 

Let the compressed segment be defined between a starting point $(t_i, w_i)$ and an ending point $(t_j, w_j)$. The structural ramp rate $R_"wind"$ is then computationally derived as the mathematical slope of the compressed segment:

$ R_"wind" = (w_j - w_i) / (t_j - t_i) $

By converting raw, high-frequency wind velocity into discrete OpSDA ramp rates, the feature engineering pipeline isolates sustained physical momentum shifts from irrelevant localized turbulence, directly providing the LightGBM engine with a clean derivative of meteorological stress.

=== System Inertia and Risk Proxy Integration

A critical, often-overlooked dimension of modern frequency forecasting is the dynamic variation of system inertia. Standard machine learning models treat frequency volatility as a stationary process. However, the physical reality governed by the Swing Equation dictates that identical wind ramps or frequency drops carry drastically different risk profiles depending on the total kinetic energy currently spinning on the grid.

To mathematically inject this physical context into the predictive model, the system pulls half-hourly system inertia telemetry (`NESO_INERTIA_HALFHOURLY_RESOURCE_ID`) directly from the NESO database. This raw metric is processed into three explicit stability indicators:
1. `inertia_value`: The absolute GVA·s measurement.
2. `low_inertia_flag`: A binary trigger activated when $H < 100$ GVA·s.
3. `rocof_inertia_risk`: A synthetic derivative calculated as $R_"risk" = "RoCoF" / H$.

By explicitly providing the model with the denominator of the Swing Equation ($H$), the algorithm can dynamically adjust its sensitivity to sudden frequency drops. The gradient boosting trees mathematically learn to correctly identify that a moderately steep RoCoF during a low-inertia, high-wind night is significantly more dangerous than the same RoCoF during a high-inertia midday period.

=== Target Variable Formulation

Unlike standard forecasting that predicts a continuous value, this system predicts a binary instability state (`target_is_unstable`). The target variable was engineered directly from physical grid constraints. A future state ($t + 10$ seconds) is classified as unstable if any of the following Boolean conditions are met:

1.  *Statutory Breach:* The future frequency drops below $49.85$ Hz or exceeds $50.15$ Hz.
2.  *Momentum Failure:* The current smoothed RoCoF is worse than $-0.02$ Hz/s while the frequency is below $49.95$ Hz.
3.  *Turbulence:* The 10-second rolling frequency standard deviation exceeds $0.03$.

This encoding forces the machine learning models to learn the boundaries of physical failure rather than merely minimizing generic prediction error.

== Machine Learning Architecture

=== Temporal Hold-Out Validation

Standard $k$-fold cross-validation is invalid for time-series data due to temporal leakage. The evaluation utilizes a strict chronological hold-out split. The models were trained exclusively on data prior to August 9, 2019 (`2019-08-09 00:00:00`). The August 9 blackout event functions as a completely unseen testing set, rigorously evaluating out-of-sample generalization.

=== Primary Predictor: LightGBM Quantile Regression

LightGBM @ke2017lightgbm serves as the primary inference engine due to its histogram-based tree construction, which meets the sub-second latency requirements of grid operations. The model optimizes for quantile regression using the asymmetric Pinball Loss function ($L_alpha$). For a target quantile $alpha in (0, 1)$, true value $y$, and prediction $hat(y)$:

$ L_alpha(y, hat(y)) = cases(
  alpha (y - hat(y)) & "if" y >= hat(y),
  (1 - alpha) (hat(y) - y) & "if" y < hat(y)
) $

Rather than automated Bayesian optimization, structural regularization was applied manually. Hyperparameters were constrained to shallow trees (`max_depth = 6`) with explicit L1 and L2 penalties (`reg_alpha = 0.1`, `reg_lambda = 1.0`). This forces the trees to rely on primary physical drivers (RoCoF) and prevents the memorization of transient sensor noise.

=== Secondary Predictor: LSTM Dual-Architecture Baseline

To definitively evaluate the computational and predictive efficiency of the LightGBM engine, a Long Short-Term Memory (LSTM) network was engineered as a dual-purpose baseline architecture: acting both as a secondary binary classifier for structural anomalies and as a probabilistic quantile comparator.

The LSTM network was strictly constrained to match the operational context of the control room. The topology processes sequential inputs of length 30 (representing a 30-second sliding window of historical telemetry). The architecture consists of a single recurrent layer containing 50 hidden units, optimized using the Adam optimizer @kingma2014adam (learning rate = 0.001) and binary cross-entropy loss. To prevent over-parameterization on the time-series data, training was restricted to a maximum of 5 epochs with early stopping (patience = 3). 

For its probabilistic role, Monte Carlo (MC) Dropout @srivastava2014dropout was implemented. Standard neural networks statically disable dropout layers during inference, yielding deterministic point predictions. By explicitly maintaining a dropout rate of $0.2$ during active inference (`training=True`), the network's forward passes are forced to generate a stochastic, Gaussian distribution of predictions for the exact same input matrix. The system samples 25 parallel forward passes per observation. The variance across these 25 samples is computationally derived to quantify the network's predictive uncertainty, allowing for a direct statistical comparison against the LightGBM quantile boundaries.

=== Post-hoc Isotonic Recalibration

Initial evaluation demonstrated that the LightGBM quantile outputs exhibited pessimistic calibration bias. Post-hoc Isotonic Recalibration @kuleshov2018accurate was applied. Isotonic regression fits a monotonically increasing mapping function to a held-out calibration set (`2019-08-07` to `2019-08-09`). This non-parametric transformation recalibrates the raw outputs to ensure that a predicted 10% risk corresponds to a mathematically true 10% empirical risk.
