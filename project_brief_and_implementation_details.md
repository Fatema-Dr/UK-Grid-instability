# GridGuardian v2: Project Brief & Implementation Details

## 1. Project Purpose & Domain

**GridGuardian v2** is a physics-informed machine-learning prototype that monitors and predicts instability in the UK power grid. It addresses the **Inertia Crisis**: as the UK replaces heavy synchronous generators (coal, gas) with lightweight, weather-dependent renewables (wind, solar), the grid loses its natural physical shock absorber — *system inertia*. Without sufficient inertia, sudden generation losses propagate as rapid frequency drops that can cascade into national blackouts.

The system targets the **National Energy System Operator (NESO) Control Room** use case. Instead of simply classifying the grid state as stable/unstable, it uses **quantile regression** to forecast a probabilistic uncertainty band for grid frequency **10 seconds into the future**, triggering an alert when the predicted worst-case scenario breaches a critical threshold (e.g., 49.8 Hz).

### Why This Matters

- The UK grid must be maintained at **50.0 Hz** within a statutory limit of **±0.2 Hz**.
- NGESO spends hundreds of millions procuring synthetic inertia (Stability Pathfinder programmes: >£650M).
- On **August 9, 2019**, a lightning strike plus low inertia caused frequency to collapse to 48.8 Hz, cutting power to over 1.1 million customers.
- A **10-second warning** is operationally significant: Firm Frequency Response (FFR) batteries discharge within 1–2 seconds, so 10 seconds allows automated prevention.

---

## 2. Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| **Language** | Python 3.13 | Wide ML ecosystem, rapid prototyping |
| **Data Pipeline** | Polars | High-performance columnar engine; `join_asof` merges multi-resolution time series at 1-second scale far faster than Pandas |
| **ML — Gradient Boosting** | LightGBM | Fast tree-based quantile regression; two models produce 10th/90th percentile bounds (80% confidence interval). Also used for a binary classifier with automatic class weighting (`scale_pos_weight`) |
| **ML — Deep Learning** | TensorFlow/Keras (LSTM) | Captures temporal dependencies; used as a residual monitor that flags "High Model Uncertainty" when it disagrees with LightGBM. Trained with `EarlyStopping` (patience=3) on validation loss |
| **XAI** | SHAP (TreeExplainer) | Model-agnostic feature attribution displayed in real time; operators see *why* an alert fires, not just *that* it fires |
| **Dashboard** | Streamlit + Plotly | Interactive "Control Room" UI with dark theme, real-time KPI cards, uncertainty-band charts, LSTM probability timeline, and intervention simulation |
| **Data Sources** | NESO CKAN API (frequency, inertia — daily and half-hourly), Open-Meteo API (weather) | Live, programmatic ingestion with retry logic (`tenacity`), response caching (`requests-cache`), and type/range validation |
| **Calibration** | Isotonic Regression (scikit-learn) | Post-hoc quantile recalibration via `src/calibration.py`; improves systematic pessimistic bias in the raw quantile predictions |
| **Caching** | Parquet files with source-code hashing | Processed datasets are cached as `.parquet`; an MD5 hash of `src/` auto-invalidates cache when logic changes |
| **Testing** | pytest | 3 test modules (31 tests) covering feature engineering, alert logic, and evaluation metrics |
| **Environment** | `uv` (package manager) | Fast, lockfile-based dependency resolution |

---

## 3. Project Structure & Data Flow

```
Implementation/
├── app.py                     # Streamlit Control Room dashboard (entry point)
├── run_pipeline.py            # End-to-end CLI: fetch → engineer → train → calibrate → export
├── evaluate_models.py         # Offline evaluation: metrics + plots for dissertation
├── src/
│   ├── config.py              # All constants: paths, features, hyperparams, API config
│   ├── data_loader.py         # API fetchers with retry, caching & validation
│   ├── feature_engineering.py # Merge + feature creation (RoCoF, OpSDA, lags, etc.)
│   ├── model_trainer.py       # LightGBM classifier, quantile regressors, LSTM trainer
│   ├── calibration.py         # Isotonic regression post-hoc recalibration
│   └── opsda.py               # Optimized Swinging Door Algorithm implementation
├── tests/
│   ├── test_feature_engineering.py  # 9 tests for create_features()
│   ├── test_alert_logic.py          # 11 tests for alert + intervention logic
│   └── test_metrics.py             # 11 tests for pinball loss, PICP, MPIW, calibration
├── notebooks/                 # Saved model artifacts (.pkl, .keras), evaluation outputs
│   └── evaluation_outputs/    # Dissertation-ready plots from evaluate_models.py
└── data/                      # Raw data cache and processed Parquet cache
    └── processed_cache/       # Source-hash-keyed Parquet caches
```

### Pipeline Flow

1. **Data Ingestion** (`data_loader.py`) — Fetches 1-second frequency data (NESO CKAN — automatically selecting monthly resource IDs for the requested date range), hourly weather data (Open-Meteo — interpolated to 1-second resolution), daily inertia cost data (NESO CKAN), and half-hourly system inertia data (`fetch_inertia_data_halfhourly()` — interpolated to 1-second resolution).
2. **Merging** — Uses Polars `join_asof` (backward strategy) to align the time series by closest preceding timestamp.
3. **Feature Engineering** (`feature_engineering.py`) — Creates domain-specific features from the merged data (see §4).
4. **Model Training** (`model_trainer.py` / `run_pipeline.py`) — Trains a LightGBM binary classifier (with `scale_pos_weight`), two LightGBM quantile regressors (α=0.1 and α=0.9), and an LSTM binary classifier (with `EarlyStopping`). Exports all artifacts to `notebooks/`.
5. **Post-Hoc Calibration** (`run_pipeline.py` / `calibration.py`) — Fits isotonic regression calibrators on a held-out calibration window (Aug 7–9), then saves them alongside the models.
6. **Dashboard** (`app.py`) — Loads models, calibrators, and data; performs live predictions per time step; renders UI.
7. **Offline Evaluation** (`evaluate_models.py`) — Computes pinball loss, PICP, MPIW, MAE, RMSE, calibration scores, and generates dissertation-ready plots. Supports CLI arguments `--start-date`, `--end-date`, and `--calibrated`.

---

## 4. Key Concepts & Feature Engineering

### 4.1 Quantile Regression (Core Alert Mechanism)

Two LightGBM models predict the **10th percentile** (lower bound) and **90th percentile** (upper bound) of grid frequency 10 seconds ahead. This generates an 80% prediction interval — an "uncertainty band." The alert fires when *either* the predicted lower bound drops below the threshold *or* the current grid frequency is already below the threshold. This ensures alerts cover both forecasted risk and present-moment distress.

**Why quantile regression over binary classification?** Binary classification answers "Will it be unstable?" with a yes/no. Quantile regression answers "How bad could it get?" — directly mapping to how grid operators think about risk margins. The binary classifier is still trained for comparison and is displayed in the Model Health tab.

### 4.2 Rate of Change of Frequency (RoCoF)

RoCoF measures the first derivative of grid frequency — how fast the frequency is accelerating or decelerating. Raw 1-second `diff()` is noisy due to sensor micro-jitter, so the implementation applies a **5-second rolling average** to smooth RoCoF:

```python
df["rocof"] = df["grid_frequency"].diff().rolling(window=5, min_periods=1).mean().fillna(0)
```

RoCoF is consistently the **most important feature** in LightGBM split counts, confirming the model has learned the physical relationship between frequency momentum and near-term stability.

### 4.3 Wind Ramp Rate (OpSDA — Optimized Swinging Door Algorithm)

Raw wind speed does not capture sudden generation changes (ramps). The Swinging Door Algorithm compresses the wind speed time series into a piecewise-linear representation, retaining only the inflection points where the trend changes significantly. The *slope between compressed points* gives the **wind ramp rate** — a measure of volatility in wind generation.

This feature **significantly outranks raw `wind_speed`** in model importance, validating that the algorithm isolates the physical "jolts" that destabilise the grid.

### 4.4 Renewable Penetration Ratio (Inertia Proxy)

Daily inertia cost data is too coarse to inform 1-second predictions (the model sees a constant for 86,400 consecutive rows). As a mitigation, a **dynamic renewable penetration proxy** was introduced:

```python
renewable_penetration_ratio = (wind_speed × 3000 MW capacity) / 35000 MW demand
```

This varies every second with interpolated wind speed, acting as a physics-based approximation of how much asynchronous (non-inertia-contributing) generation is on the grid at any moment.

### 4.5 Weather Data Interpolation

Hourly weather data is **linearly interpolated to 1-second resolution** using Pandas `resample('1s').interpolate(method='time')`. This eliminates the "staircase" effect where feature values jump abruptly every 3,600 seconds, providing smoother and more physically accurate inputs.

### 4.6 Additional Features

| Feature | Description |
|---|---|
| `volatility_10s` | 10-second rolling standard deviation of frequency — measures short-term turbulence |
| `volatility_30s` | 30-second rolling standard deviation of frequency — captures medium-term oscillations |
| `volatility_60s` | 60-second rolling standard deviation of frequency — captures longer-term instability trends |
| `solar_radiation` | Direct solar radiation (W/m²) from Open-Meteo — captures solar generation variability |
| `lag_1s`, `lag_5s`, `lag_60s` | Lagged frequency values providing auto-regressive context |
| `hour` | Hour-of-day captures diurnal demand patterns (peak vs off-peak) |
| `minute` | Minute-of-hour provides finer temporal context |
| `target_freq_next` | Regression target — actual frequency 10 seconds ahead |
| `target_is_unstable` | Classification target — 1 if future frequency is outside 49.8–50.2 Hz |

### 4.7 LSTM Residual Monitor

The LSTM was originally trained as a standalone deep-learning benchmark. It has been integrated into the live dashboard as a **Residual Monitor**: if LightGBM and LSTM *disagree* on the grid state, the dashboard displays **"HIGH MODEL UNCERTAINTY"** instead of a definitive alert or stable status. This uses the LSTM's temporal-pattern recognition as a cross-check on the gradient-boosting model.

The LSTM is trained with `EarlyStopping` (patience=3, monitoring `val_loss`, restoring best weights) to prevent overfitting. It uses the same `SPLIT_DATE`-based temporal train/test split as the LightGBM models to ensure methodological consistency.

| LightGBM Alert | LSTM Alert | Dashboard Status |
|---|---|---|
| ✅ Yes | ✅ Yes | ⚠️ **INSTABILITY ALERT** |
| ✅ Yes | ❌ No | ⚠️ **HIGH MODEL UNCERTAINTY** |
| ❌ No | ✅ Yes | ⚠️ **HIGH MODEL UNCERTAINTY** |
| ❌ No | ❌ No | ✅ **SYSTEM STABLE** |

---

## 5. Dashboard Features

### 5.1 Time Navigation

- **Date range pickers** to select the overall data range for API fetching.
- **Date picker** to jump between available days within the loaded range.
- **Step buttons** (±1 second, ±1 minute) for fine-grained scrubbing.
- **Exact time input** (HH:MM:SS UTC) to jump directly to a specific moment.
- **Autoplay mode** with selectable speed (1×, 5×, 10×, 30×, 60×) — auto-pauses at end of day.
- Automatic timezone handling and closest-index snapping.

### 5.2 Alert Configuration

- **Time to Alert slider** (5–60 seconds ahead) — configures the display label (model always predicts at the trained 10-second horizon).
- **Instability Threshold slider** (49.5–49.9 Hz) — configures when an alert triggers.
- **Dual-trigger alert logic** — alerts when the predicted lower bound is below threshold *or* the current frequency is already below threshold.

### 5.3 Intervention Simulator (Prescriptive Analytics — Swing Equation)

A "What-If" slider lets operators simulate injecting synthetic inertia (0–5000 MW). When activated, the model dynamically recalculates predictions using the **swing equation**:

$$\Delta f = \frac{\Delta P \times f_0}{2 \times H \times S_{base}}$$

Where:
- $\Delta P$ = injected power (MW)
- $f_0$ = nominal frequency (50 Hz)
- $H$ = system inertia constant (4.0 s)
- $S_{base}$ = total system capacity (35,000 MW)

Additionally, the `renewable_penetration_ratio` feature is reduced proportionally ($\text{reduction} = \Delta P / S_{base}$). Operators can visualise how battery storage or demand response would move the predicted lower bound back into the safe zone.

### 5.4 SHAP Risk Drivers (XAI Panel)

A real-time horizontal bar chart shows SHAP values for the lower bound prediction. Red bars indicate factors pushing frequency *down* (increasing risk); blue bars indicate factors pushing it *up* (stabilising). This allows operators to immediately see *why* an alert is firing — e.g., "Negative wind ramp + elevated RoCoF."

### 5.5 Real-Time Frequency Monitor

A Plotly chart on a dark "Control Room" theme shows:
- Cyan line — actual grid frequency (300-second rolling window).
- Orange shaded band — LightGBM uncertainty envelope.
- Red dashed line — alert threshold.
- Red dot — current time position marker.

Below the main chart, a secondary **LSTM P(Unstable) Timeline** displays the LSTM's instability probability over the same window, with a yellow dashed line at the 0.5 alert threshold.

### 5.6 Model Health Tab

Displays dynamically computed calibration metrics for the selected day:
- **Pinball Loss** for both lower (α=0.1) and upper (α=0.9) quantiles.
- **Prediction Interval Coverage Probability (PICP)** — what fraction of actual values fall within the bands.
- **Mean Prediction Interval Width (MPIW)** — average width of the uncertainty band.
- **Calibration scores** — observed fraction of actuals below each quantile prediction (compared to nominal targets of 10% and 90%).
- **Evaluation sample count** for the selected day.
- Automatic pass/fail indication (green ✅ if PICP ≥ 80%, amber ⚠️ otherwise).

Also displays **LightGBM Classifier performance** for the selected day:
- **Confusion Matrix** (Actual Stable/Unstable vs Predicted Stable/Unstable).
- **Precision, Recall, and F1-Score** for the "Unstable" class.
- **AUC-ROC** (when both classes are present in the day's data).

---

## 6. Evaluation & Validation Results

### 6.1 Model Performance (August 2019 Dataset)

| Metric | Lower Bound (α=0.1) | Upper Bound (α=0.9) |
|---|---|---|
| **Pinball Loss** | 0.0142 | 0.0168 |
| **MAE** | 0.033 Hz | 0.018 Hz |

These error margins are an order of magnitude smaller than standard operational safety buffers (~0.2 Hz).

### 6.2 Uncertainty Calibration

| Metric | Value | Target |
|---|---|---|
| **PICP (80% CI)** | ~77.5–81.5% | ≥ 80% |
| **Calibration (obs. fraction below lower α=0.1)** | ~1.8% | 10% |
| **Calibration (obs. fraction below upper α=0.9)** | ~79.3% | 90% |

**Interpretation — Pessimistic Bias:** Both models are shifted downward. The lower bound is *too low* (only 1.8% of actuals fall below it vs the target 10%). In safety-critical infrastructure, this is a **desirable "fail-safe"** property: false positives (predicting a dip that doesn't occur) are vastly preferable to false negatives (missing a crash). Post-hoc isotonic recalibration (`src/calibration.py`) is available to adjust this bias; the `--calibrated` flag in `evaluate_models.py` enables evaluation with recalibrated predictions.

### 6.3 Feature Importance Validation

RoCoF is the dominant feature. OpSDA `wind_ramp_rate` significantly outranks raw `wind_speed`. Inertia cost ranks lowest due to its daily-resolution granularity mismatch. These rankings align with grid physics expectations, validating that the model has learned meaningful physical relationships.

### 6.4 Blackout Day Stress Test

During spot-checks of the **August 9, 2019 blackout**, the predicted lower bound crossed the 49.8 Hz threshold *before* the actual frequency collapsed, confirming the system's efficacy as an early-warning tool. The extreme magnitude of the blackout exceeded the 80% confidence bounds (as expected for an out-of-distribution Black Swan event).

### 6.5 Evaluation Script Outputs

The `evaluate_models.py` script generates four dissertation-ready plots in `notebooks/evaluation_outputs/`:
1. `aug9_timeseries.png` — Blackout day frequency vs predicted uncertainty band.
2. `residual_distributions.png` — Prediction error histograms for both quantile models.
3. `calibration_plot.png` — Quantile calibration check (observed fraction vs nominal quantile).
4. `feature_importance.png` — LightGBM feature importance for both quantile models.

The script also performs **dashboard output validation** — spot-checking random timestamps plus the blackout event to verify predictions match model outputs and alert logic is correct.

---

## 7. Analysis of Current Limitations & Improvements Needed

### 7.1 Inertia Data Granularity Gap

**Issue:** Daily inertia cost is merged into 86,400 identical rows per day, preventing the model from learning the dynamic relationship between inertia levels and frequency drops.

**Current Mitigation:** The `renewable_penetration_ratio` proxy provides per-second variation. Additionally, `fetch_inertia_data_halfhourly()` has been implemented to fetch half-hourly system inertia data (NESO resource `2f2dbaa1-3047-4e48-85f2-ec24e669678f`), interpolated to 1-second resolution. However, this data source is not yet integrated into the main pipeline or dashboard data flow — only the daily inertia cost is currently used in `run_pipeline.py` and `app.py`.

### 7.2 Evaluation Bias — Single Training Period

**Issue:** All training uses August 2019 data. The model may be an "August 2019 Detector" — having learned seasonal patterns rather than generalisable grid physics.

**Current Mitigation:** `run_pipeline.py` includes a **winter validation step** that fetches December 2019 data, runs the evaluation suite, and reports PICP/MPIW. The `evaluate_models.py` script also supports `--start-date` and `--end-date` CLI arguments to evaluate on any period. However, the model itself is still **trained only on August 2019 data** — cross-season training remains a future improvement.

### 7.3 Quantile Calibration

**Issue:** The model exhibits a systematic pessimistic bias (1.8% vs 10% target for α=0.1).

**Current Mitigation:**
- **Reliability Diagram**: `evaluate_models.py` generates a calibration plot across the two quantile levels.
- **Post-hoc Recalibration**: `src/calibration.py` implements isotonic regression recalibration. Calibrators are fitted on a held-out window (Aug 7–9) during `run_pipeline.py` and are loaded by both `app.py` and `evaluate_models.py`.
- **Remaining improvement**: Multi-point calibration across more quantile levels (e.g., α = 0.05, 0.25, 0.5, 0.75, 0.95) would provide a more comprehensive reliability diagram.

### 7.4 Intervention Simulator — Physical Fidelity

**Issue:** The simulator uses the swing equation, which is a first-order approximation. The inertia constant ($H$ = 4.0 s) and system capacity ($S_{base}$ = 35,000 MW) are static estimates.

**Improvement Needed:** Use time-varying inertia estimates (from the half-hourly data source already implemented in `data_loader.py`) to make $H$ dynamic. This would make the simulator's predictions more credible to domain experts.

### 7.5 Half-Hourly Inertia Integration

**Issue:** `fetch_inertia_data_halfhourly()` exists in `data_loader.py` and is fully implemented (including interpolation to 1-second resolution), but is not yet called by `run_pipeline.py` or `app.py`.

**Improvement Needed:** Replace the daily inertia cost merge with the half-hourly `system_inertia_mws` merge in the pipeline and dashboard, or use both in parallel. This would provide sub-daily inertia variation as a model feature.

### 7.6 Dissertation Justification for TTA (10 Seconds)

**Issue:** The choice of a 10-second prediction horizon is not formally justified in the codebase.

**Justification:** Most Fast Frequency Response (FFR) services can inject maximum power within 1–2 seconds. A 10-second lead time provides ample time for automated battery clusters, demand flexibility services, and operator-initiated actions. It also represents a balance between prediction accuracy (which degrades with longer horizons) and operational usefulness.

---

## 8. Summary — What Has Been Completed vs What Remains

### ✅ Completed

- [x] Live API integration (NESO CKAN for frequency + inertia, Open-Meteo for weather)
- [x] Dynamic monthly resource ID selection for frequency data (auto-selects correct NESO resource per month)
- [x] High-performance data pipeline with Polars `join_asof`
- [x] Weather interpolation to 1-second resolution (removing staircase artefacts)
- [x] Smoothed RoCoF (5-second rolling average)
- [x] Multi-window volatility features (10s, 30s, 60s rolling standard deviations)
- [x] OpSDA wind ramp rate feature
- [x] Renewable penetration ratio as dynamic inertia proxy
- [x] LightGBM quantile regression (α=0.1 and α=0.9)
- [x] LightGBM binary classifier with automatic class weighting (`scale_pos_weight`)
- [x] LSTM trained as residual monitor with `EarlyStopping` and disagreement-based uncertainty alerts
- [x] LSTM uses same temporal train/test split as LightGBM (SPLIT_DATE-based)
- [x] SHAP XAI integration in live dashboard
- [x] Intervention Simulator — physics-informed (swing equation: Δf = ΔP×f₀ / 2H×S)
- [x] Model Health tab with dynamic metric computation (Pinball Loss, PICP, MPIW, Calibration)
- [x] LightGBM Classifier performance in Model Health tab (Confusion Matrix, Precision, Recall, F1, AUC-ROC)
- [x] LSTM P(Unstable) probability timeline chart in dashboard
- [x] Dual-trigger alert logic (predicted lower bound OR current frequency below threshold)
- [x] Dashboard autoplay mode with selectable speed
- [x] Parquet caching with source-code hash invalidation
- [x] Offline evaluation suite (`evaluate_models.py`) with dissertation-ready plots (4 plots)
- [x] Dark "Control Room" CSS theme
- [x] Automated unit tests for pipeline, alert logic, and metrics (31 tests — pytest)
- [x] Out-of-season validation support (CLI args: `--start-date`, `--end-date`, dynamic resource ID selection)
- [x] Winter validation step in `run_pipeline.py` (December 2019 — PICP/MPIW evaluation)
- [x] Post-hoc quantile recalibration (isotonic regression via `src/calibration.py`)
- [x] Calibrator integration in dashboard (`app.py` loads and applies calibrators)
- [x] Sub-daily inertia data fetcher implemented (`fetch_inertia_data_halfhourly()` — half-hourly, interpolated to 1s)
- [x] Evaluation script supports `--calibrated` flag for recalibrated metrics

### 🔲 Remaining

- [ ] Integrate `fetch_inertia_data_halfhourly()` into the main pipeline and dashboard data flow
- [ ] Use time-varying inertia constant ($H$) in the Intervention Simulator
- [ ] Multi-point quantile calibration diagram (more quantile levels beyond α=0.1 and α=0.9)
- [ ] Cross-season model training (currently trains on August 2019 only; evaluates but does not train on other periods)