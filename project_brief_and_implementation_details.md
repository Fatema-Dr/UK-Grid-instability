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
| **ML — Gradient Boosting** | LightGBM | Fast tree-based quantile regression; two models produce 10th/90th percentile bounds (80% confidence interval) |
| **ML — Deep Learning** | TensorFlow/Keras (LSTM) | Captures temporal dependencies; used as a residual monitor that flags "High Model Uncertainty" when it disagrees with LightGBM |
| **XAI** | SHAP (TreeExplainer) | Model-agnostic feature attribution displayed in real time; operators see *why* an alert fires, not just *that* it fires |
| **Dashboard** | Streamlit + Plotly | Interactive "Control Room" UI with dark theme, real-time KPI cards, uncertainty-band charts, and intervention simulation |
| **Data Sources** | NESO CKAN API (frequency, inertia), Open-Meteo API (weather) | Live, programmatic ingestion with retry logic (`tenacity`), response caching (`requests-cache`), and type/range validation |
| **Caching** | Parquet files with source-code hashing | Processed datasets are cached as `.parquet`; an MD5 hash of `src/` auto-invalidates cache when logic changes |
| **Environment** | `uv` (package manager) | Fast, lockfile-based dependency resolution |

---

## 3. Project Structure & Data Flow

```
Implementation/
├── app.py                  # Streamlit Control Room dashboard (entry point)
├── run_pipeline.py         # End-to-end CLI: fetch → engineer → train → export
├── evaluate_models.py      # Offline evaluation: metrics + plots for dissertation
├── src/
│   ├── config.py           # All constants: paths, features, hyperparams, API config
│   ├── data_loader.py      # API fetchers with retry, caching & validation
│   ├── feature_engineering.py  # Merge + feature creation (RoCoF, OpSDA, lags, etc.)
│   ├── model_trainer.py    # LightGBM classifier, quantile regressors, LSTM trainer
│   └── opsda.py            # Optimized Swinging Door Algorithm implementation
├── notebooks/              # Saved model artifacts (.pkl, .keras) + evaluation plots
└── data/                   # Raw data cache and processed Parquet cache
```

### Pipeline Flow

1. **Data Ingestion** (`data_loader.py`) — Fetches 1-second frequency data (NESO CKAN), hourly weather data (Open-Meteo — interpolated to 1-second resolution), and daily inertia cost data (NESO CKAN).
2. **Merging** — Uses Polars `join_asof` (backward strategy) to align the three time series by closest preceding timestamp.
3. **Feature Engineering** (`feature_engineering.py`) — Creates domain-specific features from the merged data (see §4).
4. **Model Training** (`model_trainer.py` / `run_pipeline.py`) — Trains LightGBM classifier, two quantile regressors (α=0.1 and α=0.9), and an LSTM binary classifier. Exports all artifacts to `notebooks/`.
5. **Dashboard** (`app.py`) — Loads models and data, performs live predictions per time step, renders UI.
6. **Offline Evaluation** (`evaluate_models.py`) — Computes pinball loss, PICP, MPIW, calibration scores, and generates dissertation-ready plots.

---

## 4. Key Concepts & Feature Engineering

### 4.1 Quantile Regression (Core Alert Mechanism)

Two LightGBM models predict the **10th percentile** (lower bound) and **90th percentile** (upper bound) of grid frequency 10 seconds ahead. This generates an 80% prediction interval — an "uncertainty band." The alert fires when the *predicted lower bound drops below the threshold*, meaning the system alerts on **forecasted risk**, not current state.

**Why quantile regression over binary classification?** Binary classification answers "Will it be unstable?" with a yes/no. Quantile regression answers "How bad could it get?" — directly mapping to how grid operators think about risk margins.

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
| `lag_1s`, `lag_5s`, `lag_60s` | Lagged frequency values providing auto-regressive context |
| `hour` | Hour-of-day captures diurnal demand patterns (peak vs off-peak) |
| `target_freq_next` | Regression target — actual frequency 10 seconds ahead |
| `target_is_unstable` | Classification target — 1 if future frequency is outside 49.8–50.2 Hz |

### 4.7 LSTM Residual Monitor

The LSTM was originally trained as a standalone deep-learning benchmark but **was not used** in the live dashboard. It has now been integrated as a **Residual Monitor**: if LightGBM and LSTM *disagree* on the grid state, the dashboard displays **"HIGH MODEL UNCERTAINTY"** instead of a definitive alert or stable status. This uses the LSTM's temporal-pattern recognition as a cross-check on the gradient-boosting model.

| LightGBM Alert | LSTM Alert | Dashboard Status |
|---|---|---|
| ✅ Yes | ✅ Yes | ⚠️ **INSTABILITY ALERT** |
| ✅ Yes | ❌ No | ⚠️ **HIGH MODEL UNCERTAINTY** |
| ❌ No | ✅ Yes | ⚠️ **HIGH MODEL UNCERTAINTY** |
| ❌ No | ❌ No | ✅ **SYSTEM STABLE** |

---

## 5. Dashboard Features

### 5.1 Time Navigation

- **Date picker** to jump between available days.
- **Step buttons** (±1 second, ±1 minute) for fine-grained scrubbing.
- **Exact time input** (HH:MM:SS UTC) to jump directly to a specific moment.
- Automatic timezone handling and closest-index snapping.

### 5.2 Alert Configuration

- **Time to Alert slider** (5–60 seconds ahead) — configures the prediction horizon.
- **Instability Threshold slider** (49.5–49.9 Hz) — configures when an alert triggers.

### 5.3 Intervention Simulator (Prescriptive Analytics)

A "What-If" slider lets operators simulate injecting synthetic inertia (0–5000 MW). When activated, the model dynamically recalculates predictions by reducing the `renewable_penetration_ratio` feature proportionally (every 1000 MW reduces the ratio by 0.05). Operators can visualise how battery storage or demand response would move the predicted lower bound back into the safe zone.

### 5.4 SHAP Risk Drivers (XAI Panel)

A real-time horizontal bar chart shows SHAP values for the lower bound prediction. Red bars indicate factors pushing frequency *down* (increasing risk); blue bars indicate factors pushing it *up* (stabilising). This allows operators to immediately see *why* an alert is firing — e.g., "Negative wind ramp + elevated RoCoF."

### 5.5 Model Health Tab

Displays calibration metrics for the selected day:
- **Pinball Loss** for both lower (α=0.1) and upper (α=0.9) quantiles.
- **Prediction Interval Coverage Probability (PICP)** — what fraction of actual values fall within the bands.

### 5.6 Real-Time Frequency Monitor

A Plotly chart on a dark "Control Room" theme shows:
- Cyan line — actual grid frequency (300-second rolling window).
- Orange shaded band — LightGBM uncertainty envelope.
- Red dashed line — alert threshold.
- Red dot — current time position marker.

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

**Interpretation — Pessimistic Bias:** Both models are shifted downward. The lower bound is *too low* (only 1.8% of actuals fall below it vs the target 10%). In safety-critical infrastructure, this is a **desirable "fail-safe"** property: false positives (predicting a dip that doesn't occur) are vastly preferable to false negatives (missing a crash).

### 6.3 Feature Importance Validation

RoCoF is the dominant feature. OpSDA `wind_ramp_rate` significantly outranks raw `wind_speed`. Inertia cost ranks lowest due to its daily-resolution granularity mismatch. These rankings align with grid physics expectations, validating that the model has learned meaningful physical relationships.

### 6.4 Blackout Day Stress Test

During spot-checks of the **August 9, 2019 blackout**, the predicted lower bound crossed the 49.8 Hz threshold *before* the actual frequency collapsed, confirming the system's efficacy as an early-warning tool. The extreme magnitude of the blackout exceeded the 80% confidence bounds (as expected for an out-of-distribution Black Swan event).

---

## 7. Analysis of Current Limitations & Improvements Needed

### 7.1 Inertia Data Granularity Gap

**Issue:** Daily inertia cost is merged into 86,400 identical rows per day, preventing the model from learning the dynamic relationship between inertia levels and frequency drops.

**Current Mitigation:** The `renewable_penetration_ratio` proxy provides per-second variation. However, accessing sub-daily inertia data (e.g., from Elexon's Balancing Mechanism reports or NESO System Inertia API — half-hourly resolution) would significantly improve the model's physical fidelity.

### 7.2 Evaluation Bias — Single Test Period

**Issue:** All training and testing uses August 2019 data. The model may be an "August 2019 Detector" — having learned seasonal patterns rather than generalisable grid physics.

**Improvement Needed:** Fetch data for a contrasting period (e.g., a winter month with different demand profiles and renewable generation mix) and run the same evaluation suite. This would prove robustness.

### 7.3 Quantile Calibration

**Issue:** The model exhibits a systematic pessimistic bias (1.8% vs 10% target for α=0.1).

**Improvement Needed:**
- **Reliability Diagram**: Generate a multi-point calibration plot across several quantile levels.
- **Post-hoc Recalibration**: Apply techniques like isotonic regression or Platt scaling to adjust the predicted quantiles.
- The Model Health tab currently shows **placeholder metrics**. It should compute pinball loss, PICP, and calibration dynamically for the selected day's data.

### 7.4 Model Health Tab — Placeholder Metrics

**Issue:** The Model Health tab displays static, hardcoded metric values rather than computing them from the current data in real time.

**Improvement Needed:** Implement on-the-fly computation of pinball loss and PICP by running model predictions on the full selected day and comparing against `target_freq_next`, mirroring the logic in `evaluate_models.py`.

### 7.5 Intervention Simulator — Physical Fidelity

**Issue:** The simulator uses a simple linear relationship (1000 MW → 0.05 reduction in renewable penetration ratio). This is a coarse approximation.

**Improvement Needed:** Use a more physics-informed transfer function, ideally backed by the swing equation: $\Delta f \propto \frac{\Delta P}{2H \cdot f_0}$, where $H$ is the system inertia constant. This would make the simulator's predictions more credible to domain experts.

### 7.6 Missing Unit Tests

**Issue:** There are no automated unit tests for the feature engineering pipeline, data validation logic, or alert threshold logic.

**Improvement Needed:** Create a `tests/` directory with tests for:
- `merge_datasets` with known overlaps/gaps.
- `create_features` with manually verified RoCoF, lag, and volatility values.
- Alert threshold boundary conditions (just above, at, and just below 49.8 Hz).

### 7.7 Dissertation Justification for TTA (10 Seconds)

**Issue:** The choice of a 10-second prediction horizon is not formally justified in the codebase.

**Justification:** Most Fast Frequency Response (FFR) services can inject maximum power within 1–2 seconds. A 10-second lead time provides ample time for automated battery clusters, demand flexibility services, and operator-initiated actions. It also represents a balance between prediction accuracy (which degrades with longer horizons) and operational usefulness.

---

## 8. Summary — What Has Been Completed vs What Remains

### ✅ Completed

- [x] Live API integration (NESO CKAN for frequency + inertia, Open-Meteo for weather)
- [x] High-performance data pipeline with Polars `join_asof`
- [x] Weather interpolation to 1-second resolution (removing staircase artefacts)
- [x] Smoothed RoCoF (5-second rolling average)
- [x] OpSDA wind ramp rate feature
- [x] Renewable penetration ratio as dynamic inertia proxy
- [x] LightGBM quantile regression (α=0.1 and α=0.9)
- [x] LSTM trained as residual monitor with disagreement-based uncertainty alerts
- [x] SHAP XAI integration in live dashboard
- [x] Intervention Simulator ("What-If" slider)
- [x] Model Health tab (placeholder metrics)
- [x] Parquet caching with source-code hash invalidation
- [x] Offline evaluation suite (`evaluate_models.py`) with dissertation-ready plots
- [x] Dark "Control Room" CSS theme

### ⬚ Remaining Improvements

- [ ] Replace placeholder Model Health metrics with dynamic computation
- [ ] Out-of-season validation (winter month test)
- [ ] Post-hoc quantile recalibration
- [ ] Physics-informed intervention simulator (swing equation)
- [ ] Automated unit tests for pipeline and alert logic
- [ ] Sub-daily inertia data source integration (if available)