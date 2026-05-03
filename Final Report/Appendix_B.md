## Appendix B: Full Model Evaluation Tables

This appendix presents comprehensive model evaluation metrics for the August 2019 training/validation period and December 2019 out-of-season testing. All values are derived from the real trained LightGBM models.

## B.1 Quantile Regression Performance Metrics

### Table B.1: Complete Quantile Metrics (August 2019)

| Metric | Lower Bound (α=0.1) | Upper Bound (α=0.9) | Target | Status | Notes |
|--------|---------------------|---------------------|--------|--------|-------|
| **Pinball Loss** | 0.00268 | 0.00260 | <0.02 | (Pass) | Excellent quantile estimation |
| **MAE (Hz)** | 0.0208 | 0.0207 | <0.05 | (Pass) | Highly accurate point estimation |
| **RMSE (Hz)** | 0.0260 | 0.0263 | <0.10 | (Pass) | Acceptable for 10s horizon |
| **PICP (%)** | 82.1 | — | ≥80% | (Pass) | Meets nominal coverage target |
| **MPIW (Hz)** | 0.0387 | — | <0.2 | (Pass) | Precise uncertainty bands |
| **Calibration (α=0.1)** | 8.9% | — | 10% | (Good) | Near-nominal lower tail coverage |
| **Calibration (α=0.9)** | 91.0% | — | 90% | (Good) | Near-nominal upper tail coverage |
| **Quantile Crossing** | 0.0% | — | 0% | (Pass) | Monotonicity preserved |

*Note: All metrics calculated using 1-second resolution data from August 1–31, 2019 (2,588,390 observations).*

## B.2 Binary Classifier Performance

### Table B.2: Instability Detection Metrics (August 2019 Blackout Day)

| Metric | Value | Target | Status | Interpretation |
|--------|-------|--------|--------|----------------|
| **Precision** | 0.889 | >0.85 | (Pass) | 88.9% of alerts are true instability |
| **Recall** | 0.210 | >0.80 | (Fail) | Only catches 21% of fast transients |
| **F1-Score** | 0.339 | >0.85 | (Fail) | Low overall performance on extreme events |
| **AUC-ROC** | 0.978 | >0.90 | (Pass) | Excellent ranking of instability risk |
| **Specificity** | 0.999 | >0.90 | (Pass) | Virtually zero false alerts during stable periods |
| **False Positive Rate** | 0.01% | <0.15 | (Pass) | Negligible operator fatigue risk |
| **False Negative Rate** | 79.0% | <0.20 | (Fail) | Significant risk of missed alerts for fast events |

*Instability defined as frequency below 49.8 Hz. Binary classifier triggers when predicted lower bound < 49.8 Hz. High AUC but low recall indicates the model is highly discriminating but overly conservative in its absolute frequency predictions during extreme transients.*

## B.3 Out-of-Season Validation

### Table B.3: December 2019 Performance (Winter Conditions)

| Metric | August (Full Month) | December (Testing) | Change | Assessment |
|--------|--------------------|-------------------|--------|------------|
| **PICP (%)** | 82.1 | 87.9 | +5.8 pp | ✅ Improved coverage |
| **Pinball Loss (Lower)** | 0.00268 | 0.00251 | −6.3% | ✅ Robust |
| **Pinball Loss (Upper)** | 0.00260 | 0.00244 | −6.2% | ✅ Robust |
| **MAE Lower (Hz)** | 0.0208 | 0.0215 | +3.4% | ✅ Stable |
| **Calibration (α=0.1)** | 10.1% | 5.9% | −4.2 pp | ⚠️ More conservative |
| **MPIW (Hz)** | 0.0387 | 0.0407 | +5.2% | ✅ Acceptable |

*The model generalizes exceptionally well to December conditions, with an improvement in coverage at the cost of slightly wider intervals. This indicates that the learned physical relationships are seasonally robust.*

## B.4 Computational Performance

### Table B.4: Real-Time System Latency

| Operation | Mean (s) | Std (s) | Min (s) | Max (s) | Requirement | Status |
|-----------|----------|---------|---------|---------|-------------|--------|
| **Data Refresh** | 0.12 | 0.03 | 0.08 | 0.21 | <1.0 | (Pass) |
| **Feature Engineering** | 0.08 | 0.02 | 0.05 | 0.14 | <0.5 | (Pass) |
| **Model Inference** | 0.08 | 0.01 | 0.06 | 0.12 | <0.5 | (Pass) |
| **SHAP Explanation** | 0.15 | 0.04 | 0.10 | 0.28 | <0.5 | (Pass) |
| **Dashboard Render** | 0.05 | 0.01 | 0.03 | 0.09 | <1.0 | (Pass) |
| **Total Cycle** | **0.40** | **0.08** | **0.28** | **0.62** | **<2.0** | ✅ **Pass** |

*Measurements conducted on Intel i7-1165G7 (2.8 GHz), 16GB RAM. Python 3.11, Polars 0.20.x, LightGBM 4.1.x.*

## B.5 Feature Importance Stability (Temporal Windows)

### Table B.5: SHAP Feature Importance Stability Across August 9 Time Windows

| Feature | 00:00–08:00 | 08:00–16:00 | 16:00–24:00 | Mean (%) | Std |
|---------|-------------|-------------|-------------|----------|-----|
| **Grid Frequency** | 85.2 | 88.4 | 85.7 | 86.4 | 1.41 |
| **Lag 1s** | 6.1 | 4.9 | 7.8 | 6.3 | 1.19 |
| **Wind Speed** | 2.2 | 1.7 | 0.9 | 1.6 | 0.54 |
| **Hour of Day** | 3.0 | 1.2 | 1.6 | 1.9 | 0.77 |
| **Lag 60s** | 0.7 | 0.9 | 0.9 | 0.8 | 0.09 |
| **RoCoF (5s smoothed)** | 0.6 | 0.8 | 0.9 | 0.8 | 0.12 |
| **Volatility (30s)** | 0.7 | 0.6 | 0.8 | 0.7 | 0.08 |

*Metrics represent Mean Absolute SHAP importance for the lower bound (α=0.1) model. Results demonstrate that frequency signals remain the dominant predictors across all periods of the day, with low variance (Std < 1.5%) in feature rankings.*

## B.6 August 9, 2019 Blackout Event Metrics

### Table B.6: Real Blackout Prediction Timeline

| Time (UTC) | Event | Actual Freq (Hz) | Pred Lower Bound (Hz) | Advance Warning (s) | Status |
|------------|-------|------------------|----------------------|---------------------|--------|
| 15:52:30 | Pre-disturbance | 50.006 | 49.982 | — | Stable |
| 15:52:35 | **Actual threshold breach** | **49.790** | 49.970 | — | 🔴 **Event** |
| 15:52:40 | **Model alert triggered** | 49.372 | **49.796** | **−5** | 🚨 **Late Warning** |
| 15:53:49 | **Actual Nadir reached** | **48.787** | — | **69** | 🔴 **Blackout** |

*Alert threshold: 49.80 Hz. Actual nadir: 48.787 Hz. The model was 5 seconds late in detecting the initial threshold breach but provided 69 seconds of advance warning before the system nadir, which is the critical window for secondary containment actions.*


---

## References

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

Ucar, F. (2023) 'Explainable AI for smart grid stability prediction: Enhancing operator trust through SHAP analysis', *Energy and AI*, 14, p. 100256. doi: 10.1016/j.egyai.2023.100256.

Zhang, Y., Wang, J. and Chen, X. (2021) 'A review of machine learning applications in power system frequency forecasting', *Renewable and Sustainable Energy Reviews*, 145, p. 111156. doi: 10.1016/j.rser.2021.111156.
