## Appendix B: Full Model Evaluation Tables

This appendix presents comprehensive model evaluation metrics for the August 2019 training/validation period and December 2019 out-of-season testing.

## B.1 Quantile Regression Performance Metrics

### Table B.1: Complete Quantile Metrics (August 2019)

| Metric | Lower Bound (α=0.1) | Upper Bound (α=0.9) | Target | Status | Notes |
|--------|---------------------|---------------------|--------|--------|-------|
| **Pinball Loss** | 0.0142 | 0.0168 | <0.02 | ✅ Pass | Excellent quantile estimation |
| **MAE (Hz)** | 0.033 | 0.018 | <0.05 | ✅ Pass | Order of magnitude below safety buffer |
| **RMSE (Hz)** | 0.047 | 0.029 | <0.10 | ✅ Pass | Acceptable for 10s horizon |
| **PICP (%)** | 77.5–81.5 | — | ≥80% | ⚠️ Marginal | Slightly below nominal target |
| **MPIW (Hz)** | 0.12 | 0.15 | <0.2 | ✅ Pass | Reasonable precision-uncertainty balance |
| **Calibration (Observed < α=0.1)** | 1.8% | — | 10% | ⚠️ Pessimistic | Desirable for safety-critical applications |
| **Calibration (Observed < α=0.9)** | 79.3% | — | 90% | ⚠️ Slightly Low | Upper bound coverage acceptable |
| **Quantile Crossing** | 0.0% | — | 0% | ✅ Pass | No invalid predictions |

*Note: All metrics calculated using 1-second resolution data from August 1–31, 2019. Training/validation split: 80%/20% chronological.*

## B.2 Binary Classifier Performance

### Table B.2: Instability Detection Metrics (August 2019)

| Metric | Value | Target | Status | Interpretation |
|--------|-------|--------|--------|----------------|
| **Precision** | 0.92 | >0.85 | ✅ Pass | 92% of alerts are true instability |
| **Recall** | 0.88 | >0.80 | ✅ Pass | Catches 88% of actual instability events |
| **F1-Score** | 0.90 | >0.85 | ✅ Pass | Balanced precision-recall performance |
| **AUC-ROC** | 0.96 | >0.90 | ✅ Pass | Excellent discrimination capability |
| **Specificity** | 0.94 | >0.90 | ✅ Pass | Correctly identifies stable periods |
| **False Positive Rate** | 0.06 | <0.15 | ✅ Pass | Low unnecessary alert burden |
| **False Negative Rate** | 0.12 | <0.20 | ✅ Pass | Acceptable miss rate for safety system |
| **Alert Latency (s)** | 7.2 | <10 | ✅ Pass | Meets advance warning requirement |

*Instability defined as frequency below 49.8 Hz. Binary classifier derived from quantile predictions (alert when predicted lower bound < 49.8 Hz).*

## B.3 Out-of-Season Validation

### Table B.3: December 2019 Performance (Winter Conditions)

| Metric | August (Training) | December (Testing) | Change | Assessment |
|--------|------------------|-------------------|--------|------------|
| **PICP (%)** | 79.5 | 74.2 | -5.3 pp | ⚠️ Below target |
| **Pinball Loss** | 0.0155 | 0.0172 | +10.9% | ✅ Acceptable |
| **Pinball Loss (Upper)** | 0.0168 | 0.0189 | +12.5% | ✅ Acceptable |
| **MAE (Hz)** | 0.033 | 0.041 | +24.2% | ⚠️ Degradation |
| **Calibration (α=0.1)** | 1.8% | 0.9% | -0.9 pp | ⚠️ Increased pessimism |
| **Calibration (α=0.9)** | 79.3% | 75.6% | -3.7 pp | ⚠️ Below target |
| **F1-Score** | 0.90 | 0.84 | -6.7% | ⚠️ Reduced accuracy |

*Performance degradation in December indicates seasonal overfitting. Model trained exclusively on August data lacks exposure to winter generation/demand patterns.*

## B.4 Computational Performance

### Table B.4: Real-Time System Latency

| Operation | Mean (s) | Std (s) | Min (s) | Max (s) | Requirement | Status |
|-----------|----------|---------|---------|---------|-------------|--------|
| **Data Refresh** | 0.12 | 0.03 | 0.08 | 0.21 | <1.0 | ✅ Pass |
| **Feature Engineering** | 0.08 | 0.02 | 0.05 | 0.14 | <0.5 | ✅ Pass |
| **Model Inference** | 0.08 | 0.01 | 0.06 | 0.12 | <0.5 | ✅ Pass |
| **SHAP Explanation** | 0.15 | 0.04 | 0.10 | 0.28 | <0.5 | ✅ Pass |
| **Dashboard Render** | 0.05 | 0.01 | 0.03 | 0.09 | <1.0 | ✅ Pass |
| **Total Cycle** | **0.40** | **0.08** | **0.28** | **0.62** | **<2.0** | ✅ **Pass** |

*Measurements conducted on Intel i7-1165G7 (2.8 GHz), 16GB RAM, SSD storage. Python 3.11, Polars 0.20.x, LightGBM 4.1.x.*

## B.5 Feature Importance Stability

### Table B.5: Top Feature Importance Across Folds

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std | CV (%) |
|---------|--------|--------|--------|--------|--------|------|-----|--------|
| **RoCoF (5s smoothed)** | 38.5 | 37.8 | 39.2 | 38.1 | 37.5 | 38.2 | 0.6 | 1.6 |
| **OpSDA Wind Ramp** | 21.2 | 22.5 | 21.8 | 21.0 | 22.1 | 21.7 | 0.6 | 2.8 |
| **Renewable Penetration** | 15.8 | 14.9 | 15.2 | 15.6 | 15.0 | 15.3 | 0.4 | 2.6 |
| **Time of Day** | 8.5 | 9.2 | 8.8 | 8.9 | 9.1 | 8.9 | 0.3 | 3.4 |
| **Wind Speed** | 8.1 | 7.5 | 7.9 | 8.0 | 7.5 | 7.8 | 0.3 | 3.8 |
| **Daily Inertia Cost** | 5.1 | 5.4 | 5.0 | 5.5 | 5.2 | 5.3 | 0.2 | 3.8 |
| **Solar Radiation** | 2.4 | 2.5 | 2.6 | 2.4 | 2.6 | 2.5 | 0.1 | 4.0 |
| **Temperature** | 0.4 | 0.2 | 0.5 | 0.3 | 0.2 | 0.3 | 0.1 | 33.3 |

*CV = Coefficient of Variation (Std/Mean × 100). Low CV (<5%) for top features indicates stable importance across cross-validation folds.*

## B.6 August 9, 2019 Blackout Event Metrics

### Table B.6: Blackout Prediction Timeline

| Time (UTC) | Event | Actual Freq (Hz) | Pred Lower Bound (Hz) | Advance Warning (s) | Status |
|------------|-------|------------------|----------------------|---------------------|--------|
| 16:52:30 | Initial disturbance | 49.95 | 49.92 | — | Stable |
| 16:52:33 | **Alert threshold crossed** | 49.91 | **49.80** | **7** | 🚨 **Warning** |
| 16:52:35 | Frequency declining | 49.85 | 49.76 | 5 | ⚠️ Critical |
| 16:52:40 | **Actual threshold breach** | **49.80** | 49.72 | 0 | 🔴 **Event** |
| 16:52:45 | Nadir approaching | 48.85 | 48.90 | — | 🔴 Emergency |
| 16:52:48 | **Nadir reached** | **48.80** | — | — | 🔴 **Blackout** |

*Alert threshold: 49.80 Hz. Actual nadir: 48.80 Hz. Automatic load shedding triggered at 48.80 Hz.*

---

## References

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

Ucar, F. (2023) 'Explainable AI for smart grid stability prediction: Enhancing operator trust through SHAP analysis', *Energy and AI*, 14, p. 100256. doi: 10.1016/j.egyai.2023.100256.

Zhang, Y., Wang, J. and Chen, X. (2021) 'A review of machine learning applications in power system frequency forecasting', *Renewable and Sustainable Energy Reviews*, 145, p. 111156. doi: 10.1016/j.rser.2021.111156.
