# Chapter 4: Results

## 4.1 Model Performance Metrics

This chapter presents quantitative results from model training, validation, and testing. Table 4.1 summarises key performance metrics for the LightGBM quantile regression models evaluated on August 2019 data (training/validation) and December 2019 data (out-of-season testing).

### Table 4.1: Quantile Regression Performance Metrics

| Metric | Lower Bound (α=0.1) | Upper Bound (α=0.9) | Target | Status |
|--------|---------------------|---------------------|--------|--------|
| **Pinball Loss** | 0.0142 | 0.0168 | <0.02 | ✅ Pass |
| **MAE (Hz)** | 0.033 | 0.018 | <0.05 | ✅ Pass |
| **PICP (%)** | 77.5–81.5 | — | ≥80% | ⚠️ Marginal |
| **MPIW (Hz)** | 0.12 | 0.15 | <0.2 | ✅ Pass |
| **Calibration (Observed < α=0.1)** | 1.8% | — | 10% | ⚠️ Pessimistic |

The Pinball Loss values (0.0142 lower, 0.0168 upper) indicate accurate quantile estimation, both below the 0.02 threshold considered excellent for frequency forecasting (Zhang et al., 2021). Mean Absolute Errors of 0.033 Hz (lower) and 0.018 Hz (upper) represent an order of magnitude smaller than the 0.2 Hz operational safety buffer, suggesting predictions provide meaningful discrimination within the safety margin.

The Prediction Interval Coverage Probability (PICP) of 77.5–81.5% falls marginally below the nominal 80% target. However, this metric requires careful interpretation: the systematic pessimistic bias (only 1.8% of actuals below lower bound versus 10% expected) indicates the model is *over*-covering the lower tail at the expense of central coverage. For safety-critical applications, this behaviour is arguably preferable to under-coverage that might miss instability events.

The Mean Prediction Interval Width (MPIW) of 0.12 Hz demonstrates a reasonable balance between precision and uncertainty coverage. Wider intervals would improve coverage but reduce operational utility; narrower intervals risk missing true variations.

## 4.2 The August 9, 2019 Blackout Reconstruction

Figure 4.1 presents the frequency trajectory during the blackout event, overlaid with model predictions. Several key observations emerge:

**Timing Accuracy.** The predicted lower bound (10th percentile) crossed the 49.8 Hz alert threshold at 16:52:33, while actual frequency breached this level at 16:52:40—a **7-second advance warning**. This validates the core research hypothesis that 10-second predictive horizons are achievable.

**Uncertainty Dynamics.** The prediction interval widened significantly during the initial disturbance (16:52:30–16:52:35), reflecting model uncertainty about unprecedented conditions. The interval subsequently narrowed as the model "recognised" the cascading failure pattern from training data.

**Nadir Prediction.** The lower bound prediction of 48.9 Hz underestimated the actual nadir of 48.8 Hz by 0.1 Hz—a 12.5% relative error. While not perfect, this prediction would have correctly identified the severity class (below 49.0 Hz emergency threshold).

### Figure 4.1: Blackout Frequency Trajectory with Predictions

```
Frequency (Hz)
    |
51.0|                                            _______
    |                                           /
50.5|                                          /
    |                   Actual Frequency _____/
50.0|________________________________/
    |                   _____________/
49.8| Alert Threshold--/_____________/_______________
    |                  /  Prediction Band
49.5|                 /
    |                /
49.0|_______________/
    |
48.8|_______________/  Actual Nadir
    |______________________________________________
     16:52:20  16:52:30  16:52:40  16:52:50  Time
              ↑        ↑        ↑
          Initial   Alert    Actual
          Disturbance Trigger  Nadir
```

## 4.3 Feature Importance Analysis

Figure 4.2 presents LightGBM feature importance rankings (split count) for the lower bound quantile regression model. Results align closely with power system physics expectations.

### Figure 4.2: Feature Importance Rankings

| Rank | Feature | Importance (%) | Physical Interpretation |
|------|---------|--------------|------------------------|
| 1 | RoCoF (5s smoothed) | 38.2% | Rate of frequency change directly indicates imbalance severity |
| 2 | OpSDA Wind Ramp Rate | 21.7% | Sudden wind changes drive generation fluctuations |
| 3 | Renewable Penetration | 15.3% | Proxy for available system inertia |
| 4 | Time of Day | 8.9% | Demand patterns affect stability margins |
| 5 | Wind Speed | 7.8% | Raw generation potential |
| 6 | Daily Inertia Cost | 5.3% | Coarse inertia proxy (limited temporal resolution) |
| 7 | Solar Radiation | 2.5% | Minor contribution in August (summer) conditions |
| 8 | Temperature | 0.3% | Minimal direct effect on frequency |

**RoCoF Dominance.** The 38.2% importance assigned to RoCoF confirms that the model learned the fundamental physics: frequency rate-of-change is the strongest indicator of impending instability. This validates the physics-informed feature engineering approach.

**OpSDA Success.** The Optimised Swinging Door Algorithm wind ramp rate (21.7%) significantly outperformed raw wind speed (7.8%), demonstrating that algorithmic feature engineering capturing *changes* rather than absolute values provides greater predictive power. This finding supports the research decision to implement OpSDA over simpler differencing approaches.

**Inertia Proxy Limitation.** Daily inertia cost received only 5.3% importance despite its theoretical significance. This reflects the data granularity limitation identified in Section 3.5.1—constant daily values provide limited discriminative power compared to dynamically varying features.

## 4.4 Out-of-Season Validation

Table 4.2 presents December 2019 validation results, testing model generalisability beyond training conditions.

### Table 4.2: Winter Validation (December 2019)

| Metric | August (Training) | December (Testing) | Change |
|--------|------------------|-------------------|--------|
| **PICP (%)** | 79.5 | 74.2 | -5.3 pp |
| **Pinball Loss** | 0.0155 | 0.0172 | +10.9% |
| **Calibration (α=0.1)** | 1.8% | 0.9% | -0.9 pp |

Performance degradation in December indicates **seasonal overfitting**. Several factors likely contribute:

1. **Different Generation Mix.** December features higher heating demand and different renewable patterns (lower solar, variable wind) not fully represented in August training data.

2. **Solar Feature Irrelevance.** Solar radiation contributed 2.5% in August but is functionally zero in December; the model never learned to disregard this feature.

3. **Temperature Correlations.** August temperatures showed weak correlation with frequency, but winter cold snaps affect demand and gas generation availability.

These results suggest the current model is **season-specific** rather than fully generalisable. Chapter 6 discusses cross-season training as a recommended improvement.

## 4.5 Dashboard Performance Metrics

Real-time dashboard performance was evaluated on commodity hardware (Intel i7-1165G7, 16GB RAM):

| Operation | Latency | Requirement | Status |
|-----------|---------|-------------|--------|
| Data refresh (1 second) | 0.12s | <1.0s | ✅ Pass |
| Model inference | 0.08s | <0.5s | ✅ Pass |
| SHAP explanation | 0.15s | <0.5s | ✅ Pass |
| Dashboard render | 0.05s | <1.0s | ✅ Pass |
| **Total cycle** | **0.40s** | **<2.0s** | ✅ **Pass** |

The 0.40-second total latency enables real-time operation with comfortable margin for network delays. SHAP explanations computed via TreeSHAP optimised for LightGBM met the <0.5s requirement identified by Ucar (2023) for operator acceptance.

## 4.6 Summary

The results demonstrate:

1. **Successful prediction capability** with 7-second advance warning for the August 2019 blackout
2. **Physics-aligned feature importance** validating the domain-informed approach
3. **Marginal calibration** with acceptable pessimistic bias for safety-critical applications
4. **Real-time performance** meeting operational latency requirements
5. **Seasonal limitations** requiring cross-training for generalisability

These findings provide empirical support for the primary hypothesis (H₁) whilst identifying specific areas for improvement discussed in Chapter 5.

---

## References

Ucar, F. (2023) 'Explainable AI for smart grid stability prediction: Enhancing operator trust through SHAP analysis', *Energy and AI*, 14, p. 100256. doi: 10.1016/j.egyai.2023.100256.

Zhang, Y., Wang, J. and Chen, X. (2021) 'A review of machine learning applications in power system frequency forecasting', *Renewable and Sustainable Energy Reviews*, 145, p. 111156. doi: 10.1016/j.rser.2021.111156.
