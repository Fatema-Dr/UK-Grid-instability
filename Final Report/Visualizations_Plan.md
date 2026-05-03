# GridGuardian: Dissertation Visualization & Table Plan

This document serves as a high-signal checklist for the most important figures and tables to include in the final dissertation report to demonstrate technical depth and physics-informed understanding.

---

## Chapter 1: Introduction
- [ ] **Figure 1.1: The Frequency "Safety Zone"**
    - *Content:* Conceptual diagram showing 50.0 Hz nominal line and ±0.2 Hz (49.8–50.2 Hz) statutory limits.
    - *Goal:* Visually define the "Problem Space."
- [ ] **Figure 1.2: The August 9, 2019 Baseline**
    - *Content:* Plot of raw 1-second frequency data for the blackout day.
    - *Goal:* Highlight the lightning strike moment and the 48.8 Hz nadir.

---

## Chapter 3: Methodology
- [ ] **Figure 3.1: High-Level System Architecture**
    - *Content:* Formal version of the data flow (APIs → Polars → Models → Dashboard).
    - *Goal:* Demonstrate modular, production-ready engineering.
- [ ] **Figure 3.2: OpSDA Logic Visualization (The "Physics" Proof)**
    - *Content:* "Before vs. After" plot showing noisy wind speed and the compressed piecewise-linear representation.
    - *Goal:* Prove the value of the Optimized Swinging Door Algorithm.
- [ ] **Figure 3.3: Quantile Regression Concept**
    - *Content:* Diagram showing a predicted "band" (α=0.1 to α=0.9) around a center-line.
    - *Goal:* Explain probabilistic vs. binary forecasting.
- [ ] **Table 3.1: Dataset Split Strategy**
    - *Content:* Training (Aug 1–6), Calibration (Aug 7–8), Validation (Aug 9).
    - *Goal:* Demonstrate academic rigor and temporal validation.

---

## Chapter 4: Results
- [ ] **Table 4.1: Model Performance Matrix**
    - *Content:* Comparing LightGBM (Lower/Upper) and LSTM baseline across Pinball Loss, MAE, and RMSE.
    - *Goal:* Quantitative proof of model superiority.
- [ ] **Table 4.2: Uncertainty Calibration Table**
    - *Content:* Target Quantile (10%, 90%) vs. Observed Frequency (e.g., 1.8% and 79.3%).
    - *Goal:* Discuss the "Pessimistic Bias" (fail-safe property).
- [ ] **Figure 4.1: Feature Importance Plot**
    - *Content:* Bar chart of feature gains (RoCoF, OpSDA Ramps, Lags, etc.).
    - *Goal:* Confirm that physics-based features are the primary drivers.
- [ ] **Figure 4.2: Prediction vs. Actual (Stable Period)**
    - *Content:* "Zoom-in" plot of a stable hour showing the frequency staying within the 80% band.
    - *Goal:* Prove model reliability during normal operations.

---

## Chapter 5: Analysis & Discussion
- [ ] **Figure 5.1: THE BLACKOUT ALERT (The "Money Shot")**
    - *Content:* Frequency dropping, Uncertainty Band widening, and Lower Bound crossing 49.8 Hz.
    - *Goal:* Visualize the **7-second advance warning** provided before the threshold breach.
- [ ] **Figure 5.2: SHAP Waterfall Plot (The "Why")**
    - *Content:* Feature attribution for the exact second the alert fired.
    - *Goal:* Demonstrate "Trusted Advisor" explainability.
- [ ] **Figure 5.3: Intervention Simulation Plot**
    - *Content:* "What-If" plot showing frequency with vs. without 2000MW synthetic inertia.
    - *Goal:* Prove the value of the physics-informed simulator.

---

## Appendices
- [ ] **Table A.1: Test Case Matrix**
    - *Content:* Results of the 31 `pytest` tests (ID, Feature, Expected, Result, Pass/Fail).
    - *Goal:* Verify code integrity and logic.
- [ ] **Table A.2: Hyperparameter Table**
    - *Content:* Final LightGBM and LSTM parameters for reproducibility.
    - *Goal:* Scientific transparency.

---

### **A+ Captioning Strategy**
Ensure every figure has a caption that explains **what it proves**, not just what it is.
*Example:* "Figure 5.1: Real-time instability alert triggered 7 seconds prior to the frequency threshold breach, demonstrating the system's ability to provide operationally significant lead-time for BESS intervention."
