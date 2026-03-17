# Chapter 2: Literature Review (Integrated Final Version)

[... Sections 2.1 to 2.5 as previously drafted ...]

## 2.6 Synthesis of Existing Research and the GridGuardian Research Gap

While the literature provides a robust foundation for individual components of grid stability, a significant gap remains in the integration of these methodologies into a cohesive, real-time decision-support framework tailored for the Great Britain (GB) context. The table below synthesizes the key research findings and identifies the specific "high-water mark" set by this project.

| Study | Model Approach | Lead Time | Uncertainty Included? | XAI (SHAP) Included? | UK-Specific Context? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Amamra (2025)** | Empirical Analysis | N/A (Ex-post) | No | No | Yes |
| **Zhou et al. (2025)** | LightGBM | 1-5 Seconds | No | No | No |
| **Dey et al. (2025)** | LightGBM/LSTM | 24 Hours | No | Yes | Yes |
| **Pandit et al. (2025)** | LSTM + OpSDA | 10-60 Minutes | No | No | No |
| **GridGuardian (This Work)** | **Quantile LightGBM** | **10 Seconds** | **Yes (10th/90th)** | **Yes** | **Yes** |

### 2.7 Conclusion
GridGuardian bridges the gap between ex-post empirical analysis (Amamra, 2025) and black-box point forecasts (Zhou et al., 2025). By unifying **physics-informed feature engineering (OpSDA)** with **probabilistic forecasting (Quantile Regression)** and **operational explainability (SHAP)**, this research provides the first holistic, risk-aware alerting tool for the UK's evolving low-carbon energy landscape.
