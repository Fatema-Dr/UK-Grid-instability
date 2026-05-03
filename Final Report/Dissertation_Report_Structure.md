# Dissertation Report: Structure and Requirements (GridGuardian )

This document outlines the final structure, word limits, and section-specific requirements for the GridGuardian dissertation, based on UEL project module standards and the technical scope of the system.

**Total Target Word Count:** 10,000 words.

---

## 1. Title Page (Front Cover)
- **Requirements:** Must use the official UEL template.
- **Details:** Project Title: *GridGuardian: Proactive AI for Low-Inertia Power Grids*, Student Name, ID, Supervisor Name, and Submission Date.

## 2. Abstract
- **Word Limit:** **250 to 500 words.**
- **Key Points:** 
    - Purpose: Addressing the UK "Inertia Crisis" and the August 9, 2019 blackout.
    - Methodology: Physics-informed ML using LightGBM Quantile Regression and SHAP.
    - Findings: Achievement of a 10-second predictive horizon and successful blackout re-run validation.

## 3. Acknowledgement
- **Word Limit:** Maximum 500 words.
- **Details:** Thanking the supervisor, NESO for open data, and open-source contributors (Polars, LightGBM, SHAP).

## 4. Table of Contents / List of Figures / List of Tables
- **Requirements:** Professional formatting with automated page numbering.

---

## Chapter 1: Introduction 
- **Target:** ~1,000 words.
- **Sub-sections:**
    - **1.1 Introduction & Background:** The shift to renewables and the loss of physical inertia.
    - **1.2 Rationale:** The economic cost of instability (>£650M in procurement) and the social impact of blackouts.
    - **1.3 Research Question & Hypotheses:** Can quantile regression provide reliable 10s warnings? (H₁ vs H₀).
    - **1.4 Research Aim and Objectives:** 
        - **Aim:** Design and implement a physics-informed early warning system.
        - **Objectives:** Data pipeline (Polars), Feature Engineering (OpSDA), Probabilistic Modeling (Quantile LGBM), XAI Integration (SHAP), and Blackout Validation.
    - **1.5 Research Outline:** Roadmap of the 6 chapters.

---

## Chapter 2: Literature Review 
- **Word Limit:** **Maximum 3,000 words.**
- **Sub-sections:**
    - **2.1 Physics of Grid Stability:** The Swing Equation and the mechanics of mechanical inertia.
    - **2.2 The UK Context & August 2019 Blackout:** Analysis of the event as a cascading failure.
    - **2.3 Machine Learning in Power Systems:** Comparing statistical models (ARIMA) vs. Deep Learning (LSTM) vs. Gradient Boosting (LightGBM).
    - **2.4 Explainable AI (XAI) & Uncertainty:** Theory of Shapley values and Quantile Regression for risk-averse forecasting.
    - **2.7 Gaps in the Literature:** Lack of integrated real-time probabilistic alerting for the UK grid.

---

## Chapter 3: Methodology
- **Target:** ~1,500 – 2,000 words.
- **Sub-sections:**
    - **3.1 Introduction:** Overview of the dual-approach: **Practical Implementation** and **Theoretical Field Research**.
    - **3.2 Research Design:** Defining the connection between physics-informed feature engineering and predictive stability.
    - **3.3 Practical Approach (Implementation & SDLC):** 
        - **3.3.1 SDLC:** Justification for **Agile** (iterative model development and feature engineering sprints).
        - **3.3.2 Technical Method:** Selection of high-performance tools (**Polars, LightGBM, Streamlit**) for real-time inference.
    - **3.4 Theoretical Approach (Field Research):** 
        - **3.4.1 Literature Strategy:** Process for identifying and evaluating grid stability theories.
        - **3.4.2 Quantitative Research:** Detailed strategy for using **Secondary Datasets** (NESO CKAN & Open-Meteo) as the "Field" data.
        - **3.4.3 Data Acquisition:** API integration logic and frequency/weather data synchronization strategy.
    - **3.5 Challenges and Limitations:** Addressing the inertia data granularity gap and sensor micro-jitter in RoCoF.
    - **3.9 Ethical & Legal Considerations:** Evidence of ethical approval, confidentiality of grid data (if applicable), and data protection compliance (**GDPR 2018**).

---

## Chapter 4: Results / Findings
- **Target:** ~1,500 words.
- **Sub-sections:**
    - **4.1 Statistical Outcomes:** Pinball Loss, PICP (Coverage), and MPIW (Width) results.
    - **4.2 The Blackout Re-Run:** Visual evidence of the "Uncertainty Band" during the 2019 collapse.
    - **4.3 Feature Importance:** Visualizing RoCoF and OpSDA dominance in the model.
    - **4.4 Dashboard Artifacts:** Screenshots of the "Control Room" UI, SHAP risk drivers, and Intervention Simulator.
- **Note:** No interpretation in this chapter; present the raw metrics and plots.

---

## Chapter 5: Analysis / Discussion
- **Target:** ~2,000 words.
- **Sub-sections:**
    - **5.1 Performance Reflection:** Why LightGBM outperformed LSTM baselines for tabular grid data.
    - **5.2 Safety Analysis:** Discussing the "Pessimistic Bias" as a desirable fail-safe feature for critical infrastructure.
    - **5.3 XAI Interpretation:** How SHAP values explain the "Hidden Risks" during the blackout.
    - **5.4 Objective Review:** Assessing how the 10-second warning fulfills operational requirements.

---

## Chapter 6: Conclusion
- **Target:** ~1,000 words.
- **Sub-sections:**
    - **6.1 Summary of Key Findings:** Proactive alerting is achievable with physics-informed features.
    - **6.2 Limitations:** Granularity of inertia data and single-season training scope.
    - **6.3 Lessons Learned:** The importance of Polars for high-resolution time-series performance.
    - **6.4 Future Recommendations:** Integration of half-hourly inertia data and cross-season model training.

---

## References
- **Requirements:** **Harvard Referencing Style.**
- **Note:** Ensure all key references from the project document are cited.

## Appendices
- **Appendix A:** Sample OpSDA Algorithm Implementation (Python).
- **Appendix B:** Full Model Evaluation Tables (Aug 2019).
- **Appendix C:** Dashboard User Manual & Additional Screenshots.

---

## Key General Requirements
- **Academic Integrity:** Use your own words. Turnitin similarity should be <20%. 
- **Referencing:** All claims must be cited in-text and referenced at the end using Harvard style.
- **Validation:** Changes must be verified against historical blackout data (Aug 9, 2019).
