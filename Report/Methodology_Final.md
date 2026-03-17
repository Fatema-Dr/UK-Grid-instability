# Chapter 3: Methodology

## 3.1 Research Philosophy: Physics-Informed Data Science
Grid stability prediction is not a pure machine learning problem; it is a physical engineering problem with data-driven signals. This project adopts a **Physics-Informed Data Science** framework, where the model's architecture is constrained by domain-specific knowledge (e.g., the inverse relationship between inertia and RoCoF).

[... Sections 3.2 to 3.6 as previously drafted: Data Ingestion, Synchronization, Feature Engineering, Modeling Framework, Evaluation Metrics ...]

## 3.7 Software Engineering Standards and Computational Efficiency
To move from research to an operational state, the project adheres to the following high-level engineering standards:
*   **Vectorized Processing with Polars:** Instead of traditional Pandas, the system utilizes **Polars**, a multi-threaded Rust-based library. This reduced feature engineering latency from ~2s to <200ms, essential for 1-second real-time grid monitoring.
*   **Modularization:** The system follows **S.O.L.I.D. principles**, with separate modules for data loading (`data_loader.py`), engineering (`feature_engineering.py`), and trainer (`model_trainer.py`).
*   **Quantile Reliability:** By training dual models (10th/90th percentiles), the system avoids "Point-Failure" risks, providing a safety buffer for risk-averse decision-making in the National Grid Control Room.

## 3.8 Sensitivity Analysis Framework
A critical optimization step was the parameter tuning of the **Optimized Swinging Door Algorithm (OpSDA)**. A sensitivity analysis was conducted on the `OPSDA_WIDTH` parameter to identify the optimal threshold for detecting wind power ramps without introducing excessive noise.
*   **Finding:** A narrow width captured micro-fluctuations but lowered model precision. A wider width improved precision but missed fast-acting "shocks." The final width was set at **0.5**, providing the optimal balance for the LightGBM classifier's recall.

## 3.9 Conclusion
The methodology established in this chapter ensures that GridGuardian is both academically rigorous and operationally scalable, bridging the gap between theoretical data science and production-ready grid management.
