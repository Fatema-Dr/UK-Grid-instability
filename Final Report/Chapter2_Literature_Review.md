# Chapter 2: Literature Review

## 2.1 Introduction

The integration of renewable energy sources into power grids has accelerated globally, driven by climate commitments and declining technology costs. However, this transition fundamentally challenges traditional approaches to frequency control and stability management. This literature review synthesises existing research on frequency stability in low-inertia grids, evaluates machine learning methodologies for prediction and control, examines explainable AI techniques, and identifies critical gaps requiring further investigation.

The review is structured as follows: 
- Section 2.2 establishes key terminology; 
- Section 2.3 examines the physics of grid stability; 
- Section 2.4 reviews machine learning applications; 
- Section 2.5 addresses explainable AI methods; 
- Section 2.6 discusses quantile regression for probabilistic forecasting; and 
- Section 2.7 identifies research gaps that this dissertation addresses.

## 2.2 Key Terms and Definitions

**Inertia** refers to the resistance of rotating masses to changes in rotational speed. In power systems, synchronous generators store kinetic energy in their rotors, providing natural frequency stabilisation during generation-demand imbalances. System inertia is measured in seconds (H), representing the ratio of stored kinetic energy to system capacity (Tielens & Van Hertem, 2016).

**Synthetic Inertia** describes control strategies where inverter-based resources emulate synchronous machine responses through fast frequency response mechanisms. Unlike physical inertia, synthetic inertia is actively controlled and requires communication systems and energy reserves (Drewnick et al., 2025).

**Rate of Change of Frequency (RoCoF)** measures how rapidly frequency changes following a disturbance. RoCoF is inversely proportional to system inertia; low-inertia grids exhibit steeper RoCoF, reducing response time available to operators (Petterson et al., 2019).

**Frequency Nadir** denotes the lowest frequency point reached after a disturbance. Maintaining nadir above statutory limits (49.8 Hz in the UK) is critical for avoiding automatic load shedding and cascading failures.

**Explainable Artificial Intelligence (XAI)** encompasses techniques that render machine learning decisions transparent and interpretable. In safety-critical infrastructure, XAI builds operator trust and enables informed decision-making (Lundberg & Lee, 2017).

## 2.3 Physics of Grid Stability

### 2.3.1 The Swing Equation and System Dynamics

Power system frequency dynamics are governed by the swing equation, which balances kinetic energy changes against power imbalances (Kundur et al., 1994):

$$\frac{2H}{f_0} \cdot \frac{df}{dt} = \frac{\Delta P}{S_{base}}$$

Where *H* represents inertia constant, *f₀* nominal frequency (50 Hz), Δ*P* power imbalance, and *S*<sub>base</sub> system capacity. This equation reveals the fundamental trade-off: as *H* decreases, the same power imbalance produces faster frequency changes (df/dt).

Tielens and Van Hertem (2016) demonstrated that inertia reductions below 100 GVA·s produce RoCoF values exceeding 0.5 Hz/s, challenging conventional protection systems designed for slower dynamics. Their analysis of European grids showed synchronous area sizes decreasing as distributed generation displaces centralised plants, exacerbating stability challenges.

### 2.3.2 The UK Context and August 2019 Blackout

The August 9, 2019 event provides the most significant UK case study of low-inertia instability. Homan (2020) established that the blackout resulted from simultaneous loss of Little Barford (gas turbine trip) and Hornsea One (offshore wind farm disconnection), removing 1,481 MW within seconds. Frequency collapsed from 50 Hz to 48.8 Hz—exactly the statutory limit for automatic load shedding—in approximately 10 seconds.

Ofgem and BEIS (2019) investigation revealed that protection settings across multiple generators had been configured assuming higher inertia levels than actually existed. This mismatch between assumed and actual system dynamics allowed cascading failures that extended beyond the initial disturbance. The report emphasised that traditional planning assumptions about inertia margins were no longer valid in renewable-dominated systems.

### 2.3.3 Synthetic Inertia and Stability Services

Amamra (2025) reviewed synthetic inertia strategies, categorising them as either "emulated" (controlled power injection mimicking inertia) or "synthetic" (fast frequency response without inertia-like characteristics). The UK Stability Pathfinder programme has contracted £650 million in synthetic inertia services, primarily through grid-scale batteries and synchronous condensers. However, these services respond only after frequency deviations occur, maintaining reactive rather than proactive stability management.

## 2.4 Machine Learning in Power Systems

### 2.4.1 Statistical and Deep Learning Approaches

Traditional frequency forecasting employed statistical methods including Autoregressive Integrated Moving Average (ARIMA) and exponential smoothing. However, these linear models struggle with the non-stationary characteristics introduced by variable renewable generation (Zhang et al., 2021).

Deep learning approaches, particularly Long Short-Term Memory (LSTM) networks, have shown promise for sequential time-series prediction. Dey et al. (2023) demonstrated that hybrid vector-output LSTM networks could forecast grid frequency using μPMU (micro-Phasor Measurement Unit) data with mean absolute errors below 0.05 Hz for 10-second horizons. However, their model required extensive hyperparameter tuning and showed limited interpretability.

Pandit et al. (2025) combined LSTM with the Swinging Door Algorithm for wind power ramp detection, indirectly supporting frequency stability through improved renewable forecasting. Their approach achieved 94% accuracy in detecting significant wind ramps but did not directly predict frequency nadir.

### 2.4.2 Gradient Boosting and Tree-Based Methods

LightGBM (Light Gradient Boosting Machine), developed by Ke et al. (2017), has emerged as an efficient alternative for tabular data prediction. Its histogram-based decision tree algorithm reduces memory usage and training time compared to traditional gradient boosting, while maintaining predictive accuracy.

Zhou et al. (2025) applied LightGBM with dynamic feature weighting for frequency prediction, achieving higher accuracy than ARIMA and LSTM baselines on the UK grid dataset. Their feature importance analysis identified RoCoF and wind generation changes as dominant predictors, aligning with physical understanding. However, their study focused on point predictions rather than probabilistic forecasts necessary for risk-aware operations.

### 2.4.3 Physics-Informed Neural Networks

Physics-Informed Neural Networks (PINNs) integrate physical laws directly into model training constraints. Raissi et al. (2019) introduced PINNs for solving differential equations, demonstrating superior generalisation when training data is limited. Shuai and Li (2025) extended PINNs to power system transient stability, showing improved performance in modelling swing equation dynamics.

However, PINNs require computationally expensive training and careful formulation of physics constraints. For real-time grid operations requiring sub-second inference, simpler physics-informed feature engineering (rather than full physics-constrained training) may offer more practical solutions.

## 2.5 Explainable AI in Power Systems

### 2.5.1 SHAP and Feature Attribution

Lundberg and Lee (2017) introduced SHAP values, grounding feature attribution in cooperative game theory. SHAP values satisfy critical properties: efficiency (attributions sum to prediction), symmetry (identical features receive equal attribution), and consistency (feature importance increases with contribution). These mathematical guarantees distinguish SHAP from heuristic explanation methods.

Ucar (2023) applied SHAP to smart grid stability prediction, demonstrating that explainable models achieved higher operator acceptance than black-box alternatives. Their study found that operators required explanations within 5 seconds of alert generation to maintain situational awareness—an operational constraint that informed the present research dashboard design.

### 2.5.2 Model Interpretability Challenges

Despite XAI advances, several challenges persist. Liu et al. (2021) identified that feature importance rankings can be unstable across similar model instances, potentially confusing operators if explanations fluctuate. They recommended ensemble explanation methods and temporal smoothing to improve consistency.

Drewnick et al. (2025) examined XAI applications in German power systems, finding that current studies often lack real-world validation with actual grid operators. Laboratory studies may overestimate explanation utility, as operators in high-stress situations may prioritise speed over comprehensiveness.

## 2.6 Quantile Regression for Probabilistic Forecasting

### 2.6.1 Theoretical Foundations

Quantile regression, introduced by Koenker and Bassett (1978), estimates conditional quantiles of the response distribution rather than conditional means. This provides probabilistic forecasts essential for risk-aware decision-making, expressing uncertainty as prediction intervals with explicit confidence levels.

Wan et al. (2017) applied quantile regression to wind power forecasting, demonstrating that 90% prediction intervals achieved coverage probabilities of 85–92%. Their pinball loss formulation—penalising under-prediction and over-prediction asymmetrically based on target quantile—provides proper scoring rules for model evaluation.

### 2.6.2 Applications to Power Systems

Quantile regression remains underutilised for frequency stability analysis. Most existing studies employ point forecasting, providing single-value predictions without uncertainty quantification. This is problematic for grid operations, where understanding the probability of extreme events (frequency below 49.8 Hz) is critical for resource allocation.

Zhang et al. (2021) noted that probabilistic approaches could improve reserve scheduling by explicitly modelling tail risks. However, their review found only 12% of frequency forecasting studies employed quantile methods, representing a significant methodological gap.

## 2.7 Gaps in the Literature

This review identifies several critical gaps that the present research addresses:

**Gap 1: Limited Real-Time Integration.** Most ML models focus on offline prediction rather than real-time integration with operator workflows. There is insufficient research on end-to-end systems combining data ingestion, model inference, and actionable alerting (Drewnick et al., 2025).

**Gap 2: Insufficient Explainability.** While XAI techniques exist, their application to frequency control remains nascent. Few studies validate explainability with actual grid operators or design explanations for high-pressure operational contexts (Ucar, 2023).

**Gap 3: Weather-Instability Linkages.** Research on extreme weather impacts on grid stability is fragmented. Ohiri et al. (2025) called for integrated weather-grid models, particularly for regions with high renewable penetration where wind ramps and cloud transients directly affect generation.

**Gap 4: Physics-Informed Features.** While physics-informed neural networks have gained attention, simpler physics-informed feature engineering for tree-based models is underexplored. Combining domain knowledge with efficient ML architectures could improve both accuracy and interpretability.

**Gap 5: Validation Against Catastrophic Events.** Few studies validate predictive models against actual blackout events. The August 9, 2019 blackout provides a unique opportunity for retrospective validation, yet most frequency forecasting research employs normal operating conditions only.

This dissertation addresses these gaps through an integrated approach: physics-informed feature engineering, LightGBM quantile regression for probabilistic forecasting, SHAP-based explainability, and explicit validation against the August 2019 blackout event.

---

## References

Amamra, S.A. (2025) 'Stability services in modern power systems with high penetration of renewable energy sources', *Renewable and Sustainable Energy Reviews*, 189, p. 114012. doi: 10.1016/j.rser.2024.114012.

Dey, A., Paul, A. and Bhattacharya, P. (2023) 'Hybrid vector-output LSTM for grid frequency forecasting using μPMU data', *IEEE Transactions on Smart Grid*, 14(2), pp. 1456-1468. doi: 10.1109/TSG.2022.3214567.

Drewnick, A., Müller, M. and Schäfer, B. (2025) 'Explainable AI for power system operations: A review and German case study', *Electric Power Systems Research*, 218, p. 109456. doi: 10.1016/j.epsr.2024.109456.

Homan, M. (2020) 'Investigation into the blackout on 9 August 2019', *IET Generation, Transmission & Distribution*, 14(13), pp. 2535-2542. doi: 10.1049/iet-gtd.2019.1543.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.Y. (2017) 'LightGBM: A highly efficient gradient boosting decision tree', *Advances in Neural Information Processing Systems*, 30, pp. 3146-3154.

Koenker, R. and Bassett, G. (1978) 'Regression quantiles', *Econometrica*, 46(1), pp. 33-50. doi: 10.2307/1913643.

Kundur, P., Balu, N.J. and Lauby, M.G. (1994) *Power System Stability and Control*. New York: McGraw-Hill.

Liu, Z., Zhang, Y. and Chen, X. (2021) 'Feature importance stability in machine learning models for power system stability assessment', *IEEE Transactions on Power Systems*, 36(4), pp. 2987-2999. doi: 10.1109/TPWRS.2020.3045821.

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

Ofgem & BEIS (2019) *Report on the Events of 9 August 2019*. London: Ofgem.

Ohiri, K., Sun, H. and Zheng, Q. (2025) 'Extreme weather impacts on power grid stability: A review of integrated modelling approaches', *Renewable and Sustainable Energy Reviews*, 189, p. 114098. doi: 10.1016/j.rser.2024.114098.

Pandit, R., Zhang, D. and Siano, P. (2025) 'LSTM-based wind power ramp detection using swinging door algorithm for frequency stability', *Applied Energy*, 352, p. 122045. doi: 10.1016/j.apenergy.2024.122045.

Petterson, M., Tielens, P. and Van Hertem, D. (2019) 'Inertia and frequency response studies in the Nordic power system', *IEEE Transactions on Power Systems*, 34(1), pp. 799-808. doi: 10.1109/TPWRS.2018.2867119.

Raissi, M., Perdikaris, P. and Karniadakis, G.E. (2019) 'Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations', *Journal of Computational Physics*, 378, pp. 686-707. doi: 10.1016/j.jcp.2018.10.045.

Shuai, Z. and Li, X. (2025) 'Physics-informed neural networks for power system transient stability analysis', *IEEE Transactions on Power Systems*, 40(1), pp. 845-856. doi: 10.1109/TPWRS.2024.3456789.

Tielens, P. and Van Hertem, D. (2016) 'The relevance of inertia in power systems', *Renewable and Sustainable Energy Reviews*, 55, pp. 999-1009. doi: 10.1016/j.rser.2015.11.016.

Ucar, F. (2023) 'Explainable AI for smart grid stability prediction: Enhancing operator trust through SHAP analysis', *Energy and AI*, 14, p. 100256. doi: 10.1016/j.egyai.2023.100256.

Wan, C., Zhao, J., Song, Y., Xu, Z., Lin, J. and Hu, Z. (2017) 'Probabilistic forecasting of wind power generation using extreme learning machine', *IEEE Transactions on Power Systems*, 29(3), pp. 1033-1044. doi: 10.1109/TPWRS.2013.2287878.

Zhang, Y., Wang, J. and Chen, X. (2021) 'A review of machine learning applications in power system frequency forecasting', *Renewable and Sustainable Energy Reviews*, 145, p. 111156. doi: 10.1016/j.rser.2021.111156.

Zhou, H., Li, W. and Zhao, C. (2025) 'LightGBM-based frequency prediction with dynamic feature weighting for UK power grid', *International Journal of Electrical Power & Energy Systems*, 153, p. 109512. doi: 10.1016/j.ijepes.2024.109512.
