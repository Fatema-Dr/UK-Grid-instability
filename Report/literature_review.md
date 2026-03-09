# Chapter 2: Literature Review

## 2.1 Introduction

The global energy landscape is currently navigating a profound and irreversible shift, characterised by the systematic decarbonisation of power systems to mitigate the effects of anthropogenic climate change. This transition is fundamentally altering the operational dynamics of electrical grids worldwide, moving from a paradigm of centralised, fossil-fuel-based synchronous generation to one increasingly dominated by decentralised, asynchronous renewable energy sources (RES). While this shift is imperative for achieving net-zero emissions targets, it introduces a complex set of engineering challenges that threaten the very stability and reliability the modern world depends upon from its electrical infrastructure.

The purpose of this chapter is to critically examine the existing body of academic and industry literature pertaining to three interconnected domains: the physical challenges posed by the energy transition to grid stability, the data-driven forecasting methodologies being developed to address these challenges, and the requirements for translating raw predictions into trustworthy, actionable decision-support tools. By synthesising findings across these areas, this review establishes the theoretical and methodological foundation for the GridGuardian alert system developed in this project, and identifies the specific research gap that this work aims to address.

The review is structured as follows. Section 2.2 examines the fundamental challenges of grid stability in the context of the energy transition, with a particular focus on the Great Britain (GB) power system. Section 2.3 discusses the quantification of grid instability, moving from simple thresholds to dynamic proxy variables. Section 2.4 reviews the data-driven forecasting methodologies applicable to this problem, critically comparing statistical, machine learning, and deep learning approaches. Section 2.5 addresses the critical requirements of alert systems, including explainability, probabilistic forecasting, and validation. Finally, Section 2.6 identifies the research gap that this project addresses and presents the conclusions of the review.

## 2.2 The Energy Transition and Grid Stability

### 2.2.1 The Growth of Variable Renewable Energy Sources

The growth of Variable Renewable Energy Sources (VRES) such as wind and solar photovoltaic (PV) systems is accelerating rapidly across the globe, driven by the dual imperatives of enhancing energy security and reducing greenhouse-gas emissions (Fearn, 2025). However, unlike traditional synchronous generators that provide stable and dispatchable power, energy generation from VRES is inherently dependent upon varying and unpredictable meteorological conditions. Che et al. (2025) describe this variability as introducing a "chaos" into the power system, where supply does not naturally align with demand patterns, fundamentally complicating the supply-demand balance required for stable grid operation.

This challenge is not merely theoretical. Zhao et al. (2024) have recognised that the increasing penetration of weather-dependent renewable energy sources (WD-RES) has raised legitimate and material concerns for the functioning of electric power systems, particularly during abnormal weather events. However, it is important to note a nuanced finding from their research: the fabricated perception that VRES are inherently unstable and therefore inevitably create grid instability is not fully supported by the evidence. Rather, Zhao et al. (2024) indicate that the intensity of blackouts and susceptibility to extreme weather may actually be *reduced* in grids with high WD-RES penetration, provided they are operated with advanced control strategies and sophisticated forecasting capabilities. This finding underscores the critical importance of the data-driven approach adopted in this project; the challenge is not the renewable generation itself, but the sophistication of the management systems governing its integration.

### 2.2.2 Frequency Stability and the Inertia Crisis in Low-Inertia Power Systems

A critical consequence of displacing traditional synchronous generators (SGs) with inverter-based resources (IBRs) such as wind turbines and solar panels is the systematic reduction of system inertia. Tielens and Van Hertem (2016) defined inertia as the resistance of the system's aggregate rotating mass to changes in frequency. In a traditional power system, the kinetic energy stored in the heavy rotors of synchronous generators provides an immediate, natural, and automatic response to any power imbalance. When demand momentarily exceeds supply, the rotors slow down fractionally, releasing their stored kinetic energy and thereby slowing the Rate of Change of Frequency (RoCoF). This physical buffer provides a crucial window of time for primary frequency response mechanisms to detect and correct the imbalance.

As this mechanical inertia is progressively replaced by static power electronic inverters, which have no rotating mass, the system becomes fundamentally "lighter" and more volatile. Saleem et al. (2024) provide a comprehensive assessment of this phenomenon, demonstrating that in low-inertia grids, frequency deviations following a disturbance occur more rapidly and to a greater depth. The reduced inertial buffer means that the time available for corrective action is significantly curtailed, creating an environment where even moderate disturbances can lead to cascading failures. Consequently, there is a critical and growing need for "synthetic" or "virtual" inertia control strategies. These strategies employ power electronics to mimic the inertial behaviour of synchronous machines, maintaining stability within permissible limits (Saleem et al., 2024; Tielens and Van Hertem, 2016).

A particularly important characteristic of low-inertia systems, highlighted by Amamra (2025), is that the RoCoF does not vary smoothly but can shift abruptly. This non-linear behaviour poses profound operational challenges for real-time system management, as operators and automated control systems must react to sudden, discontinuous changes rather than gradual drifts. The implication for this project is clear: an effective alert system must be capable of predicting these abrupt shifts, not merely extrapolating smooth trends.

### 2.2.3 The 9 August 2019 UK Blackout: A Defining Case Study

The risks associated with the Inertia Crisis are starkly exemplified by the major power disruption that occurred on 9 August 2019 in the United Kingdom. This event was triggered by a lightning strike on the Eaton Socon–Wymondley circuit, which caused the near-simultaneous loss of two large generation units: the Hornsea One offshore wind farm (which tripped due to an internal software fault exacerbated by the voltage disturbance) and the Little Barford gas-fired power station. The combined loss of approximately 1.4 GW of generation resulted in a rapid frequency drop to 48.8 Hz, breaching the Low Frequency Demand Disconnection (LFDD) relay settings and causing the automatic disconnection of approximately 1.1 million customers across England and Wales (Ofgem, 2019; Amamra, 2025).

This incident is widely regarded by the power systems community as a watershed event. Post-event analysis revealed that reduced system inertia, resulting directly from the increased penetration of renewable energy sources, was a major contributing factor to the severity of the frequency excursion (Saleem et al., 2024). The event demonstrated that the grid's inherent resilience, once provided naturally by the physical properties of synchronous generation, had diminished to the point where a plausible set of concurrent failures could lead to a nationally significant disruption. The 9 August 2019 blackout therefore serves as the primary validation case study for this project's alert system, providing the ultimate "stress test" against which the system's predictive capabilities are evaluated.

### 2.2.4 The Great Britain Grid as a Unique Case Study

The power system of Great Britain (GB) serves as a premier and particularly demanding case study for these challenges. The GB transmission system faces unique frequency control challenges for several interconnected reasons. Firstly, its renewable resources are non-uniformly distributed, with a preponderance of wind generation in Scotland and a concentration of demand in England (Hong et al., 2021). Secondly, GB's lower total system capacity, compared to the vast interconnected continental European grid, means that the loss of a single large infeed (such as Hornsea One during the 2019 event) represents a far larger proportional disturbance. This makes the GB system inherently more sensitive to individual contingencies (Hong et al., 2021).

Historical analysis by Homan and Brown (2020) provides quantitative evidence that the GB system has been experiencing an increasing number of frequency excursions outside operational limits. Their analysis of frequency event data reveals a clear trend: periods of high renewable energy supply have coincided with decreasing levels of system inertia, thus making the system progressively more susceptible to frequency-related events. This historical trajectory reinforces the urgency of developing proactive, predictive tools for grid operators—a need that forms the central motivation for this research.

Furthermore, GB's physical isolation as an island grid (connected to continental Europe and Ireland only via HVDC interconnectors) means that it cannot draw upon the vast inertial reserves of the continental system in the same way that, for example, France or Germany can. This isolation makes it uniquely vulnerable to the effects of low inertia, and consequently, a valuable laboratory for studying and developing solutions to these challenges.

## 2.3 Quantifying Grid Instability

To move from reactive protection to proactive alerting, the definition of instability must evolve from simple frequency deviation thresholds to dynamic, leading proxy variables that can provide advance warning of an impending event.

### 2.3.1 Operational Definitions and Regulatory Thresholds

Maintaining stability in the electric grid is fundamentally accomplished by keeping the system frequency within defined regulatory and operational constraints. The nominal frequency for the UK National Grid Electricity System Operator (ESO) is 50.00 Hz. The statutory limits, defined by the Electricity Supply Regulations, require the frequency to be maintained within ±1% (i.e., 49.5 Hz to 50.5 Hz) at all times. However, the operational limits are significantly tighter. Using research data, the key operational threshold has been established at ±0.2 Hz (i.e., from 49.8 Hz to 50.2 Hz), which serves as the primary threshold for identifying a deviation event that requires some level of intervention (Homan and Brown, 2020). Any frequency excursion beyond this ±0.2 Hz range is considered an "event" and is logged for analysis, with more severe deviations triggering automated protection mechanisms such as Low Frequency Demand Disconnection (at 48.8 Hz).

The definition and management of stability is also evolving with the development of smart grid technologies. New conceptual frameworks, such as Decentralized Smart Grid Control (DSGC), are emerging which link electricity pricing directly to grid frequency dynamics, creating market-based incentives for consumers and producers to self-regulate their behaviour in support of frequency stability (Ucar, 2023). This evolution demonstrates that instability is not only a physical phenomenon governed by electromechanical dynamics but also an economic one, where supply-demand imbalances can be anticipated through market signals.

Crucially, instability does not only occur when the frequency violates physical limits. A state of pre-instability can exist when there is a growing disparity between supply and demand that, while not yet causing a frequency breach, indicates an increasing vulnerability. Data-driven models that utilise predictive analytics can help identify these precursor states, facilitating early intervention through mechanisms such as Demand-Side Management (DSM) before the system reaches the point of violating physical limits (Ucar, 2023).

### 2.3.2 Proxy Variables for Frequency Stability

Two primary physical metrics serve as proxies for frequency stability in low-inertia systems, providing more granular and leading indicators than the frequency value alone:

-   **Rate of Change of Frequency (RoCoF):** RoCoF is the instantaneous rate at which frequency changes following a disturbance, calculated as df/dt. It is identified in the literature as the most critical metric for low-inertia grids because high RoCoF values can trigger "Loss of Mains" (LoM) protection relays on distributed generators, causing them to trip offline and thereby further exacerbating the initial power imbalance in a cascading manner (Amamra, 2025; Hong et al., 2021). As RoCoF is a derivative of frequency with respect to time, it serves as a leading indicator that captures the *dynamics* of a frequency event, not merely its current state. This property makes it an exceptionally valuable feature for predictive models.

-   **Frequency Nadir:** The frequency nadir is the minimum frequency value attained by the system following a disturbance, before the primary frequency response mechanisms arrest the decline and begin recovery. Accurate prediction of the nadir is of paramount operational importance, as it determines whether automatic load shedding will be triggered. In the UK, the LFDD threshold is set at 48.8 Hz. Recent studies have proposed a Frequency Nadir Index (FNI) that identifies linear characteristics between inertia and reserve capacity in relation to the nadir, facilitating faster and more practical estimation (Son et al., 2024). Furthermore, Lekshmi et al. (2024) demonstrate that online estimation of disturbance size is a prerequisite for accurate nadir prediction, enabling operators to deploy faster and more targeted frequency control.

### 2.3.3 Market-Based Inertia Costs as a Financial Proxy

A novel, pre-event proxy for grid fragility that has emerged in the recent literature is the cost of procuring inertia services from the market. Interest in this area has been driven by the National Grid ESO's Stability Pathfinder programme, which has established competitive markets for procuring stability services (including inertia) from non-traditional providers.

Dey et al. (2025) demonstrate that forecasting system inertia service costs allows operators to quantify the "financial stress" of the grid at any given time. Their analysis of National Grid ESO cost data from the period 2017 to 2024 reveals a steady upward trend in inertia procurement costs, with a sharp and significant increase observed between 2020 and 2022. This trend directly reflects the growing reliance on flexible, inertia-providing ancillary services as synchronous generation is displaced, and serves as a quantitative measure of the grid's increasing structural fragility.

High predicted inertia costs therefore serve as a proxy for periods of critically low physical inertia, providing a pre-event warning of potential instability before any physical disturbance has even occurred. This financial dimension of grid stability is incorporated into the GridGuardian system as the `inertia_cost` feature, bridging the domains of physical grid management and energy market dynamics.

## 2.4 Data-Driven Forecasting Methodologies

The transition from physics-based modelling to data-driven forecasting is necessitated by the growing complexity and inherent non-linearity of modern power systems. As the number of generation sources, their geographical distribution, and their dependency on weather conditions increase, the computational cost and modelling difficulty of purely physics-based approaches become prohibitive for real-time operational use. This section reviews the major data-driven forecasting paradigms relevant to grid stability prediction.

### 2.4.1 Benchmarking Against Naïve Persistence

To validate the efficacy of any advanced forecasting model, it must first be benchmarked against a Naïve Persistence baseline. This model operates on the simplest possible assumption: that the future value of a variable will equal its most recently observed value (ŷ_{t+1} = y_t). While conceptually trivial, the persistence model is a remarkably difficult baseline to outperform for very short-term forecasting horizons in power systems, where conditions often change slowly and autocorrelation is high.

However, in the context of renewable energy forecasting and grid stability prediction, where the specific challenge is to anticipate *changes* and disruptions, advanced models have been shown to offer significant improvements. Gaboitaolelwe et al. (2024) demonstrate that models such as Random Forest and Gradient Boosting significantly outperform persistence baselines in terms of Root Mean Squared Error (RMSE) and Skill Score, thereby justifying their computational cost and complexity. This benchmarking step is a methodological necessity, ensuring that the added complexity of machine learning models translates into genuine predictive value rather than unnecessary overhead.

### 2.4.2 Classical Statistical Models: SARIMAX

The SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model remains a widely recognised standard statistical baseline for time-series forecasting. SARIMAX is valued in the literature for its mathematical interpretability, its well-understood statistical properties, and its inherent ability to model seasonality and trend through its autoregressive and moving average components (Ashtar et al., 2025).

However, the fundamental limitation of SARIMAX lies in its linearity. The model assumes that the relationship between past values and future values is linear, an assumption that is increasingly violated in complex power systems where non-linear interactions between weather, generation, demand, and grid dynamics dominate the signal. In a comprehensive 2025 study comparing multiple forecasting approaches for electricity demand in The Netherlands, Ashtar et al. (2025) found that classical statistical models like SARIMAX were consistently outperformed by deep learning models in capturing the complex temporal dependencies inherent in modern energy data.

Despite these limitations, SARIMAX retains practical value as a benchmark. Its interpretability provides a useful point of comparison for understanding where and why more complex models improve upon it. In this project, SARIMAX serves as the statistical baseline against which the performance of LightGBM and LSTM is evaluated.

### 2.4.3 Gradient Boosting Machines: LightGBM

LightGBM (Light Gradient Boosting Machine) has emerged in the recent literature as a particularly powerful method for power grid frequency prediction and related forecasting tasks. LightGBM belongs to the family of Gradient Boosting Decision Tree (GBDT) algorithms but differentiates itself through several computational innovations, including a leaf-wise (rather than level-wise) tree growth strategy and histogram-based algorithms for feature binning. These innovations deliver a significant improvement in both training speed and memory efficiency, making it well-suited for the large-scale, high-frequency datasets characteristic of grid stability analysis.

Two recent studies provide compelling evidence for LightGBM's effectiveness in this domain. Dey et al. (2025), in their study on forecasting system inertia service costs for Great Britain, found that LightGBM achieved the highest predictive accuracy among all tested models, outperforming LSTM, Residual LSTM, and XGBoost in terms of both RMSE and MAE. This finding is particularly relevant to the present research, as it demonstrates LightGBM's superiority on a closely related UK grid dataset.

Independently, Zhou et al. (2025) developed a LightGBM-based framework specifically for grid frequency prediction, incorporating a novel dynamic feature-weighting scheme. Their framework reduced the RMSE by 5.2% to 10.4% compared with benchmark methods, further substantiating LightGBM's position as a leading algorithm for this application. The combination of high accuracy and low inference latency makes LightGBM especially attractive for real-time alerting systems where decisions must be made within seconds—a core requirement of the GridGuardian system.

### 2.4.4 Deep Learning: Long Short-Term Memory (LSTM) Networks

Deep learning architectures, specifically Long Short-Term Memory (LSTM) networks, represent a fundamentally different approach to time-series forecasting. Unlike tree-based models such as LightGBM which process each observation independently (requiring explicit lag features to encode temporal context), LSTM networks are designed with internal memory cells and gating mechanisms that enable them to learn and retain long-term temporal dependencies directly from sequential data (Mollasalehi and Farhadi, 2025).

This architectural advantage makes LSTMs particularly well-suited for forecasting complex waveforms in wind and solar generation, where patterns may span many time steps and involve subtle, non-linear interactions between variables over time. Ashtar et al. (2025) demonstrated that LSTM models achieved the highest accuracy (1.88% MAPE) for long-horizon electricity demand forecasting, outperforming all other tested approaches.

However, LSTMs come with significant practical trade-offs. They are computationally more intensive to train and deploy than tree-based models, requiring more memory, longer training times, and often GPU acceleration. For stability prediction in deep learning models, the evaluation time can be considerable, making them potentially unsuitable for deployment in highly constrained, near-real-time operational forecasting scenarios (Ashtar et al., 2025). Furthermore, the study by Dey et al. (2025) found that LSTM's smooth activation functions (tanh and sigmoid) tend to underestimate extreme fluctuations in highly volatile time series such as inertia costs, leading to relatively poorer performance compared to ensemble methods on such signals.

### 2.4.5 Hybrid and Ensemble Modelling Approaches

The literature increasingly points toward hybrid and ensemble models as a strategy for maximising forecasting performance by combining the complementary strengths of different paradigms. Ashtar et al. (2025) demonstrate that a SARIMAX-LSTM hybrid model can outperform standalone statistical models by combining the linear trend capture capabilities of SARIMAX with the non-linear residual modelling capabilities of LSTM. In this approach, SARIMAX first captures the deterministic, seasonal component of the signal, and the LSTM is subsequently trained on the residuals to capture the remaining complex, non-linear patterns.

Additionally, Sequence-to-Sequence (Seq2Seq) deep learning architectures have achieved the highest accuracy for long-horizon forecasting in specific energy demand studies, suggesting that attention-based mechanisms may offer further improvements for complex temporal modelling (Ashtar et al., 2025). The success of these hybrid approaches supports the methodological decision in this project to benchmark multiple modelling paradigms rather than committing exclusively to a single approach.

### 2.4.6 Comparative Summary of Model Performance

The following table synthesises the comparative performance findings from the reviewed literature:

| Model | Best Use Case | Key Performance Findings |
| :--- | :--- | :--- |
| SARIMAX | Linear and seasonal trends | Reliable and interpretable baseline; outperformed by AI models in long-term non-linear forecasting (Ashtar et al., 2025) |
| LSTM | Complex sequential time-series | Achieved highest accuracy (1.88% MAPE) for long-horizon demand forecasting; computationally intensive (Ashtar et al., 2025) |
| LightGBM | High-frequency tabular data | Achieved lowest RMSE for inertia cost forecasting (Dey et al., 2025); reduced RMSE by up to 10.4% for frequency prediction (Zhou et al., 2025) |
| Hybrid (SARIMAX-LSTM) | Combined linear + non-linear | Outperformed standalone models by capturing complementary patterns (Ashtar et al., 2025) |

When evaluating specifically for the requirements of a real-time grid stability alert system, the literature strongly suggests that LightGBM offers the optimal trade-off between predictive accuracy and the low latency required for operational deployment (Dey et al., 2025; Zhou et al., 2025).

### 2.4.7 Data Integration and Feature Engineering

The integration of diverse, multi-resolution datasets is a critical prerequisite for effective grid stability forecasting. Grid stability is not determined by any single variable but by the complex interplay of generation patterns, weather conditions, demand profiles, and market dynamics. Studies on the Nigerian power system by Ohiri et al. (2025) emphasise that integrating diverse datasets—including generation mix, weather data, and load profiles—is necessary to accurately assess the impact of renewable penetration levels on stability indicators such as voltage and frequency fluctuations.

A critical insight from the literature is that raw weather data is insufficient for predicting stability; features must be engineered to expose the underlying physical phenomena that drive grid dynamics. Two categories of feature engineering are particularly relevant:

-   **Ramp Rate Extraction using the Optimized Swinging Door Algorithm (OpSDA):** Sudden changes in renewable output, rather than the absolute output level, are the primary drivers of frequency deviations. To model this phenomenon, the Optimized Swinging Door Algorithm (OpSDA) has been introduced in the wind power forecasting literature. Pandit, Mu and Astolfi (2025) demonstrate the use of the Swinging Door Algorithm to detect wind power ramp events, enabling the extraction of explicit ramp features such as the ramp rate and ramp duration. These features allow a predictive model to anticipate the physical "jolt" imparted to the grid by a sudden change in wind output, rather than merely reacting to the absolute power level—a substantially more informative representation.

-   **Lag Features for Tree-Based Models:** For models like LightGBM, which do not inherently process sequence order or temporal context, the inclusion of explicit lag features is essential. Dey et al. (2025) demonstrate that incorporating time-based features (such as hour of day) and historical lag features (e.g., the frequency value at t−1, t−5, and t−60 seconds) significantly improves forecasting accuracy by explicitly capturing temporal trends and autocorrelation. In their inertia cost forecasting study, seven lag variables were generated to capture temporal persistence, proving indispensable for model performance.

### 2.4.8 Critical Analysis of Forecasting Methodologies

A synthesis of the reviewed literature reveals a distinct and pragmatically important divergence in modelling philosophies for grid stability application. While deep learning architectures such as LSTM and Sequence-to-Sequence models demonstrate superior accuracy in capturing long-term temporal dependencies in certain forecasting scenarios (Ashtar et al., 2025), they often suffer from high computational latency, significant memory requirements, and a lack of transparency—challenges that are particularly acute in real-time operational settings.

In contrast, Gradient Boosting Machines (specifically LightGBM) have emerged in the 2024–2025 literature as the optimal "operational" solution for grid-related forecasting tasks. Zhou et al. (2025) and Dey et al. (2025) independently confirm that LightGBM consistently offers the best trade-off between predictive accuracy (achieving the lowest RMSE for both inertia costs and grid frequency) and inference speed, a characteristic that is critical and non-negotiable for real-time fault detection and alerting systems.

Furthermore, a significant methodological limitation identified across the broader forecasting literature is the predominant reliance on deterministic "point forecasts"—models that predict a single expected value. As emphasised by Saleem et al. (2024), low-inertia grids are inherently stochastic systems where the "tail risk" (the small probability of a catastrophic event such as a blackout) is far more operationally relevant than the average system behaviour. The majority of existing studies evaluate models using standard regression metrics such as RMSE and MAE, which measure average prediction error; however, these metrics fundamentally fail to capture the severity and consequence of missing a rare but critical stability event. This limitation has led to growing advocacy within the research community for the adoption of probabilistic evaluation metrics, such as the Pinball Loss function (Osone and Kodaira, 2025), which evaluates the calibration and reliability of prediction intervals rather than the accuracy of a single point estimate.

## 2.5 Alert Systems, Explainability (XAI), and Validation

Moving from raw model predictions to a deployable decision-support tool requires addressing three additional critical requirements: uncertainty quantification to express the confidence associated with predictions, model interpretability to explain why specific alerts are triggered, and rigorous validation to ensure the system performs reliably under extreme conditions.

### 2.5.1 Probabilistic and Uncertainty-Aware Forecasting

The limitations of deterministic forecasting are particularly acute in the context of managing risks associated with Variable Renewable Energy Sources. A stochastic approach—one that explicitly incorporates and quantifies the uncertainties inherent in RES generation and load demand—is advocated by Saleem et al. (2024) as essential for effective stability assessment.

**Quantile Regression (QR)** provides an elegant and practical framework for achieving this. Rather than predicting the expected (mean) value of the target variable, quantile regression estimates specific percentiles of the target distribution. A recent 2025 study on the UK energy market by Osone and Kodaira (2025) demonstrated that a quantile regression model achieved high predictive accuracy and stability, providing reliable prediction intervals for volatile market metrics. The GridGuardian system implements this approach by training two separate LightGBM models to predict the 10th and 90th percentiles of the frequency 10 seconds into the future, thereby producing an 80% confidence interval.

Critically, the performance of probabilistic forecasts cannot be evaluated using standard metrics such as RMSE. Instead, the **Pinball Loss function** (also known as the quantile loss) must be used, which measures the calibration and sharpness of the predicted probability distributions. This metric penalises both under-prediction and over-prediction asymmetrically depending on the quantile, ensuring that the model's uncertainty estimates are statistically well-calibrated (Osone and Kodaira, 2025). For inner-bound quality assessment, the **Prediction Interval Coverage Probability (PICP)** and **Mean Prediction Interval Width (MPIW)** provide complementary measures of how often actual values fall within the predicted bounds and how wide those bounds are, respectively.

### 2.5.2 Explainable Artificial Intelligence (XAI) in Grid Operations

Black-box models such as LSTM and LightGBM, despite their predictive power, often face resistance from grid operators and regulators due to their fundamental lack of transparency. In safety-critical infrastructure, operators must understand *why* a model is predicting instability before they can be expected to trust and act upon that prediction. This need for a "Right to Explanation" is both an ethical imperative and a practical necessity for the adoption of machine learning in control room environments.

To address this challenge, **SHAP (SHapley Additive exPlanations)** values have been successfully integrated into smart grid stability prediction and analysis. SHAP provides a unified, theoretically grounded measure of feature importance based on concepts from cooperative game theory, allowing the system to explain the contribution of each input feature to any individual prediction (Ucar, 2023). This interpretability is essential for building trust and facilitating the adoption of ML models by grid operators who bear the responsibility for operational decisions.

The power of SHAP analysis in this domain is demonstrated across multiple studies:

-   **Identifying Physical Drivers:** Kruse et al. (2021) utilise SHAP values to identify the key features and risk factors influencing frequency stability. Their analysis identifies solar ramps as critical drivers in explaining patterns in the Rate of Change of Frequency (RoCoF), demonstrating SHAP's ability to surface physically meaningful relationships.

-   **Distinguishing Deterministic from Stochastic Drivers:** Drewnick et al. (2025), in their analysis of grid frequency dynamics, use SHAP to distinguish between deterministic drivers (such as scheduled generation changes that predictably affect frequency drift) and stochastic drivers (such as weather noise that affects frequency diffusion), identifying that total generation and load influence the drift component while calendar features are critical for the diffusion component.

-   **Model Validation:** Dey et al. (2025) analyse their LightGBM models for inertia forecasting using SHAP, confirming that the importance assigned to specific temporal features is consistent with known grid physics. This use of SHAP as a validation tool—confirming that the model is learning realistic and physically plausible relationships rather than exploiting spurious correlations—is particularly valuable in ensuring the scientific integrity of data-driven approaches.

### 2.5.3 Validation and Alert System Performance Metrics

Validating an alert system for critical infrastructure requires a rigorous evaluation framework that extends far beyond simple accuracy metrics. Gayathri and Jena (2025) emphasise that an alert system's performance must be assessed through the lens of its operational consequences. Specifically, the system must be evaluated on two dimensions:

-   **Precision** measures the proportion of triggered alerts that correspond to genuine instability events, thereby quantifying the false alarm rate. Excessive false alarms erode operator trust and lead to "alarm fatigue."
-   **Recall** measures the proportion of genuine instability events that are successfully detected by the system. In the context of grid stability, where failing to detect an impending blackout (a False Negative) represents a potentially catastrophic failure, **Recall is the paramount metric.**

The ultimate validation for an operational alert system is the stress test against known extreme events. Validating against historical crises, such as the 9 August 2019 blackout, serves as the definitive test of the system's ability to predict the "tail events" that matter most (Gayathri and Jena, 2025). The system's success is defined not only by its aggregate statistical performance but also by its ability to provide advance warning for specific, known, catastrophic failure modes.

### 2.5.4 Developing Decision Support Systems and Dashboards

The literature supports the use of interactive dashboards as the primary mechanism for converting predictive intelligence into actionable operational information. Dey et al. (2025) note that machine learning outputs, such as forecasted inertia costs and stability predictions, can be directly used for strategic planning and market participation, implying the need for interfaces that present these forecasts in an accessible and time-critical manner to trading and control room teams.

Dashboards that integrate probabilistic forecasting enable decision-makers to weigh the potential costs and benefits of different actions under varying scenarios, such as determining the optimal unit commitment strategy or energy storage dispatch schedule. Effectively presenting metrics such as "Time to Alert" and instability probabilities allows for the proactive deployment of reserves or demand response services before the system reaches its critical limits (Ucar, 2023). The GridGuardian dashboard, built using Streamlit and Plotly, is designed to fulfil this role by providing a real-time "control room" interface with integrated predictions, uncertainty bands, and SHAP-based explanations.

## 2.6 The Research Gap and Conclusion

### 2.6.1 Identifying the Research Gap

Despite the extensive and growing body of work on individual aspects of grid stability, forecasting, and explainability, a critical gap remains in the integration of these disparate methodologies into a unified, cohesive decision-support framework specifically tailored for the UK context. Existing studies typically focus on isolated components of the problem:

-   Amamra (2025) provides a rigorous quantification of the need for synthetic inertia in the UK grid but does not build a predictive alert system from these findings.
-   Dey et al. (2025) successfully forecast inertia costs using machine learning but do not directly predict physical stability metrics such as RoCoF or frequency deviation.
-   Zhou et al. (2025) develop an effective LightGBM framework for frequency prediction but do not integrate the explainability and probabilistic uncertainty quantification required for operator trust and risk-aware decision-making.
-   Pandit, Mu and Astolfi (2025) demonstrate the value of the Swinging Door Algorithm for wind ramp detection but do not integrate this technique into a broader stability prediction pipeline.

Critically, there is currently no unified framework in the literature that:

1.  Combines physics-informed feature engineering (specifically, OpSDA-based wind ramp detection) with market-based inertia proxies to predict grid instability.
2.  Benchmarks LightGBM against LSTM specifically for the UK's islanded grid constraints, characterised by low inertia and high wind penetration.
3.  Provides probabilistic "Time to Alert" warnings, validated by SHAP-based explainability, to ensure that the predictive "black box" is interpretable and trustworthy for control room operators.

### 2.6.2 Conclusion

This literature review establishes that the decarbonisation of the Great Britain electricity grid has precipitated a fundamental "Inertia Crisis," rendering the system increasingly susceptible to rapid Rate of Change of Frequency (RoCoF) events and deep frequency nadirs (Amamra, 2025; Hong et al., 2021). The 9 August 2019 blackout serves as a definitive case study, highlighting the inadequacy of traditional protection mechanisms in a grid increasingly dominated by asynchronous, inverter-based renewable generation.

The review of forecasting methodologies reveals a clear paradigm shift from linear statistical models (SARIMAX) to advanced machine learning techniques. While LSTM networks excel in capturing long-term sequential dependencies (Ashtar et al., 2025), the literature consistently suggests that LightGBM offers a superior balance of predictive accuracy and computational efficiency for high-frequency grid data (Dey et al., 2025; Zhou et al., 2025), making it the most viable candidate for real-time alerting. However, accuracy alone is insufficient; the stochastic nature of wind and solar generation necessitates a fundamental shift from deterministic forecasting to probabilistic uncertainty quantification using Quantile Regression (Osone and Kodaira, 2025).

Furthermore, the "black box" nature of advanced machine learning models remains a significant barrier to their adoption in safety-critical control room environments. The integration of Explainable AI (XAI) through SHAP values has been identified as a mandatory requirement to provide operators with the "why" behind every alert, ensuring transparency, trust, and accountability (Ucar, 2023; Kruse et al., 2021; Drewnick et al., 2025).

Consequently, this project—GridGuardian—aims to fill the identified research gap by developing a holistic, end-to-end Data-Driven Alert System. By integrating physics-informed feature engineering (OpSDA for wind ramp detection), benchmarking LightGBM against LSTM on the GB grid, employing quantile regression for probabilistic forecasting, and utilising SHAP for model interpretability, this research provides a robust, transparent, and risk-aware tool to forecast and mitigate grid instability in the UK's evolving low-carbon energy landscape.