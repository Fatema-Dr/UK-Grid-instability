# Chapter 2: Literature Review

## 2.1 Introduction

This chapter provides a comprehensive review of the literature relevant to the development of GridGuardian. The review is structured to establish the theoretical foundation for the research, covering four interconnected domains: (1) power grid stability and the inertia crisis; (2) machine learning approaches for power system forecasting; (3) quantile regression for uncertainty quantification; and (4) explainable AI in critical infrastructure.

The chapter begins with an examination of the physical principles underlying power grid stability, focusing on the concept of system inertia and its role in frequency regulation. It then reviews the UK's specific challenges, including the August 9, 2019 blackout, which serves as a critical case study throughout this dissertation. The literature on machine learning for power systems is surveyed, with particular attention to gradient boosting machines, deep learning approaches, and their comparative performance. The chapter then explores quantile regression as a methodology for probabilistic forecasting, followed by a review of explainable AI techniques, with emphasis on SHAP. Finally, the chapter identifies the research gap that GridGuardian addresses.

## 2.2 The Inertia Crisis: Technical Background

### 2.2.1 System Inertia in Power Systems

System inertia is a fundamental physical property of synchronous power systems. It refers to the kinetic energy stored in the rotating masses of synchronous generators, motors, and other rotating equipment connected to the grid. This stored energy provides an immediate, automatic response to imbalances between generation and demand, acting as a "shock absorber" that resists rapid frequency changes (Kundur, 1994).

The relationship between system inertia and frequency dynamics is described by the swing equation:

$$2H \frac{df}{dt} = P_m - P_e - D(f - f_0)$$

Where:
- $H$ is the system inertia constant (seconds)
- $f$ is the system frequency (Hz)
- $P_m$ is mechanical power input (per unit)
- $P_e$ is electrical power output (per unit)
- $D$ is the damping coefficient
- $f_0$ is the nominal frequency (50 Hz in GB)

From this equation, it follows that the Rate of Change of Frequency (RoCoF) following a power imbalance $\Delta P$ is inversely proportional to system inertia:

$$\text{RoCoF} = \frac{df}{dt} \approx \frac{\Delta P}{2H}$$

This relationship demonstrates that lower inertia results in faster frequency changes for a given power imbalance, reducing the time available for control systems to respond (Milano et al., 2018).

### 2.2.2 The Transition to Low-Inertia Systems

The global transition toward renewable energy has fundamentally altered the inertia characteristics of power systems worldwide. Traditional synchronous generators—coal, gas, nuclear, and hydro—contribute mechanical inertia through their rotating turbines. In contrast, most renewable energy sources connect to the grid through power electronic inverters and do not inherently contribute mechanical inertia (Bevrani et al., 2014).

Wind turbines, particularly those using full-scale power converters (Type 3 and Type 4), are decoupled from the grid frequency and do not provide natural inertia. While some modern wind turbines can be configured to provide "synthetic inertia" through control algorithms that temporarily increase power output in response to frequency drops, this capability is not universally deployed and requires active control (Ulbig et al., 2014).

Solar photovoltaic systems, which have no rotating components, provide no inherent inertia. Like wind turbines, they can be equipped with grid-forming inverters that mimic inertial response, but such configurations remain relatively rare (Rocabert et al., 2012).

The implications of reduced inertia have been extensively documented. Ulbig et al. (2014) demonstrated that low-inertia systems exhibit:
1. Higher RoCoF following disturbances
2. Lower frequency nadirs (minimum frequency reached)
3. Increased susceptibility to cascading failures
4. Reduced stability margins

### 2.2.3 Rate of Change of Frequency (RoCoF)

RoCoF has emerged as a critical metric for assessing grid stability in low-inertia systems. In high-inertia systems, RoCoF following a typical disturbance might be 0.1-0.2 Hz/s. In low-inertia systems, RoCoF can exceed 1 Hz/s, triggering protective relays designed to disconnect generators or load (Johns et al., 2018).

The GB Grid Code specifies RoCoF limits to ensure system security. Following the August 2019 blackout, National Grid ESO introduced enhanced RoCoF monitoring and response requirements (National Grid ESO, 2020a). Generators are now required to withstand RoCoF events up to 1 Hz/s without tripping, and new fast frequency response (FFR) services have been procured to arrest frequency declines within 1-2 seconds.

Research by Tang et al. (2020) demonstrated that RoCoF-based protection schemes must be carefully coordinated to avoid unnecessary tripping during low-inertia conditions. Overly sensitive RoCoF relays can exacerbate frequency declines by disconnecting generation precisely when it is most needed.

### 2.2.4 Frequency Nadir and Overshoot

The frequency nadir—the minimum frequency reached following a disturbance—is another critical stability metric. In high-inertia systems, the nadir typically occurs several seconds after the disturbance, allowing time for governor response. In low-inertia systems, the nadir can occur within 1-2 seconds, before conventional control systems can respond (Koch et al., 2019).

The relationship between inertia, RoCoF, and frequency nadir is complex. While higher inertia reduces RoCoF, it does not necessarily improve the nadir if the total energy imbalance is large. The nadir depends on:
1. The magnitude of the power imbalance
2. The system inertia
3. The speed and magnitude of primary frequency response
4. The characteristics of load-frequency dependence

Strbac et al. (2020) developed analytical models to predict frequency nadir in low-inertia systems, demonstrating that conventional primary response mechanisms may be insufficient to prevent nadir breaches during large disturbances.

### 2.2.5 Market-Based Inertia Proxies

As inertia has become a critical constraint, market mechanisms have emerged to value and procure inertia services. National Grid ESO's Stability Pathfinder programme, launched in 2019, has awarded over £650 million in contracts for synthetic inertia and other stability services (National Grid ESO, 2022).

Inertia cost data, published by NESO, provides a market-based proxy for the scarcity of inertia at different times. High inertia costs indicate periods of low system inertia and elevated stability risk. While inertia costs are financial metrics rather than physical measurements, they correlate with the underlying physical conditions that affect grid stability.

Research by O'Malley et al. (2021) demonstrated that market-based inertia proxies can be useful indicators for stability risk assessment, particularly when direct inertia measurements are unavailable. However, the relationship between inertia costs and physical stability metrics is complex and influenced by market dynamics beyond pure physics.

---

## 2.3 UK Grid Stability Challenges

### 2.3.1 Historical Context of UK Grid Stability

The UK power grid, one of the oldest and most interconnected in the world, was originally designed in the 1930s with synchronous generation as the dominant technology. The National Grid was established in 1935, with a design philosophy centred on large, centralised power stations connected via high-voltage transmission lines (Hughes, 2004).

For decades, this architecture provided robust stability characteristics. The abundance of synchronous generators ensured high system inertia, and the relatively predictable nature of fossil-fuel generation facilitated effective planning and operation. Frequency stability was rarely a concern, with the grid maintaining frequency within ±0.1 Hz of nominal 99.9% of the time (Ofgem, 2018).

The liberalisation of the UK electricity market in the 1990s introduced competition but did not fundamentally alter the physical characteristics of the grid. Coal and gas remained dominant, and nuclear provided a stable baseload. The inertia crisis began to emerge only in the 2010s, as renewable energy penetration increased and coal generation declined.

### 2.3.2 Major Power Outages: Lessons Learned

Several significant power outages have occurred in the UK over the past two decades, each providing lessons for grid stability management.

**Table 2.1: Major UK Power Outages 2003-2019**

| Date | Event | Cause | Customers Affected | Duration |
|------|-------|-------|-------------------|----------|
| August 2003 | South London Outage | Equipment failure, incorrect settings | 500,000 | 2 hours |
| May 2008 | National Scale Outage | Multiple generator trips | 500,000 | 30 minutes |
| August 2019 | Great Britain Blackout | Wind farm + gas plant trip | 1.1 million | 45 minutes |

The August 2003 South London outage was caused by equipment failure at a substation combined with incorrect circuit breaker settings. The incident highlighted the importance of robust operational procedures and correctly configured protection systems (Health and Safety Executive, 2004).

The May 2008 national-scale outage resulted from simultaneous failures at a Scottish coal-fired power station and the Sizewell B nuclear reactor. The incident demonstrated the importance of diverse generation and system resilience to multiple concurrent events (Ofgem, 2008).

However, it was the August 2019 blackout that fundamentally changed the understanding of grid stability risks in the UK.

### 2.3.3 The August 9, 2019 Blackout: Detailed Analysis

The August 9, 2019 blackout remains the most significant power system disturbance in UK history. The official investigation report, published by Ofgem (2020), provides a comprehensive account of the events and their causes.

**Timeline of Events:**

- **17:52:45:** A lightning strike on the 400kV transmission network near Wymondley, Hertfordshire, caused a voltage dip.
- **17:52:46:** The voltage dip triggered the disconnection of Hornsea One offshore wind farm (290 MW) due to its protection settings.
- **17:52:47:** Within one second, Little Barford CCGT gas plant (726 MW) tripped due to control system issues unrelated to the lightning strike.
- **17:52:48:** The simultaneous loss of 1,016 MW of generation caused grid frequency to begin declining rapidly.
- **17:52:49:** Frequency dropped below 49.5 Hz, triggering automatic load shedding.
- **17:52:50:** Frequency reached a nadir of 48.8 Hz, the lowest recorded since the 1970s.
- **17:53-18:30:** Gradual restoration of generation and reconnection of load.

**Key Findings:**

1. **Design Standard Exceeded:** The GB system is designed to withstand the loss of the single largest infeed (typically 1,000-1,200 MW). The simultaneous loss of two generation units (1,016 MW total) exceeded this design standard (Ofgem, 2020).

2. **Low System Inertia:** At the time of the event, system inertia was approximately 115,000 MVAs, significantly below historical levels. This low inertia contributed to the rapid frequency decline (National Grid ESO, 2019).

3. **Wind Farm Disconnection:** The disconnection of Hornsea One following a voltage dip was consistent with its grid code requirements at the time. However, this highlighted the vulnerability of wind farms to voltage disturbances (Ofgem, 2020).

4. **Insufficient Fast Response:** The available fast frequency response was insufficient to arrest the frequency decline before the nadir was reached. This led to automatic load shedding to prevent system collapse.

5. **Distribution Generator Disconnection:** An unexpected aspect of the event was the disconnection of distributed generators at the distribution level, which was not fully anticipated by the system operator (Ofgem, 2020).

### 2.3.4 Regulatory and Operational Responses

The August 2019 blackout triggered significant regulatory and operational responses:

**Grid Code Modifications:** Ofgem (2020) recommended, and National Grid ESO implemented, modifications to the Grid Code requiring wind farms to remain connected during voltage dips and to provide frequency response services.

**Enhanced RoCoF Requirements:** New RoCoF requirements were introduced, mandating that generators withstand RoCoF events up to 1 Hz/s without tripping (National Grid ESO, 2020a).

**Stability Pathfinder Programme:** National Grid ESO accelerated its Stability Pathfinder programme, awarding £328 million in Phase 1 (2019) and £323 million in Phase 2 (2022) contracts for synthetic inertia and other stability services (National Grid ESO, 2022).

**Electricity System Restoration Standard:** A new standard was introduced, requiring 100% of national demand to be restored within five days following a major outage (Ofgem, 2020).

### 2.3.5 Ongoing Challenges

Despite these responses, significant challenges remain:

**Increasing Renewable Penetration:** As wind and solar penetration continues to increase, system inertia will further decline. National Grid ESO's Future Energy Scenarios project that inertia could fall to 80,000 MVAs during certain periods by 2030 (National Grid ESO, 2023).

**Distributed Energy Resources:** The proliferation of distributed energy resources (DERs)—rooftop solar, electric vehicles, home batteries—presents new challenges for visibility and control at the distribution level (National Grid ESO, 2021).

**Cybersecurity:** An increasingly digitised and interconnected grid is more vulnerable to cyber-attacks, which could potentially cause or exacerbate stability events (NCSC, 2022).

**Market Design:** Current electricity markets were designed around fossil-fuel generation and may not adequately value the stability services required in a low-inertia system (IEA, 2021).

---

## 2.4 Machine Learning for Power System Forecasting

### 2.4.1 Time-Series Forecasting in Power Systems

Time-series forecasting has long been a critical tool for power system operation and planning. Traditional approaches include statistical methods such as Autoregressive Integrated Moving Average (ARIMA) and its seasonal variant SARIMAX (Box et al., 2015). These methods model time-series data as a linear combination of past values and error terms, with parameters estimated from historical data.

ARIMA-based approaches have been widely applied to load forecasting, price forecasting, and renewable generation forecasting. Hong et al. (2016) demonstrated that SARIMAX can achieve competitive performance for short-term load forecasting, particularly when combined with exogenous variables such as temperature and humidity.

However, ARIMA and SARIMAX have limitations:
1. They assume linear relationships between variables
2. They struggle with non-stationary data
3. They require careful parameter selection
4. They may not capture complex temporal dependencies

These limitations have motivated the exploration of machine learning approaches for power system forecasting.

### 2.4.2 Neural Networks and Deep Learning

Artificial neural networks (ANNs) were among the first machine learning approaches applied to power system forecasting. Early work by Hippert et al. (2001) demonstrated that feedforward neural networks could outperform traditional statistical methods for short-term load forecasting.

Recurrent neural networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, have become the dominant deep learning approach for time-series forecasting. LSTMs address the vanishing gradient problem that plagues standard RNNs, enabling the learning of long-range temporal dependencies (Hochreiter and Schmidhuber, 1997).

**LSTM Architecture:**

An LSTM cell contains three gates:
1. **Forget Gate:** Determines which information to discard from the cell state
2. **Input Gate:** Determines which new information to store
3. **Output Gate:** Determines which information to output

The cell state acts as a "conveyor belt" that carries information across time steps, with gates controlling the flow of information.

LSTMs have been successfully applied to:
- Load forecasting (Zheng et al., 2017)
- Renewable generation forecasting (Wan et al., 2017)
- Price forecasting (Weron, 2014)
- Frequency forecasting (Liu et al., 2020)

Liu et al. (2020) demonstrated that LSTM models can outperform ARIMA for UK grid frequency forecasting, particularly during periods of high volatility. However, LSTMs require substantial data and computational resources, and their "black box" nature can limit operator trust.

### 2.4.3 Gradient Boosting Machines

Gradient Boosting Machines (GBMs) are ensemble learning methods that build predictive models by sequentially adding weak learners (typically decision trees) to correct the errors of previous learners (Friedman, 2001). Each new tree is trained on the residuals of the current model, gradually improving prediction accuracy.

**LightGBM:**

LightGBM, developed by Microsoft, is an efficient implementation of gradient boosting that uses two innovative techniques:
1. **Gradient-based One-Side Sampling (GOSS):** Retains data points with large gradients while randomly sampling those with small gradients
2. **Exclusive Feature Bundling (EFB):** Bundles mutually exclusive features to reduce dimensionality

These techniques enable LightGBM to achieve faster training speeds and lower memory usage compared to traditional GBM implementations, while maintaining or improving accuracy (Ke et al., 2017).

LightGBM has been widely adopted for power system applications:
- Load forecasting (Fan et al., 2020)
- Renewable generation forecasting (Qin et al., 2021)
- Fault detection (Wang et al., 2020)
- Price forecasting (Nowotarski and Weron, 2018)

**Advantages of LightGBM for Power Systems:**

1. **Efficiency:** Fast training and inference, suitable for real-time applications
2. **Accuracy:** Competitive or superior performance to deep learning on tabular data
3. **Feature Importance:** Built-in feature importance metrics for interpretability
4. **Handling Missing Data:** Robust to missing values without imputation
5. **Quantile Regression:** Native support for quantile loss functions

### 2.4.4 Comparative Studies

Several studies have compared different machine learning approaches for power system forecasting:

**Load Forecasting:**
- Tibshirani et al. (2019) compared ARIMA, SVM, Random Forest, and XGBoost for load forecasting, finding that gradient boosting methods achieved the lowest error.
- Kong et al. (2020) demonstrated that LSTM outperformed traditional methods for short-term load forecasting but required more computational resources.

**Renewable Generation Forecasting:**
- Qin et al. (2021) compared LightGBM, XGBoost, and LSTM for wind power forecasting, finding that LightGBM achieved comparable accuracy to LSTM with significantly faster training.
- Zhang et al. (2022) demonstrated that ensemble methods combining multiple models could outperform individual models.

**Frequency Forecasting:**
- Liu et al. (2020) compared ARIMA, LSTM, and Prophet for UK grid frequency forecasting, finding that LSTM achieved the lowest error but ARIMA was more interpretable.
- Chen et al. (2021) demonstrated that gradient boosting methods could effectively predict frequency deviations using exogenous variables.

**Table 2.2: Comparison of Machine Learning Models for Power Systems**

| Model | Strengths | Limitations | Best Use Case |
|-------|-----------|-------------|---------------|
| ARIMA/SARIMAX | Interpretable, fast, well-understood | Linear assumptions, struggles with non-stationarity | Short-term forecasting with stable patterns |
| LSTM | Captures long-range dependencies, handles non-linearity | Data-hungry, computationally intensive, black box | Complex temporal patterns, abundant data |
| LightGBM | Fast, accurate on tabular data, feature importance | May overfit on small datasets | Tabular data with engineered features |
| Random Forest | Robust, handles non-linearity, feature importance | Less accurate than boosting, large models | Baseline models, feature selection |
| XGBoost | High accuracy, regularisation, efficient | Slower than LightGBM on large datasets | High-accuracy requirements |

### 2.4.5 Physics-Informed Machine Learning

A growing body of literature advocates for the integration of domain knowledge into machine learning models, termed "physics-informed machine learning" (Raissi et al., 2019). This approach recognises that purely data-driven models may fail to capture fundamental physical constraints and may generalise poorly to out-of-distribution scenarios.

Physics-informed approaches include:
1. **Feature Engineering:** Incorporating domain-specific features that capture physical relationships
2. **Architecture Design:** Designing model architectures that respect physical constraints
3. **Loss Functions:** Adding physics-based regularisation terms to the loss function
4. **Hybrid Models:** Combining physical models with machine learning components

For power system applications, physics-informed feature engineering has proven particularly effective. Features such as RoCoF, ramp rates, and renewable penetration ratios encode physical understanding of grid dynamics and have been shown to improve prediction accuracy (Tang et al., 2020).

The Optimized Swinging Door Algorithm (OpSDA), used in this dissertation for wind ramp rate detection, represents a physics-informed approach to feature engineering. Originally developed for time-series compression (Kao et al., 2007), OpSDA has been adapted for power systems to detect significant ramp events that may indicate stability risks (Zhao et al., 2020).

---

## 2.5 Quantile Regression for Uncertainty Quantification

### 2.5.1 Limitations of Point Forecasting

Traditional forecasting approaches typically produce point estimates—single predicted values for future observations. While point forecasts are useful for many applications, they do not convey uncertainty, which is critical for risk-based decision making.

In power system operations, uncertainty quantification is essential for:
1. **Risk Assessment:** Understanding the probability of extreme events
2. **Resource Allocation:** Determining appropriate reserve margins
3. **Decision Making:** Supporting operators in making informed choices under uncertainty

Point forecasts can be misleading, particularly during periods of high volatility. A forecast of 50.0 Hz frequency provides no information about the likelihood of breaching the 49.8 Hz threshold.

### 2.5.2 Principles of Quantile Regression

Quantile regression, introduced by Koenker and Bassett (1978), is a statistical technique that estimates conditional quantiles of a response variable. Unlike ordinary least squares regression, which estimates the conditional mean, quantile regression can estimate any quantile $\tau \in (0, 1)$.

The quantile regression estimator minimises the pinball loss (also known as quantile loss):

$$L_\tau(y, \hat{y}) = \begin{cases} \tau(y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-\tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

Where:
- $y$ is the actual value
- $\hat{y}$ is the predicted quantile
- $\tau$ is the target quantile (e.g., 0.1 for the 10th percentile)

The pinball loss is asymmetric, penalising under-prediction and over-prediction differently depending on the target quantile. For the 10th percentile ($\tau = 0.1$), under-prediction is penalised less than over-prediction, encouraging the model to produce conservative (low) estimates.

### 2.5.3 Applications in Energy Forecasting

Quantile regression has been widely applied to energy forecasting:

**Load Forecasting:**
- Cejar et al. (2019) demonstrated that quantile regression can provide probabilistic load forecasts that support risk-based reserve allocation.
- Hong et al. (2016) showed that quantile regression ensembles can outperform point forecasts for probabilistic load forecasting.

**Renewable Generation Forecasting:**
- Wan et al. (2017) applied quantile regression to wind power forecasting, demonstrating improved uncertainty quantification compared to point forecasts.
- Pinson et al. (2013) developed quantile regression models for wind and solar forecasting that are now used operationally in several European countries.

**Price Forecasting:**
- Nowotarski and Weron (2018) demonstrated that quantile regression averages can provide accurate probabilistic electricity price forecasts.
- Unland et al. (2019) showed that quantile regression can capture the heavy-tailed distribution of electricity prices.

**Frequency Forecasting:**
- Liu et al. (2020) applied quantile regression to UK grid frequency forecasting, demonstrating that probabilistic forecasts can support early warning systems.
- Chen et al. (2021) developed quantile regression models for frequency nadir prediction, supporting stability assessment.

### 2.5.4 Evaluation Metrics for Quantile Forecasts

Evaluating quantile forecasts requires specialised metrics:

**Pinball Loss:**
The average pinball loss across all predictions measures the accuracy of quantile forecasts. Lower pinball loss indicates better performance.

**Prediction Interval Coverage Probability (PICP):**
For a prediction interval defined by quantiles $\tau_1$ and $\tau_2$ (e.g., 0.1 and 0.9), the PICP is the fraction of actual values that fall within the interval:

$$\text{PICP} = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(y_i \in [\hat{y}_{\tau_1}, \hat{y}_{\tau_2}])$$

For an 80% prediction interval ($\tau_1 = 0.1, \tau_2 = 0.9$), a well-calibrated model should have PICP ≈ 0.80.

**Mean Prediction Interval Width (MPIW):**
The average width of prediction intervals measures the sharpness of forecasts:

$$\text{MPIW} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_{\tau_2} - \hat{y}_{\tau_1})$$

Narrower intervals are preferred, but only if coverage is maintained.

**Calibration:**
A quantile forecast is calibrated if the fraction of observations below the predicted $\tau$ quantile equals $\tau$. For example, 10% of observations should fall below the predicted 10th percentile. Calibration can be assessed using reliability diagrams (Gneiting and Katzfuss, 2014).

### 2.5.5 LightGBM Quantile Regression

LightGBM provides native support for quantile regression through the 'regression_quantile' objective function. This enables efficient training of quantile models without custom loss functions.

Research by Qin et al. (2021) demonstrated that LightGBM quantile regression can achieve competitive performance for wind power forecasting, with the advantage of faster training compared to deep learning approaches.

For grid frequency forecasting, quantile regression offers several advantages:
1. **Uncertainty Bands:** Provides upper and lower bounds that visualise prediction uncertainty
2. **Risk-Based Alerts:** Enables alerts based on worst-case scenarios (e.g., 10th percentile)
3. **Fail-Safe Bias:** Can be tuned to produce conservative predictions for safety-critical applications

---

## 2.6 Explainable AI in Critical Infrastructure

### 2.6.1 The Black Box Problem

Machine learning models, particularly complex models like deep neural networks and gradient boosting machines, are often described as "black boxes" because their internal workings are not easily interpretable by humans. This lack of interpretability can limit the adoption of ML in safety-critical domains, including power system operations.

Grid operators, responsible for maintaining the reliability and security of the electricity supply, require more than accurate predictions—they need to understand *why* a prediction was made. This understanding supports:
1. **Trust:** Operators are more likely to act on predictions they understand
2. **Validation:** Operators can verify that predictions are based on physically plausible reasoning
3. **Learning:** Understanding model behaviour can improve operator expertise
4. **Accountability:** Decisions can be explained and justified to stakeholders

### 2.6.2 Explainable AI Techniques

Explainable AI (XAI) encompasses a range of techniques for making machine learning models more interpretable:

**Model-Specific Methods:**
- **Linear Models:** Coefficients provide direct interpretation of feature effects
- **Decision Trees:** Rules can be traced from root to leaf
- **Tree Ensembles:** Feature importance based on split frequency and quality

**Model-Agnostic Methods:**
- **Permutation Importance:** Measures feature importance by shuffling each feature
- **Partial Dependence Plots:** Shows the marginal effect of features on predictions
- **LIME (Local Interpretable Model-agnostic Explanations):** Fits local surrogate models (Ribeiro et al., 2016)
- **SHAP (SHapley Additive exPlanations):** Uses game theory to attribute predictions to features (Lundberg and Lee, 2017)

### 2.6.3 SHAP: A Unified Framework

SHAP (SHapley Additive exPlanations), introduced by Lundberg and Lee (2017), provides a unified framework for interpreting machine learning predictions. SHAP values are based on Shapley values from cooperative game theory, which fairly distribute the "payout" (prediction) among the "players" (features).

**Shapley Values:**

For a prediction $f(x)$, the Shapley value $\phi_i$ for feature $i$ is:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

Where:
- $F$ is the set of all features
- $S$ is a subset of features not including $i$
- $f(S)$ is the model prediction using only features in $S$

The Shapley value represents the average marginal contribution of a feature across all possible feature combinations.

**Properties of SHAP:**

1. **Efficiency:** The sum of SHAP values equals the difference between the prediction and the baseline (expected) value
2. **Symmetry:** Features that contribute equally receive equal SHAP values
3. **Dummy:** Features that do not affect predictions receive zero SHAP values
4. **Additivity:** SHAP values are consistent across different model formulations

**SHAP Variants:**

- **TreeExplainer:** Optimised for tree-based models (LightGBM, XGBoost, Random Forest)
- **KernelExplainer:** Model-agnostic, uses weighted sampling
- **DeepExplainer:** Optimised for deep learning models
- **LinearExplainer:** Optimised for linear models

### 2.6.4 XAI in Power Systems

The application of XAI in power systems is an emerging research area:

**Load Forecasting:**
- Wang et al. (2021) used SHAP to explain load forecasting models, identifying key drivers such as temperature, time of day, and day of week.

**Renewable Generation Forecasting:**
- Zhang et al. (2022) applied SHAP to wind power forecasting, demonstrating that SHAP values could identify periods of high uncertainty.

**Fault Detection:**
- Liu et al. (2021) used SHAP to explain fault classification models, improving operator trust in automated diagnostics.

**Grid Stability:**
- Chen et al. (2021) applied SHAP to frequency stability prediction, demonstrating that SHAP values could identify the physical drivers of instability.

### 2.6.5 Real-Time Explainability

Most XAI research focuses on post-hoc analysis, where explanations are generated after model training. However, operational applications require real-time explainability—explanations that are generated alongside predictions during live operation.

Real-time SHAP computation presents challenges:
1. **Computational Cost:** TreeExplainer has complexity $O(n \cdot m \cdot d)$, where $n$ is the number of samples, $m$ is the number of trees, and $d$ is the tree depth
2. **Latency:** Explanations must be generated within the prediction latency budget
3. **Interpretability:** Explanations must be presented in a format that operators can quickly understand

Despite these challenges, real-time XAI is feasible for many applications. The GridGuardian dashboard demonstrates that SHAP values can be computed and visualised in real-time for grid stability prediction.

---

## 2.7 Feature Engineering for Power Systems

### 2.7.1 The Importance of Feature Engineering

Feature engineering—the process of creating informative input features for machine learning models—is critical for achieving high prediction accuracy. Well-designed features can capture domain-specific patterns that raw data may not explicitly contain.

For power system applications, feature engineering is particularly important because:
1. **Physical Understanding:** Features can encode physical relationships that improve generalisation
2. **Data Efficiency:** Good features can reduce the amount of training data required
3. **Interpretability:** Physically meaningful features are easier for operators to understand

### 2.7.2 Rate of Change of Frequency (RoCoF)

RoCoF is a fundamental metric for grid stability, measuring the rate at which frequency changes over time. In this dissertation, RoCoF is calculated as the first difference of frequency:

$$\text{RoCoF}_t = f_t - f_{t-1}$$

However, raw RoCoF calculated from 1-second data can be noisy due to measurement errors and high-frequency oscillations. Smoothing techniques such as moving averages or Savitzky-Golay filters can reduce noise while preserving the underlying signal (Savitzky and Golay, 1964).

Research by Tang et al. (2020) demonstrated that smoothed RoCoF features improve the accuracy of frequency prediction models, particularly during periods of high volatility.

### 2.7.3 Wind Ramp Detection

Wind ramp events—rapid changes in wind power output—can significantly impact grid stability. Detecting and characterising these events is important for forecasting and operational planning.

The Swinging Door Algorithm (OpSDA) is a time-series compression technique that identifies significant changes in data while filtering out noise (Kao et al., 2007). The algorithm works by:
1. Starting with the first data point as a pivot
2. Forming a "door" (angular bounds) from the pivot to the current point
3. Checking if intermediate points fall within the door
4. If a point falls outside, recording the previous point and starting a new door

The ramp rate is then calculated as the slope between compressed points:

$$\text{Ramp Rate} = \frac{\Delta \text{Wind Speed}}{\Delta \text{Time}}$$

Zhao et al. (2020) demonstrated that OpSDA-based ramp rate features improve wind power forecasting accuracy compared to raw wind speed features.

### 2.7.4 Lag Features

Lag features—past values of the target variable—are commonly used in time-series forecasting. For frequency prediction, lag features capture the autoregressive nature of frequency dynamics:

$$\text{lag}_k = f_{t-k}$$

Where $k$ is the lag interval (e.g., 1 second, 5 seconds, 60 seconds).

Lag features are particularly effective for short-term forecasting, where recent history is a strong predictor of future values. However, too many lag features can lead to overfitting and increased computational cost.

### 2.7.5 Temporal Features

Temporal features encode time-related information that may affect grid behaviour:
- **Hour of Day:** Load and generation patterns vary by time of day
- **Day of Week:** Weekend patterns differ from weekday patterns
- **Season:** Seasonal variations in load and renewable generation

These features enable models to capture periodic patterns without requiring explicit modelling of seasonality.

### 2.7.6 Renewable Penetration Proxies

In low-inertia systems, the proportion of renewable generation is a key indicator of stability risk. However, direct measurements of renewable penetration may not be available in real-time.

Synthetic proxies can be constructed from available data:
- **Wind Speed / Load Ratio:** Higher wind speeds relative to load indicate higher wind penetration
- **Solar Radiation / Load Ratio:** Higher solar radiation relative to load indicates higher solar penetration
- **Inertia Cost:** Market-based proxy for inertia scarcity

These proxies provide surrogate measures for physical quantities that are difficult to measure directly.

---

## 2.8 Research Gap Identification

The literature review has established the following key points:

1. **The Inertia Crisis:** The transition to renewable energy has reduced system inertia, increasing the risk of rapid frequency deviations and blackouts. The August 9, 2019 blackout demonstrated the real-world consequences of these dynamics.

2. **Machine Learning for Forecasting:** Machine learning approaches, particularly gradient boosting machines and LSTMs, have demonstrated superior performance compared to traditional statistical methods for power system forecasting. However, most applications focus on point forecasts rather than probabilistic predictions.

3. **Quantile Regression:** Quantile regression provides a principled approach to uncertainty quantification, but its application to grid frequency forecasting remains limited.

4. **Explainable AI:** SHAP and other XAI techniques can improve operator trust in ML predictions, but real-time integration in operational dashboards is rare.

5. **Feature Engineering:** Physics-informed features such as RoCoF and ramp rates improve prediction accuracy, but the integration of domain-specific algorithms like OpSDA for grid stability prediction is novel.

**Identified Research Gaps:**

1. **Integration Gap:** No existing work integrates physics-informed feature engineering, quantile regression, and real-time explainability into a unified framework for grid stability prediction.

2. **Operational Gap:** Most ML research focuses on offline analysis rather than real-time operational deployment. There is a lack of functional prototypes that demonstrate feasibility for control room use.

3. **Validation Gap:** Limited validation against historical blackout events makes it difficult to assess the practical utility of proposed methods.

4. **Granularity Gap:** Most studies use aggregated data (e.g., 5-minute or 15-minute resolution) rather than high-resolution (1-second) data that captures rapid frequency dynamics.

**GridGuardian Contribution:**

This dissertation addresses these gaps by:
1. Integrating OpSDA-based wind ramp detection with quantile regression for probabilistic frequency forecasting
2. Implementing real-time SHAP explainability in a Streamlit dashboard
3. Validating against the August 9, 2019 blackout event
4. Using 1-second resolution frequency data for high-fidelity modelling

---

## 2.9 Chapter Summary

This chapter has reviewed the literature relevant to the development of GridGuardian. The key findings are:

1. **System Inertia:** Inertia is a fundamental property that resists frequency changes. The transition to renewable energy has reduced system inertia, increasing the risk of rapid frequency deviations.

2. **UK Grid Challenges:** The August 9, 2019 blackout demonstrated the vulnerabilities of low-inertia systems and triggered significant regulatory and operational responses.

3. **Machine Learning:** Gradient boosting machines (particularly LightGBM) and LSTMs have demonstrated superior performance for power system forecasting compared to traditional statistical methods.

4. **Quantile Regression:** Quantile regression provides principled uncertainty quantification, enabling risk-based decision making. However, its application to grid frequency forecasting remains limited.

5. **Explainable AI:** SHAP provides a unified framework for interpreting ML predictions. Real-time integration in operational dashboards is feasible but rare.

6. **Feature Engineering:** Physics-informed features such as RoCoF, ramp rates, and renewable penetration proxies improve prediction accuracy and operator understanding.

7. **Research Gaps:** No existing work integrates physics-informed feature engineering, quantile regression, and real-time explainability into a unified framework for operational grid stability monitoring.

The next chapter describes the methodology used in this research, including the research design philosophy, software development lifecycle, data sources, feature engineering strategy, model selection rationale, and evaluation metrics.

---

