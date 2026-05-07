= Literature Review

== Introduction

The structural transition of electricity grids toward inverter-based renewable energy sources presents a fundamental challenge to frequency stability. The displacement of synchronous generation reduces system inertia, fundamentally modifying the physical dynamics of frequency response and severely compressing the time available for operator intervention. This review systematically synthesises current research concerning grid stability in low-inertia environments. It evaluates the comparative efficacy of machine learning forecasting architectures and the critical integration of physical constraints into data-driven models. The chapter examines the specific operational requirements for predictive alerting systems and the absolute necessity of algorithmic interpretability in safety-critical grid management.

== The Inertial Challenge and Physical Dynamics

=== The Historical Evolution of Frequency Control

Historically, power system frequency stability was guaranteed by the inherent physical properties of massive synchronous generators (coal, gas, and nuclear plants). The kinetic energy stored in the rotating turbines provided a natural, instantaneous buffer against supply-demand imbalances, a phenomenon formally known as "synchronous inertia." Under this traditional paradigm, frequency deviations were relatively slow, allowing grid operators to rely on "droop control"—a primary frequency response mechanism where turbine governors mechanically adjusted steam valves in proportion to frequency deviations. The response times required were on the order of tens of seconds, making frequency control a manageable, predominantly mechanical problem.

However, the rapid decarbonisation of the energy sector has rendered this paradigm obsolete. As modern grids transition to renewable energy sources, synchronous mass is actively being displaced.

=== The Swing Equation and Inverter-Based Resources

Power system frequency dynamics are governed by the swing equation, which mathematically formalises the relationship between kinetic energy in rotating masses and active power imbalances @kundur1994power. The fundamental equation dictates that the rate of change of frequency ($"df"/"dt"$) following a disturbance is inversely proportional to the system's synchronous inertia ($H$). Therefore, reductions in inertia directly produce steeper frequency trajectories during generation losses.

@milano2018foundations established that modern networks exhibit increasingly volatile frequency deviations as decentralised, Inverter-Based Resources (IBRs) replace centralised thermal plants. Because IBRs (such as wind and solar photovoltaics) connect to the grid via power electronics rather than direct electromechanical coupling, they inherently provide zero natural synchronous inertia. @tielens2016relevance demonstrated that operating below critical inertia thresholds produces Rate of Change of Frequency (RoCoF) values exceeding standard protection limits. This physical constraint dictates that contemporary grid monitoring systems must evaluate continuous frequency derivatives rather than static frequency thresholds alone.

=== Empirical Evidence: The August 2019 Blackout

The August 9, 2019 UK blackout provides a definitive empirical case study for low-inertia vulnerability. The simultaneous disconnection of the Little Barford gas plant and the Hornsea One wind farm removed 1,481 MW of generation, causing the grid frequency to decline to an unprecedented 48.8 Hz within 10 seconds @homan2020august.

The subsequent technical investigation by the Office of Gas and Electricity Markets @ofgem2019report identified a critical misalignment between existing protection configurations and actual system inertia. Generators were calibrated for historical inertia margins that were mathematically invalid for a wind-dominated grid profile. This catastrophic event demonstrated the inadequacy of reactive, threshold-based protection mechanisms when system momentum is insufficient to arrest frequency decay prior to automated under-frequency load shedding.

=== Frequency Response Mechanisms

To mitigate declining physical inertia, grid operators must procure synthetic stability services. The National Energy System Operator's operability strategy @nationalgrid2020 categorises these interventions strictly by response velocity. Traditional Firm Frequency Response (FFR) requires up to 10 seconds to reach maximum active power output. In contrast, Enhanced Frequency Response (EFR), typically provided by grid-scale Battery Energy Storage Systems (BESS), must deliver full output within 1 second. The operational utility of any predictive alerting system is strictly defined by these mechanical deployment constraints; a forecasting horizon must perfectly align with the activation parameters of the available response technology.

== Machine Learning Methodologies for Power Systems

As the physical complexity of the grid outpaces the capabilities of traditional analytical solvers, data-driven approaches have gained prominence for dynamic security assessment @zhao2019machine.

=== Predictive Horizons in Power Systems

A critical distinction in power system forecasting literature is the temporal predictive horizon. The vast majority of existing ML applications focus on "day-ahead" or "hour-ahead" unit commitment forecasting, utilizing autoregressive integrated moving average (ARIMA) models or deep learning to predict overall load curves for energy market trading. These models operate in minutes or hours and prioritize average long-term accuracy.

Conversely, "sub-second" transient stability forecasting addresses a fundamentally different problem: detecting structural collapses instantaneously. Models operating in this domain face extreme latency constraints and must prioritize worst-case boundary predictions over average expected values. The literature reveals a distinct gap where advanced ML techniques successfully deployed for day-ahead forecasting fail dramatically when constrained by sub-second latency budgets.

#figure(
  table(
    columns: (1.5fr, 1.5fr, 2fr, 2fr),
    align: left,
    [*Horizon*], [*Latency Budget*], [*Typical Architectures*], [*Primary Application*],
    [Day-Ahead], [Hours], [ARIMA, Transformers], [Unit Commitment \& Trading],
    [Intra-Hour], [Minutes], [Deep Learning (LSTM)], [Economic Dispatch],
    [Sub-Second], [$< 1$ Second], [Tree Ensembles (LightGBM)], [Transient Stability \& EFR]
  ),
  caption: [Comparative summary of forecasting horizons and corresponding computational constraints in power systems.]
)

=== Sequential Modeling and Deep Learning

Recurrent architectures, specifically Long Short-Term Memory (LSTM) networks, have dominated sequential forecasting in energy systems due to their capacity to capture complex temporal dependencies @goodfellow2016deep. However, while LSTMs excel at mapping high-dimensional inputs to point-predictions, they present significant operational challenges.

The primary limitation of deep learning approaches in active grid operations is not predictive accuracy, but rather the computational latency and structural opacity of the models. @zhao2019machine highlighted that while neural networks can achieve minimal Mean Absolute Errors (MAE) on historical datasets, the dense matrix multiplications required for sub-second inference often exceed the latency budgets of real-time control room environments. Furthermore, LSTMs require the network to independently infer underlying physical laws purely from statistical distributions, risking catastrophic failure during unprecedented "black swan" structural events.

=== Gradient Boosting and Tree-Based Ensembles

While deep neural networks suffer from high inference latency, tree-based ensemble methods provide a computationally efficient alternative for tabular forecasting. However, not all tree architectures are suitable for sub-second control room deployment.

Random Forest (RF) constructs multiple decision trees independently using bootstrap aggregation (bagging). While highly robust against overfitting, RF requires deep trees to capture complex non-linearities. During inference, traversing hundreds of deep, independent trees simultaneously introduces unacceptable computational bottlenecks. Furthermore, standard RF cannot natively optimize for asymmetric loss functions like quantile bounds without significant post-hoc structural modifications.

Extreme Gradient Boosting (XGBoost) resolves the depth issue by building trees sequentially, minimizing residual errors. However, XGBoost utilizes a level-wise (depth-first) tree growth strategy. When processing the 2.6 million high-resolution data points required for continuous grid monitoring, XGBoost's exact greedy algorithm for finding optimal split points becomes memory-bound.

The LightGBM algorithm @ke2017lightgbm resolves these specific bottlenecks, making it uniquely suited for the strict 1-second inference constraints of EFR battery deployment. LightGBM utilizes histogram-based decision tree construction, binning continuous variables into discrete buckets. This reduces memory allocation by over $80%$ compared to exact greedy algorithms. Critically, LightGBM employs a leaf-wise (best-first) growth strategy, expanding the leaf with the maximum delta loss rather than growing level-by-level. This produces deeper, more asymmetrical trees that achieve lower error rates with significantly fewer splits. Furthermore, Gradient-based One-Side Sampling (GOSS) allows LightGBM to selectively drop data instances with small gradients during training, maintaining predictive accuracy while drastically reducing the computational burden.

@qiu2020ensemble demonstrated the efficacy of ensemble machine learning for frequency response prediction, noting that tree-based ensembles excel when provided with carefully engineered, domain-specific features. Because the histogram-based tree traversal of LightGBM is computationally trivial compared to backpropagation networks or deep Random Forests, it emerges as the optimal architecture for real-time edge deployment.

=== Probabilistic Forecasting via Quantile Regression

Operational grid management necessitates uncertainty quantification. Grid dispatchers must evaluate the probability of a fault against the economic cost of intervention. Conventional implementations of gradient boosting default to point-estimation (minimizing Mean Squared Error), predicting expected values without corresponding probability distributions.

Quantile regression @koenker1978regression resolves this by computing conditional quantiles, yielding explicit probability intervals. By minimizing the asymmetric Pinball Loss function, models penalize underestimations and overestimations differently @hastie2009elements. The integration of quantile regression into efficient tree-based algorithms presents an optimal mechanism for generating real-time, statistically bounded stability forecasts, moving beyond deterministic predictions to true risk quantification.

== Feature Engineering and Physics-Informed Machine Learning (PIML)

=== Automated vs. Domain-Specific Feature Extraction

In modern data science, automated feature extraction libraries, such as `tsfresh` @christ2018time, have gained immense popularity. These algorithms automatically generate thousands of statistical features (e.g., Fourier coefficients, wavelet transforms, kurtosis) from raw time-series data, allowing models to detect hidden patterns without requiring domain expertise. 

While statistically powerful, automated extraction is entirely detached from physical reality. A grid operator cannot act upon an alert triggered by a "shift in the third-order wavelet coefficient." In contrast, domain-specific feature engineering deliberately constructs variables that represent known physical phenomena. Transforming raw frequency into Rate of Change of Frequency (RoCoF) directly encodes the grid's acceleration—a metric instantly actionable and legally codified in grid operation standards.

=== Physics-Informed Neural Networks (PINNs)

Standard machine learning models treat power grids as pure statistical distributions, ignoring fundamental mechanical realities. Physics-Informed Machine Learning (PIML) resolves this epistemological flaw by embedding domain principles into the learning architecture @karniadakis2021physics.

Physics-Informed Neural Networks explicitly penalise deviations from differential equations (such as the swing equation) during the loss calculation @raissi2019physics. By adding a physical residual term to the objective function, PINNs ensure that predictions obey the laws of conservation. While PINNs demonstrate excellent generalisation in transient stability assessments, enforcing continuous mathematical constraints during real-time, sub-second inference introduces substantial computational overhead. 

=== Physics-Informed Feature Engineering

An alternative, highly efficient strategy involves physics-informed feature engineering. Rather than constraining the model's internal architecture, the input domain is structurally modified to include explicit physical derivatives. Proxying inertia via renewable penetration ratios allows the model to learn the swing equation indirectly. This approach leverages the sub-second computational efficiency of gradient boosting @ke2017lightgbm while securely grounding the statistical predictions in mechanical reality.

#figure(
  table(
    columns: (1.5fr, 2fr, 2fr),
    align: left,
    [*Methodology*], [*Primary Advantage*], [*Operational Drawback*],
    [Pure Data-Driven (Black Box)], [High capacity for non-linear mapping], [Ignores physical laws; prone to hallucination],
    [Physics-Informed Neural Networks (PINNs)], [Mathematically guarantees physical conservation limits], [High inference latency; struggles with sub-second bounds],
    [Physics-Informed Feature Engineering], [Sub-second inference; utilizes high-speed tree traversal], [Requires expert domain knowledge for feature creation]
  ),
  caption: [Comparative evaluation of physics-informed integration strategies for grid forecasting.]
)

== Explainable AI (XAI) in Safety-Critical Infrastructure

=== The Transparency Requirement

Black-box models face insurmountable adoption barriers in power system operations due to the inability to verify the logic preceding automated interventions. Algorithmic transparency is an explicit, non-negotiable requirement for integrating machine learning systems into live grid control rooms @machlev2022explainable.

=== The Mathematical Axioms of SHAP

SHAP (SHapley Additive exPlanations) resolves the mathematical inconsistencies of traditional heuristic attribution methods (such as simple feature permutation) by employing cooperative game theory to distribute exact feature contributions @lundberg2017unified. In safety-critical contexts, an attribution algorithm must not merely be "interpretable"—it must be mathematically provable. SHAP is the only additive feature attribution method that mathematically guarantees four critical axioms:

1.  *Efficiency:* The feature attributions must sum precisely to the difference between the model's current prediction and the expected baseline prediction. No fractional attribution is lost.
2.  *Symmetry:* If two features contribute equally to all possible coalitions, their SHAP values must be identical.
3.  *Dummy (Null Effect):* If a feature never changes the predicted value regardless of the coalition it joins, its SHAP attribution is guaranteed to be exactly zero.
4.  *Additivity:* For a random forest or gradient boosting ensemble, the SHAP value for a feature across the entire ensemble is exactly equal to the sum of its SHAP values calculated for each individual tree.

These mathematical guarantees distinguish SHAP from heuristic explanation methods like LIME, which approximate local decision boundaries and can suffer from severe local instability. Furthermore, exact tree-based SHAP implementations (TreeSHAP) leverage the internal split structures of LightGBM to compute these values in low-order polynomial time, providing distinct operational latency advantages over the slow, model-agnostic permutation methods required by neural networks.

However, modern XAI evaluation demands an analysis of robustness and adversarial vulnerability. @slack2020fooling demonstrated that post-hoc explanation methods like SHAP can be manipulated or become mathematically unstable when presented with highly correlated out-of-distribution data. When dealing with highly correlated grid features (e.g., wind speed and renewable generation percentage), SHAP values can exhibit temporal instability, distributing attribution arbitrarily between correlated variables. If an attribution algorithm requires multiple seconds to compute, or if its explanations oscillate wildly during a cascading fault, its utility to a grid operator is zero.

== Synthesis and Research Gaps

This critical review identifies three persistent gaps in contemporary grid stability literature:

1. *Absence of Real-Time Probabilistic Bounds:* While deep learning architectures offer strong point-prediction accuracy for day-ahead markets, the sub-second generation of reliable, calibrated quantile bounds remains limited, restricting risk-aware decision making in the control room during transient instability.
2. *Computational Feasibility of Physics Constraints:* PINNs successfully embed physical laws but struggle to meet the strict inference constraints of EFR battery deployment. The efficacy of physics-informed feature engineering coupled with high-speed tree ensembles remains inadequately benchmarked against formal neural architectures.
3. *Adversarial and Discontinuous Event Validation:* The overwhelming majority of forecasting models and their XAI explanations are evaluated on normal, continuous grid operations. Systematic validation of model predictions and SHAP stability against catastrophic, discontinuous faults (such as the August 2019 blackout) is exceedingly rare.

The present research addresses these specific methodological deficiencies by engineering GridGuardian: a hybrid, physics-informed LightGBM architecture designed to generate probabilistic frequency boundaries, explicitly validated against the non-linear dynamics of a major historical blackout.
