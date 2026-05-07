#import "template.typ": appendix, project

#show: project.with(
  title: "GridGuardian: Evaluating Physics-Informed Machine Learning for Grid Instability Early-Warning Systems",
  author: "Fatema Doctor",
  student_id: "2604383",
  degree: "Data Science and Artificial Intelligence",
  supervisor: "Ms. Dhara Parekh",
  date: datetime.today(),
  abstract: [
    Declining system inertia from inverter-based renewables alters UK grid dynamics, compressing the window for corrective intervention to sub-second scales. This dissertation evaluates GridGuardian, a prototype for the probabilistic forecasting of grid instability. The system integrates LightGBM quantile regression with physics-based heuristics, including the Swinging Door Algorithm (OpSDA) for wind ramp extraction, inertia proxies, and post-hoc isotonic recalibration. This hybrid architecture addresses the limitations of traditional machine learning in safety-critical environments. Evaluation was conducted using a 1-second telemetry dataset from the National Energy System Operator (NESO), focused on the August 2019 blackout event. While the tree-based architecture achieved sub-second inference latency ($<0.20$s) and a high recall rate on the hold-out set, forensic stress-testing revealed significant performance ceilings. The model achieved a Prediction Interval Coverage Probability (PICP) of 73.5%, falling short of the nominal 80% safety target, and exhibited reactive rather than predictive behavior during the final frequency nadir. Seasonal analysis further identifies a need for dynamic re-calibration to maintain reliability across shifting grid topologies. This work establishes the performance ceiling of physics-informed tree ensembles for sub-second grid forecasting and identifies calibration reliability as the primary barrier to operational deployment.
  ],
  acknowledgments: [
    I would like to express my sincere gratitude to my supervisor, Ms. Dhara Parekh, for her invaluable guidance, continuous feedback, and patience throughout the research and development of this dissertation.

    I am also grateful to the University of East London and the open-data providers, specifically the National Energy System Operator (NESO) and Open-Meteo, without whose resources this project would not have been possible.

    Finally, my deepest appreciation goes to my parents, family and friends. Thank you for your endless encouragement, understanding, and unwavering support during the many long hours spent working on this project. I could not have reached this milestone without you.
  ],
)

#include "chapters/01_introduction.typ"
#include "chapters/02_lit_review.typ"
#include "chapters/03_methodology.typ"
#include "chapters/04_implementation.typ"
#include "chapters/05_evaluation.typ"
#include "chapters/06_conclusion.typ"

#bibliography("works.bib", style: "apa")

#include "chapters/Appendix.typ"
