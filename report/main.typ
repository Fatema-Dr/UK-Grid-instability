#import "template.typ": project, appendix

#show: project.with(
  title: "GridGuardian: Physics-Informed Machine Learning for Grid Instability Early-Warning Systems",
  author: "Fatema Doctor",
  student_id: "2604383",
  degree: "Data Science and Artificial Intelligence",
  supervisor: "Ms. Dhara Parekh",
  date: datetime.today(),
  abstract: [
    Declining system inertia from inverter-based renewables fundamentally alters UK grid dynamics, compressing the window for corrective intervention to sub-second scales. This dissertation presents GridGuardian, a functional early-warning prototype for the probabilistic forecasting of grid instability. The system integrates high-resolution LightGBM quantile regression with physics-based heuristics, including the Swinging Door Algorithm (OpSDA) for wind ramp extraction, inertia proxies, and post-hoc isotonic recalibration. This hybrid architecture seeks to address the limitations of traditional 'black-box' machine learning in safety-critical environments. Evaluation was conducted using a strict 1-second telemetry dataset from the National Energy System Operator (NESO), focused on the catastrophic August 2019 blackout event. Empirical results demonstrate a highly lightweight and operationally promising engine, with the tree-based architecture achieving sub-second inference latency ($<0.20$s) and a 99.2% recall rate on the hold-out set, significantly outperforming a deeper LSTM baseline. However, forensic stress-testing revealed critical performance ceilings: the model achieved an in-distribution Prediction Interval Coverage Probability (PICP) of 73.5%, falling short of the statutory 80% safety target, and exhibited reactive rather than predictive behavior during the final frequency nadir. Furthermore, seasonal analysis identifies a necessity for dynamic re-calibration to maintain reliability across shifting grid topologies. This research concludes that while a physics-informed ensemble offers a computationally viable and interpretable framework for grid monitoring, its operational deployment remains conditional upon meeting rigorous calibration and stability benchmarks.
  ],
  acknowledgments: [
    I would like to express my sincere gratitude to my supervisor, Ms. Dhara Parekh, for her guidance, feedback, and support throughout the research and development of this dissertation.
    
    I am grateful to the University of East London, School of Architecture, Computing and Engineering, for providing the academic environment and resources that made this work possible.
    
    This research would not have been possible without access to publicly available datasets provided by the National Energy System Operator (NESO) via the CKAN open data platform, and meteorological data from the Open-Meteo API. I acknowledge these organisations for their commitment to open data in the energy sector.
  ]
)

#include "chapters/01_introduction.typ"
#include "chapters/02_lit_review.typ"
#include "chapters/03_methodology.typ"
#include "chapters/04_implementation.typ"
#include "chapters/05_evaluation.typ"
#include "chapters/06_conclusion.typ"

#bibliography("works.bib", style: "apa")
