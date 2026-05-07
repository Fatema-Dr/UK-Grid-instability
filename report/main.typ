#import "template.typ": project, appendix

#show: project.with(
  title: "GridGuardian: Hardware-Accelerated Physics-Informed Machine Learning for Grid Instability Early-Warning Systems",
  author: "Fatema Doctor",
  student_id: "2604383",
  degree: "BSc (Hons) in Data Science and Artificial Intelligence",
  supervisor: "Ms. Dhara Parekh",
  date: datetime.today(),
  abstract: [
    Declining system inertia caused by the displacement of synchronous generation with inverter-based renewables has fundamentally altered UK power grid dynamics, reducing the time available for corrective action from minutes to seconds. This dissertation presents GridGuardian, a hybrid early-warning system designed to predict and detect power grid instability. By integrating high-resolution, hardware-accelerated LightGBM forecasting models with physics-based heuristics (such as the Rate of Change of Frequency, RoCoF), the system overcomes the limitations of pure machine learning approaches in predicting discontinuous events like lightning strikes. Evaluated against the August 2019 UK blackout, the system successfully demonstrates a 1-second predictive warning threshold. This aligns perfectly with the deployment of grid-scale Enhanced Frequency Response (EFR) battery systems, maximizing true positive recall while minimizing false alerts.
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
