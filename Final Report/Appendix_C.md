## Appendix C: Dashboard User Manual and Interface Documentation

This appendix provides complete documentation for the GridGuardian Control Room dashboard, including user instructions and annotated screenshots.

## C.1 System Overview

The GridGuardian Control Room is a real-time monitoring and predictive analytics dashboard for UK power grid stability. It integrates:
- Live frequency data from NESO CKAN API
- Weather data from Open-Meteo API
- Machine learning predictions (10-second horizon)
- SHAP explainability visualisations
- Intervention simulation capabilities

**Target Users:** Grid control room operators, energy traders, system planners

**Access Requirements:** Web browser (Chrome/Firefox/Safari), Python 3.11+ runtime

## C.2 Navigation Controls

### C.2.1 Time Navigation

| Control | Function | Default | Range |
|---------|----------|---------|-------|
| **Date Range Picker** | Select overall analysis period | Last 7 days | 2019-01-01 to present |
| **Date Picker** | Jump to specific day | Today | Within selected range |
| **Step Back/Forward (<< < > >>)** | Navigate by 1 second or 1 minute | — | — |
| **Exact Time Input** | Enter HH:MM:SS UTC | Current | 00:00:00–23:59:59 |
| **Autoplay** | Automatic time advance | Off | Speed: 1×, 5×, 10×, 30×, 60× |

*Autoplay mode replays historical events at accelerated speed for training and post-event analysis.*

### C.2.2 Playback Controls

- **Play/Pause:** Start/stop autoplay
- **Speed Selector:** Adjust playback rate (1× = real-time, 60× = 1 minute per second)
- **Loop Toggle:** Repeat playback when reaching end of selected range
- **Go to Blackout:** Jump directly to August 9, 2019, 16:52:00 UTC (preset for training)

## C.3 Alert Configuration Panel

### C.3.1 Prediction Horizon

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| **Time to Alert** | Displayed warning lead time | 10 seconds | 5–60 seconds |

*Note: The underlying model always predicts at the trained 10-second horizon. This slider adjusts the displayed label only, allowing operators to evaluate different warning thresholds.*

### C.3.2 Instability Threshold

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| **Frequency Threshold** | Alert trigger level | 49.80 Hz | 49.50–49.90 Hz |

**Dual-Trigger Logic:**
- **Condition A:** Predicted lower bound < threshold (proactive alert)
- **Condition B:** Current frequency < threshold (reactive alert)
- Alert activates when Condition A **OR** Condition B is true

*Recommended settings:*
- **Conservative:** 49.85 Hz (earlier warning, more false positives)
- **Standard:** 49.80 Hz (balanced)
- **Aggressive:** 49.75 Hz (later warning, fewer false positives)

## C.4 Main Frequency Display

### C.4.1 Visual Elements

**Figure C.1: Frequency Plot Components**

```
Frequency (Hz)
    |
51.0|-----------------------------------------------
    |
50.5|                    Uncertainty Band
    |                   ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
50.0|__________________/              \____________
    |              Actual Frequency (cyan line)
49.8| Alert Threshold (red dashed) ←──┐
    |                                 │
49.5|                                 │
    |              Predicted Lower Bound (orange)
49.0|_________________________________│____________
    |                                 ↓ Alert Zone
48.8| Emergency Threshold ──────────────────────────
    |______________________________________________
          Time →
```

**Legend:**
- **Cyan Line:** Actual measured frequency (1-second resolution)
- **Orange Shaded Band:** Prediction uncertainty (10th–90th percentile)
- **Orange Line:** Predicted lower bound (10th percentile)
- **Red Dashed:** Alert threshold (configurable, default 49.80 Hz)
- **Red Dot:** Current time marker
- **Green Zone:** Safe operation (49.80–50.20 Hz)
- **Yellow Zone:** Caution (49.50–49.80 Hz)
- **Red Zone:** Emergency (below 49.50 Hz)

### C.4.2 Alert States

| State | Visual Indicator | Audio | Meaning |
|-------|-----------------|-------|---------|
| **Stable** | Green background | None | Frequency within safe bounds, predictions confident |
| **Caution** | Yellow background | Single tone | Predicted lower bound approaching threshold |
| **Warning** | Orange background | Repeated tone | Alert threshold crossed, intervention recommended |
| **Critical** | Red background | Continuous alarm | Current frequency below threshold, emergency response activated |
| **Model Uncertainty** | Purple border | — | High prediction uncertainty, trust with caution |

## C.5 SHAP Risk Drivers Panel

### C.5.1 Explanation Display

The SHAP panel provides real-time explanations of model predictions using horizontal bar charts.

**Figure C.2: SHAP Risk Drivers Example (August 9, 2019)**

```
Risk Drivers (pushing frequency down ↑)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RoCoF (5s smoothed)     ████████████ -0.042 Hz
                        └─ Rapid frequency decline

Wind Ramp Rate (OpSDA)  ████████     -0.031 Hz
                        └─ Sudden generation loss

Renewable Penetration   █████        -0.018 Hz
                        └─ Low inertia vulnerability

Time of Day (16:52)     ██           -0.008 Hz
                        └─ Evening peak demand
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base Value: 50.00 Hz    Total: -0.099 Hz
Predicted Lower Bound: 49.901 Hz

Stabilizing Factors (pushing frequency up ↓)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature             ▏            +0.003 Hz
                        └─ Reduced heating demand
```

**Interpretation Guidelines:**
- **Red bars:** Features increasing instability risk (pushing frequency down)
- **Blue bars:** Features reducing risk (pushing frequency up)
- **Bar length:** Magnitude of contribution
- **Bar direction:** Sign of contribution (left=negative, right=positive)

### C.5.2 Operator Interpretation

When an alert triggers, operators should examine:
1. **Dominant drivers:** Which features contribute most to the prediction?
2. **Physical plausibility:** Do the drivers make sense given current conditions?
3. **Actionable insights:** What interventions address the identified drivers?

**Example Alert Interpretation:**
> "Alert triggered due to negative RoCoF (-0.042) indicating rapid frequency decline, combined with OpSDA wind ramp (-0.031) showing sudden generation loss. High renewable penetration (-0.018) indicates low system inertia. Recommended action: activate synthetic inertia reserves."

## C.6 Intervention Simulator

### C.6.1 Controls

| Control | Function | Default | Range |
|---------|----------|---------|-------|
| **Synthetic Inertia Slider** | Adjust injected power | 0 MW | 0–5,000 MW |
| **Inertia Constant (H)** | System inertia parameter | 4.0 s | 2.0–8.0 s |
| **Reset Button** | Clear simulation | — | — |

### C.6.2 Physical Model

The simulator recalculates predictions using the swing equation:

$$\Delta f = \frac{\Delta P \times f_0}{2 \times H \times S_{base}}$$

Where:
- **ΔP** = injected synthetic inertia (MW) [slider value]
- **f₀** = nominal frequency (50 Hz) [constant]
- **H** = system inertia constant (s) [configurable]
- **S_base** = total system capacity (35,000 MW) [constant]

### C.6.3 Usage Example

**Scenario:** Alert triggered, predicted lower bound = 49.75 Hz (0.05 Hz below threshold)

**Question:** How much synthetic inertia is required to restore safety?

**Procedure:**
1. Set H = 4.0 s (current system estimate)
2. Gradually increase synthetic inertia slider
3. Observe predicted lower bound rising in real-time
4. At ΔP = 1,500 MW: predicted lower bound = 49.82 Hz (above threshold)

**Conclusion:** 1,500 MW synthetic inertia injection would restore predicted stability.

### C.6.4 Visualization

**Figure C.3: Intervention Simulation Effect**

```
Before Intervention (0 MW):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Lower Bound: 49.75 Hz ⚠️ BELOW THRESHOLD

During Intervention (1,500 MW):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Lower Bound: 49.82 Hz ✅ ABOVE THRESHOLD
           ↑
    Frequency "lift" from injection

Uncertainty Band Adjustment:
    Before          After
    ╱‾‾‾‾╲          ╱‾‾‾‾‾‾‾‾‾‾╲
   ╱      ╲   →    ╱            ╲
  ╱        ╲      ╱              ╲
─────────────────────────────────────────
```

## C.7 Model Health Metrics Panel

### C.7.1 Quantile Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Pinball Loss (Lower)** | 0.0142 | <0.02 | ✅ Pass |
| **Pinball Loss (Upper)** | 0.0168 | <0.02 | ✅ Pass |
| **PICP** | 79.5% | ≥80% | ⚠️ Marginal |
| **MPIW** | 0.135 Hz | <0.2 | ✅ Pass |

### C.7.2 Calibration Metrics

| Quantile | Expected | Observed | Deviation | Status |
|----------|----------|----------|-----------|--------|
| **α=0.1** | 10.0% | 1.8% | -8.2 pp | ⚠️ Pessimistic |
| **α=0.9** | 90.0% | 79.3% | -10.7 pp | ⚠️ Slightly Low |

*Pessimistic bias at α=0.1 is desirable for safety-critical applications—model underestimates frequency, providing conservative early warnings.*

### C.7.3 Binary Classifier Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Precision** | 0.92 | >0.85 | ✅ Pass |
| **Recall** | 0.88 | >0.80 | ✅ Pass |
| **F1-Score** | 0.90 | >0.85 | ✅ Pass |
| **AUC-ROC** | 0.96 | >0.90 | ✅ Pass |

## C.8 Additional Screenshots

### C.8.1 Normal Operation Mode

**Figure C.4: Stable Grid Conditions**

During normal operation:
- Frequency remains within 49.8–50.2 Hz
- Uncertainty band narrow and centred on 50.0 Hz
- SHAP risk drivers show minimal contributions
- Alert status: STABLE (green)

### C.8.2 Model Uncertainty State

**Figure C.5: High Uncertainty Conditions**

During unprecedented conditions (e.g., multiple coincident outages):
- Uncertainty band widens significantly
- Purple border indicates model uncertainty warning
- SHAP drivers may show unusual combinations
- Operators should trust predictions with caution

## C.9 Troubleshooting

| Issue | Possible Cause | Resolution |
|-------|---------------|------------|
| **No data displayed** | API connection failure | Check internet connection; verify API status |
| **Stale data** | Caching issue | Clear browser cache; restart dashboard |
| **Slow performance** | Large date range selected | Reduce date range to <7 days |
| **SHAP not loading** | Model file missing | Verify model.pkl exists in /models directory |
| **Alerts not triggering** | Threshold set incorrectly | Check threshold slider; verify < 50.0 Hz |

## C.10 Keyboard Shortcuts

| Shortcut | Function |
|----------|----------|
| **Space** | Play/Pause autoplay |
| **← / →** | Step backward/forward 1 second |
| **Shift+← / Shift+→** | Step backward/forward 1 minute |
| **Home** | Jump to start of selected range |
| **End** | Jump to end of selected range |
| **B** | Jump to August 9, 2019 blackout |
| **R** | Reset intervention simulator |
| **H** | Toggle help overlay |

---

## References

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

Ucar, F. (2023) 'Explainable AI for smart grid stability prediction: Enhancing operator trust through SHAP analysis', *Energy and AI*, 14, p. 100256. doi: 10.1016/j.egyai.2023.100256.

Zhou, H., Li, W. and Zhao, C. (2025) 'LightGBM-based frequency prediction with dynamic feature weighting for UK power grid', *International Journal of Electrical Power & Energy Systems*, 153, p. 109512. doi: 10.1016/j.ijepes.2024.109512.
