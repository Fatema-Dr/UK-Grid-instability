#!/usr/bin/env python
"""
generate_real_dissertation_figures.py
======================================
Generates ALL dissertation figures from REAL trained models and REAL data.
Replaces the synthetic np.random figures in Final Report/figures/.

Run from Implementation/ directory:
    uv run python generate_real_dissertation_figures.py

Outputs saved to: ../Final Report/figures/
Metrics summary saved to: /tmp/real_metrics_summary.txt
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

# ── Paths ───────────────────────────────────────────────────────────────────
IMPL_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(IMPL_DIR, "..", "Final Report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LOWER_MODEL_PATH = os.path.join(IMPL_DIR, "notebooks", "lgbm_quantile_lower.pkl")
UPPER_MODEL_PATH = os.path.join(IMPL_DIR, "notebooks", "lgbm_quantile_upper.pkl")
DEMO_DATA_PATH   = os.path.join(IMPL_DIR, "notebooks", "demo_data_aug9.csv")
METRICS_OUT      = "/tmp/real_metrics_summary.json"

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "font.family": "DejaVu Sans",
})

COLORS = {
    "lower": "#E53935", "upper": "#1E88E5",
    "actual": "#00BCD4", "band": "#FF8F00",
    "green": "#43A047", "red": "#E53935",
}

sys.path.insert(0, IMPL_DIR)
from src.config import LGBM_FEATURE_COLS, TARGET_FREQ_NEXT, QUANTILE_ALPHAS


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Saved: {name}")
    return path


def pinball(y_true, y_pred, alpha):
    e = y_true - y_pred
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


# ════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════
print("Loading models and data...")
lo_m = joblib.load(LOWER_MODEL_PATH)
up_m = joblib.load(UPPER_MODEL_PATH)

df = pd.read_csv(DEMO_DATA_PATH, parse_dates=["timestamp"])
df = df.dropna(subset=LGBM_FEATURE_COLS + [TARGET_FREQ_NEXT])
print(f"  Aug 9 data: {len(df):,} rows")

X      = df[LGBM_FEATURE_COLS]
y_true = df[TARGET_FREQ_NEXT].values
lo_pred = lo_m.predict(X)
up_pred = up_m.predict(X)

# Core metrics
pb_lo  = pinball(y_true, lo_pred, QUANTILE_ALPHAS[0])
pb_up  = pinball(y_true, up_pred, QUANTILE_ALPHAS[1])
mae_lo = float(np.mean(np.abs(y_true - lo_pred)))
mae_up = float(np.mean(np.abs(y_true - up_pred)))
rmse_lo = float(np.sqrt(np.mean((y_true - lo_pred)**2)))
rmse_up = float(np.sqrt(np.mean((y_true - up_pred)**2)))
covered = ((y_true >= lo_pred) & (y_true <= up_pred))
picp   = float(np.mean(covered))
mpiw   = float(np.mean(up_pred - lo_pred))
cal_lo = float(np.mean(y_true < lo_pred))
cal_up = float(np.mean(y_true < up_pred))

print(f"\n  === REAL AUGUST 2019 METRICS ===")
print(f"  Pinball Lower : {pb_lo:.6f}")
print(f"  Pinball Upper : {pb_up:.6f}")
print(f"  MAE Lower     : {mae_lo:.6f} Hz")
print(f"  MAE Upper     : {mae_up:.6f} Hz")
print(f"  RMSE Lower    : {rmse_lo:.6f} Hz")
print(f"  RMSE Upper    : {rmse_up:.6f} Hz")
print(f"  PICP (80% CI) : {picp:.4f}")
print(f"  MPIW          : {mpiw:.6f} Hz")
print(f"  Calib α=0.1   : {cal_lo:.4f}  (target 0.10)")
print(f"  Calib α=0.9   : {cal_up:.4f}  (target 0.90)")

aug_metrics = {
    "pb_lo": pb_lo, "pb_up": pb_up,
    "mae_lo": mae_lo, "mae_up": mae_up,
    "rmse_lo": rmse_lo, "rmse_up": rmse_up,
    "picp": picp, "mpiw": mpiw,
    "cal_lo": cal_lo, "cal_up": cal_up,
}


# ════════════════════════════════════════════════════════════════════════════
# FIG 4.2 — REAL FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 4.2: Feature Importance (REAL)...")

FEATURE_LABELS = {
    "grid_frequency":            "Grid Frequency",
    "rocof":                     "RoCoF (5s smoothed)",
    "volatility_10s":            "Volatility (10s)",
    "volatility_30s":            "Volatility (30s)",
    "volatility_60s":            "Volatility (60s)",
    "wind_speed":                "Wind Speed",
    "wind_ramp_rate":            "Wind Ramp Rate (OpSDA)",
    "solar_radiation":           "Solar Radiation",
    "hour":                      "Hour of Day",
    "renewable_penetration_ratio": "Renewable Penetration",
    "lag_1s":                    "Lag 1s",
    "lag_5s":                    "Lag 5s",
    "lag_60s":                   "Lag 60s",
}

imp_lo = lo_m.feature_importances_
imp_up = up_m.feature_importances_
imp_mean = (imp_lo + imp_up) / 2.0
total = imp_mean.sum()
imp_pct = (imp_mean / total * 100) if total > 0 else imp_mean

labels = [FEATURE_LABELS.get(f, f) for f in LGBM_FEATURE_COLS]
order = np.argsort(imp_pct)

# Colour by category
PHYSICS = {"RoCoF (5s smoothed)", "Wind Ramp Rate (OpSDA)", "Renewable Penetration"}
VOLATILITY = {"Volatility (10s)", "Volatility (30s)", "Volatility (60s)"}
LAG = {"Lag 1s", "Lag 5s", "Lag 60s"}
bar_colors = []
for lbl in [labels[i] for i in order]:
    if lbl in PHYSICS:
        bar_colors.append("#E53935")
    elif lbl in VOLATILITY:
        bar_colors.append("#FB8C00")
    elif lbl in LAG:
        bar_colors.append("#8E24AA")
    else:
        bar_colors.append("#1E88E5")

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh([labels[i] for i in order], imp_pct[order],
               color=bar_colors, edgecolor="black", linewidth=0.4)

for bar, val in zip(bars, imp_pct[order]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Feature Importance (% Split Count — averaged lower/upper models)", fontweight="bold")
ax.set_title("Figure 4.2: LightGBM Feature Importance\n(13-feature model, real trained weights)", fontweight="bold", pad=15)

legend_els = [
    mpatches.Patch(facecolor="#E53935", label="Physics features (RoCoF, OpSDA, Renewable)"),
    mpatches.Patch(facecolor="#FB8C00", label="Volatility features (10s / 30s / 60s std)"),
    mpatches.Patch(facecolor="#8E24AA", label="Lag features (1s / 5s / 60s)"),
    mpatches.Patch(facecolor="#1E88E5", label="Context features (Frequency, Wind, Solar, Hour)"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=9)
ax.set_xlim(0, imp_pct[order].max() * 1.2)
ax.grid(True, alpha=0.3, axis="x")

save_fig(fig, "figure_4_2_feature_importance.png")

# Print the real importance table for dissertation update
print("\n  === REAL FEATURE IMPORTANCE TABLE ===")
order_desc = np.argsort(imp_pct)[::-1]
for rank, idx in enumerate(order_desc, 1):
    print(f"  {rank:2d}. {labels[idx]:<35} {imp_pct[idx]:.1f}%")

feat_importance_data = {
    labels[i]: round(float(imp_pct[i]), 1) for i in order_desc
}


# ════════════════════════════════════════════════════════════════════════════
# FIG 4.4 — REAL RESIDUAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 4.4: Residual Analysis (REAL)...")

res_lo = y_true - lo_pred
res_up = y_true - up_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, res, color, title, mae_val in [
    (axes[0], res_lo, COLORS["lower"], f"Lower Bound (α=0.1)\nMAE = {mae_lo:.5f} Hz", mae_lo),
    (axes[1], res_up, COLORS["upper"], f"Upper Bound (α=0.9)\nMAE = {mae_up:.5f} Hz", mae_up),
]:
    ax.hist(res, bins=120, density=True, alpha=0.75, color=color, edgecolor="none")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero error")
    ax.axvline(np.mean(res), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {np.mean(res):.4f} Hz")
    ax.set_xlabel("Residual: Actual − Predicted (Hz)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(f"({('A' if color == COLORS['lower'] else 'B')}) {title}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Figure 4.4: Prediction Residual Distributions (August 9, 2019)",
             fontweight="bold", fontsize=13)
save_fig(fig, "figure_4_4_residual_analysis.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5.4 — REAL CALIBRATION (2-point + annotated)
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 5.4: Calibration Reliability (REAL 2-point)...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Left: calibration dot plot
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", zorder=1)
alphas_tested = [QUANTILE_ALPHAS[0], QUANTILE_ALPHAS[1]]
obs_tested    = [cal_lo, cal_up]
ax.scatter(alphas_tested, obs_tested, s=180, color=[COLORS["lower"], COLORS["upper"]],
           zorder=5, edgecolors="black", linewidth=1.5)
for a, o, label in zip(alphas_tested, obs_tested, ["α=0.10", "α=0.90"]):
    ax.annotate(f"  {label}\n  nominal={a:.0%}\n  observed={o:.1%}",
                (a, o), fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.fill_between([0, alphas_tested[0]], [0, obs_tested[0]], [0, alphas_tested[0]],
                alpha=0.12, color=COLORS["lower"], label="Pessimistic bias region")
ax.set_xlabel("Nominal Quantile (α)", fontweight="bold")
ax.set_ylabel("Observed Fraction Below Prediction", fontweight="bold")
ax.set_title("(A) Quantile Calibration Check\n(2 quantile levels evaluated)", fontweight="bold")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Right: cost asymmetry table — the safety argument
ax2 = axes[1]
ax2.axis("off")
headers = ["Consequence", "False Negative\n(Missed Alert)", "False Positive\n(Unnecessary Alert)"]
rows = [
    ["Economic Cost",   "£100M – £1B",     "£1K – £10K"],
    ["Public Safety",   "Hospital failures,\ntransport disruption", "None"],
    ["Regulatory",      "Licence review,\nfines",  "None"],
    ["Reputational",    "National media\ncoverage", "Internal note"],
]
tbl = ax2.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1.0, 2.2)
for j in range(3):
    tbl[(0, j)].set_facecolor("#263238"); tbl[(0, j)].set_text_props(color="white", fontweight="bold")
for i in range(1, 5):
    tbl[(i, 1)].set_facecolor("#FFCDD2")
    tbl[(i, 2)].set_facecolor("#C8E6C9")
ax2.set_title("(B) Safety Argument: Why Pessimistic Bias is Acceptable\n"
              "False negatives cost orders of magnitude more than false positives",
              fontweight="bold", fontsize=10)

fig.suptitle("Figure 5.4: Quantile Calibration Reliability & Safety Justification",
             fontweight="bold", fontsize=13)
save_fig(fig, "figure_5_4_calibration_reliability.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5.5 — REAL SHAP BEESWARM
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 5.5: SHAP Beeswarm (REAL TreeExplainer)...")

# Use a sample of 2000 rows for speed
np.random.seed(0)
sample_idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
X_sample = X.iloc[sample_idx]

explainer   = shap.TreeExplainer(lo_m)
shap_values = explainer.shap_values(X_sample)

fig, ax = plt.subplots(figsize=(11, 7))
feature_labels_list = [FEATURE_LABELS.get(f, f) for f in LGBM_FEATURE_COLS]

# Manual beeswarm (shap.plots.beeswarm needs newer API; do it manually for control)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs_shap)
y_pos = np.arange(len(LGBM_FEATURE_COLS))

for yi, fi in enumerate(order):
    vals = shap_values[:, fi]
    feat_vals = X_sample.iloc[:, fi].values
    feat_norm = (feat_vals - feat_vals.min()) / (np.ptp(feat_vals) + 1e-9)
    y_jitter  = yi + np.random.uniform(-0.3, 0.3, len(vals))
    sc = ax.scatter(vals, y_jitter, c=feat_norm, cmap="RdYlGn_r",
                    s=12, alpha=0.5, edgecolors="none", vmin=0, vmax=1)
    ax.axhline(yi, color="gray", linewidth=0.3, alpha=0.4)
    ax.scatter([np.mean(vals)], [yi], color="black", s=60, marker="|", zorder=6)

ax.set_yticks(y_pos)
ax.set_yticklabels([feature_labels_list[i] for i in order], fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP Value (impact on predicted lower bound frequency, Hz)", fontweight="bold")
ax.set_title("Figure 5.5: Global SHAP Feature Importance — Beeswarm\n"
             f"Lower bound model (α=0.1) | n={len(X_sample):,} samples | Real TreeExplainer",
             fontweight="bold", pad=15)
cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Feature value\n(red=high, green=low)", fontsize=8)
ax.grid(True, alpha=0.2, axis="x")
save_fig(fig, "figure_5_5_shap_summary_beeswarm.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5.2 — REAL SHAP WATERFALL (Blackout alert second)
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 5.2: SHAP Waterfall (blackout second, REAL)...")

# Find the alert second — first row where lower prediction drops below 49.8 Hz
alert_mask = lo_pred < 49.8
if alert_mask.any():
    alert_idx = int(np.where(alert_mask)[0][0])
else:
    alert_idx = int(np.argmin(lo_pred))

X_alert = X.iloc[[alert_idx]]
alert_ts = df["timestamp"].iloc[alert_idx]
sv_alert = explainer(X_alert)

fig, ax = plt.subplots(figsize=(10, 7))

# Manual waterfall
sv_arr   = sv_alert.values[0]
base_val = float(sv_alert.base_values[0])
pred_val = base_val + sv_arr.sum()

order_w  = np.argsort(np.abs(sv_arr))[::-1]
top_n    = min(10, len(order_w))
top_idx  = order_w[:top_n]
top_labels = [FEATURE_LABELS.get(LGBM_FEATURE_COLS[i], LGBM_FEATURE_COLS[i]) for i in top_idx]
top_sv   = sv_arr[top_idx]

# Waterfall bars
running = pred_val
bar_starts, bar_heights, bar_colors_w = [], [], []
for sv in top_sv:
    bar_starts.append(running - sv)
    bar_heights.append(sv)
    bar_colors_w.append(COLORS["red"] if sv < 0 else COLORS["green"])
    running -= sv

y_pos_w = np.arange(top_n)
ax.barh(y_pos_w, top_sv, left=[s for s in bar_starts],
        color=bar_colors_w, edgecolor="white", linewidth=0.5, height=0.6)
ax.axvline(pred_val, color="orange", linestyle="--", linewidth=1.5, label=f"Pred lower: {pred_val:.4f} Hz")
ax.axvline(49.8, color="red", linestyle=":", linewidth=1.5, label="Alert threshold: 49.80 Hz")
ax.axvline(base_val, color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"Base value: {base_val:.4f} Hz")

for i, (sv, start) in enumerate(zip(top_sv, bar_starts)):
    ax.text(start + sv/2, i, f"{sv:+.4f}", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

ax.set_yticks(y_pos_w)
ax.set_yticklabels(top_labels, fontsize=9)
ax.set_xlabel("SHAP Value (Hz impact on predicted lower bound)", fontweight="bold")
ax.set_title(f"Figure 5.2: SHAP Waterfall — Alert Trigger\n"
             f"Timestamp: {alert_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
             f"Predicted lower bound: {pred_val:.4f} Hz",
             fontweight="bold", pad=15)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3, axis="x")
save_fig(fig, "figure_5_2_shap_waterfall.png")
print(f"  Alert fired at: {alert_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Predicted lower bound: {pred_val:.4f} Hz  (base: {base_val:.4f} Hz)")


# ════════════════════════════════════════════════════════════════════════════
# TEMPORAL WINDOW STABILITY (replaces fake 5-fold CV table)
# ════════════════════════════════════════════════════════════════════════════
print("\nComputing temporal window feature importance stability...")

# Use three non-overlapping weeks of Aug 9 data (sub-windows by hour)
hours = df["timestamp"].dt.hour if hasattr(df["timestamp"].dt, "hour") else pd.to_datetime(df["timestamp"]).dt.hour
windows = {
    "00:00–08:00": (hours >= 0) & (hours < 8),
    "08:00–16:00": (hours >= 8) & (hours < 16),
    "16:00–24:00": (hours >= 16),
}

temporal_importance = {}
for win_name, mask in windows.items():
    if mask.sum() < 10:
        continue
    imp_lo_w = lo_m.feature_importances_
    imp_up_w = up_m.feature_importances_
    # Feature importances don't change per window (they're model-global)
    # Instead, compute mean |SHAP| per window
    X_win = X[mask.values]
    if len(X_win) > 500:
        X_win = X_win.sample(500, random_state=42)
    sv_win = explainer.shap_values(X_win)
    mean_abs = np.abs(sv_win).mean(axis=0)
    total_w  = mean_abs.sum()
    pct_w    = mean_abs / total_w * 100 if total_w > 0 else mean_abs
    temporal_importance[win_name] = {LGBM_FEATURE_COLS[i]: round(float(pct_w[i]), 1)
                                      for i in range(len(LGBM_FEATURE_COLS))}

print("\n  === TEMPORAL WINDOW |SHAP| IMPORTANCE (%) ===")
print(f"  {'Feature':<35} {'00-08h':>8} {'08-16h':>8} {'16-24h':>8} {'Std':>8}")
print("  " + "-"*65)
for i, feat in enumerate(LGBM_FEATURE_COLS):
    vals = [temporal_importance.get(w, {}).get(feat, 0) for w in windows.keys()]
    lbl  = FEATURE_LABELS.get(feat, feat)
    std  = float(np.std(vals))
    print(f"  {lbl:<35} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}% {std:>7.2f}")


# ════════════════════════════════════════════════════════════════════════════
# SAVE METRICS SUMMARY
# ════════════════════════════════════════════════════════════════════════════
summary = {
    "aug_metrics": aug_metrics,
    "feat_importance": feat_importance_data,
    "temporal_importance": temporal_importance,
    "alert_timestamp": str(alert_ts),
    "alert_pred_lower_hz": round(pred_val, 4),
    "alert_base_hz": round(base_val, 4),
}
with open(METRICS_OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Metrics summary saved to {METRICS_OUT}")

print("\n" + "="*60)
print("  ALL FIGURES COMPLETE")
print("="*60)
print(f"  Output directory: {FIG_DIR}")
print("  Files generated:")
for fn in [
    "figure_4_2_feature_importance.png",
    "figure_4_4_residual_analysis.png",
    "figure_5_2_shap_waterfall.png",
    "figure_5_4_calibration_reliability.png",
    "figure_5_5_shap_summary_beeswarm.png",
]:
    path = os.path.join(FIG_DIR, fn)
    size = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
    print(f"    {'✅' if size > 0 else '❌'} {fn} ({size} KB)")
