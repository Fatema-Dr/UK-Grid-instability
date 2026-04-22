"""
GridGuardian Command Deck v3 — Operator-First Decision-Support Dashboard
=========================================================================
Storytelling flow: STATUS → CHART → RISK → ACTION
Corrected UTC timestamps for all event presets.

Run:  uv run streamlit run app_command_deck.py
"""

import streamlit as st
import pandas as pd
import polars as pl
import joblib
import plotly.graph_objects as go
import numpy as np
import os
import shap
import hashlib
from pathlib import Path
import tensorflow as tf
from src.config import (
    TTA_SECONDS,
    LGBM_FEATURE_COLS,
    LSTM_FEATURE_COLS,
    LSTM_TIME_STEPS,
    WEATHER_API_DEFAULT_START_DATE,
    WEATHER_API_DEFAULT_END_DATE,
    TARGET_FREQ_NEXT,
    TARGET_COL,
    QUANTILE_ALPHAS,
    LOWER_CALIBRATOR_PATH,
    UPPER_CALIBRATOR_PATH,
    LGBM_QUANTILE_LOWER_PATH,
    LGBM_QUANTILE_UPPER_PATH,
    LSTM_MODEL_PATH,
    SCALER_PATH,
    LGBM_MODEL_PATH,
)
from src.data_loader import fetch_frequency_data, fetch_inertia_data, fetch_weather_data
from src.calibration import calibrate_predictions
from src.feature_engineering import create_features
from datetime import date, datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_INERTIA_H = 4.0
NOMINAL_FREQ = 50.0
TOTAL_SYSTEM_CAPACITY = 35000

# Event Presets — ALL TIMES IN UTC
EVENT_PRESETS = {
    "— Select Preset —": None,
    "⚡ Aug 9: Blackout Pre-Event Fragility (15:52 UTC)": ("2019-08-09", "15:52:35"),
    "💥 Aug 9: Blackout Nadir — 48.79 Hz (15:53 UTC)": ("2019-08-09", "15:53:49"),
    "📉 Aug 9: Early Morning Dip (07:12 UTC)": ("2019-08-09", "07:12:24"),
    "🔄 Aug 10: Grid Recovery Phase": ("2019-08-10", "00:05:00"),
    "☀️ Aug 9: Stable Mid-Day Reference": ("2019-08-09", "12:00:00"),
}

# Metadata Briefs for Incidents
EVENT_BRIEFS = {
    "⚡ Aug 9: Blackout Pre-Event Fragility (15:52 UTC)": "Before the crash — 1.1GW Lightning Strike imminent.",
    "💥 Aug 9: Blackout Nadir — 48.79 Hz (15:53 UTC)": "Frequency drops to lowest point (48.79 Hz) as Hornsea and Little Barford generators trip.",
    "📉 Aug 9: Early Morning Dip (07:12 UTC)": "Minor frequency dip below 49.8Hz threshold; effectively managed by balancing services.",
    "🔄 Aug 10: Grid Recovery Phase": "Normal system conditions restored 12 hours after the cascade. High system inertia.",
    "☀️ Aug 9: Stable Mid-Day Reference": "System is highly stable, operating at nominal 50.0 Hz with high prediction confidence.",
}

# Human-readable feature labels for SHAP
SHAP_LABELS = {
    "grid_frequency": "Grid Frequency",
    "rocof": "Rate of Change (RoCoF)",
    "wind_speed": "Wind Speed",
    "wind_ramp_rate": "Wind Ramp Rate (OpSDA)",
    "solar_radiation": "Solar Radiation",
    "temperature": "Temperature",
    "volatility_10s": "Volatility (10s)",
    "volatility_30s": "Volatility (30s)",
    "volatility_60s": "Volatility (60s)",
    "renewable_penetration_ratio": "Renewable Penetration",
    "hour": "Hour of Day",
    "minute": "Minute",
    "lag_1s": "Lag 1s",
    "lag_5s": "Lag 5s",
    "lag_10s": "Lag 10s",
    "lag_30s": "Lag 30s",
    "lag_60s": "Lag 60s",
    "inertia_cost": "Inertia Cost Proxy",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/processed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_src_hash():
    hasher = hashlib.md5()
    for fp in sorted(Path("src").glob("**/*.py")):
        if "__pycache__" in str(fp):
            continue
        with open(fp, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def clear_invalid_cache(h):
    for cf in CACHE_DIR.glob("*.parquet"):
        if h not in cf.name:
            try:
                cf.unlink()
            except Exception:
                pass


def get_row_safe(df_local, idx):
    if df_local.empty or not (0 <= idx < len(df_local)):
        return None
    return df_local.iloc[idx]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridGuardian Command Deck",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Dark theme, glassmorphism, alerts
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(pulse: bool = False):
    pulse_css = ""
    if pulse:
        pulse_css = """
        @keyframes border-pulse {
            0%   { box-shadow: inset 0 0 0 4px rgba(255,50,50,0.0); }
            50%  { box-shadow: inset 0 0 0 4px rgba(255,50,50,0.85); }
            100% { box-shadow: inset 0 0 0 4px rgba(255,50,50,0.0); }
        }
        .stApp { animation: border-pulse 1.4s ease-in-out infinite; }
        """

    st.markdown(
        f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        .stApp {{
            background: linear-gradient(145deg, #0a0e1a 0%, #101829 50%, #0d1321 100%);
            color: #e0e6f0;
            font-family: 'Inter', sans-serif;
        }}
        .block-container {{ padding-top: 0.5rem; padding-bottom: 0.5rem; }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0d1321 0%, #131b2e 100%);
            border-right: 1px solid #1e2a42;
        }}

        /* KPI Cards */
        .kpi-card {{
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 160, 255, 0.12);
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
        }}
        .kpi-label {{
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #7b8fb8;
            margin-bottom: 4px;
        }}
        .kpi-value {{
            font-size: 1.5rem;
            font-weight: 800;
            line-height: 1.15;
        }}
        .kpi-delta {{
            font-size: 0.75rem;
            margin-top: 2px;
        }}

        /* Glass Cards */
        .glass-card {{
            background: rgba(15, 23, 42, 0.55);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 160, 255, 0.12);
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 8px;
        }}
        .glass-card h4 {{
            margin: 0 0 6px 0;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #7b8fb8;
        }}

        /* Alert Banner */
        .alert-banner {{
            border-radius: 8px;
            padding: 10px 18px;
            text-align: center;
            margin-bottom: 8px;
            font-weight: 800;
            letter-spacing: 2px;
            font-size: 1rem;
        }}
        .alert-critical {{
            background: linear-gradient(90deg, rgba(255,50,50,0.25), rgba(255,50,50,0.08));
            border: 1px solid rgba(255,60,60,0.55);
            color: #ff4b4b;
        }}
        .alert-warning {{
            background: linear-gradient(90deg, rgba(255,179,71,0.18), rgba(255,179,71,0.05));
            border: 1px solid rgba(255,179,71,0.4);
            color: #ffb347;
        }}

        /* Tank */
        .tank-outer {{
            background: rgba(10, 15, 30, 0.7);
            border: 1px solid rgba(100, 160, 255, 0.15);
            border-radius: 10px;
            padding: 8px;
            height: 140px;
            position: relative;
            overflow: hidden;
        }}
        .tank-fill {{
            position: absolute; bottom: 0; left: 0; right: 0;
            border-radius: 0 0 9px 9px;
            transition: height 0.5s ease;
        }}
        .tank-text {{
            position: relative; z-index: 2;
            text-align: center;
        }}

        /* Recommendation */
        .rec-card {{
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 8px;
        }}
        .rec-critical {{
            background: linear-gradient(135deg, rgba(255,75,75,0.12), rgba(255,75,75,0.03));
            border: 1px solid rgba(255,75,75,0.3);
        }}
        .rec-stable {{
            background: linear-gradient(135deg, rgba(0,230,138,0.08), rgba(0,230,138,0.02));
            border: 1px solid rgba(0,230,138,0.2);
        }}

        /* Countdown */
        .countdown {{
            background: linear-gradient(135deg, rgba(255,60,60,0.2), rgba(200,40,40,0.08));
            border: 1px solid rgba(255,60,60,0.4);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            margin-bottom: 8px;
        }}

        /* Metric override */
        div[data-testid="metric-container"] {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(100, 160, 255, 0.1);
            border-radius: 8px;
            padding: 8px;
        }}
        div[data-testid="metric-container"] > label {{ color: #7b8fb8; }}
        div[data-testid="metric-container"] {{ color: #e0e6f0; }}

        /* Metric label visibility fix */
        .stMetric label {{
            color: #7b8fb8 !important; 
            font-weight: 600;
            font-size: 0.75rem;
        }}
        .stMetric .stMarkdown {{
            color: #e0e6f0;
        }}

        /* Footer */
        .footer-bar {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(100, 160, 255, 0.08);
            border-radius: 8px;
            padding: 10px 16px;
            display: flex;
            justify-content: space-around;
            align-items: center;
            gap: 16px;
            margin-top: 6px;
        }}
        .footer-item {{
            text-align: center;
        }}
        .footer-label {{
            font-size: 0.62rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #556;
        }}
        .footer-value {{
            font-size: 1rem;
            font-weight: 700;
        }}

        {pulse_css}

        /* Mobile Responsive */
        @media (max-width: 768px) {{
            .stColumns > div {{ margin-bottom: 10px; }}
            .kpi-card {{ padding: 8px 10px; }}
            .kpi-value {{ font-size: 1.2rem; }}
            .kpi-label {{ font-size: 0.6rem; }}
            section[data-testid="stSidebar"] {{ width: 100%; }}
            .footer-bar {{ flex-wrap: wrap; gap: 8px; }}
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Keyboard navigation helper
st.markdown(
    """
<script>
document.addEventListener('keydown', function(e) {{
    const key = e.key;
    if (key === 'ArrowLeft') {{
        const btn = parent.document.querySelector('button:has-text("◀ -1s")');
        if (btn) btn.click();
    }} else if (key === 'ArrowRight') {{
        const btn = parent.document.querySelector('button:has-text("+1s ▶")');
        if (btn) btn.click();
    }} else if (key === ' ') {{
        const btn = parent.document.querySelector('button:has-text("▶ Play"), button:has-text("⏸ Pause")');
        if (btn) btn.click();
    }}
}});
</script>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading models and data…")
def load_all(start_str, end_str):
    src_hash = get_src_hash()
    clear_invalid_cache(src_hash)
    cache_file = (
        CACHE_DIR / f"processed_data_{start_str}_to_{end_str}_{src_hash}.parquet"
    )

    required = [
        LGBM_QUANTILE_LOWER_PATH,
        LGBM_QUANTILE_UPPER_PATH,
        LSTM_MODEL_PATH,
        SCALER_PATH,
        LGBM_MODEL_PATH,
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        st.error(f"🚨 Missing: {', '.join(missing)}")
        st.stop()

    lo_m = joblib.load(LGBM_QUANTILE_LOWER_PATH)
    up_m = joblib.load(LGBM_QUANTILE_UPPER_PATH)
    cls_m = joblib.load(LGBM_MODEL_PATH)
    lstm_m = tf.keras.models.load_model(LSTM_MODEL_PATH)
    sc = joblib.load(SCALER_PATH)

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        df_f = fetch_frequency_data(start_str, end_str)
        df_w = fetch_weather_data(start_str, end_str)
        df_i = fetch_inertia_data(start_str, end_str)
        for name, d in [("Frequency", df_f), ("Weather", df_w), ("Inertia", df_i)]:
            if d.is_empty():
                st.error(f"🚨 {name} data empty.")
                st.stop()
        df_f = df_f.sort("timestamp")
        df_w = df_w.sort("timestamp")
        df_i = df_i.sort("timestamp_date")
        merged = df_f.join_asof(df_w, on="timestamp", strategy="backward")
        df_i = df_i.with_columns(
            pl.col("timestamp_date")
            .cast(pl.Datetime(time_unit="us", time_zone="UTC"))
            .alias("timestamp")
        )
        merged = merged.join_asof(
            df_i.select(["timestamp", "inertia_cost"]),
            on="timestamp",
            strategy="backward",
        )
        merged = merged.drop_nulls().to_pandas()
        df = create_features(merged)
        if df.empty:
            st.error("🚨 No data after processing.")
            st.stop()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.to_parquet(cache_file)

    explainer = shap.TreeExplainer(lo_m)
    lo_cal = hi_cal = None
    if os.path.exists(LOWER_CALIBRATOR_PATH) and os.path.exists(UPPER_CALIBRATOR_PATH):
        lo_cal = joblib.load(LOWER_CALIBRATOR_PATH)
        hi_cal = joblib.load(UPPER_CALIBRATOR_PATH)

    return lo_m, up_m, cls_m, lstm_m, sc, df, explainer, lo_cal, hi_cal


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Navigation & Lab
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar.container(border=True):
    st.markdown("##### 📅 Data Range")
    default_s = pd.to_datetime(WEATHER_API_DEFAULT_START_DATE).date()
    default_e = pd.to_datetime(WEATHER_API_DEFAULT_END_DATE).date()
    c1, c2 = st.columns(2)
    with c1:
        sel_start = st.date_input("Start", value=default_s)
    with c2:
        sel_end = st.date_input("End", value=default_e)

lo_m, up_m, cls_m, lstm_m, scaler, df_data, explainer, lo_cal, hi_cal = load_all(
    sel_start.strftime("%Y-%m-%d"), sel_end.strftime("%Y-%m-%d")
)
if df_data.empty:
    st.error("🚨 No data.")
    st.stop()

# Event presets with Scenario Runner
SCENARIOS = {
    "— Select Scenario —": None,
    "1️⃣ Stable Mid-Day (Aug 9, 12:00)": (
        "2019-08-09",
        "12:00:00",
        "Normal grid operation at nominal frequency",
    ),
    "2️⃣ Morning Dip (Aug 9, 07:12)": (
        "2019-08-09",
        "07:12:24",
        "Minor frequency dip below 49.8Hz threshold",
    ),
    "3️⃣ Pre-Event Warning (Aug 9, 15:52)": (
        "2019-08-09",
        "15:52:35",
        "High-risk state before lightning strike",
    ),
    "4️⃣ Blackout Nadir (Aug 9, 15:53)": (
        "2019-08-09",
        "15:53:49",
        "Lowest frequency point - 48.79 Hz",
    ),
    "5️⃣ Grid Recovery (Aug 10, 00:05)": (
        "2019-08-10",
        "00:05:00",
        "System restored, high inertia",
    ),
}

with st.sidebar.container(border=True):
    st.markdown("##### 🎬 Scenario Runner")
    scenario_label = st.selectbox("Select Scenario", list(SCENARIOS.keys()), index=0)
    scenario_val = SCENARIOS[scenario_label]
    if scenario_val is not None:
        st.success(f"📋 {scenario_val[2]}")

# Event presets
with st.sidebar.container(border=True):
    st.markdown("##### 🎯 Incident Reviewer")
    _prev = st.session_state.get("_last_preset", "— Select Preset —")
    preset_label = st.selectbox("Event Preset", list(EVENT_PRESETS.keys()), index=0)
    preset_val = EVENT_PRESETS[preset_label]
    if preset_label in EVENT_BRIEFS:
        st.info(EVENT_BRIEFS[preset_label])

# Time navigation
with st.sidebar.container(border=True):
    st.markdown("##### ⏱ Time Scrubbing")
    available_dates = sorted(df_data["timestamp"].dt.date.unique().tolist())

if (
    "nav_date" not in st.session_state
    or st.session_state.nav_date not in available_dates
):
    st.session_state.nav_date = available_dates[len(available_dates) // 2]

# Scenario runner handler
_scenario_prev = st.session_state.get("_last_scenario", "— Select Scenario —")
if scenario_val is not None and scenario_label != _scenario_prev:
    pd_obj = pd.to_datetime(scenario_val[0]).date()
    if pd_obj in available_dates:
        st.session_state.nav_date = pd_obj
        st.session_state["_preset_time"] = scenario_val[1]
        st.session_state["_skip_ti"] = True
st.session_state["_last_scenario"] = scenario_label

if preset_val is not None and preset_label != _prev:
    pd_obj = pd.to_datetime(preset_val[0]).date()
    if pd_obj in available_dates:
        st.session_state.nav_date = pd_obj
        st.session_state["_preset_time"] = preset_val[1]
st.session_state["_last_preset"] = preset_label

sel_nav_date = st.sidebar.date_input(
    "📅 Jump to Date",
    value=st.session_state.nav_date,
    min_value=available_dates[0],
    max_value=available_dates[-1],
)
if sel_nav_date != st.session_state.nav_date:
    st.session_state.nav_date = sel_nav_date
    st.session_state.pop("time_nav_index", None)

day_mask = df_data["timestamp"].dt.date == sel_nav_date
day_idxs = np.where(day_mask)[0].tolist()
day_ts = df_data["timestamp"].iloc[day_idxs].tolist()
if not day_idxs:
    st.sidebar.warning(f"No data for {sel_nav_date}")
    st.stop()

if "time_nav_index" not in st.session_state:
    st.session_state.time_nav_index = len(day_idxs) // 2
st.session_state.time_nav_index = min(
    max(0, st.session_state.time_nav_index), len(day_idxs) - 1
)

# Preset time jump
if "_preset_time" in st.session_state:
    pt = st.session_state.pop("_preset_time")
    parts = pt.split(":")
    target = pd.Timestamp.combine(
        sel_nav_date,
        pd.Timestamp(f"{parts[0]}:{parts[1]}:{parts[2]}").time(),  # type: ignore[attr-defined]
    )
    ts_series = pd.Series(day_ts)
    if hasattr(ts_series.iloc[0], "tzinfo") and ts_series.iloc[0].tzinfo is not None:
        target = target.tz_localize(ts_series.iloc[0].tzinfo)  # type: ignore[attr-defined]
    st.session_state.time_nav_index = int((ts_series - target).abs().idxmin())
    st.session_state["_skip_ti"] = True

# Step buttons
b1, b2, b3, b4 = st.sidebar.columns(4)
with b1:
    if st.button("◀ -1m"):
        st.session_state.time_nav_index = max(0, st.session_state.time_nav_index - 60)
        st.rerun()
with b2:
    if st.button("◀ -1s"):
        st.session_state.time_nav_index = max(0, st.session_state.time_nav_index - 1)
        st.rerun()
with b3:
    if st.button("+1s ▶"):
        st.session_state.time_nav_index = min(
            len(day_idxs) - 1, st.session_state.time_nav_index + 1
        )
        st.rerun()
with b4:
    if st.button("+1m ▶"):
        st.session_state.time_nav_index = min(
            len(day_idxs) - 1, st.session_state.time_nav_index + 60
        )
        st.rerun()

# Autoplay
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False
ap1, ap2 = st.sidebar.columns(2)
with ap1:
    if st.button("▶ Play" if not st.session_state.autoplay else "⏸ Pause"):
        st.session_state.autoplay = not st.session_state.autoplay
        st.rerun()
with ap2:
    ap_speed = st.selectbox(
        "Speed", [1, 5, 10, 30, 60], index=0, label_visibility="collapsed"
    )
if st.session_state.autoplay:
    import time as _time

    _time.sleep(1.0)
    st.session_state.time_nav_index = min(
        len(day_idxs) - 1, st.session_state.time_nav_index + ap_speed
    )
    if st.session_state.time_nav_index >= len(day_idxs) - 1:
        st.session_state.autoplay = False
    st.rerun()

# Exact time text input (HH:MM:SS)
cur_t = day_ts[st.session_state.time_nav_index].time()
cur_t_str = cur_t.strftime("%H:%M:%S")
sel_t_str = st.sidebar.text_input("⌨️ Exact Time (HH:MM:SS UTC)", value=cur_t_str)
if st.session_state.pop("_skip_ti", False):
    pass
elif sel_t_str != cur_t_str:
    try:
        p = sel_t_str.strip().split(":")  # type: ignore[union-attr]
        from datetime import time as dt_time

        pt = dt_time(int(p[0]), int(p[1]), int(p[2]) if len(p) > 2 else 0)
        tgt = pd.Timestamp.combine(sel_nav_date, pt)
        ts_s = pd.Series(day_ts)
        if hasattr(ts_s.iloc[0], "tzinfo") and ts_s.iloc[0].tzinfo is not None:
            tgt = tgt.tz_localize(ts_s.iloc[0].tzinfo)  # type: ignore[attr-defined]
        c = int((ts_s - tgt).abs().idxmin())
        if st.session_state.time_nav_index != c:
            st.session_state.time_nav_index = c
            st.rerun()
    except (ValueError, IndexError):
        st.sidebar.caption("⚠️ Format: HH:MM:SS")

time_index = day_idxs[st.session_state.time_nav_index]
sel_ts = day_ts[st.session_state.time_nav_index]
st.sidebar.caption(f"📍 **{sel_ts.strftime('%Y-%m-%d  %H:%M:%S')} UTC**")

current_row = get_row_safe(df_data, time_index)
if current_row is None:
    st.stop()
current_time = current_row["timestamp"]

# Alert config
with st.sidebar.container(border=True):
    st.markdown("##### ⚙️ Alert Configuration")
    alert_hz = st.slider(
        "Instability Threshold (Hz)",
        49.5,
        49.95,
        49.8,
        0.05,
        help="UK Statutory Operational Limit is typically 49.5 Hz. Below 49.8 Hz triggers standard remedial action.",
    )

# Intervention
with st.sidebar.container(border=True):
    st.markdown("##### 🕹️ Intervention Lab (Simulated)")
    synth_mw = st.slider(
        "Inject Synthetic Inertia (MW)",
        0,
        5000,
        0,
        100,
        help="Simulate battery / demand response to correct frequency drops.",
    )
    if synth_mw > 0:
        est_cost = synth_mw * current_row.get("inertia_cost", 12.5)
        st.info(f"💸 Est. Intervention Cost: **£{est_cost:,.0f}**")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
X_in = pd.DataFrame([current_row[LGBM_FEATURE_COLS].values], columns=LGBM_FEATURE_COLS)

swing_df = 0.0
if synth_mw > 0:
    swing_df = (synth_mw * NOMINAL_FREQ) / (
        2 * SYSTEM_INERTIA_H * TOTAL_SYSTEM_CAPACITY
    )
    X_in["renewable_penetration_ratio"] = np.maximum(
        0, X_in["renewable_penetration_ratio"] - synth_mw / TOTAL_SYSTEM_CAPACITY
    )

lo_raw = lo_m.predict(X_in)[0]
up_raw = up_m.predict(X_in)[0]
if lo_cal is not None and hi_cal is not None:
    lo_raw = calibrate_predictions(lo_cal, np.array([lo_raw]))[0]
    up_raw = calibrate_predictions(hi_cal, np.array([up_raw]))[0]

lo_pred = lo_raw + swing_df
up_pred = up_raw + swing_df
mid_pred = (lo_pred + up_pred) / 2.0

freq = current_row["grid_frequency"]
rocof = current_row["rocof"]

is_alert = lo_pred < alert_hz or freq < alert_hz

# LSTM
w_start = max(0, time_index - LSTM_TIME_STEPS + 1)
lstm_df = df_data.iloc[w_start : time_index + 1]
lstm_alert = False
lstm_prob = 0.0
if len(lstm_df) == LSTM_TIME_STEPS:
    lstm_sc = scaler.transform(lstm_df[LSTM_FEATURE_COLS])
    lstm_prob = float(lstm_m.predict(np.array([lstm_sc]), verbose=0)[0][0])
    lstm_alert = lstm_prob > 0.5

# Combined status
if is_alert and lstm_alert:
    status = "INSTABILITY ALERT"
    status_class = "alert-critical"
    status_icon = "🚨"
elif is_alert or lstm_alert:
    status = "HIGH MODEL UNCERTAINTY"
    status_class = "alert-warning"
    status_icon = "⚠️"
else:
    status = "SYSTEM STABLE"
    status_class = ""
    status_icon = "✅"

is_breach = status == "INSTABILITY ALERT"

# SHAP
sv_raw = explainer.shap_values(X_in)
sv_arr = sv_raw[0][0] if isinstance(sv_raw, list) else sv_raw[0]
shap_dict = dict(zip(LGBM_FEATURE_COLS, sv_arr))

# Trust score
trust = 100.0
model_conflict = is_alert != lstm_alert

if model_conflict:
    trust = 0.0
else:
    if current_row.get("volatility_10s", 0) > 0.015:
        trust -= min(20.0, current_row.get("volatility_10s", 0) * 200)
    if abs(rocof) > 0.01:
        trust -= min(15.0, abs(rocof) * 150)
    u = 1.0 - abs(lstm_prob - 0.5) * 2
    if u > 0.3:
        trust -= u * 20.0
    fd = abs(freq - 50.0)
    if fd > 0.05:
        trust -= min(15.0, fd * 100)
trust = max(0.0, min(100.0, trust))

# MW recommendation
req_mw = 0.0
if lo_pred < alert_hz:
    uplift = alert_hz - lo_pred + 0.01
    req_mw = (uplift * 2 * SYSTEM_INERTIA_H * TOTAL_SYSTEM_CAPACITY) / NOMINAL_FREQ

# Countdown
countdown_s = None
if is_alert:
    ahead = min(time_index + 300, len(df_data))
    for fi in range(time_index, ahead):
        if df_data.iloc[fi]["grid_frequency"] < alert_hz:
            countdown_s = fi - time_index
            break

# Tank
rp_raw = current_row.get("renewable_penetration_ratio", 0.5)
rp = min(1.0, max(0.0, rp_raw))
tank_pct = max(0, min(100, (1 - rp) * 100))

# ─────────────────────────────────────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────────────────────────────────────
inject_css(pulse=is_breach)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

# ── Quick Stats Summary ─────────────────────────────────────────────────────
st.markdown("### 📈 Daily Overview", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

day_data = df_data[df_data["timestamp"].dt.date == sel_nav_date]
if not day_data.empty:
    day_freq = day_data["grid_frequency"]
    qs_min = day_freq.min()
    qs_max = day_freq.max()
    qs_avg = day_freq.mean()
    qs_breaches = (day_freq < alert_hz).sum()

    qs_col1, qs_col2, qs_col3, qs_col4 = st.columns(4, gap="small")
    with qs_col1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">MIN FREQUENCY</div>
            <div class="kpi-value" style="color:#ff4b4b;">{qs_min:.3f} Hz</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with qs_col2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">MAX FREQUENCY</div>
            <div class="kpi-value" style="color:#00e68a;">{qs_max:.3f} Hz</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with qs_col3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">AVG FREQUENCY</div>
            <div class="kpi-value" style="color:#00CCFF;">{qs_avg:.3f} Hz</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with qs_col4:
        breach_color = "#ff4b4b" if qs_breaches > 0 else "#00e68a"
        st.markdown(
            f"""
        <div class="kpi-card" style="box-shadow: 0 0 15px {breach_color}25;">
            <div class="kpi-label">THRESHOLD BREACHES</div>
            <div class="kpi-value" style="color:{breach_color};">{qs_breaches}</div>
        </div>""",
            unsafe_allow_html=True,
        )

# ── Data Export ────────────────────────────────────────────────────────────
export_col1, export_col2 = st.columns([1, 4])
with export_col1:
    st.download_button(
        label="📥 Export Data",
        data=day_data.to_csv(index=False),
        file_name=f"gridguardian_{sel_nav_date}.csv",
        mime="text/csv",
        help="Download current day's data as CSV",
    )

# ── Alert Banner ─────────────────────────────────────────────────────────
if status_class:
    st.markdown(
        f'<div class="alert-banner {status_class}">{status_icon} {status}</div>',
        unsafe_allow_html=True,
    )

# ── Title ────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='margin:0;padding:0 0 6px 0;font-size:1.7rem;font-weight:900;'>"
    f"⚡ COMMAND DECK "
    f"<span style='font-weight:400;font-size:1rem;color:#7b8fb8;'>"
    f"— {current_time.strftime('%H:%M:%S UTC')} — {sel_nav_date.strftime('%d %b %Y')}"
    f"</span></h1>",
    unsafe_allow_html=True,
)

# ── TABS ─────────────────────────────────────────────────────────────────
tabs = st.tabs(["🎛️ Command Deck", "📊 Model Health"])

with tabs[0]:
    # ══════════════════════════════════════════════════════════════════════
    # ROW 1 — KPI Cards
    # ══════════════════════════════════════════════════════════════════════
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    # Hz colour
    if freq >= 49.8 and freq <= 50.2:
        hz_col = "#00e68a"
    elif freq >= 49.6:
        hz_col = "#ffb347"
    else:
        hz_col = "#ff4b4b"

    delta_f = freq - 50.0
    delta_sign = "▲" if delta_f >= 0 else "▼"
    delta_col = "#00e68a" if delta_f >= 0 else "#ff4b4b"

    with k1:
        st.markdown(
            f"""
        <div class="kpi-card" style="box-shadow: 0 0 15px {hz_col}25;">
            <div class="kpi-label">FREQUENCY</div>
            <div class="kpi-value" style="color:{hz_col};">{freq:.3f} Hz</div>
            <div class="kpi-delta" style="color:{delta_col};">{delta_sign} {abs(delta_f):.3f} Hz</div>
        </div>""",
            unsafe_allow_html=True,
        )

    rocof_col = (
        "#ff4b4b"
        if abs(rocof) > 0.02
        else ("#ffb347" if abs(rocof) > 0.005 else "#00e68a")
    )
    rocof_arrow = "↗" if rocof > 0.005 else ("↘" if rocof < -0.005 else "→")
    with k2:
        st.markdown(
            f"""
        <div class="kpi-card" style="box-shadow: 0 0 15px {rocof_col}25;">
            <div class="kpi-label">ROCOF (MOMENTUM)</div>
            <div class="kpi-value" style="color:{rocof_col};">{rocof_arrow} {rocof:+.4f}</div>
            <div class="kpi-delta" style="color:#7b8fb8;">Hz/s</div>
        </div>""",
            unsafe_allow_html=True,
        )

    lo_col = (
        "#ff4b4b"
        if lo_pred < alert_hz
        else ("#ffb347" if lo_pred < alert_hz + 0.05 else "#00e68a")
    )
    with k3:
        st.markdown(
            f"""
        <div class="kpi-card" style="box-shadow: 0 0 15px {lo_col}25;">
            <div class="kpi-label">SAFETY FLOOR (10s Ahead)</div>
            <div class="kpi-value" style="color:{lo_col};">{lo_pred:.3f} Hz</div>
            <div class="kpi-delta" style="color:#7b8fb8;">Predicted bound</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
        <div class="kpi-card" style="box-shadow: 0 0 15px #00CCFF25;">
            <div class="kpi-label">PRED UPPER (90th)</div>
            <div class="kpi-value" style="color:#00CCFF;">{up_pred:.3f} Hz</div>
            <div class="kpi-delta" style="color:#7b8fb8;">Ceiling</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st_col = (
        "#ff4b4b"
        if is_breach
        else ("#ffb347" if status != "SYSTEM STABLE" else "#00e68a")
    )
    if model_conflict:
        st_col = "#b366ff"  # Purple for model conflict
        status_icon = "❓"

    with k5:
        st.markdown(
            f"""
        <div class="kpi-card" style="border-color:{st_col}40; box-shadow: 0 0 15px {st_col}25;">
            <div class="kpi-label">STATUS</div>
            <div class="kpi-value" style="color:{st_col};font-size:1.1rem;">{status_icon} {status.split()[-1]}</div>
            <div class="kpi-delta" style="color:#7b8fb8;">LSTM: {lstm_prob:.1%}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2 — HERO CHART (Safety Tunnel)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("")  # spacing
    window_back = 120  # 2 min behind
    window_fwd = 30  # 30s ahead
    s_idx = max(0, time_index - window_back)
    e_idx = min(len(df_data), time_index + window_fwd + 1)
    chart_df = df_data.iloc[s_idx:e_idx]

    if not chart_df.empty:
        cX = chart_df[LGBM_FEATURE_COLS]
        c_lo = lo_m.predict(cX) + swing_df
        c_up = up_m.predict(cX) + swing_df

        fig = go.Figure()

        # Threshold
        fig.add_hline(
            y=alert_hz,
            line_dash="dash",
            line_color="#ff4b4b",
            line_width=1.5,
            annotation_text=f"Alert: {alert_hz} Hz",
            annotation_font_color="#ff8888",
            annotation_font_size=11,
            annotation_position="top left",
        )

        # Uncertainty band
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp"],
                y=c_up,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp"],
                y=c_lo,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255,165,0,0.18)",
                name="80% Prediction Interval",
            )
        )

        # "What-If" Ghost Line
        if synth_mw > 0:
            c_lo_orig = c_lo - swing_df
            fig.add_trace(
                go.Scatter(
                    x=chart_df["timestamp"],
                    y=c_lo_orig,
                    mode="lines",
                    line=dict(color="rgba(150,150,150,0.8)", width=2, dash="dot"),
                    name="Original Lower Bound (No Action)",
                )
            )

        # Actual frequency
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp"],
                y=chart_df["grid_frequency"],
                mode="lines",
                line=dict(color="#00CCFF", width=2.5),
                name="Actual Frequency",
                hovertemplate="%{x|%H:%M:%S}<br>%{y:.3f} Hz<extra></extra>",
            )
        )

        # Current marker
        fig.add_trace(
            go.Scatter(
                x=[current_time],
                y=[freq],
                mode="markers",
                marker=dict(
                    color="#ff4b4b",
                    size=10,
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                name="Now",
                hoverinfo="skip",
            )
        )

        # Ghost prediction marker
        fig.add_trace(
            go.Scatter(
                x=[current_time],
                y=[mid_pred],
                mode="markers",
                marker=dict(
                    color="rgba(0,204,255,0.5)",
                    size=10,
                    symbol="diamond-open",
                    line=dict(color="#00CCFF", width=2),
                ),
                name=f"Predicted ({mid_pred:.3f} Hz)",
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=50, r=20, t=35, b=40),
            title=dict(
                text="FREQUENCY & SAFETY TUNNEL",
                font=dict(size=13, color="#7b8fb8"),
                x=0.01,
            ),
            yaxis=dict(
                title="Frequency (Hz)",
                title_font=dict(size=11, color="#7b8fb8"),
                gridcolor="rgba(100,160,255,0.06)",
            ),
            xaxis=dict(
                tickformat="%H:%M:%S",
                gridcolor="rgba(100,160,255,0.06)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.5)",
            legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, key="hero_chart")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 3 — RISK DRIVERS (Left) + ACTION PANEL (Right)
    # ══════════════════════════════════════════════════════════════════════
    col_risk, col_act = st.columns([1, 1], gap="medium")

    # ── SHAP Horizontal Bar Chart ────────────────────────────────────────
    with col_risk:
        st.markdown(
            '<div class="glass-card"><h4>RISK DRIVERS — SHAP ATTRIBUTION</h4></div>',
            unsafe_allow_html=True,
        )

        # Sort by absolute SHAP, take top 8
        sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[
            :8
        ]
        feat_names = [SHAP_LABELS.get(f, f) for f, _ in sorted_feats]
        feat_vals = [v for _, v in sorted_feats]
        bar_colors = ["#ff4b4b" if v < 0 else "#00e68a" for v in feat_vals]

        top_risk_driver = "None"
        for name, val in zip(feat_names, feat_vals):
            if val < 0:
                top_risk_driver = name
                break
        st.markdown(
            f"<div style='font-size:0.8rem; color:#7b8fb8; margin-bottom:8px;'>Current instability risk primarily driven by: <b>{top_risk_driver}</b></div>",
            unsafe_allow_html=True,
        )

        fig_shap = go.Figure()
        fig_shap.add_trace(
            go.Bar(
                y=feat_names[::-1],
                x=feat_vals[::-1],
                orientation="h",
                marker_color=bar_colors[::-1],
                text=[f"{v:+.4f}" for v in feat_vals[::-1]],
                textposition="outside",
                textfont=dict(size=10, color="#e0e6f0"),
                hovertemplate="%{y}: %{x:.5f}<extra></extra>",
            )
        )
        fig_shap.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=60, t=8, b=8),
            xaxis=dict(
                title="SHAP Value",
                title_font_size=10,
                gridcolor="rgba(100,160,255,0.06)",
            ),
            yaxis=dict(tickfont=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.3)",
        )
        fig_shap.add_vline(x=0, line_color="rgba(200,200,200,0.3)", line_width=1)
        st.plotly_chart(fig_shap, use_container_width=True, key="shap_chart")

        st.caption("🔴 Red = destabilising | 🟢 Green = stabilising")

    # ── Action Panel ─────────────────────────────────────────────────────
    with col_act:
        # Countdown
        if is_alert and countdown_s is not None:
            cd = f"T-{countdown_s}s" if countdown_s > 0 else "NOW"
            st.markdown(
                f"""
            <div class="countdown">
                <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:1.5px;color:#ff8888;">COUNTDOWN TO IMPACT</div>
                <div style="font-size:2.2rem;font-weight:900;color:#ff4b4b;
                            font-variant-numeric:tabular-nums;">{cd}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        elif is_alert:
            st.markdown(
                """
            <div class="countdown">
                <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:1.5px;color:#ff8888;">⚠️ THRESHOLD BREACHED</div>
                <div style="font-size:1.3rem;font-weight:900;color:#ff4b4b;">BREACH ACTIVE</div>
            </div>""",
                unsafe_allow_html=True,
            )

        # MW Recommendation
        if req_mw > 0:
            reasoning = (
                f"to lift lower bound above {alert_hz} Hz · Δf needed: +{(alert_hz - lo_pred + 0.01):.4f} Hz"
                if lo_pred < alert_hz
                else f"to lift actual frequency above {alert_hz} Hz · Δf needed: +{(alert_hz - freq + 0.01):.4f} Hz"
            )
            st.markdown(
                f"""
            <div class="rec-card rec-critical">
                <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:1.2px;color:#ff8888;">⚡ ACTION REQUIRED</div>
                <div style="font-size:1.5rem;font-weight:800;color:#ff4b4b;margin:4px 0;">
                    Inject {req_mw:,.0f} MW</div>
                <div style="font-size:0.75rem;color:#aa7777;">
                    {reasoning}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="rec-card rec-stable">
                <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:1.2px;color:#00cc7a;">✅ NO ACTION REQUIRED</div>
                <div style="font-size:0.82rem;color:#00e68a;margin-top:4px;">
                    Frequency and prediction safely above threshold ({alert_hz} Hz)</div>
            </div>""",
                unsafe_allow_html=True,
            )

        # Intervention status
        if synth_mw > 0:
            st.markdown(
                f"""
            <div class="glass-card">
                <h4>INTERVENTION ACTIVE</h4>
                <div style="font-size:1.1rem;font-weight:700;color:#00CCFF;">
                    +{synth_mw:,} MW · Δf: +{swing_df:.4f} Hz</div>
            </div>""",
                unsafe_allow_html=True,
            )

        # Inertia Tank
        if tank_pct > 60:
            t_grad = "linear-gradient(0deg, #00e68a, #00cc7a)"
            t_col = "#00e68a"
        elif tank_pct > 30:
            t_grad = "linear-gradient(0deg, #ffb347, #ff9500)"
            t_col = "#ffb347"
        else:
            t_grad = "linear-gradient(0deg, #ff4b4b, #cc3333)"
            t_col = "#ff4b4b"

        st.markdown(
            f"""
        <div class="tank-outer">
            <div class="tank-fill" style="height:{tank_pct}%;background:{t_grad};"></div>
            <div class="tank-text">
                <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:1px;color:#7b8fb8;padding-top:4px;">INERTIA RESILIENCE</div>
                <div style="font-size:1.6rem;font-weight:800;color:{t_col};padding-top:16px;">
                    {tank_pct:.0f}%</div>
                <div style="font-size:0.68rem;color:#7b8fb8;padding-top:4px;">
                    Ren. Penetration: {rp:.1%}</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # FOOTER — Trust, LSTM, Model Agreement
    # ══════════════════════════════════════════════════════════════════════
    if model_conflict:
        tr_col = "#b366ff"
        agree_txt = "CONFLICT ⚠️"
        agree_col = "#b366ff"
    else:
        tr_col = "#00e68a" if trust >= 80 else ("#ffb347" if trust >= 50 else "#ff4b4b")
        agree_txt = "AGREE ✓"
        agree_col = "#00e68a"

    lstm_col = (
        "#ff4b4b" if lstm_prob > 0.5 else ("#ffb347" if lstm_prob > 0.3 else "#00e68a")
    )

    st.markdown(
        f"""
    <div class="footer-bar">
        <div class="footer-item">
            <div class="footer-label">System Trust</div>
            <div class="footer-value" style="color:{tr_col};">{trust:.0f}%</div>
        </div>
        <div class="footer-item">
            <div class="footer-label">LSTM P(Unstable)</div>
            <div class="footer-value" style="color:{lstm_col};">{lstm_prob:.1%}</div>
        </div>
        <div class="footer-item">
            <div class="footer-label">Model Agreement</div>
            <div class="footer-value" style="color:{agree_col};">{agree_txt}</div>
        </div>
        <div class="footer-item">
            <div class="footer-label">Prediction Band</div>
            <div class="footer-value" style="color:#00CCFF;">{(up_pred - lo_pred):.4f} Hz</div>
        </div>
        <div class="footer-item">
            <div class="footer-label">Volatility (10s)</div>
            <div class="footer-value" style="color:#7b8fb8;">{current_row.get("volatility_10s", 0):.5f}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL HEALTH TAB
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing model metrics...", ttl=0)
def compute_model_metrics(
    _date_str,
    _df,
    _lo_m,
    _up_m,
    _cls_m,
    _target_freq_next,
    _target_col,
    _lgbm_feature_cols,
    _quantile_alphas,
):
    day_eval = _df.dropna(subset=[_target_freq_next, _target_col])
    if day_eval.empty or len(day_eval) < 10:
        return None

    y_t = day_eval[_target_freq_next].values
    X_e = day_eval[_lgbm_feature_cols]
    lo_d = _lo_m.predict(X_e)
    up_d = _up_m.predict(X_e)
    a_lo, a_up = _quantile_alphas
    e_lo = y_t - lo_d
    pb_lo = np.mean(np.maximum(a_lo * e_lo, (a_lo - 1) * e_lo))
    e_up = y_t - up_d
    pb_up = np.mean(np.maximum(a_up * e_up, (a_up - 1) * e_up))
    cov = ((y_t >= lo_d) & (y_t <= up_d)).astype(int)
    picp = np.mean(cov)
    mpiw = np.mean(up_d - lo_d)
    cal_lo = np.mean(y_t < lo_d)
    cal_up = np.mean(y_t > up_d)

    y_cls = day_eval[_target_col].values
    y_p = _cls_m.predict(day_eval[_lgbm_feature_cols])
    y_pr = _cls_m.predict_proba(day_eval[_lgbm_feature_cols])[:, 1]

    return {
        "pb_lo": pb_lo,
        "pb_up": pb_up,
        "picp": picp,
        "mpiw": mpiw,
        "cal_lo": cal_lo,
        "cal_up": cal_up,
        "y_cls": y_cls,
        "y_p": y_p,
        "y_pr": y_pr,
        "len": len(y_t),
    }


with tabs[1]:
    st.subheader("Model Reliability & Calibration")
    st.write(f"Metrics for **{sel_nav_date}**.")

    day_eval = df_data[df_data["timestamp"].dt.date == sel_nav_date]

    if day_eval.empty or len(day_eval) < 10:
        st.warning("Not enough data.")
    else:
        metrics = compute_model_metrics(
            str(sel_nav_date),
            day_eval,
            lo_m,
            up_m,
            cls_m,
            TARGET_FREQ_NEXT,
            TARGET_COL,
            LGBM_FEATURE_COLS,
            QUANTILE_ALPHAS,
        )

        if metrics is None:
            st.warning("Not enough valid data for metrics.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Pinball Loss (α=0.1)", f"{metrics['pb_lo']:.6f}")
            m2.metric("Pinball Loss (α=0.9)", f"{metrics['pb_up']:.6f}")
            m3, m4 = st.columns(2)
            m3.metric("PICP (80% CI)", f"{metrics['picp']:.2%}")
            m4.metric("MPIW", f"{metrics['mpiw']:.5f} Hz")
            m5, m6 = st.columns(2)
            cal_lo_pass = metrics["cal_lo"] <= 0.15
            cal_up_pass = metrics["cal_up"] >= 0.85
            m5.metric(
                "Calibration α=0.1 (target: ≤15%)",
                f"{metrics['cal_lo']:.2%}",
                delta="✓ OK" if cal_lo_pass else "⚠ high",
                delta_color="normal" if cal_lo_pass else "inverse",
            )
            m6.metric(
                "Calibration α=0.9 (target: ≥85%)",
                f"{metrics['cal_up']:.2%}",
                delta="✓ OK" if cal_up_pass else "⚠ low",
                delta_color="normal" if cal_up_pass else "inverse",
            )
            st.metric("Samples", f"{metrics['len']:,}")
            if metrics["picp"] >= 0.80:
                st.success("✅ PICP meets 80% target.")
            else:
                st.warning(f"⚠️ PICP ({metrics['picp']:.2%}) below 80%.")

            st.markdown("---")
            st.subheader("LightGBM Classifier — Confusion Matrix")
            from sklearn.metrics import (
                confusion_matrix,
                classification_report,
                roc_auc_score,
                roc_curve,
            )

            y_cls = metrics["y_cls"]
            y_p = metrics["y_p"]
            y_pr = metrics["y_pr"]
            cm = confusion_matrix(y_cls, y_p)
            cr = classification_report(
                y_cls, y_p, target_names=["Stable", "Unstable"], output_dict=True
            )
            cm_df = pd.DataFrame(
                cm,
                index=["Act. Stable", "Act. Unstable"],
                columns=["Pred. Stable", "Pred. Unstable"],
            )
            st.dataframe(cm_df, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            cr_dict = cr  # type: ignore[assignment]
            c1.metric("Precision (Unstable)", f"{cr_dict['Unstable']['precision']:.4f}")
            c2.metric("Recall (Unstable)", f"{cr_dict['Unstable']['recall']:.4f}")
            c3.metric("F1 (Unstable)", f"{cr_dict['Unstable']['f1-score']:.4f}")
            try:
                st.metric("AUC-ROC", f"{roc_auc_score(y_cls, y_pr):.4f}")
            except ValueError:
                st.info("AUC-ROC unavailable — single class.")

            st.markdown("---")
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_cls, y_pr)
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line=dict(color="#00CCFF", width=2),
                    name=f"ROC (AUC={roc_auc_score(y_cls, y_pr):.4f})",
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                    name="Random",
                )
            )
            fig_roc.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=50, r=20, t=30, b=40),
                xaxis=dict(
                    title="False Positive Rate", gridcolor="rgba(100,160,255,0.06)"
                ),
                yaxis=dict(
                    title="True Positive Rate", gridcolor="rgba(100,160,255,0.06)"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,26,0.3)",
                legend=dict(x=0.02, y=0.98),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.72rem;color:#445;'>"
    f"GridGuardian Command Deck v3 · Quantile Regression · "
    f"{TTA_SECONDS}s horizon · All times UTC"
    "</div>",
    unsafe_allow_html=True,
)
