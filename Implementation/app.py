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
    TTA_SECONDS, LGBM_FEATURE_COLS, LSTM_FEATURE_COLS, LSTM_TIME_STEPS,
    WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE,
    TARGET_FREQ_NEXT, TARGET_COL, QUANTILE_ALPHAS,
    LOWER_CALIBRATOR_PATH, UPPER_CALIBRATOR_PATH,
    LGBM_QUANTILE_LOWER_PATH, LGBM_QUANTILE_UPPER_PATH,
    LSTM_MODEL_PATH, SCALER_PATH, LGBM_MODEL_PATH
)
from src.data_loader import fetch_frequency_data, fetch_inertia_data_halfhourly, fetch_weather_data
from src.feature_engineering import create_features, merge_datasets
from datetime import date

# -----------------------------------------------------------------------------
# 0. CACHE & HASHING HELPERS
# -----------------------------------------------------------------------------
CACHE_DIR = Path("data/processed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_src_hash():
    """Calculates a hash of all python files in the src directory to detect logic changes."""
    hasher = hashlib.md5()
    src_path = Path("src")
    # Sort files to ensure consistent hashing
    for file_path in sorted(src_path.glob("**/*.py")):
        if "__pycache__" in str(file_path):
            continue
        with open(file_path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()

def clear_invalid_cache(current_hash: str):
    """Deletes cache files that don't match the current source code hash."""
    for cache_file in CACHE_DIR.glob("*.parquet"):
        if current_hash not in cache_file.name:
            try:
                cache_file.unlink()
            except Exception as e:
                st.warning(f"Could not delete old cache file {cache_file}: {e}")

# Helper function to safely get current row
def get_current_row_safely(df_data_local, time_index_local):
    if df_data_local.empty:
        st.error("🚨 Attempted to access row from empty data. This should not happen. Please reload the app or adjust your date range.")
        return None
    if not (0 <= time_index_local < len(df_data_local)):
        st.error(f"🚨 Internal Error: time_index ({time_index_local}) is out of bounds for df_data (length {len(df_data_local)}). Please report this issue.")
        return None
    return df_data_local.iloc[time_index_local]

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GridGuardian v2: UK Frequency Alert System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Mode" Control Room feel
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        .stMetric {background-color: #1E1E1E; border: 1px solid #333; padding: 10px; border-radius: 5px;}
        div[data-testid="metric-container"] > label {color: #888;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD RESOURCES
# ----------------------------------------------------------------------# -----------------------------------------------------------------------------
# 2. CACHED LOADING FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_resource
def load_models(model_stamp: tuple):
    """Load model objects. Re-runs ONLY if model files on disk change."""
    lower_model = joblib.load(LGBM_QUANTILE_LOWER_PATH)
    upper_model = joblib.load(LGBM_QUANTILE_UPPER_PATH)
    classifier_model = joblib.load(LGBM_MODEL_PATH)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    quantiles_all_path = "notebooks/lgbm_quantiles_all.pkl"
    quantile_models = joblib.load(quantiles_all_path) if os.path.exists(quantiles_all_path) else None
    
    return lower_model, upper_model, classifier_model, lstm_model, scaler, quantile_models

@st.cache_data(show_spinner="Loading and processing dataset...")
def load_processed_data(start_date_str: str, end_date_str: str, src_hash: str):
    """Fetch and process data. Re-runs if dates or src/ code changes."""
    # Logic to fetch from API and merge
    df_freq = fetch_frequency_data(start_date_str, end_date_str)
    df_weather = fetch_weather_data(start_date_str, end_date_str)
    df_inertia = fetch_inertia_data_halfhourly(start_date_str, end_date_str)
    
    df_merged_pl = merge_datasets(df_freq, df_weather, df_inertia)
    df_merged = df_merged_pl.to_pandas()
    df_data = create_features(df_merged)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    
    # Pre-calculate calibrators and explainer
    lower_calibrator, upper_calibrator = None, None
    if os.path.exists(LOWER_CALIBRATOR_PATH) and os.path.exists(UPPER_CALIBRATOR_PATH):
        lower_calibrator = joblib.load(LOWER_CALIBRATOR_PATH)
        upper_calibrator = joblib.load(UPPER_CALIBRATOR_PATH)
    
    # We use a placeholder for the explainer as it depends on the loaded model
    # (will be initialized at top level)
    return df_data, lower_calibrator, upper_calibrator


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ GridGuardian Controls")
st.sidebar.subheader("Data Selection")

# Timezone Note for Dissertation
st.sidebar.info("🌐 **Timezone Note**: Dashboard operates in **UTC**. The Aug 9, 2019 event occurred at **15:52 UTC** (16:52 BST).")

default_start_date = pd.to_datetime(WEATHER_API_DEFAULT_START_DATE).date()
default_end_date = pd.to_datetime(WEATHER_API_DEFAULT_END_DATE).date()

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    selected_start_date = st.date_input("Start Date", value=default_start_date)
with col_date2:
    selected_end_date = st.date_input("End Date", value=default_end_date)

# --- EXECUTE LOADING ---
# 1. Models (Keyed by file timestamps)
model_stamp = (
    os.path.getmtime(LGBM_QUANTILE_LOWER_PATH),
    os.path.getmtime(LGBM_QUANTILE_UPPER_PATH),
    os.path.getmtime(LGBM_MODEL_PATH),
    os.path.getmtime(LSTM_MODEL_PATH),
    os.path.getmtime(SCALER_PATH),
)
lower_model, upper_model, classifier_model, lstm_model, scaler, quantile_models = load_models(model_stamp)

# 2. Data (Keyed by dates and code version)
df_data, lower_calibrator, upper_calibrator = load_processed_data(
    selected_start_date.strftime("%Y-%m-%d"),
    selected_end_date.strftime("%Y-%m-%d"),
    get_src_hash()
)

# 3. Explainer (Static per session)
explainer = shap.TreeExplainer(lower_model)

# Defensive check: Ensure df_data is not empty after loading
if df_data.empty:
    st.error("🚨 No data available for the selected date range after loading and processing. Please select a different range.")
else: # Only proceed if df_data is not empty
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱ Time Navigation")

    # 1. Date picker for day selection
    available_dates = df_data['timestamp'].dt.date.unique().tolist()
    available_dates.sort()
    
    if "nav_date" not in st.session_state or st.session_state.nav_date not in available_dates:
        st.session_state.nav_date = available_dates[len(available_dates) // 2]
    
    selected_nav_date = st.sidebar.date_input(
        "📅 Jump to Date",
        value=st.session_state.nav_date,
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )
    
    # Reset time slider if date changed
    if selected_nav_date != st.session_state.nav_date:
        st.session_state.nav_date = selected_nav_date
        if "time_nav_index" in st.session_state:
            del st.session_state["time_nav_index"]

    # 2. Filter data to selected date
    day_mask = df_data['timestamp'].dt.date == selected_nav_date
    global_indices_for_day = np.where(day_mask)[0].tolist()
    day_timestamps = df_data['timestamp'].iloc[global_indices_for_day].tolist()
    
    if not global_indices_for_day:
        st.sidebar.warning(f"No data for {selected_nav_date}")
        st.stop()

    default_day_idx = len(global_indices_for_day) // 2
    if "time_nav_index" not in st.session_state:
        st.session_state.time_nav_index = default_day_idx
        
    # Ensure index is within bounds of current day
    st.session_state.time_nav_index = min(max(0, st.session_state.time_nav_index), len(global_indices_for_day) - 1)

    # 3. Step buttons (Simplified: ±5s and ±1s)
    btn_col1, btn_col2, btn_col3, btn_col4 = st.sidebar.columns(4)
    with btn_col1:
        if st.button("◀ -5s"):
            st.session_state.time_nav_index = max(0, st.session_state.time_nav_index - 5)
            st.rerun()
    with btn_col2:
        if st.button("◀ -1s"):
            st.session_state.time_nav_index = max(0, st.session_state.time_nav_index - 1)
            st.rerun()
    with btn_col3:
        if st.button("+1s ▶"):
            st.session_state.time_nav_index = min(len(global_indices_for_day) - 1, st.session_state.time_nav_index + 1)
            st.rerun()
    with btn_col4:
        if st.button("+5s ▶"):
            st.session_state.time_nav_index = min(len(global_indices_for_day) - 1, st.session_state.time_nav_index + 5)
            st.rerun()

    # Autoplay toggle
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False

    autoplay_col1, autoplay_col2 = st.sidebar.columns(2)
    with autoplay_col1:
        if st.button("▶ Play" if not st.session_state.autoplay else "⏸ Pause"):
            st.session_state.autoplay = not st.session_state.autoplay
            st.rerun()
    with autoplay_col2:
        autoplay_speed = st.selectbox("Speed", [1, 5, 10, 30, 60], index=0, label_visibility="collapsed")

    if st.session_state.autoplay:
        import time
        time.sleep(1.0)
        st.session_state.time_nav_index = min(
            len(global_indices_for_day) - 1,
            st.session_state.time_nav_index + autoplay_speed
        )
        if st.session_state.time_nav_index >= len(global_indices_for_day) - 1:
            st.session_state.autoplay = False  # Stop at end of day
        st.rerun()

    # 4. Exact Time Input (Replaces heavy slider)
    current_time_val = day_timestamps[st.session_state.time_nav_index].time()
    
    selected_time = st.sidebar.time_input("⌨️ Exact Time (UTC)", value=current_time_val, step=60)
    
    if selected_time != current_time_val:
        # Find closest index to the inputted time
        target_dt = pd.Timestamp.combine(selected_nav_date, selected_time)
        series_ts = pd.Series(day_timestamps)
        
        # safely handle timezone if present
        if hasattr(series_ts.iloc[0], 'tzinfo') and series_ts.iloc[0].tzinfo is not None:
             target_dt = target_dt.tz_localize(series_ts.iloc[0].tzinfo)
             
        # Calculate absolute difference and get index of smallest difference
        closest_idx = int((series_ts - target_dt).abs().idxmin())
        
        if st.session_state.time_nav_index != closest_idx:
            st.session_state.time_nav_index = closest_idx
            st.rerun()
    
    # Map back to global index for the rest of the app
    time_index = global_indices_for_day[st.session_state.time_nav_index]

    # Caption showing the full selected timestamp
    sel_ts = day_timestamps[st.session_state.time_nav_index]
    st.sidebar.caption(f"📍 **{sel_ts.strftime('%Y-%m-%d  %H:%M:%S')} UTC**")

    current_row = get_current_row_safely(df_data, time_index)

    if current_row is None:
        st.stop()  # Stop further execution if current_row could not be retrieved safely

    current_time = current_row['timestamp']

    # --- Alert Configuration (TTA + Threshold grouped) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Alert Configuration")
    
    tta_seconds_user = st.sidebar.slider(
        "Time to Alert (seconds ahead)", 5, 60, TTA_SECONDS, 5
    )
    st.sidebar.caption(f"ℹ️ Model trained at {TTA_SECONDS}s horizon.")

    alert_threshold_hz = st.sidebar.slider("Instability Threshold (Hz)", 49.5, 49.95, 49.85, 0.05)

    # --- Intervention Simulator ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕹️ Intervention Simulator")
    synthetic_inertia_mw = st.sidebar.slider(
        "Inject Synthetic Inertia (MW)", 0, 5000, 0, 100,
        help="Simulate adding battery or demand response to improve grid stability."
    )

    # -----------------------------------------------------------------------------
    # 4. PREDICTION LOGIC (Calculated early for UI feedback)
    # -----------------------------------------------------------------------------
    input_lgbm = pd.DataFrame([current_row[LGBM_FEATURE_COLS].values], columns=LGBM_FEATURE_COLS)

    # Swing-equation constants for physics-informed intervention
    SYSTEM_INERTIA_H = 4.0    # Typical UK grid inertia constant (seconds)
    NOMINAL_FREQ = 50.0       # Nominal grid frequency (Hz)
    TOTAL_SYSTEM_CAPACITY = 35000  # Approximate UK system capacity (MW)

    # Apply Intervention Simulator logic using the Swing Equation:
    # Δf = (ΔP × f₀) / (2 × H × S_base)
    swing_delta_f = 0.0
    if synthetic_inertia_mw > 0:
        # 1. Physics-based frequency uplift from the swing equation
        swing_delta_f = (synthetic_inertia_mw * NOMINAL_FREQ) / (2 * SYSTEM_INERTIA_H * TOTAL_SYSTEM_CAPACITY)
        # 2. Adjust the renewable penetration proxy
        intervention_effect = (synthetic_inertia_mw / TOTAL_SYSTEM_CAPACITY)
        input_lgbm['renewable_penetration_ratio'] = np.maximum(0, input_lgbm['renewable_penetration_ratio'] - intervention_effect)

    lower_bound_raw = lower_model.predict(input_lgbm)[0]
    upper_bound_raw = upper_model.predict(input_lgbm)[0]
    
    # Apply calibration if calibrators are available
    if lower_calibrator is not None and upper_calibrator is not None:
        from src.calibration import calibrate_predictions
        lower_bound_raw = calibrate_predictions(lower_calibrator, np.array([lower_bound_raw]))[0]
        upper_bound_raw = calibrate_predictions(upper_calibrator, np.array([upper_bound_raw]))[0]
    
    lower_bound_pred = lower_bound_raw + swing_delta_f
    upper_bound_pred = upper_bound_raw + swing_delta_f

    # LSTM Prediction for Residual Monitoring
    window_start_idx = max(0, time_index - LSTM_TIME_STEPS + 1)
    lstm_input_df = df_data.iloc[window_start_idx : time_index + 1]
    
    lstm_alert = False
    lstm_prob = 0.0
    if len(lstm_input_df) == LSTM_TIME_STEPS:
        lstm_input_scaled = scaler.transform(lstm_input_df[LSTM_FEATURE_COLS])
        lstm_input_seq = np.array([lstm_input_scaled])
        lstm_prob = lstm_model.predict(lstm_input_seq, verbose=0)[0][0]

    # (uses classifier as primary signal):
    input_lgbm_cls = pd.DataFrame([current_row[LGBM_FEATURE_COLS].values], columns=LGBM_FEATURE_COLS)
    classifier_prob = classifier_model.predict_proba(input_lgbm_cls)[0][1]
    
    # --- Robust Proactive Alert Logic ---
    freq_now = current_row['grid_frequency']
    
    # 1. Emergency (Red): Imminent Blackout Risk
    # Trigger if breach is predicted OR if high-probability instability is detected while freq < 50.1
    emergency_trigger = (
        lower_bound_pred < alert_threshold_hz 
        or freq_now < alert_threshold_hz
        or (lstm_prob > 0.4 and freq_now < 50.1)
        or (classifier_prob > 0.4 and freq_now < 50.1)
    )
    
    # 2. Warning (Yellow): General Instability / High Frequency
    warning_trigger = (
        freq_now > 50.15
        or (classifier_prob > 0.25)
        or (lstm_prob > 0.25)
    )

    # --- Alert Persistence Logic ---
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = None
    
    if emergency_trigger:
        st.session_state.last_alert_time = current_time
    
    persistent_alert = False
    if st.session_state.last_alert_time is not None:
        time_since_alert = (current_time - st.session_state.last_alert_time).total_seconds()
        if 0 <= time_since_alert <= 30:
            persistent_alert = True

    # --- Sidebar UI components ---
    with st.sidebar.expander("🔍 Alert Debugging"):
        st.write(f"Threshold: {alert_threshold_hz} Hz")
        st.write(f"Current Freq: {freq_now:.4f} Hz")
        st.write(f"LGBM Lower: {lower_bound_pred:.4f} Hz")
        st.write(f"Emergency Trigger: {'✅' if emergency_trigger else '❌'}")
        st.write(f"Warning Trigger: {'✅' if warning_trigger else '❌'}")
        st.write(f"LSTM Prob: {lstm_prob:.4f}")
        st.write(f"Classifier P(unstable): {classifier_prob:.4f}")
    
    if synthetic_inertia_mw > 0:
        st.sidebar.caption(f"⚡ Swing Eq: Δf = ({synthetic_inertia_mw} × {NOMINAL_FREQ}) / (2 × {SYSTEM_INERTIA_H} × {TOTAL_SYSTEM_CAPACITY}) = **+{swing_delta_f:.4f} Hz**")



    # -----------------------------------------------------------------------------
    # 5. MAIN DASHBOARD UI
    # -----------------------------------------------------------------------------
    st.title(f"Control Room: {current_time.strftime('%H:%M:%S UTC')}")

    # Tabs for Control Room and Model Health
    tab_main, tab_health = st.tabs(["Control Room", "Model Health"])

    with tab_main:
        # Top KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Grid Frequency", f"{current_row['grid_frequency']:.3f} Hz", f"{current_row['rocof']:.4f} Hz/s", delta_color="inverse")
        col2.metric("Predicted Lower Bound", f"{lower_bound_pred:.3f} Hz")
        col3.metric("Predicted Upper Bound", f"{upper_bound_pred:.3f} Hz")
    
        if persistent_alert and emergency_trigger:
            col4.error(f"⚠️ INSTABILITY ALERT")
            col5.metric("Time to Alert", f"{tta_seconds_user} sec", "⚠️")
        elif persistent_alert:
            col4.warning(f"⚠️ HIGH MODEL UNCERTAINTY")
            col5.metric("Time to Alert", f"{tta_seconds_user} sec", "⚠️")
        else:
            col4.success(f"✅ SYSTEM STABLE")
            col5.metric("Time to Alert", "N/A")

    # Main Layout: Graph + Interpretation
    col_main, col_xai = st.columns([2, 1])

    with col_main:
        st.subheader("Real-Time Frequency Monitor with Uncertainty Bands")
        
        window = 300 # seconds
        start_idx = max(0, time_index - window)
        # Ensure chart_data slicing is also safe
        end_idx = time_index + TTA_SECONDS + 1
        if end_idx > len(df_data):
            end_idx = len(df_data)
        
        chart_data = df_data.iloc[start_idx : end_idx] 
        
        if chart_data.empty:
            st.warning("Chart data is empty for the current window. Adjust time or date range.")
        else:
            # Predict bounds for the whole window
            window_input = chart_data[LGBM_FEATURE_COLS]
            # Ensure the order of columns matches LGBM_FEATURE_COLS to prevent warnings
            window_input = window_input.reindex(columns=LGBM_FEATURE_COLS)
            lower_preds = lower_model.predict(window_input)
            upper_preds = upper_model.predict(window_input)

            fig = go.Figure()
            
            # Alert Threshold line
            fig.add_hline(y=alert_threshold_hz, line_dash="dash", line_color="red", annotation_text="Alert Threshold")

            if quantile_models is not None:
                from src.calibration import calibrate_predictions
                alphas = sorted(quantile_models.keys())
                num_pairs = len(alphas) // 2
                for i in range(num_pairs):
                    a_low = alphas[i]
                    a_high = alphas[-(i+1)]
                    raw_low = quantile_models[a_low].predict(window_input)
                    raw_high = quantile_models[a_high].predict(window_input)
                    
                    # Apply calibration if we have matching calibrators (usually 0.1 and 0.9)
                    if a_low == 0.10 and lower_calibrator:
                        pred_low = calibrate_predictions(lower_calibrator, raw_low)
                    else:
                        pred_low = raw_low
                        
                    if a_high == 0.90 and upper_calibrator:
                        pred_high = calibrate_predictions(upper_calibrator, raw_high)
                    else:
                        pred_high = raw_high

                    # Apply simple intervention logic delta
                    pred_low += swing_delta_f
                    pred_high += swing_delta_f
                    
                    opacity = 0.1 + (0.3 * (i / num_pairs))
                    
                    if i == 0:
                        name = f'{int((a_high - a_low)*100)}% CI'
                        fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=pred_high, fill=None, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False))
                        fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=pred_low, fill='tonexty', mode='lines', line_color='rgba(255,255,255,0)', fillcolor=f'rgba(255, 165, 0, {opacity})', name=name))
                    else:
                        fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=pred_high, fill=None, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False, hoverinfo='skip'))
                        fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=pred_low, fill='tonexty', mode='lines', line_color='rgba(255,255,255,0)', fillcolor=f'rgba(255, 165, 0, {opacity})', showlegend=False, hoverinfo='skip'))
            else:
                from src.calibration import calibrate_predictions
                raw_low = lower_model.predict(window_input)
                raw_high = upper_model.predict(window_input)
                
                if lower_calibrator and upper_calibrator:
                    lower_preds = calibrate_predictions(lower_calibrator, raw_low) + swing_delta_f
                    upper_preds = calibrate_predictions(upper_calibrator, raw_high) + swing_delta_f
                else:
                    lower_preds = raw_low + swing_delta_f
                    upper_preds = raw_high + swing_delta_f

                # Uncertainty Band
                fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=upper_preds, fill=None, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False))
                fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=lower_preds, fill='tonexty', mode='lines', line_color='rgba(255,255,255,0)', fillcolor='rgba(255, 165, 0, 0.2)', name='Uncertainty Band'))
            
            # Actual Frequency Line
            fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=chart_data['grid_frequency'], mode='lines', name='Actual Frequency', line=dict(color='#00CCFF', width=2)))
            
            # Current Point Marker
            fig.add_trace(go.Scatter(x=[current_time], y=[current_row['grid_frequency']], mode='markers', marker=dict(color='red', size=10), name='Current Time'))

            fig.update_layout(template="plotly_dark", yaxis_title="Frequency (Hz)", margin=dict(l=0, r=0, t=0, b=0), height=400, yaxis_range=[49.4, 50.4])
            st.plotly_chart(fig, width='stretch')

            # --- LSTM Probability Timeline ---
            st.caption("LSTM P(Unstable) Timeline")
            lstm_probs = []
            lstm_timestamps = []
            # Sample every 10th point to keep it fast
            step = max(1, len(chart_data) // 60)
            for i in range(0, len(chart_data), step):
                global_idx = start_idx + i
                w_start = max(0, global_idx - LSTM_TIME_STEPS + 1)
                lstm_window = df_data.iloc[w_start : global_idx + 1]
                if len(lstm_window) == LSTM_TIME_STEPS:
                    lstm_sc = scaler.transform(lstm_window[LSTM_FEATURE_COLS])
                    prob = lstm_model.predict(np.array([lstm_sc]), verbose=0)[0][0]
                    lstm_probs.append(float(prob))
                    lstm_timestamps.append(chart_data.iloc[i]['timestamp'])

            if lstm_timestamps:
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(
                    x=lstm_timestamps, y=lstm_probs,
                    mode='lines', name='P(Unstable)',
                    line=dict(color='#FF6B6B', width=2),
                    fill='tozeroy', fillcolor='rgba(255,107,107,0.15)'
                ))
                fig_lstm.add_hline(y=0.5, line_dash="dash", line_color="yellow", annotation_text="Alert Threshold (0.5)")
                fig_lstm.update_layout(
                    template="plotly_dark", height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis_title="P(Unstable)", yaxis_range=[0, 1]
                )
                st.plotly_chart(fig_lstm, width='stretch')

    with col_xai:
        st.subheader(f"Risk Drivers for Instability ({TTA_SECONDS}s Ahead)")
        
        shap_values = explainer.shap_values(input_lgbm)
        
        # Ensure shap_values is a 2D array for single sample regressors
        if isinstance(shap_values, list):
            # If it's a list (e.g., from multi-output or older shap versions), take the first element for a single regressor
            shap_values_for_plot = shap_values[0]
        else:
            # Otherwise, assume it's already the 2D array
            shap_values_for_plot = shap_values

        contrib_df = pd.DataFrame({
            "Factor": LGBM_FEATURE_COLS,
            "SHAP Value": shap_values_for_plot[0] # Access the first (and only) row for the single sample
        }).sort_values("SHAP Value", ascending=False)

        # Filter out dominant autoregressive features (grid_frequency and lags)
        # to reveal the underlying physics drivers (inertia, wind, etc.)
        mask = ~contrib_df["Factor"].str.contains("grid_frequency|lag_")
        contrib_plot_df = contrib_df[mask].copy()

        # Sort by absolute impact to show most important drivers first
        contrib_plot_df["Abs Impact"] = contrib_plot_df["SHAP Value"].abs()
        contrib_plot_df = contrib_plot_df.sort_values("Abs Impact", ascending=False).head(12)

        # Color mapping: Red for negative drivers (pushing freq lower/higher risk), blue for positive
        contrib_plot_df['Color'] = np.where(contrib_plot_df["SHAP Value"] < 0, '#FF4B4B', '#00CCFF')

        fig_bar = go.Figure(go.Bar(
            x=contrib_plot_df["SHAP Value"],
            y=contrib_plot_df["Factor"],
            orientation='h',
            marker_color=contrib_plot_df['Color']
        ))
        
        fig_bar.update_layout(
            template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0),
            title="Red bars are pushing the prediction lower (increasing risk)",
            yaxis=dict(autorange="reversed"),
            xaxis_title="SHAP Value (impact on frequency prediction)"
        )
        st.plotly_chart(fig_bar, width='stretch')

    with tab_health:
        st.subheader("Model Reliability & Calibration")
        st.write("This tab shows how well calibrated the uncertainty bands are for the selected day.")

        # --- Dynamic metric computation for the selected day ---
        day_data = df_data[df_data['timestamp'].dt.date == selected_nav_date].copy()
        day_eval = day_data.dropna(subset=[TARGET_FREQ_NEXT])

        if day_eval.empty or len(day_eval) < 10:
            st.warning("Not enough data on this day to compute calibration metrics.")
        else:
            with st.spinner("Computing calibration metrics for this day..."):
                y_true_day = day_eval[TARGET_FREQ_NEXT].values
                X_day = day_eval[LGBM_FEATURE_COLS]
                lower_day = lower_model.predict(X_day)
                upper_day = upper_model.predict(X_day)

                alpha_lower = 0.10
                alpha_upper = 0.90

                # Pinball Loss
                err_lower = y_true_day - lower_day
                pb_lower = np.mean(np.maximum(alpha_lower * err_lower, (alpha_lower - 1) * err_lower))
                err_upper = y_true_day - upper_day
                pb_upper = np.mean(np.maximum(alpha_upper * err_upper, (alpha_upper - 1) * err_upper))

                # PICP & MPIW
                covered = ((y_true_day >= lower_day) & (y_true_day <= upper_day)).astype(int)
                picp = np.mean(covered)
                mpiw = np.mean(upper_day - lower_day)

                # Calibration scores
                cal_lower = np.mean(y_true_day < lower_day)
                cal_upper = np.mean(y_true_day < upper_day)

                # Display metrics
                m1, m2 = st.columns(2)
                m1.metric("Pinball Loss (Lower α=0.1)", f"{pb_lower:.6f}")
                m2.metric("Pinball Loss (Upper α=0.9)", f"{pb_upper:.6f}")

                m3, m4 = st.columns(2)
                m3.metric("PICP (80% CI Coverage)", f"{picp:.2%}")
                m4.metric("MPIW (Band Width)", f"{mpiw:.5f} Hz")

                m5, m6 = st.columns(2)
                m5.metric("Calibration α=0.1 (target: 10%)", f"{cal_lower:.2%}")
                m6.metric("Calibration α=0.9 (target: 90%)", f"{cal_upper:.2%}")

                st.metric("Evaluation Samples", f"{len(y_true_day):,}")

                if picp >= 0.80:
                    st.success("✅ PICP meets the 80% target — uncertainty bands are well-calibrated.")
                else:
                    st.warning(f"⚠️ PICP ({picp:.2%}) is below 80% — bands may be too narrow or biased.")

        # --- LightGBM Classifier Performance ---
        st.markdown("---")
        st.subheader("LightGBM Classifier — Confusion Matrix")
        st.write("Classification performance of the LightGBM binary classifier for the selected day.")

        if day_eval.empty or len(day_eval) < 10:
            st.warning("Not enough data on this day to compute classifier metrics.")
        else:
            from sklearn.metrics import confusion_matrix, classification_report
            y_true_cls = day_eval[TARGET_COL].values
            y_pred_cls = classifier_model.predict(day_eval[LGBM_FEATURE_COLS])
            y_pred_proba_cls = classifier_model.predict_proba(day_eval[LGBM_FEATURE_COLS])[:, 1]

            cm = confusion_matrix(y_true_cls, y_pred_cls)
            cls_report = classification_report(y_true_cls, y_pred_cls, target_names=["Stable", "Unstable"], output_dict=True)

            # Display confusion matrix as a table
            cm_df = pd.DataFrame(cm, index=["Actual Stable", "Actual Unstable"], columns=["Pred Stable", "Pred Unstable"])
            st.dataframe(cm_df, use_container_width=True)

            # Display classification report as metrics
            cr1, cr2, cr3 = st.columns(3)
            cr1.metric("Precision (Unstable)", f"{cls_report['Unstable']['precision']:.4f}")
            cr2.metric("Recall (Unstable)", f"{cls_report['Unstable']['recall']:.4f}")
            cr3.metric("F1-Score (Unstable)", f"{cls_report['Unstable']['f1-score']:.4f}")

            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true_cls, y_pred_proba_cls)
                st.metric("AUC-ROC", f"{auc:.4f}")
            except ValueError:
                st.info("AUC-ROC unavailable — only one class present in this day's data.")

    st.markdown("---")
    st.info("The dashboard is now running in **Quantile Regression Mode**. The alert is triggered when the predicted lower bound of the frequency drops below the set threshold.")