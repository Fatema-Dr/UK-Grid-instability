import streamlit as st
import pandas as pd
import polars as pl
import joblib
import plotly.graph_objects as go
import numpy as np
import os
import shap
from src.config import TTA_SECONDS, LGBM_FEATURE_COLS, WEATHER_API_DEFAULT_START_DATE, WEATHER_API_DEFAULT_END_DATE
from src.data_loader import fetch_frequency_data, fetch_inertia_data, fetch_weather_data
from src.feature_engineering import create_features #
from datetime import date # Import date for date_input widget

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
# -----------------------------------------------------------------------------
# Using st.cache_data for this function as it depends on user input dates.
# The hash will change when dates change, triggering a re-run.
@st.cache_data(show_spinner="Loading models and data for selected date range...")
def load_resources_and_data(start_date_str: str, end_date_str: str):
    # Define paths
    lower_model_path = "notebooks/lgbm_quantile_lower.pkl"
    upper_model_path = "notebooks/lgbm_quantile_upper.pkl"
    
    # Check for models
    required_models = [lower_model_path, upper_model_path]
    missing = [f for f in required_models if not os.path.exists(f)]
    if missing:
        st.error(f"🚨 MISSING MODEL FILES ERROR: {', '.join(missing)}")
        st.warning("Please ensure you have run the `data_ingestion.ipynb` notebook to generate the models.")
        st.stop()

    # Load Models
    lower_model = joblib.load(lower_model_path)
    upper_model = joblib.load(upper_model_path)
    
    # --- Data Ingestion via API ---
    # Fetch frequency data
    df_freq = fetch_frequency_data(
        start_date=start_date_str,
        end_date=end_date_str
    )
    if df_freq.is_empty():
        st.error("🚨 Failed to load frequency data from API. Please check your date range or API connection.")
        st.stop()

    # Fetch weather data
    df_weather = fetch_weather_data(
        start_date=start_date_str,
        end_date=end_date_str
    )
    if df_weather.is_empty():
        st.error("🚨 Failed to load weather data from API. Please check your date range or API connection.")
        st.stop()

    # Fetch inertia data
    df_inertia = fetch_inertia_data(
        start_date=start_date_str,
        end_date=end_date_str
    )
    if df_inertia.is_empty():
        st.error("🚨 Failed to load inertia data from API. Please check your date range or API connection.")
        st.stop()

    # --- Merge Datasets ---
    df_freq = df_freq.sort("timestamp")
    df_weather = df_weather.sort("timestamp")
    df_inertia = df_inertia.sort("timestamp_date")

    df_merged = df_freq.join_asof(
        df_weather,
        on="timestamp",
        strategy="backward"
    )
    df_inertia = df_inertia.with_columns(
        pl.col("timestamp_date").cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias("timestamp")
    )
    #st.write(f"DEBUG: df_merged shape after joining with df_weather: {df_merged.shape}")
    #st.write(f"DEBUG: df_merged null count before joining with df_inertia: {df_merged.null_count()}")

    df_merged = df_merged.join_asof(
        df_inertia.select(["timestamp", "inertia_cost"]),
        on="timestamp",
        strategy="backward"
    )
    #st.write(f"DEBUG: df_merged shape after joining with df_inertia: {df_merged.shape}")
    #st.write(f"DEBUG: df_merged null count before drop_nulls(): {df_merged.null_count()}")
    
    df_merged = df_merged.drop_nulls().to_pandas() # Convert to Pandas DataFrame for SHAP and joblib models
    #st.write(f"DEBUG: df_merged Pandas shape after drop_nulls() and to_pandas(): {df_merged.shape}")
    
    # --- Feature Engineering ---
    df_data = create_features(df_merged)

    if df_data.empty:
        st.error("🚨 After merging and feature engineering, no data remains. Please adjust your date range or data sources.")
        st.stop()
    #st.write(f"DEBUG: df_data shape after create_features: {df_data.shape}")

    # Convert timestamp to datetime if not already (after feature engineering)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    
    # Create explainer for the lower bound model
    explainer = shap.TreeExplainer(lower_model)
    
    return lower_model, upper_model, df_data, explainer


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ GridGuardian Controls")
st.sidebar.subheader("Data Selection")

# Default dates for the date picker
default_start_date = pd.to_datetime(WEATHER_API_DEFAULT_START_DATE).date()
default_end_date = pd.to_datetime(WEATHER_API_DEFAULT_END_DATE).date()

# Date range pickers
col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    selected_start_date = st.date_input("Start Date", value=default_start_date)
with col_date2:
    selected_end_date = st.date_input("End Date", value=default_end_date)

# Load resources based on selected dates
lower_model, upper_model, df_data, explainer = load_resources_and_data(
    selected_start_date.strftime("%Y-%m-%d"),
    selected_end_date.strftime("%Y-%m-%d")
)

# Defensive check: Ensure df_data is not empty after loading
if df_data.empty:
    st.error("🚨 No data available for the selected date range after loading and processing. Please select a different range.")
else: # Only proceed if df_data is not empty
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")

    # Allow user to "scrub" through time to find the blackout
    # Default value for slider will be roughly the middle of the loaded data
    default_slider_value = int(len(df_data) * 0.5)
    slider_max_value = max(0, len(df_data) - 1) # Ensure max value is at least 0

    slider_value = st.sidebar.slider(
        "Simulation Time (relative index)", 0, slider_max_value, default_slider_value, 1
    )

    # Add a text input for precise time navigation
    time_input = st.sidebar.text_input("Go to time (HH:MM:SS UTC)", value="")

    # Add a slider for TTA_SECONDS
    tta_seconds_user = st.sidebar.slider(
        "Time to Alert (seconds ahead)", 5, 60, TTA_SECONDS, 5
    )
    # Override TTA_SECONDS with user's selection
    TTA_SECONDS = tta_seconds_user

    # Determine the final time_index based on text input or slider
    time_index = slider_value
    if time_input:
        try:
            # Assuming the time input is for the first day of the selected range, for simplicity
            # This part might need more robust handling if multi-day selection is fully supported
            target_time_str = f"{selected_start_date.strftime('%Y-%m-%d')} {time_input}"
            target_datetime = pd.to_datetime(target_time_str).tz_localize('UTC')
            
            # Find the closest index
            # Check if df_data['timestamp'] is empty before calling argmin
            if not df_data['timestamp'].empty:
                closest_idx = (df_data['timestamp'] - target_datetime).abs().argmin()
                # Ensure closest_idx is within bounds
                if 0 <= closest_idx < len(df_data):
                    time_index = closest_idx
                else:
                    st.sidebar.warning("Calculated time index out of bounds, using slider value.")
            else:
                st.sidebar.warning("Cannot search for time input as timestamp data is empty.")
            
            # Update slider value for consistency, but this might be an issue with Streamlit's state management
            # For now, let's just use time_index from text input if valid
        except Exception as e:
            st.sidebar.error(f"Invalid time format: {e}. Please use HH:MM:SS.")

    current_row = get_current_row_safely(df_data, time_index)

    if current_row is None:
        st.stop() # Stop further execution if current_row could not be retrieved safely

    current_time = current_row['timestamp']

    st.sidebar.markdown("---")
    st.sidebar.subheader("Alerting Threshold")
    alert_threshold_hz = st.sidebar.slider("Instability Threshold (Hz)", 49.5, 49.9, 49.8, 0.05)


    # -----------------------------------------------------------------------------
    # 4. PREDICTION LOGIC
    # -----------------------------------------------------------------------------
    input_lgbm = pd.DataFrame([current_row[LGBM_FEATURE_COLS].values], columns=LGBM_FEATURE_COLS)
    lower_bound_pred = lower_model.predict(input_lgbm)[0]
    upper_bound_pred = upper_model.predict(input_lgbm)[0]

    is_alert = lower_bound_pred < alert_threshold_hz


    # -----------------------------------------------------------------------------
    # 5. MAIN DASHBOARD UI
    # -----------------------------------------------------------------------------
    st.title(f"Control Room: {current_time.strftime('%H:%M:%S UTC')}")

    # Top KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Grid Frequency", f"{current_row['grid_frequency']:.3f} Hz", f"{current_row['rocof']:.4f} Hz/s", delta_color="inverse")
    col2.metric("Predicted Lower Bound", f"{lower_bound_pred:.3f} Hz")
    col3.metric("Predicted Upper Bound", f"{upper_bound_pred:.3f} Hz")

    if is_alert:
        col4.error(f"⚠️ INSTABILITY ALERT")
        col5.metric("Time to Alert", f"{TTA_SECONDS} sec", "⚠️")
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

            # Uncertainty Band
            fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=upper_preds, fill=None, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=lower_preds, fill='tonexty', mode='lines', line_color='rgba(255,255,255,0)', fillcolor='rgba(255, 165, 0, 0.2)', name='Uncertainty Band'))
            
            # Actual Frequency Line
            fig.add_trace(go.Scatter(x=chart_data['timestamp'], y=chart_data['grid_frequency'], mode='lines', name='Actual Frequency', line=dict(color='#00CCFF', width=2)))
            
            # Current Point Marker
            fig.add_trace(go.Scatter(x=[current_time], y=[current_row['grid_frequency']], mode='markers', marker=dict(color='red', size=10), name='Current Time'))

            fig.update_layout(template="plotly_dark", yaxis_title="Frequency (Hz)", margin=dict(l=0, r=0, t=0, b=0), height=400, yaxis_range=[49.4, 50.4])
            st.plotly_chart(fig, width='stretch')

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

        # Positive SHAP values push prediction higher, negative push lower.
        # For instability (low frequency), we are interested in what pushes the prediction DOWN.
        contrib_df['Color'] = np.where(contrib_df["SHAP Value"] < 0, '#FF4B4B', '#00CCFF') # Red for negative drivers, blue for positive
        
        fig_bar = go.Figure(go.Bar(
            x=contrib_df["SHAP Value"],
            y=contrib_df["Factor"],
            orientation='h',
            marker_color=contrib_df['Color']
        ))
        
        fig_bar.update_layout(
            template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0),
            title="Red bars are pushing the prediction lower (increasing risk)",
            yaxis=dict(autorange="reversed"),
            xaxis_title="SHAP Value (impact on frequency prediction)"
        )
        st.plotly_chart(fig_bar, width='stretch')

    st.markdown("---")
    st.info("The dashboard is now running in **Quantile Regression Mode**. The alert is triggered when the predicted lower bound of the frequency drops below the set threshold.")