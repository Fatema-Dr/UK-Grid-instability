### 1. Data Ingestion Pipeline

**Current Implementation Details:**
The data ingestion pipeline is responsible for gathering raw frequency, weather, and inertia data, and is primarily handled by functions in `src/data_loader.py` and orchestrated by `run_pipeline.py`.

*   **Grid Frequency Data:** Loaded from `data/f-2019-aug/f 2019 8.csv`. This is 1-second resolution data, where columns `dtm` (datetime) and `f` (frequency) are renamed to `timestamp` and `grid_frequency` respectively. The `timestamp` is parsed with a specific format (`%Y-%m-%d %H:%M:%S %z`) to include timezone awareness.
*   **Weather Data:** Fetched hourly from the Open-Meteo API for an approximate UK center (Lat 54.0, Lon -2.0) for a specified date range (e.g., "2019-08-01" to "2019-08-31"). It includes hourly variables such as temperature, precipitation, wind speed, solar radiation. `requests_cache.CachedSession` is used for caching API calls. The fetched Pandas DataFrame is converted to Polars, with timestamps localized to "Europe/London" then converted to "UTC" and cast to microseconds for consistency.
*   **System Inertia Data:** Loaded from `data/inertia_costs19.csv`. This is daily data, where "Settlement Date" is parsed as `pl.Date` (format `%d/%m/%Y`) and renamed `Cost` to `inertia_cost`. The date is then cast to `pl.Datetime` at midnight UTC.
*   **Tools:** Polars library is used extensively for data manipulation and `openmeteo-requests` for API interaction.

**How it should be implemented (Detailed Steps):**

1.  **Configuration (`src/config.py`):**
    *   `DATA_DIR`: Base directory for raw data.
    *   `FREQUENCY_DATA_FILE`, `INERTIA_DATA_FILE`: Paths to CSV files.
    *   API specific parameters like `WEATHER_API_LATITUDE`, `WEATHER_API_LONGITUDE`, `WEATHER_API_TIMEZONE`, and `WEATHER_API_HOURLY_VARS` should ideally be defined here for easy modification. (Currently, lat/lon are hardcoded in `src/data_loader.py`).
    *   Date ranges for fetching weather data (`start_date`, `end_date`) can also be configured.
2.  **`load_frequency_data(filepath)` (`src/data_loader.py`):**
    *   Use `pl.scan_csv(filepath, ignore_errors=True, truncate_ragged_lines=True)` for efficient and robust reading of potentially malformed CSVs.
    *   `q.rename({"dtm": "timestamp", "f": "grid_frequency"})` ensures consistent column names.
    *   `q.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S %z"))` correctly parses and types the datetime column, assuming the specific format of the raw data.
    *   `q.sort("timestamp")` is crucial for time-series operations and subsequent `join_asof`.
    *   `q.collect()` materializes the lazy query into a DataFrame.
3.  **`fetch_weather_data(start_date, end_date)` (`src/data_loader.py`):**
    *   Initialize `requests_cache.CachedSession` for performance, preventing redundant API calls.
    *   Construct the Open-Meteo API `url` and `params` dictionary, including configured latitude, longitude, and hourly variables.
    *   Execute `openmeteo_requests.Client(session).weather_api(url, params)`.
    *   Extract hourly data and create a Pandas DataFrame.
    *   `pd.Series.tz_localize("Europe/London")` followed by `pl.from_pandas()` and `pl.col("timestamp").dt.convert_time_zone("UTC").cast(pl.Datetime("us", time_zone="UTC"))` is a critical step for standardizing timestamps to UTC microseconds, enabling seamless merging across datasets.
4.  **`load_inertia_data(filepath)` (`src/data_loader.py`):**
    *   Use `pl.read_csv(filepath)`.
    *   `df.with_columns(pl.col("Settlement Date").str.strptime(pl.Date, format="%d/%m/%Y").alias("timestamp_date"))` parses the date string to a Polars Date type.
    *   `df.with_columns(pl.col("timestamp_date").cast(pl.Datetime).dt.replace_time_zone("UTC"))` converts the daily date to a UTC datetime at midnight, aligning it for subsequent joins.
    *   `df.rename({"Cost": "inertia_cost"})` assigns a clear name to the cost column.

**Necessary and Relevant Improvements:**

*   **Centralized API Parameterization:** Move `latitude`, `longitude`, `timezone`, and `hourly` variables for Open-Meteo API from `src/data_loader.py` to `src/config.py`. This enhances maintainability and allows for easy experimentation with different weather locations or data.
*   **Comprehensive Data Validation:** Implement checks for expected column presence, data types, and realistic value ranges after each data load step. For example, ensure `grid_frequency` is always between 49-51 Hz. Raise warnings or errors for deviations.
*   **Scalable API Handling:** For production, consider robust API rate limiting, more sophisticated retry logic (if not fully covered by `retry_call`), and monitoring of API usage.
*   **Parallel Data Loading:** For large numbers of distinct data files (though not immediately apparent here), implement parallel loading using Polars' lazy evaluation or other multiprocessing techniques.
*   **Data Lineage/Versioning:** Integrate basic data lineage tracking (e.g., logging exact API call parameters, file hashes) to ensure reproducibility.

---

### 2. Feature Engineering (wind_ramp_event, inertia_costs)

**Current Implementation Details:**
Feature engineering combines the ingested data and creates new, more informative variables for the models. This is primarily done in `src/feature_engineering.py`.

*   **Merging Datasets (`merge_datasets`):**
    1.  All input DataFrames (`df_freq`, `df_weather`, `df_inertia`) are sorted by their respective timestamp columns.
    2.  `df_freq` is joined with `df_weather` using `join_asof(..., on="timestamp", strategy="backward")`. This effectively forward-fills hourly weather data onto the 1-second frequency data.
    3.  The result is then joined with `df_inertia` (after renaming `timestamp_date` to `timestamp`) using `join_asof(..., on="timestamp", strategy="backward")`, similarly forward-filling daily inertia costs.
    4.  Finally, `drop_nulls()` removes any rows where initial joins might have failed (e.g., very beginning of the dataset before any weather/inertia data is available).
*   **Wind Ramp Rate (`calculate_wind_ramp_rate` utilizing `src/opsda.py`):**
    1.  The `calculate_wind_ramp_rate` function extracts unique `timestamp` and `wind_speed` pairs from the merged data.
    2.  `timestamp` is converted to `unix_ts` (seconds).
    3.  Uses `opsda.compress` with `width=0.5`. This algorithm identifies "significant" changes (ramps) in `wind_speed` over time.
    4.  The compressed output (irregular `(unix_ts, wind_speed)` tuples) is converted back to a Polars DataFrame.
    5.  `wind_ramp_rate` is calculated as the `diff()` of `wind_speed` divided by the `diff()` of `timestamp` (converted to total seconds) between these compressed points.
    6.  This `wind_ramp_rate` DataFrame is then joined back to the main DataFrame using `join_asof(..., on="timestamp", strategy="backward")`.
    7.  Crucially, after joining, `pl.col("wind_ramp_rate").forward_fill()` is applied to propagate the calculated ramp rate across all 1-second intervals until the next detected ramp event.
*   **Inertia Costs (`inertia_cost`):** The `inertia_cost` feature is directly incorporated after the `join_asof` operation with the `df_inertia` DataFrame. It represents the daily cost of procuring inertia services.
*   **Other Core Features (`create_features`):**
    *   **RoCoF:** `pl.col("grid_frequency").diff().fill_null(0)`
    *   **Volatility:** `pl.col("grid_frequency").rolling_std(window_size=10).fill_null(0)`
    *   **Lag Features:** `pl.col("grid_frequency").shift(1/5/60).fill_null(50.0)` for 1s, 5s, 60s lags.
    *   **Time Embeddings:** `pl.col("timestamp").dt.hour().alias("hour")`, `pl.col("timestamp").dt.minute().alias("minute")`.
    *   **Targets:**
        *   `target_freq_next`: `pl.col("grid_frequency").shift(-TTA_SECONDS)` (10-second ahead frequency).
        *   `target_is_unstable`: Binary (1/0) based on `target_freq_next` falling outside [49.8, 50.2] Hz.

**How it should be implemented (Detailed Steps):**

1.  **`merge_datasets(df_freq, df_weather, df_inertia)`:**
    *   Ensure all input DataFrames have a clean, consistent `timestamp` column (UTC, microseconds).
    *   `df_merged = df_freq.join_asof(df_weather, on="timestamp", strategy="backward")`
    *   `df_merged = df_merged.join_asof(df_inertia.rename({"timestamp_date": "timestamp"}), on="timestamp", strategy="backward")`
    *   `df_merged = df_merged.drop_nulls()` to handle initial NaNs from shifting/joining.
2.  **`calculate_wind_ramp_rate(df)`:**
    *   `weather_data = df.select(["timestamp", "wind_speed"]).unique(subset=["timestamp"]).to_pandas()`: Extract unique hourly weather points.
    *   `weather_data['unix_ts'] = weather_data['timestamp'].view(np.int64) // 10**9`: Convert timestamps to Unix seconds for OpSDA.
    *   `compressed = opsda.compress(list(weather_data[['unix_ts', 'wind_speed']].itertuples(index=False, name=None)), width=0.5)`: Apply the OpSDA.
    *   Convert `compressed` output back to Polars, re-casting `unix_ts` to timezone-aware datetime.
    *   Calculate `wind_ramp_rate` between compressed points: `(pl.col("wind_speed").diff()) / (pl.col("timestamp").diff().dt.total_seconds())`.
    *   `df = df.join_asof(compressed_df.select(["timestamp", "wind_ramp_rate"]), on="timestamp", strategy="backward")`.
    *   `df = df.with_columns(pl.col("wind_ramp_rate").forward_fill())` to fill all 1-second entries.
3.  **`create_features(df)`:**
    *   This function integrates all feature calculations. `df` passed to it should already be the fully merged DataFrame.
    *   The `rocof`, `volatility_10s`, `lag_1s`, `lag_5s`, `lag_60s`, `hour`, `minute` features are calculated directly using Polars expressions.
    *   `target_freq_next` and `target_is_unstable` are generated based on `TTA_SECONDS` from `src/config.py`.
    *   `df_features = df_features.drop_nulls()` as a final step to clean up any remaining NaNs from feature creation (e.g., initial rows for lags, final rows for future targets).

**Necessary and Relevant Improvements:**

*   **OpSDA `width` Parameter Optimization:** The `width=0.5` parameter for the Swinging Door Algorithm is crucial. This value should be systematically optimized (e.g., via cross-validation) against prediction performance, rather than being hardcoded. Making it configurable in `src/config.py` would be a good first step.
*   **Dynamic Lag Feature Generation:** The number of lag features (1s, 5s, 60s) is fixed. Consider a configurable list of lag intervals in `config.py` to easily experiment with different historical windows.
*   **More Granular Inertia Proxy:** `inertia_cost` is currently daily. If higher resolution data on generation mix (synchronous vs. asynchronous) becomes available, implement a real-time inertia proxy calculation to capture more dynamic inertia changes. This would significantly enhance the "physics-informed" aspect.
*   **Interaction Features:** Explore creating interaction terms between highly correlated features or physically meaningful combinations (e.g., `wind_ramp_rate * inertia_cost`) if models struggle to capture complex non-linearities.
*   **Automated Feature Selection:** After initial model development, consider integrating automated feature selection techniques (e.g., recursive feature elimination, permutation importance) to reduce dimensionality and improve model interpretability/performance.
*   **Handling Missing `wind_ramp_rate`:** The `forward_fill()` for `wind_ramp_rate` assumes the last known ramp rate persists. Evaluate if other imputation strategies (e.g., interpolation, using hourly `wind_speed` directly) might be more appropriate during periods of no significant ramp events.

---

### 3. Model Training (LightGBM, LSTM, Quantile Regression)

**Current Implementation Details:**
Model training is handled in `src/model_trainer.py`, focusing on LightGBM (classifier and quantile regressors) and LSTM.

*   **Data Splitting:** A fixed chronological split is used: training on Aug 1-8, 2019, and testing on Aug 9, 2019 (the blackout day), defined by `SPLIT_DATE` and `END_TEST_DATE` in `config.py`.
*   **LightGBM Classifier (`train_and_evaluate_lgbm_classifier`):**
    1.  Uses `LGBM_FEATURE_COLS` and `TARGET_COL`.
    2.  Calculates `scale_pos_weight` (`neg / pos`) to mitigate severe class imbalance inherent in instability events.
    3.  Initializes `lgb.LGBMClassifier` with parameters from `LGBM_PARAMS` in `config.py` (e.g., `n_estimators=100`, `learning_rate=0.05`, `max_depth=10`, `random_state=42`, `n_jobs=-1`).
    4.  Evaluated using `classification_report` and `confusion_matrix`.
*   **LightGBM Quantile Regressors (`train_quantile_model`):**
    1.  Two separate `lgb.LGBMRegressor` models are trained for `alpha=0.1` (lower bound) and `alpha=0.9` (upper bound), as defined by `QUANTILE_ALPHAS` in `config.py`.
    2.  Uses `LGBM_FEATURE_COLS` to predict `TARGET_FREQ_NEXT`.
    3.  `objective='quantile'` and `alpha` are set for each regressor.
*   **LSTM Model (`train_lstm_model`):**
    1.  Uses `LSTM_FEATURE_COLS` and `TARGET_COL`.
    2.  `MinMaxScaler` is applied, fit on training features and used to transform both train/test. The scaler is saved.
    3.  Data is transformed into sequences `[samples, timesteps, features]` using `create_lstm_sequences` with `LSTM_TIME_STEPS=30`. A subset of the last 50,000 training samples is used to speed up sequence creation for the demo.
    4.  Model is a `tf.keras.models.Sequential` with `LSTM(50)`, `Dropout(0.2)`, `Dense(1, activation='sigmoid')`, with `input_shape=(None, num_features)` to allow variable sequence length.
    5.  Compiled with `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`.
    6.  Trained for `epochs=5`, `batch_size=64`, `validation_split=0.1`.
    7.  Evaluated using `classification_report`.
*   **Model Saving:** All trained models (LightGBM classifier, quantile regressors, LSTM, and scaler) are saved to designated paths in the `notebooks` directory (`config.py`).

**How it should be implemented (Detailed Steps):**

1.  **`train_and_evaluate_lgbm_classifier(df)`:**
    *   Perform chronological split using `SPLIT_DATE` and `END_TEST_DATE`.
    *   Convert selected Polars columns to Pandas for `X_train`, `y_train`, `X_test`, `y_test`.
    *   Calculate `scale_pos_weight` to pass to `LGBM_PARAMS`.
    *   Instantiate `lgb.LGBMClassifier(**LGBM_PARAMS)`.
    *   `model.fit(X_train, y_train)`.
    *   `model.predict(X_test)` and print `classification_report`, `confusion_matrix`.
2.  **`train_quantile_model(df, alpha)`:**
    *   Perform chronological split.
    *   Convert selected Polars columns to Pandas for `X_train`, `y_train`.
    *   Instantiate `lgb.LGBMRegressor(objective='quantile', alpha=alpha, **LGBM_PARAMS)`.
    *   `model.fit(X_train, y_train)`.
3.  **`train_lstm_model(df_processed)`:**
    *   Perform chronological split.
    *   Initialize `MinMaxScaler()`, `scaler.fit_transform(train_data[LSTM_FEATURE_COLS])`, and `scaler.transform(test_data[LSTM_FEATURE_COLS])`.
    *   `X_train, y_train = create_lstm_sequences(...)`
    *   Define Keras Sequential model architecture (LSTM, Dropout, Dense).
    *   `model.compile(...)`.
    *   `model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, validation_split=LSTM_VALIDATION_SPLIT, verbose=1)`.
    *   `model.predict(X_test)` and print `classification_report`.
4.  **Model Persistence:** `joblib.dump()` for LightGBM and scaler, `model.save()` for Keras models (e.g., to `.pkl` or `.keras` files).

**Necessary and Relevant Improvements:**

*   **Automated Hyperparameter Tuning (Crucial):** Implement systematic hyperparameter optimization for all models. For LightGBM, `GridSearchCV` or `RandomizedSearchCV` from scikit-learn could be used. For LSTM, `Keras Tuner` or manual exploration of network architecture (layers, units, dropout rates) is essential.
*   **Robust Evaluation Metrics:**
    *   For the classifier: Beyond `classification_report`, explicitly report **Precision, Recall, F1-score for the "Unstable" class**, and AUC-ROC. Emphasize **Recall** as paramount for an alert system (minimizing false negatives).
    *   For quantile regressors: Implement calculation of **Pinball Loss** (as mentioned in methodology), **Prediction Interval Coverage Probability (PICP)**, and **Mean Prediction Interval Width (MPIW)** to evaluate the quality of the uncertainty bands.
*   **Time-Series Cross-Validation:** Replace the single train/test split with a time-series aware cross-validation strategy (e.g., `sklearn.model_selection.TimeSeriesSplit`) for more reliable model performance estimates. This is especially important for financial/critical systems.
*   **Ensemble/Hybrid Models:** Explore combining the strengths of different models. For instance, an ensemble of the LightGBM quantile models or a hybrid approach where LSTM predicts residuals from a LightGBM forecast.
*   **Model Monitoring:** Implement mechanisms to track model performance drift in production. Retraining strategies should be considered as grid dynamics evolve.
*   **Experiment Tracking:** Use tools like MLflow or Weights & Biases to log model training runs, hyperparameters, metrics, and artifacts for better experiment management.
*   **Performance Metrics Expansion:** Beyond classification_report, consider:
    *   **Precision-Recall Curves / ROC Curves:** More robust for imbalanced datasets.
    *   **F1-score:** For balancing precision and recall.
    *   **Pinball Loss / Winkler Score:** For evaluating quantile regression models (mentioned in methodology, but unclear if implemented in evaluation scripts).
*   **Early Stopping:** Implement early stopping callbacks for LSTM to prevent overfitting and optimize training time.
*   **GPU Acceleration:** Ensure TensorFlow/Keras is configured to leverage GPUs if available for faster LSTM training.

---

### 4. Quantile Regression

**Current Implementation Details:**
The project uses two LightGBM regressors to perform quantile regression, providing an "uncertainty band."

*   **Models:** Two `lgb.LGBMRegressor` instances are trained, one for the 10th percentile (`alpha=0.1`) and one for the 90th percentile (`alpha=0.9`).
*   **Objective:** The `objective` parameter for LightGBM is set to `'quantile'`.
*   **Target:** Both models predict `target_freq_next` (the frequency 10 seconds into the future).
*   **Usage in Dashboard:** The predictions from these two models (`lower_bound_pred` and `upper_bound_pred`) are directly used to draw a shaded "Uncertainty Band" on the main frequency monitor and to trigger the "INSTABILITY ALERT" if `lower_bound_pred` drops below `alert_threshold_hz`.

**How it should be implemented (Detailed Steps):**

1.  **Configuration (`src/config.py`):** Define `QUANTILE_ALPHAS = [0.1, 0.9]` (or other desired quantiles) and ensure `LGBM_PARAMS` are suitable for regression.
2.  **`train_quantile_model(df, alpha)` (`src/model_trainer.py`):**
    *   Takes the `alpha` value as an argument to train a specific quantile.
    *   Uses a consistent data split (training on Aug 1-8).
    *   Instantiates `lgb.LGBMRegressor(objective='quantile', alpha=alpha, **LGBM_PARAMS)`.
    *   Fits the model to `X_train` (`LGBM_FEATURE_COLS`) and `y_train` (`TARGET_FREQ_NEXT`).
3.  **Prediction in `app.py`:**
    *   `lower_model` and `upper_model` are loaded via `st.cache_resource`.
    *   For the `current_row` of data, `input_lgbm = current_row[LGBM_FEATURE_COLS].values.reshape(1, -1)` prepares the input.
    *   `lower_bound_pred = lower_model.predict(input_lgbm)[0]`
    *   `upper_bound_pred = upper_model.predict(input_lgbm)[0]`
    *   For the graph window, predictions are made for a longer series: `lower_preds = lower_model.predict(window_input)`, `upper_preds = upper_model.predict(window_input)`.
4.  **Alerting Logic in `app.py`:** `is_alert = lower_bound_pred < alert_threshold_hz`.
5.  **Visualization in `app.py`:** The `go.Scatter` traces use `upper_preds` and `lower_preds` with `fill='tonexty'` to create the shaded band.

**Necessary and Relevant Improvements:**

*   **Evaluation with Pinball Loss:** Explicitly calculate and report the Pinball Loss for your quantile models. This metric directly evaluates the accuracy of quantile forecasts. While mentioned in the methodology, its implementation in evaluation scripts should be confirmed.
*   **Backtesting and Re-calibration:** Regularly backtest the coverage and width of the prediction intervals (PICP, MPIW) against unseen data. If coverage drops significantly, it might indicate concept drift, requiring model re-calibration or retraining.
*   **Dynamic Quantile Selection:** Explore if different quantiles (e.g., 0.05/0.95 or 0.01/0.99) might be more appropriate or configurable based on the desired risk tolerance.
*   **Comparison to Other QR Methods:** Benchmark LightGBM QR against other quantile regression techniques (e.g., pinball loss with other gradient boosting frameworks, or specialized time-series QR methods).
*   **Uncertainty-Aware Feature Importance:** Analyze SHAP values for quantile models to understand what drives predictions *at the extremes* (e.g., what drives the 10th percentile down) which is more relevant for alert systems than mean prediction explanations.

---

### 5. Explainability (XAI, SHAP)

**Current Implementation Details:**
The project integrates SHAP values to explain the drivers of instability alerts within the Streamlit dashboard.

*   **Explainer Initialization:** `explainer = shap.TreeExplainer(lower_model)` is created for the LightGBM lower quantile model in `app.py`. This means explanations are focused on why the *lower bound* of frequency is predicted.
*   **SHAP Value Calculation:** `shap_values = explainer.shap_values(input_lgbm)` calculates SHAP values for the current input features (`input_lgbm`).
*   **Visualization:** A horizontal bar chart is generated using Plotly. It shows `LGBM_FEATURE_COLS` as factors and their `SHAP Value` as the magnitude of impact. Features are sorted by SHAP value, and color-coded (red for negative/risk-increasing, blue for positive/risk-decreasing impact on frequency). A title explicitly states: "Red bars are pushing the prediction lower (increasing risk)."

**How it should be implemented (Detailed Steps):**

1.  **Configuration:** Ensure `LGBM_FEATURE_COLS` are clearly defined and consistent between training and explanation.
2.  **`app.py` - Explainer Loading:**
    *   `explainer = shap.TreeExplainer(lower_model)`: Instantiated once per app load using `st.cache_resource` or `st.cache_data`. This is crucial to avoid re-initializing the explainer on every dashboard refresh.
3.  **`app.py` - SHAP Calculation:**
    *   For each new `current_row` (or selected time point), extract its features: `input_lgbm = current_row[LGBM_FEATURE_COLS].values.reshape(1, -1)`.
    *   Compute `shap_values = explainer.shap_values(input_lgbm)`. For binary classification or quantile regression, `shap_values` might be a list (e.g., `shap_values[0]` for class 0, `shap_values[1]` for class 1, or for the specific quantile). The project correctly uses `shap_values[0]` assuming the structure for LightGBM's quantile output which needs further verification as `shap_values` for regressors usually return single array. (The notebook shows `shap_values[1]` which is for the positive class in a classifier). If it's a regressor, it typically returns a single array. This needs to be checked (`app.py` uses `shap_values[0]`).
4.  **`app.py` - SHAP Visualization:**
    *   Create a Pandas DataFrame (`contrib_df`) mapping `Factor` (feature name) to `SHAP Value`.
    *   Sort `contrib_df` by `SHAP Value` (e.g., `ascending=False` to show positive impacts first, or relevant risks first).
    *   Assign colors (e.g., red for negative SHAP values, blue for positive).
    *   Use `go.Figure(go.Bar(...))` to create the horizontal bar chart, setting `orientation='h'` and `marker_color`.
    *   Add informative title (e.g., "Red bars are pushing the prediction lower (increasing risk)").
    *   Render using `st.plotly_chart(fig_bar, use_container_width=True)`.

**Necessary and Relevant Improvements:**

*   **Clarity of SHAP `shap_values` Indexing:** In `app.py`, `shap_values = explainer.shap_values(input_lgbm)` and then `shap_values[0]` is accessed. For a regressor, `shap_values` is typically a single NumPy array, so directly using `shap_values` instead of `shap_values[0]` might be more accurate or require clarification based on the specific `shap` version and model output. Confirm this behavior for `lgbm.LGBMRegressor`.
*   **Dynamic Explanation Focus:** While explaining the *lower bound* prediction is relevant for alerts, consider allowing operators to switch the explanation focus (e.g., explain the upper bound, or the point prediction if a mean regressor was used) for broader understanding.
*   **SHAP Dependence Plots:** Implement interactive SHAP dependence plots for selected key features. These plots show how a feature affects the model's output and reveals potential interactions with other features, offering deeper insights.
*   **Temporal SHAP:** For time-series data, it can be useful to see how feature importances change over time, especially leading up to an instability event.
*   **SHAP Summary Plot (Global Importance):** In addition to local explanations, a global summary plot (e.g., `shap.summary_plot(shap_values, X_test)`) could be shown in a separate analytics view to give an overall understanding of feature importance across the test set.
*   **Contextual Explanation:** When an alert is triggered, in addition to the bar chart, provide a concise natural language summary of *why* the alert was triggered based on the most impactful SHAP values (e.g., "Frequency is predicted to drop due to high wind ramp rate and low inertia cost").

---

### 6. Dashboard Outputs (`app.py`)

**Current Implementation Details:**
The Streamlit application provides an interactive dashboard for visualizing grid frequency, predictions, and explanations.

*   **Page Configuration:** `st.set_page_config` sets title ("GridGuardian v2: UK Frequency Alert System"), icon, `wide` layout, and `expanded` sidebar.
*   **Custom CSS:** Injected via `st.markdown` for a "Dark Mode" control room feel, affecting `.block-container` padding and `.stMetric` styling.
*   **Cached Resources:** `st.cache_resource` loads `lower_model`, `upper_model`, `df_data`, and `explainer` once, speeding up the app.
*   **Sidebar Controls:**
    *   **Simulation Time:** `st.sidebar.slider` allows scrubbing through `df_data` indices. `st.sidebar.text_input` enables precise time jumps.
    *   **Alerting Threshold:** `st.sidebar.slider` lets users adjust the instability threshold (e.g., 49.8 Hz).
*   **Main Dashboard UI:**
    *   **Dynamic Title:** `st.title` displays current simulated timestamp.
    *   **Key Performance Indicators (KPIs):** `st.columns(5)` creates a row of metrics:
        *   "Grid Frequency" (current, with RoCoF as delta).
        *   "Predicted Lower Bound", "Predicted Upper Bound".
        *   Alert Status (`st.error` / `st.success`).
        *   "Time to Alert" (fixed 10 sec or "N/A").
    *   **Main Plot:** `st.columns([2, 1])` for graph and XAI side-by-side. Uses `plotly.graph_objects.Figure` to display:
        *   `Actual Frequency` (blue line).
        *   `Uncertainty Band` (shaded orange area between `lower_preds` and `upper_preds`).
        *   `Alert Threshold` (red dashed line).
        *   `Current Time` marker.
        *   `plotly_dark` theme, `height=400`, `yaxis_range=[49.4, 50.4]`.
    *   **XAI Plot:** Displays the SHAP bar chart for "Risk Drivers for Instability" for the `lower_model` prediction, with custom colors and a descriptive title.
    *   **Info Message:** `st.info` provides context about "Quantile Regression Mode."

**How it should be implemented (Detailed Steps):**

1.  **Streamlit App Structure:**
    *   **Setup:** `st.set_page_config` at the top for layout and aesthetics.
    *   **Caching:** Use `st.cache_resource` for loading models and large datasets to prevent re-loading on every rerun.
    *   **Sidebar:** Organize user controls (sliders, text inputs) in the sidebar using `st.sidebar`.
    *   **Main Content:** Use `st.columns` to create a responsive layout for KPIs, graphs, and XAI plots.
2.  **Interactive Controls:**
    *   **Time Navigation:** Synchronize `st.slider` and `st.text_input` for time, allowing both coarse scrubbing and precise jumps. Handle `datetime` conversions carefully.
    *   **Threshold Adjustment:** `st.slider` for `alert_threshold_hz` provides real-time sensitivity adjustment.
3.  **Data Processing & Prediction:**
    *   Slice `df_data` based on selected time index (`current_row`).
    *   Extract `LGBM_FEATURE_COLS` for the current input (`input_lgbm`).
    *   Call `lower_model.predict(input_lgbm)` and `upper_model.predict(input_lgbm)`.
    *   Determine `is_alert` based on `lower_bound_pred` vs. `alert_threshold_hz`.
4.  **Display KPIs:** Use `st.metric` for key numerical values, with `delta` for changes (e.g., RoCoF). Use `st.error` or `st.success` for alert status.
5.  **Plotly Integration:**
    *   Create `go.Figure()` objects.
    *   Add traces for `Actual Frequency`, `upper_preds`, `lower_preds` (using `fill='tonexty'` for the uncertainty band), `Alert Threshold` (`add_hline`), and `Current Time` marker.
    *   Customize layout with `template="plotly_dark"` and appropriate axis ranges/titles.
    *   Render with `st.plotly_chart(fig, use_container_width=True)`.
6.  **SHAP Visualization:**
    *   Calculate `shap_values` for the current input using the `explainer`.
    *   Process `shap_values` into a DataFrame suitable for plotting (features and their contributions).
    *   Generate a horizontal bar chart (`go.Bar`) using Plotly, clearly indicating features driving the prediction low (e.g., with specific colors).
7.  **Informative Text:** Use `st.info` or `st.markdown` for explanatory notes on the dashboard's mode or interpretation.

**Necessary and Relevant Improvements:**

*   **Real-time Data Integration:** The most significant improvement would be to transition from static `demo_data_aug9.csv` to a near real-time data feed (e.g., MQTT, Kafka, or a continuous database query). This would require backend integration to push data to the Streamlit app.
*   **Historical Data View:** Implement a feature to view historical trends, past alerts, and the associated SHAP explanations over longer periods than the current `window=300` seconds. This could involve date range pickers.
*   **Configurable TTA_SECONDS:** Make `TTA_SECONDS` configurable from the sidebar. Operators might want to adjust the look-ahead window.
*   **Multiple Scenario Simulation:** Allow users to input hypothetical changes to key features (e.g., what if wind speed drops by X, or inertia cost increases?) to see the immediate impact on predictions and SHAP explanations. This would enhance the "decision support" aspect.
*   **Detailed Event Logging:** Store alert events with more metadata (timestamp, predicted bounds, actual frequency, SHAP values at the time, operator actions) for post-event analysis and continuous improvement.
*   **Audio/Visual Alerts:** For critical alerts, integrate subtle but noticeable audio cues or more prominent visual indicators that grab operator attention without being overly distracting.
*   **User Profiles/Permissions:** In a multi-user environment, implement basic authentication and user-specific settings.
*   **Deployment Monitoring:** For a deployed system, set up application performance monitoring (APM) to track response times, errors, and resource usage of the Streamlit app.
*   **Mobile Responsiveness:** Ensure the dashboard is fully responsive and usable on various screen sizes to ensure optimal user experience.
*   **Accessibility Features:** Consider features for colorblind users or other accessibility needs in the visual design.

---

This detailed breakdown should provide a clear roadmap for understanding your project's current state and guiding its future development.
## Summary of Completed Actions and Project's Current State

**Overall Goal:** Transition data ingestion for frequency and inertia cost data from CSV files to API calls, and then verify implemented changes against the project's detailed analysis and improvements document.

**Completed Actions:**

1.  **Data Ingestion Pipeline:**
    *   **Frequency Data:** Successfully transitioned from loading CSV files to fetching 1-second resolution frequency data via the NESO CKAN API. Correct `resource_id` for August 2019 data was identified and configured.
    *   **Inertia Cost Data:** Successfully transitioned from loading CSV files to fetching daily inertia cost data via the NESO CKAN API. Correct `resource_id` for 2019 data was identified and configured.
    *   **API Configuration:** `src/config.py` was updated with the correct NESO CKAN API base URL and resource IDs. Old CSV file paths were commented out.
    *   **API Client Enhancements:** Implemented robust API handling in `src/data_loader.py` by adding a retry mechanism with exponential backoff using the `tenacity` library for network resilience.
    *   **Data Lineage:** Enhanced logging in `src/data_loader.py` to include API request URLs and parameters for better data lineage tracking.
    *   **Integration:** `notebooks/data_ingestion.py` and `app.py` were updated to utilize the new API-based data loading functions.
    *   **Testing:** `test_data_ingestion.py` was updated to test the new API data ingestion pipeline, and all tests are passing.

2.  **Feature Engineering:**
    *   **Dynamic Lag Features:** Implemented dynamic generation of lag features by configuring `LAG_INTERVALS_SECONDS` in `src/config.py` and updating `src/feature_engineering.py` to use this configuration. `LGBM_FEATURE_COLS` and `LSTM_FEATURE_COLS` in `src/config.py` were also updated to reflect these dynamic lags.
    *   **Core Feature Implementation:** All core feature engineering logic (merging, wind ramp rate calculation, inertia costs, RoCoF, Volatility, Time Embeddings, Targets) is confirmed to be correctly implemented in `src/feature_engineering.py`.

3.  **Model Training:**
    *   **Quantile Regressors Training & Saving:** Modified `notebooks/data_ingestion.py` to explicitly train the LightGBM quantile regressors (lower and upper bounds) using `src/model_trainer.py` and ensure these models are saved (`.pkl` files).
    *   **AUC-ROC for Classifier:** Implemented calculation and reporting of AUC-ROC for the LightGBM classifier within `train_and_evaluate_lgbm_classifier` in `src/model_trainer.py`.
    *   **Quantile Regression Metrics:** Implemented `pinball_loss` function and integrated calculation and reporting of Pinball Loss, Prediction Interval Coverage Probability (PICP), and Mean Prediction Interval Width (MPIW) for the LightGBM quantile regressors in `src/model_trainer.py` and `notebooks/data_ingestion.py`.
    *   **LSTM Early Stopping:** Implemented `EarlyStopping` callback for the LSTM model training in `train_lstm_model` in `src/model_trainer.py` to prevent overfitting.

4.  **Explainability (XAI, SHAP):**
    *   **SHAP Indexing Clarity:** Verified and adjusted SHAP value indexing in `app.py` to correctly handle the output of `shap.TreeExplainer` for a single-output regressor, ensuring robust display of explanations.

5.  **Dashboard Outputs (`app.py`):**
    *   **Configurable TTA_SECONDS:** Added a Streamlit slider in the sidebar of `app.py` to allow users to dynamically configure the "Time to Alert" (`TTA_SECONDS`) parameter.
    *   **Historical Data View (Date Pickers):** Implemented date range pickers in the sidebar of `app.py`, allowing users to select `start_date` and `end_date` for data loading. The data loading mechanism was adapted to fetch data dynamically based on these selections, and the simulation time slider adjusts accordingly.

**Project's Current State:**

The project has made significant progress in enhancing its data ingestion capabilities by moving to API-based data sources, improving model training robustness with better metrics and early stopping, and making the Streamlit dashboard more interactive and user-configurable. The core functionalities of data loading, feature engineering, model training, and dashboard visualization are now more robust and flexible.

All immediate and actionable improvements identified during the verification against `project_detailed_analysis_and_improvements.md` have been addressed.

**Remaining Improvements (Future Work):**

While significant progress has been made, the `project_detailed_analysis_and_improvements.md` document outlines several further improvements for future iterations. These include:

*   **Feature Engineering:** Optimizing OpSDA `width` parameter (modeling task), exploring more granular inertia proxies (requires new data), implementing interaction features, automated feature selection, and evaluating alternative `wind_ramp_rate` imputation strategies.
*   **Model Training:** Implementing automated hyperparameter tuning, time-series cross-validation, exploring ensemble/hybrid models, model monitoring, and experiment tracking.
*   **Quantile Regression:** Implementing a more automated process for regularly backtesting prediction intervals, exploring different quantiles, comparing to other QR methods, and modifying SHAP analysis for uncertainty-aware feature importance.
*   **Explainability (XAI, SHAP):** Implementing UI for dynamic explanation focus, SHAP Dependence Plots, Temporal SHAP, SHAP Summary Plots (Global Importance), and natural language contextual explanations for alerts.
*   **Dashboard Outputs:** Transitioning to real-time data integration, implementing interactive elements for scenario testing, logging alert events, integrating audio/visual alerts, implementing authentication and user settings, setting up APM, enhancing mobile responsiveness, and implementing accessibility features.