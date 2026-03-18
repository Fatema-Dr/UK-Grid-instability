import polars as pl
import openmeteo_requests
import requests_cache
import pandas as pd
import requests # Import requests for API calls
from src import config
import logging # Import logging module
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type # Import tenacity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup a cached session for API calls to avoid repeated requests
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
openmeteo = openmeteo_requests.Client(session=cache_session)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True
)
def _fetch_ckan_data(resource_id: str, date_field: str, start_date: str, end_date: str) -> pl.DataFrame:
    """
    Private helper function to fetch data from NESO CKAN API with retry logic.
    """
    logging.info(f"Attempting to fetch data from CKAN API for resource_id: {resource_id} from {start_date} to {end_date}...")
    
    # Construct the SQL query for the CKAN datastore_search_sql endpoint
    # The API expects dates in 'YYYY-MM-DD' format for filtering
    sql_query = f"SELECT * FROM \"{resource_id}\" WHERE \"{date_field}\" BETWEEN '{start_date}' AND '{end_date}'"
    
    params = {
        "sql": sql_query
    }
    
    request_url = f"{config.NESO_API_BASE_URL}/datastore_search_sql"
    logging.info(f"CKAN API Request URL: {request_url} with params: {params}")

    try:
        response = cache_session.get(request_url, params=params)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        if data.get("success"):
            records = data["result"]["records"]
            if not records:
                logging.warning(f"CKAN API returned no records for resource_id {resource_id} and date range {start_date} to {end_date}.")
                return pl.DataFrame()
            df = pl.DataFrame(records)
            logging.info(f"Successfully fetched {df.height} records from CKAN API.")
            return df
        else:
            error_message = data.get('error', 'Unknown error')
            logging.error(f"CKAN API call failed: {error_message}. Request URL: {response.url}")
            # Reraise a specific exception type to allow tenacity to retry if appropriate
            raise requests.exceptions.RequestException(f"CKAN API error: {error_message}")
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed for CKAN API: {e}")
        raise # Re-raise to allow tenacity to handle retries
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching CKAN data: {e}")
        return pl.DataFrame()

def _validate_dataframe(df: pl.DataFrame, name: str, expected_cols: dict, value_checks: dict) -> pl.DataFrame:
    """
    Performs generic validation checks on a Polars DataFrame.
    """
    if df.is_empty():
        logging.warning(f"Validation Warning: {name} DataFrame is empty after loading.")
        return df

    # Check for expected columns and types
    for col, expected_type in expected_cols.items():
        if col not in df.columns:
            logging.warning(f"Validation Warning: {name} DataFrame is missing expected column '{col}'.")
            return df # Or continue with a potentially invalid DF
        # Check if type matches or can be cast
        if not df[col].dtype == expected_type:
            logging.warning(f"Validation Warning: {name} DataFrame column '{col}' has unexpected type {df[col].dtype}, expected {expected_type}. Attempting to cast.")
            try:
                df = df.with_columns(pl.col(col).cast(expected_type))
                logging.info(f"Validation Info: Successfully cast '{col}' to {expected_type}.")
            except Exception as e:
                logging.error(f"Validation Error: Failed to cast '{col}' to {expected_type}: {e}")
                return df # Stop processing if critical column has wrong type and cannot be cast


    # Check for reasonable value ranges
    for col, check_func in value_checks.items():
        if col in df.columns and not df[col].is_empty(): # Ensure column exists and is not empty before checking values
            if not check_func(df[col]):
                logging.warning(f"Validation Warning: {name} DataFrame column '{col}' contains values outside reasonable ranges. Data might be corrupted or unexpected.")
                # For now, just warn. Could also filter or impute.

    return df


def _get_frequency_resource_ids(start_date: str, end_date: str) -> list:
    """
    Returns the list of NESO resource IDs needed to cover the date range.
    The frequency data is split by month, so a multi-month range needs
    multiple resource IDs.
    """
    from datetime import datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    resource_ids = []
    current = start
    while current <= end:
        key = current.strftime("%Y-%m")
        if key in config.NESO_FREQUENCY_RESOURCE_MAP:
            resource_ids.append((key, config.NESO_FREQUENCY_RESOURCE_MAP[key]))
        else:
            logging.warning(f"No frequency resource ID configured for {key}")
        # Move to the first day of the next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    return resource_ids


def fetch_frequency_data(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Fetches NESO 1-second frequency data from the CKAN API and performs basic validation.
    Automatically selects the correct monthly resource IDs for the date range.
    """
    logging.info(f"Fetching grid frequency data from API for {start_date} to {end_date}...")
    
    resource_entries = _get_frequency_resource_ids(start_date, end_date)
    if not resource_entries:
        logging.error(f"No frequency resource IDs found for {start_date} to {end_date}")
        return pl.DataFrame()
    
    # Fetch and concatenate data from all required monthly resources
    frames = []
    for month_key, resource_id in resource_entries:
        logging.info(f"Fetching frequency data for {month_key} (resource: {resource_id})...")
        df_part = _fetch_ckan_data(resource_id, "dtm", start_date, end_date)
        if not df_part.is_empty():
            frames.append(df_part)
    
    if not frames:
        logging.error("No frequency data returned from any resource.")
        return pl.DataFrame()
    
    df = pl.concat(frames)

    if df.is_empty():
        return df

    # API response columns might be different from CSV, adjust as needed
    # Assuming the API returns 'dtm' and 'f' based on metadata.
    df = df.rename({"dtm": "timestamp", "f": "grid_frequency"}) # API uses 'f', internal code uses 'grid_frequency'

    df = df.with_columns(
        pl.col("timestamp")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S %z")
        .dt.convert_time_zone("UTC")
        .alias("timestamp")
    ).sort("timestamp")
    
    logging.info(f"Loaded {df.height} rows of frequency data from API.")

    # Validation
    expected_cols = {"timestamp": pl.Datetime, "grid_frequency": pl.Float64}
    value_checks = {
        "grid_frequency": lambda s: s.min() > 40.0 and s.max() < 60.0
    }
    df = _validate_dataframe(df, "Frequency Data (API)", expected_cols, value_checks)
    return df





def fetch_weather_data(start_date, end_date):
    """
    Fetches hourly historical weather data for the UK using parameters from config,
    upsamples it to 1-second resolution with linear interpolation, and performs basic validation.
    """
    logging.info("Fetching weather data...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    openmeteo = openmeteo_requests.Client(session=cache_session)

    url = config.WEATHER_API_URL
    params = {
        "latitude": config.WEATHER_API_LATITUDE,
        "longitude": config.WEATHER_API_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": config.WEATHER_API_HOURLY_VARS,
        "timezone": config.WEATHER_API_TIMEZONE
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    except Exception as e:
        logging.error(f"Weather API Error: Failed to fetch data from Open-Meteo API: {e}")
        return pl.DataFrame() # Return empty DataFrame on API error

    hourly = response.Hourly()
    # Read the hourly data
    hourly_data = {"timestamp": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["timestamp"] = hourly_data["timestamp"].tz_localize(config.WEATHER_API_TIMEZONE) # Use config timezone
    hourly_data["temperature"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["precipitation"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["rain"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["snowfall"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["wind_speed"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["wind_gusts"] = hourly.Variables(5).ValuesAsNumpy()
    hourly_data["solar_radiation"] = hourly.Variables(6).ValuesAsNumpy()

    df_weather_pd = pd.DataFrame(data=hourly_data)
    
    # Upsample to 1-second resolution and interpolate
    logging.info("Interpolating weather data to 1-second resolution...")
    df_weather_pd.set_index('timestamp', inplace=True)
    df_weather_pd = df_weather_pd.resample('1s').interpolate(method='time')
    df_weather_pd.reset_index(inplace=True)

    df_weather = pl.from_pandas(df_weather_pd)
    df_weather = df_weather.with_columns(
        pl.col("timestamp")
        .dt.convert_time_zone("UTC")
        .cast(pl.Datetime("us", time_zone="UTC"))
    )
    logging.info(f"Loaded and interpolated {df_weather.height} rows of weather data.")

    # Validation
    expected_cols = {
        "timestamp": pl.Datetime,
        "temperature": pl.Float64, # Assuming float for weather
        "wind_speed": pl.Float64,
        "solar_radiation": pl.Float64
    }
    value_checks = {
        "temperature": lambda s: s.min() > -50.0 and s.max() < 50.0, # Realistic temperature
        "wind_speed": lambda s: s.min() >= 0.0, # Non-negative wind speed
        "solar_radiation": lambda s: s.min() >= 0.0 # Non-negative solar radiation
    }
    df_weather = _validate_dataframe(df_weather, "Weather Data", expected_cols, value_checks)
    return df_weather


def fetch_inertia_data(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Fetches daily inertia cost data from the CKAN API and performs basic validation.
    """
    logging.info(f"Fetching inertia cost data from API for {start_date} to {end_date}...")
    
    # Using the example resource ID from config. This should be updated dynamically in a real system.
    resource_id = config.NESO_INERTIA_RESOURCE_ID_2019
    
    # Assuming "Settlement Date" is the column to filter by in the API for inertia data
    df = _fetch_ckan_data(resource_id, "Settlement Date", start_date, end_date)

    if df.is_empty():
        return df

    # API response columns might be different from CSV, adjust as needed
    # Assuming the API returns 'Settlement Date' and 'Cost' as per apis.md example
    df = df.rename({"Settlement Date": "timestamp_date", "Cost": "inertia_cost"})

    df = df.with_columns(
        pl.col("timestamp_date").str.strptime(pl.Date, format="%Y-%m-%dT%H:%M:%S").alias("timestamp_date") # Adjust format if needed
    )
    df = df.with_columns(
        pl.col("timestamp_date").cast(pl.Datetime).dt.replace_time_zone("UTC")
    )
    logging.info(f"Loaded {df.height} rows of inertia data from API.")

    # Validation
    expected_cols = {"timestamp_date": pl.Datetime, "inertia_cost": pl.Int64}
    value_checks = {
        "inertia_cost": lambda s: s.min() >= 0
    }
    df = _validate_dataframe(df, "Inertia Data (API)", expected_cols, value_checks)
    return df


def fetch_inertia_data_halfhourly(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Fetches half-hourly system inertia data from the NESO CKAN API, interpolates
    to 1-second resolution, and performs basic validation.
    """
    logging.info(f"Fetching half-hourly system inertia data for {start_date} to {end_date}...")
    
    resource_id = config.NESO_INERTIA_HALFHOURLY_RESOURCE_ID
    df = _fetch_ckan_data(resource_id, "Settlement Date", start_date, end_date)
    
    if df.is_empty():
        logging.warning("Half-hourly inertia data is empty. Falling back to daily.")
        return pl.DataFrame()
    
    # Identify the actual column names from the API response
    logging.info(f"Half-hourly inertia columns: {df.columns}")
    
    # Rename to standard names — adjust based on actual API schema
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower and 'settlement' in col_lower:
            rename_map[col] = 'timestamp_date'
        elif 'period' in col_lower and 'settlement' in col_lower:
            rename_map[col] = 'settlement_period'
        elif 'inertia' in col_lower and 'mw' in col_lower:
            rename_map[col] = 'system_inertia_mws'
        elif 'inertia' in col_lower:
            rename_map[col] = 'system_inertia_mws'
    
    if rename_map:
        df = df.rename(rename_map)
    
    if 'system_inertia_mws' not in df.columns:
        logging.error(f"Could not find inertia column in half-hourly data. Columns: {df.columns}")
        return pl.DataFrame()
    
    # Build a proper timestamp from Settlement Date + Settlement Period
    # Each settlement period is 30 minutes: period 1 = 00:00, period 2 = 00:30, etc.
    if 'settlement_period' in df.columns and 'timestamp_date' in df.columns:
        df = df.with_columns(
            pl.col('timestamp_date').str.strptime(pl.Date, format="%Y-%m-%dT%H:%M:%S")
        )
        df = df.with_columns(
            (pl.col('timestamp_date').cast(pl.Datetime).dt.replace_time_zone("UTC") +
             (pl.col('settlement_period').cast(pl.Int64) - 1) * pl.duration(minutes=30)
            ).alias('timestamp')
        )
    else:
        # Fallback: try to parse timestamp_date directly
        df = df.with_columns(
            pl.col('timestamp_date').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S")
            .dt.replace_time_zone("UTC").alias('timestamp')
        )
    
    df = df.select(['timestamp', 'system_inertia_mws']).sort('timestamp')
    
    # Cast inertia to float
    df = df.with_columns(pl.col('system_inertia_mws').cast(pl.Float64))
    
    # Interpolate to 1-second resolution
    logging.info(f"Interpolating {df.height} half-hourly inertia records to 1-second resolution...")
    df_pd = df.to_pandas()
    df_pd.set_index('timestamp', inplace=True)
    df_pd = df_pd.resample('1s').interpolate(method='time')
    df_pd.reset_index(inplace=True)
    
    df_result = pl.from_pandas(df_pd)
    df_result = df_result.with_columns(
        pl.col('timestamp').dt.convert_time_zone('UTC').cast(pl.Datetime('us', time_zone='UTC'))
    )
    
    logging.info(f"Half-hourly inertia data interpolated to {df_result.height} rows.")
    
    expected_cols = {"timestamp": pl.Datetime, "system_inertia_mws": pl.Float64}
    value_checks = {
        "system_inertia_mws": lambda s: s.min() >= 0
    }
    df_result = _validate_dataframe(df_result, "Half-hourly Inertia", expected_cols, value_checks)
    return df_result