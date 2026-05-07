# src/model_trainer.py

import pandas as pd
import polars as pl
import lightgbm as lgb
import tensorflow as tf
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Enable memory growth for TensorFlow to prevent OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPUs")
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")

from src.config import (
    SPLIT_DATE, END_TEST_DATE, LGBM_FEATURE_COLS, TARGET_COL, TARGET_FREQ_NEXT,
    LSTM_FEATURE_COLS, LSTM_TIME_STEPS, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_VALIDATION_SPLIT, LGBM_PARAMS, LGBM_QUANTILE_PARAMS, QUANTILE_ALPHAS,
    EXPORT_DIR
)

# --- Quantile Regression Metrics ---
def pinball_loss(y_true, y_pred, alpha):
    """
    Calculates the pinball loss (also known as quantile loss).
    """
    error = y_true - y_pred
    return np.mean(np.maximum(alpha * error, (alpha - 1) * error))

def calculate_picp_mpiw(y_true, lower_bound, upper_bound, confidence_level):
    """
    Calculates Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).
    confidence_level: e.g., 0.8 for 80% confidence interval (alpha=0.1 to alpha=0.9)
    """
    # Prediction Interval Coverage Probability (PICP)
    covered = ((y_true >= lower_bound) & (y_true <= upper_bound)).astype(int)
    picp = np.mean(covered)

    # Mean Prediction Interval Width (MPIW)
    mpiw = np.mean(upper_bound - lower_bound)

    return picp, mpiw

# --- End Quantile Regression Metrics ---


def train_and_evaluate_lgbm_classifier(df):
    """
    Trains and evaluates the LightGBM classifier model.
    """
    print("Preparing data for the 'Blackout Stress Test' (Classifier)...")

    split_datetime_utc = datetime.strptime(SPLIT_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_test_datetime_utc = datetime.strptime(END_TEST_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    train = df.filter(pl.col("timestamp") < pl.lit(split_datetime_utc))
    test = df.filter(
        (pl.col("timestamp") >= pl.lit(split_datetime_utc)) &
        (pl.col("timestamp") < pl.lit(end_test_datetime_utc))
    )

    X_train = train.select(LGBM_FEATURE_COLS).to_pandas()
    y_train = train.select(TARGET_COL).to_pandas().values.ravel()

    X_test = test.select(LGBM_FEATURE_COLS).to_pandas()
    y_test = test.select(TARGET_COL).to_pandas().values.ravel()

    print("\nTraining LightGBM Classifier...")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    weight = neg / pos if pos > 0 else 1
    print(f"Calculated Class Weight: {weight:.2f}")

    params = LGBM_PARAMS.copy()
    params['scale_pos_weight'] = weight
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    print("\nEvaluating on August 9th (Blackout Day)...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class

    # Print Metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stable", "Unstable"]))
    
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC: {auc_roc:.4f}")
    except ValueError:
        print("AUC-ROC cannot be calculated if only one class is present in y_true or y_score.")
        
    return model, X_test, test

def train_quantile_model(df, alpha):
    """
    Trains a LightGBM quantile regression model for a specific alpha.
    """
    print(f"\nTraining Quantile Regression Model for alpha={alpha}...")
    
    split_datetime_utc = datetime.strptime(SPLIT_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    train = df.filter(pl.col("timestamp") < pl.lit(split_datetime_utc))
    test = df.filter(
        (pl.col("timestamp") >= pl.lit(split_datetime_utc)) &
        (pl.col("timestamp") < pl.lit(datetime.strptime(END_TEST_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)))
    )

    X_train = train.select(LGBM_FEATURE_COLS).to_pandas()
    y_train = train.select(TARGET_FREQ_NEXT).to_pandas().values.ravel()
    
    X_test = test.select(LGBM_FEATURE_COLS).to_pandas()
    y_test = test.select(TARGET_FREQ_NEXT).to_pandas().values.ravel()


    params = LGBM_QUANTILE_PARAMS.copy()
    params['objective'] = 'quantile'
    params['alpha'] = alpha
    
    model = lgb.LGBMRegressor(**params)
    
    # Upweight samples where target frequency is low (approaching instability)
    # This forces the model to care about the tails, not just average behaviour
    sample_weights = np.where(y_train < 49.95, 10.0,   # strong weight on low-freq samples
                     np.where(y_train < 50.00, 3.0,     # moderate weight approaching boundary
                     1.0))                               # normal weight otherwise
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate Pinball Loss on test set
    y_pred_quantile = model.predict(X_test)
    loss = pinball_loss(y_test, y_pred_quantile, alpha)
    print(f"Pinball Loss (alpha={alpha}): {loss:.4f}")

    print(f"Finished training for alpha={alpha}.")
    return model, X_test, y_test # Return test data for combined metrics

def train_all_quantile_models(df):
    """Trains all configured quantile models in one sweep and builds a proper reliability diagram."""
    models = {}
    results = {}
    for alpha in QUANTILE_ALPHAS:
        model, X_test, y_test = train_quantile_model(df, alpha)
        models[alpha] = model
        results[alpha] = {
            "y_pred": model.predict(X_test),
            "y_test": y_test,
            "pinball": pinball_loss(y_test, model.predict(X_test), alpha)
        }
    
    # Build proper reliability diagram
    reliability = {}
    for alpha in QUANTILE_ALPHAS:
        observed_coverage = np.mean(results[alpha]["y_test"] <= results[alpha]["y_pred"])
        reliability[alpha] = {
            "expected": alpha,
            "observed": observed_coverage,
            "deviation_pp": (observed_coverage - alpha) * 100
        }
    return models, results, reliability




def train_lstm_model(df_processed):
    """
    Trains the LSTM model.
    """
    print("Preparing data for LSTM (Deep Learning)...")
    tf.keras.backend.clear_session()  # Clear any previous graph to save memory

    data = df_processed.select(LSTM_FEATURE_COLS + [TARGET_COL, "timestamp"]).to_pandas()

    # Use SPLIT_DATE for temporal consistency with LightGBM models
    split_dt = datetime.strptime(SPLIT_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_TEST_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

    train_data = data[data['timestamp'] < split_dt].drop(columns=['timestamp'])
    test_data = data[(data['timestamp'] >= split_dt) & (data['timestamp'] < end_dt)].drop(columns=['timestamp'])

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[LSTM_FEATURE_COLS])
    test_scaled = scaler.transform(test_data[LSTM_FEATURE_COLS])

    print("Creating sequence datasets...")
    train_size = int(len(train_scaled) * (1 - LSTM_VALIDATION_SPLIT))
    
    val_scaled = train_scaled[train_size:]
    val_y = train_data[TARGET_COL].values[train_size:]
    
    train_scaled_split = train_scaled[:train_size]
    train_y = train_data[TARGET_COL].values[:train_size]

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=train_scaled_split,
        targets=train_y[LSTM_TIME_STEPS:],
        sequence_length=LSTM_TIME_STEPS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=True
    )
    val_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=val_scaled,
        targets=val_y[LSTM_TIME_STEPS:],
        sequence_length=LSTM_TIME_STEPS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False
    )
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=test_scaled,
        targets=test_data[TARGET_COL].values[LSTM_TIME_STEPS:],
        sequence_length=LSTM_TIME_STEPS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False
    )

    print(f"LSTM Input Shape: ({LSTM_TIME_STEPS}, {train_scaled.shape[1]})")

    print("Building LSTM Model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=False, input_shape=(LSTM_TIME_STEPS, train_scaled.shape[1])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=3,          # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
    )

    # Define ModelCheckpoint callback to avoid data loss and memory spikes
    os.makedirs(EXPORT_DIR, exist_ok=True)
    checkpoint_path = f"{EXPORT_DIR}/lstm_checkpoint.weights.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )

    print("Training LSTM...")
    history = model.fit(
        train_ds,
        epochs=LSTM_EPOCHS,
        validation_data=val_ds,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint] # Add callbacks
    )

    print("\nEvaluating LSTM on August 9...")
    y_pred_prob = model.predict(test_ds)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test_actual = test_data[TARGET_COL].values[LSTM_TIME_STEPS:]
    print(classification_report(y_test_actual, y_pred, target_names=["Stable", "Unstable"]))

    return model, scaler

def train_lstm_quantile_comparator(df_processed, n_mc_samples=25):
    """
    LSTM with MC Dropout for probabilistic forecasting - valid comparison to LightGBM quantile.
    """
    print("Preparing data for LSTM Quantile Comparator (MC Dropout)...")
    tf.keras.backend.clear_session()  # Clear any previous graph to save memory

    data = df_processed.select(LSTM_FEATURE_COLS + [TARGET_FREQ_NEXT, "timestamp"]).to_pandas()

    split_dt = datetime.strptime(SPLIT_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_TEST_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
    
    data = data.dropna(subset=[TARGET_FREQ_NEXT])

    train_data = data[data['timestamp'] < split_dt].drop(columns=['timestamp'])
    test_data = data[(data['timestamp'] >= split_dt) & (data['timestamp'] < end_dt)].drop(columns=['timestamp'])

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[LSTM_FEATURE_COLS])
    test_scaled = scaler.transform(test_data[LSTM_FEATURE_COLS])

    print("Creating sequence datasets...")
    train_size = int(len(train_scaled) * (1 - LSTM_VALIDATION_SPLIT))
    
    val_scaled = train_scaled[train_size:]
    val_y = train_data[TARGET_FREQ_NEXT].values[train_size:]
    
    train_scaled_split = train_scaled[:train_size]
    train_y = train_data[TARGET_FREQ_NEXT].values[:train_size]

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=train_scaled_split,
        targets=train_y[LSTM_TIME_STEPS:],
        sequence_length=LSTM_TIME_STEPS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=True
    )
    val_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=val_scaled,
        targets=val_y[LSTM_TIME_STEPS:],
        sequence_length=LSTM_TIME_STEPS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False
    )

    print("Building LSTM MC Dropout Model...")
    inputs = tf.keras.Input(shape=(LSTM_TIME_STEPS, train_scaled.shape[1]))
    x = tf.keras.layers.LSTM(50, return_sequences=False)(inputs)
    # Dropout kept ON during inference (training=True) for MC Dropout
    x = tf.keras.layers.Dropout(0.2)(x, training=True)
    outputs = tf.keras.layers.Dense(1)(x)  # regression, not classification
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mae')

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Define ModelCheckpoint callback for MC Dropout LSTM
    checkpoint_path = f"{EXPORT_DIR}/lstm_quantile_checkpoint.weights.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )

    print("Training LSTM MC Dropout Quantile Comparator...")
    model.fit(
        train_ds,
        epochs=LSTM_EPOCHS,
        validation_data=val_ds,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Build test dataset for MC Dropout evaluation
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=test_scaled,
        targets=test_data[TARGET_FREQ_NEXT].values[LSTM_TIME_STEPS:],
        sequence_length=LSTM_TIME_STEPS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False
    )

    # Run MC Dropout: sample n_mc_samples stochastic forward passes
    # training=True keeps Dropout active at inference time (MC Dropout)
    mc_preds = []
    print(f"  Generating {n_mc_samples} MC Dropout samples for each batch...")
    for i, batch in enumerate(test_ds):
        if i % 200 == 0:
            print(f"    Processing batch {i}...")
        # Optimized: Tile the batch to run all n_mc_samples in a single vectorized forward pass
        batch_size = tf.shape(batch[0])[0]
        tiled_inputs = tf.repeat(batch[0], n_mc_samples, axis=0)
        
        # Single forward pass for all stochastic samples (training=True keeps dropout active)
        tiled_preds = model(tiled_inputs, training=True)
        
        # Reshape to (batch_size, n_mc_samples, 1) and take mean across samples
        batch_samples = tf.reshape(tiled_preds, (batch_size, n_mc_samples, 1))
        batch_mean = tf.reduce_mean(batch_samples, axis=1)
        mc_preds.append(batch_mean.numpy())

    # Concatenate batches together (handles the smaller last batch correctly)
    X_test_lstm = np.concatenate(mc_preds, axis=0)
    
    # Collect ground truth
    y_test_lstm = np.concatenate([batch[1].numpy() for batch in test_ds])

    return model, scaler, X_test_lstm, y_test_lstm