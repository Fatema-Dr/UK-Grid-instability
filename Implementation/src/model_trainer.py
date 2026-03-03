# src/model_trainer.py

import polars as pl
import lightgbm as lgb
import tensorflow as tf
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping

from src.config import (
    SPLIT_DATE, END_TEST_DATE, LGBM_FEATURE_COLS, TARGET_COL, TARGET_FREQ_NEXT,
    LSTM_FEATURE_COLS, LSTM_TIME_STEPS, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_VALIDATION_SPLIT, LGBM_PARAMS, QUANTILE_ALPHAS
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


    params = LGBM_PARAMS.copy()
    params['objective'] = 'quantile'
    params['alpha'] = alpha
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate Pinball Loss on test set
    y_pred_quantile = model.predict(X_test)
    loss = pinball_loss(y_test, y_pred_quantile, alpha)
    print(f"Pinball Loss (alpha={alpha}): {loss:.4f}")

    print(f"Finished training for alpha={alpha}.")
    return model, X_test, y_test # Return test data for combined metrics


def create_lstm_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm_model(df_processed):
    """
    Trains the LSTM model.
    """
    print("Preparing data for LSTM (Deep Learning)...")
    data = df_processed.select(LSTM_FEATURE_COLS + [TARGET_COL]).to_pandas()

    split_idx = int(len(data) * (8/31))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:split_idx + 86400]

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[LSTM_FEATURE_COLS])
    test_scaled = scaler.transform(test_data[LSTM_FEATURE_COLS])

    print("Creating sequences (this may take a moment)...")
    X_train, y_train = create_lstm_sequences(train_scaled[-50000:], train_data[TARGET_COL].iloc[-50000:], time_steps=LSTM_TIME_STEPS)
    X_test, y_test = create_lstm_sequences(test_scaled, test_data[TARGET_COL], time_steps=LSTM_TIME_STEPS)

    print(f"LSTM Input Shape: {X_train.shape}")

    print("Building LSTM Model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
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

    print("Training LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_split=LSTM_VALIDATION_SPLIT,
        verbose=1,
        callbacks=[early_stopping] # Add early stopping callback
    )

    print("\nEvaluating LSTM on August 9...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Stable", "Unstable"]))

    return model, scaler