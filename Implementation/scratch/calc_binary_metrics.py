import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

model = joblib.load('notebooks/lgbm_quantile_lower.pkl')
df = pd.read_csv('notebooks/demo_data_aug9.csv')
from src.config import LGBM_FEATURE_COLS, TARGET_FREQ_NEXT

# Real August evaluation (using the whole month data if possible, but I only have Aug 9 demo)
# I'll use the Aug 9 data as a proxy for the 'Blackout Event' metrics
df = df.dropna(subset=LGBM_FEATURE_COLS + [TARGET_FREQ_NEXT])
X = df[LGBM_FEATURE_COLS]
y_true_freq = df[TARGET_FREQ_NEXT]
y_true_binary = (y_true_freq < 49.8).astype(int)

y_pred_lower = model.predict(X)
y_pred_binary = (y_pred_lower < 49.8).astype(int)

precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)
try:
    auc = roc_auc_score(y_true_binary, -y_pred_lower) # use -pred as score for lower threshold
except:
    auc = 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# False Negative / Positive
tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()

print(f"FP: {fp}, FN: {fn}")
print(f"FPR: {fp / (fp + tn):.4f}")
print(f"FNR: {fn / (fn + tp):.4f}")
