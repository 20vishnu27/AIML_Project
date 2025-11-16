# train_demand_models.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) /
                    (np.abs(y_true) + np.abs(y_pred) + 1)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE%": mape,
        "SMAPE%": smape
    }


# ---------- CONFIG ----------
DATA_PATH = "data/sales_data.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- LOAD ----------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# ---------- FEATURE LIST ----------
# Use the final features agreed upon (must exist in CSV)
feature_list = [
    'wday','month','year','wm_yr_wk',
    'snap','snap_CA','snap_TX','snap_WI',
    'sell_price',
    'lag_1','lag_7','lag_14','lag_28',
    'rolling_mean_7','rolling_std_7',
    'rolling_mean_30','rolling_std_30',
    'rolling_mean_60','rolling_std_60',
    'rolling_price_mean_30','price_ratio',
    'is_event'
]

# Verify columns exist (raise helpful error)
missing = [c for c in feature_list if c not in df.columns]
if missing:
    raise ValueError(f"Missing required features in CSV: {missing}")

target = 'sales'

# ---------- TRAIN/TEST SPLIT (time-based) ----------
df = df.sort_values('date')
train = df[df['date'] < df['date'].max() - pd.Timedelta(days=30)]
test  = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]

X_train = train[feature_list]
y_train = train[target]
X_test = test[feature_list]
y_test = test[target]

def rmse(y, yh): return np.sqrt(mean_squared_error(y, yh))

# ---------- RANDOM FOREST ----------
print("Training RandomForest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_metrics = evaluate_regression(y_test, rf.predict(X_test))
print("RF Metrics:", rf_metrics)

pickle.dump((rf, feature_list), open(os.path.join(MODELS_DIR,"rf_model.pkl"), "wb"))

# ---------- XGBOOST ----------
print("Training XGBoost...")
xg = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
xg.fit(X_train, y_train)
xg_metrics = evaluate_regression(y_test, xg.predict(X_test))
print("XGB Metrics:", xg_metrics)

pickle.dump((xg, feature_list), open(os.path.join(MODELS_DIR,"xgb_model.pkl"), "wb"))

"""
# ---------- LightGBM ----------
print("Training LightGBM...")
lg = lgb.LGBMRegressor(n_estimators=300, num_leaves=64, learning_rate=0.05)
lg.fit(X_train, y_train)
lg_rmse = rmse(y_test, lg.predict(X_test))
print("LGBM RMSE:", lg_rmse)
pickle.dump((lg, feature_list), open(os.path.join(MODELS_DIR,"lgbm_model.pkl"), "wb"))
"""

# ---------- SELECT BEST ----------
scores = {
    "RandomForest": rf_metrics["R2"], 
    "XGBoost": xg_metrics["R2"]
}

best_name = min(scores, key=scores.get)
best_model = {"RandomForest": rf, "XGBoost": xg}[best_name]
print("Best model:", best_name, scores[best_name])

# Save best model + feature_list
pickle.dump((best_model, feature_list), open(os.path.join(MODELS_DIR,"best_model.pkl"), "wb"))

# Save scores to CSV for UI
import pandas as pd
pd.DataFrame.from_dict(scores, orient='index', columns=['RMSE']).to_csv(os.path.join(MODELS_DIR,"model_scores.csv"))

print("All demand models trained and saved into", MODELS_DIR)
