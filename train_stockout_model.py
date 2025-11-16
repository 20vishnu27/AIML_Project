# train_stockout_model.py

import pandas as pd
import numpy as np
import os
import joblib
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/data_stock.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

# -------------------------------------------------------
# LOAD FORECAST MODEL
# -------------------------------------------------------
print("Loading demand forecast model...")

forecast_model, demand_features = pickle.load(
    open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb")
)

# -------------------------------------------------------
# GENERATE future_demand_30 USING FORECAST MODEL
# -------------------------------------------------------
print("Generating 30-day forecast for classifier training dataset...")

df["future_demand_30"] = forecast_model.predict(df[demand_features])

# Replace negatives if any
df["future_demand_30"] = df["future_demand_30"].clip(lower=0)

# -------------------------------------------------------
# CREATE STOCKOUT LABEL (0/1)
# -------------------------------------------------------
print("Creating stockout labels...")

df["stockout"] = (df["current_stock"] - df["future_demand_30"] <= 0).astype(int)

print(df["stockout"].value_counts())

# -------------------------------------------------------
# FEATURES FOR CLASSIFIER
# -------------------------------------------------------
stockout_features = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_30", "rolling_std_30",
    "rolling_mean_60", "rolling_std_60",
    "sell_price", "price_ratio",
    "wday", "month", "year",
    "snap", "is_event",
    "current_stock",
    "future_demand_30"
]

# Check missing features
missing = [c for c in stockout_features if c not in df.columns]
if missing:
    raise ValueError("Missing features required for stockout model: " + str(missing))

X = df[stockout_features]
y = df["stockout"]

# -------------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining XGBoost Stockout Classifier...")
clf = XGBClassifier(
    max_depth=6,
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss"
)

clf.fit(X_train, y_train)

# -------------------------------------------------------
# EVALUATION
# -------------------------------------------------------
preds = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds, digits=4))

# -------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------
pickle.dump((clf, stockout_features),
            open(os.path.join(MODELS_DIR, "xgb_stockout_model.pkl"), "wb"))

df.to_csv(os.path.join(MODELS_DIR, "stockout_training_dataset.csv"), index=False)

print("\nStockout model trained and saved successfully!")
