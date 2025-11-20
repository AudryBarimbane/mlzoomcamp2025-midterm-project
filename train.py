#!/usr/bin/env python
# coding: utf-8

# ============================================
# 1. Imports
# ============================================

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================
# 2. Load Dataset
# ============================================

df = pd.read_csv("data/cac40_features.csv")

print("Shape:", df.shape)
print(df.head())


# ============================================
# 3. Features & Target
# ============================================

target = "Target"

# Remove non-numeric columns (symbol, date)
non_numeric = df.select_dtypes(exclude=["number"]).columns
print("Removed non-numeric columns:", non_numeric.tolist())

df = df.drop(columns=non_numeric)

features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]


# ============================================
# 4. Drop missing rows
# ============================================

df = df.dropna()

X = df[features]
y = df[target]


# ============================================
# 5. Train / Validation / Test Split
# ============================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# ============================================
# 6. Scaling
# ============================================

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("Scaling completed.")


# ============================================
# 7. Train Three Models
# ============================================

results = {}

# Logistic Regression
log_params = {"C": [0.01, 0.1, 1], "penalty": ["l2"]}
log_reg = LogisticRegression(max_iter=5000)
grid_log = GridSearchCV(log_reg, log_params, cv=3, scoring="roc_auc")
grid_log.fit(X_train_scaled, y_train)
best_log = grid_log.best_estimator_
results["log_reg"] = grid_log.best_score_

# Random Forest
rf_params = {"n_estimators": [100, 300], "max_depth": [3, 5, 7]}
rf = RandomForestClassifier()
grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring="roc_auc")
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
results["rf"] = grid_rf.best_score_

# XGBoost
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.05]
}

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss"
)

grid_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring="roc_auc")
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
results["xgb"] = grid_xgb.best_score_


# ============================================
# 8. Evaluate & Pick Best Model
# ============================================

def auc_test(model):
    return roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])

test_scores = {
    "log_reg": auc_test(best_log),
    "rf": auc_test(best_rf),
    "xgb": auc_test(best_xgb)
}

best_name = max(test_scores, key=test_scores.get)
print("Best model:", best_name)

if best_name == "log_reg":
    final_model = best_log
elif best_name == "rf":
    final_model = best_rf
else:
    final_model = best_xgb


# ============================================
# 9. Save Model + Scaler + Features
# ============================================

joblib.dump(final_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("features.json", "w") as f:
    json.dump(features, f)

print("\nSaved files:")
print(" - model.pkl")
print(" - scaler.pkl")
print(" - features.json")





