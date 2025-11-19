#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ============================================
# 1. Imports & Global Settings
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib
import json

plt.style.use("ggplot")
pd.set_option("display.max_columns", 200)


# In[7]:


# ============================================
# 2. Load and Inspect Dataset
# ============================================

df = pd.read_csv("data/cac40_features.csv")

print("Shape:", df.shape)
df.head()


# In[25]:


# ============================================
# 3. Features & Target
# ============================================

target = "Target"

# Remove non-numeric columns (ticker, date, etc.)
non_numeric = df.select_dtypes(exclude=["number"]).columns
print("Removed non-numeric columns:", non_numeric)

df = df.drop(columns=non_numeric)

features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]


print("Number of features:", len(features))
features


# In[27]:


# ============================================
# 4. Basic Preprocessing
# ============================================

# Drop missing rows
df = df.dropna()

# Target variable: assume "target" is binary: 0 = down, 1 = up
target = "Target"

# Features must contain only numeric columns
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]






# In[29]:


# ============================================
# 5. Train / Validation / Test Split (60/20/20)
# ============================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


# In[31]:


# ============================================
# 6. Scaling
# ============================================

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("Scaling completed.")



# In[34]:


# ============================================
# 7. Train & Tune Models
# ============================================

results = {}

# --------------------------
# 7.1 Logistic Regression
# --------------------------
log_params = {
    "C": [0.001, 0.01, 0.1],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}

log_reg = LogisticRegression(max_iter=5000)
grid_log = GridSearchCV(log_reg, log_params, cv=3, scoring="roc_auc")
grid_log.fit(X_train_scaled, y_train)

best_log = grid_log.best_estimator_
results["Logistic Regression"] = grid_log.best_score_


# --------------------------
# 7.2 Random Forest
# --------------------------
rf_params = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 7],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestClassifier()
grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring="roc_auc")
grid_rf.fit(X_train_scaled, y_train)

best_rf = grid_rf.best_estimator_
results["Random Forest"] = grid_rf.best_score_


# --------------------------
# 7.3 XGBoost
# --------------------------
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.5, 0.8],
    "colsample_bytree": [0.7, 1.0]
}

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)

grid_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring="roc_auc")
grid_xgb.fit(X_train_scaled, y_train)

best_xgb = grid_xgb.best_estimator_
results["XGBoost"] = grid_xgb.best_score_

results


# In[36]:


# ============================================
# 8. Evaluation
# ============================================

def evaluate(model, name):
    print(f"\n========== {name} ==========")
    
    # Validation
    y_pred_val = model.predict(X_val_scaled)
    y_proba_val = model.predict_proba(X_val_scaled)[:,1]
    
    print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
    print("Validation F1:", f1_score(y_val, y_pred_val))
    print("Validation AUC:", roc_auc_score(y_val, y_proba_val))
    
    print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
    
    # Test
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:,1]
    
    print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Test F1:", f1_score(y_test, y_pred_test))
    print("Test AUC:", roc_auc_score(y_test, y_proba_test))
    
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))


evaluate(best_log, "Logistic Regression")
evaluate(best_rf, "Random Forest")
evaluate(best_xgb, "XGBoost")


# In[38]:


# ============================================
# 9. Save Best Model + Scaler + Features
# ============================================

# Choose best model by AUC
test_aucs = {
    "log_reg": roc_auc_score(y_test, best_log.predict_proba(X_test_scaled)[:,1]),
    "rf": roc_auc_score(y_test, best_rf.predict_proba(X_test_scaled)[:,1]),
    "xgb": roc_auc_score(y_test, best_xgb.predict_proba(X_test_scaled)[:,1])
}

best_name = max(test_aucs, key=test_aucs.get)
print("Best model:", best_name)

if best_name == "log_reg":
    final_model = best_log
elif best_name == "rf":
    final_model = best_rf
else:
    final_model = best_xgb


# Save model
joblib.dump(final_model, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save feature names
with open("features.json", "w") as f:
    json.dump(features, f)

print("Saved: model.pkl, scaler.pkl, features.json")


# In[24]:





# In[26]:




