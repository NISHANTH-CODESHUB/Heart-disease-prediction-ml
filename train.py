"""


HOW TO RUN:
    python train_model.py

This script:
1. Loads and cleans the dataset
2. Splits data into train/test sets
3. Scales features
4. Trains Random Forest with hyperparameter tuning
5. Evaluates and saves the model
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv("data/heart-disease-dataset.csv")

print(f"   Shape     : {df.shape}")
print(f"   Columns   : {list(df.columns)}")
print(f"   Missing   : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# STEP 2: CLEAN DATA
# ─────────────────────────────────────────────
print("\n🧹 Cleaning data...")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"   Records after cleaning: {len(df)}")

# ─────────────────────────────────────────────
# STEP 3: SEPARATE FEATURES AND TARGET
# ─────────────────────────────────────────────
# Expected columns (update names to match your CSV exactly)
FEATURE_COLUMNS = [
    "age", "sex", "chest_pain_type",
    "resting_bp_s", "cholesterol",
    "fasting_blood_sugar", "resting_ecg",
    "max_heart_rate", "exercise_angina",
    "oldpeak", "st_slope"
]
TARGET_COLUMN = "target"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


print(f"\n📊 Class distribution:\n{y.value_counts()}")

# ─────────────────────────────────────────────
# STEP 4: TRAIN-TEST SPLIT (80/20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # ensures balanced class split
)
print(f"\n🔀 Train size : {X_train.shape[0]}")
print(f"   Test size  : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# STEP 5: FEATURE SCALING
# ─────────────────────────────────────────────
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 6: HYPERPARAMETER TUNING (GridSearchCV)
# ─────────────────────────────────────────────
print("\n⚙️  Running hyperparameter tuning (this may take a minute)...")

param_grid = {
    "n_estimators"    : [100, 200, 300],
    "max_depth"       : [None, 10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf" : [1, 2],
    "max_features"    : ["sqrt", "log2"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid = param_grid,
    cv         = cv,
    scoring    = "roc_auc",
    n_jobs     = -1,
    verbose    = 1
)

grid_search.fit(X_train_scaled, y_train)

best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\n✅ Best Parameters : {best_params}")
print(f"   Best CV AUC    : {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# STEP 7: EVALUATE MODEL
# ─────────────────────────────────────────────
print("\n📈 Evaluating model on test set...")
y_pred      = best_model.predict(X_test_scaled)
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print(f"\n   Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"\n   Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
print(f"   Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ─────────────────────────────────────────────
# STEP 8: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = best_model.feature_importances_
feat_importance = sorted(
    zip(FEATURE_COLUMNS, importances),
    key=lambda x: x[1], reverse=True
)
print("\n🏆 Feature Importances:")
for name, score in feat_importance:
    bar = "█" * int(score * 50)
    print(f"   {name:<25} {bar} {score:.4f}")

# ─────────────────────────────────────────────
# STEP 9: SAVE MODEL AND SCALER
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
pickle.dump(best_model, open("model/model.pkl", "wb"))
pickle.dump(scaler,     open("model/scaler.pkl", "wb"))

print("\n💾 Model saved to   : model/model.pkl")
print("💾 Scaler saved to  : model/scaler.pkl")
print("\n✅ Training complete! Now run: python app.py\n")