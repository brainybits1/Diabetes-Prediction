"""
Train a Diabetes Prediction Model using Random Forest.
- Loads diabetes.csv from parent directory
- Replaces 0s (nulls) in feature columns with column median
- Trains a Random Forest inside a sklearn Pipeline
- Saves model to model/diabetes_model.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "diabetes.csv")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Outcome distribution:\n{df['Outcome'].value_counts()}")

# ── Replace 0 → NaN for physiologically impossible zero values ────────────────
ZERO_AS_NULL = ["Glucose", "BloodPressure", "Insulin", "BMI"]
df[ZERO_AS_NULL] = df[ZERO_AS_NULL].replace(0, np.nan)

print(f"\nMissing values after zero replacement:\n{df.isnull().sum()}")

# ── Split features / target ───────────────────────────────────────────────────
FEATURE_COLS = ["Pregnancies", "Glucose", "BloodPressure",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
X = df[FEATURE_COLS]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Build pipeline ─────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Light hyperparameter search
param_grid = {
    "model__n_estimators":   [100, 200],
    "model__max_depth":      [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

print("\nRunning GridSearchCV... (this may take ~30 s)")
gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)

best_pipeline = gs.best_estimator_
print(f"Best params: {gs.best_params_}")

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_proba)

print(f"\n✅ Test Accuracy : {accuracy:.4f}")
print(f"✅ ROC-AUC Score : {auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ── Feature importances ────────────────────────────────────────────────────────
rf_model      = best_pipeline.named_steps["model"]
importances   = rf_model.feature_importances_
feat_imp_dict = dict(zip(FEATURE_COLS, importances.tolist()))
print(f"\nFeature Importances:\n{ {k: round(v, 4) for k, v in sorted(feat_imp_dict.items(), key=lambda x: -x[1])} }")

# ── Save model + metadata ─────────────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), "model"), exist_ok=True)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "diabetes_model.pkl")

save_obj = {
    "pipeline":          best_pipeline,
    "feature_cols":      FEATURE_COLS,
    "feature_importances": feat_imp_dict,
    "accuracy":          accuracy,
    "auc":               auc,
    "zero_as_null_cols": ZERO_AS_NULL,
}
joblib.dump(save_obj, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")
