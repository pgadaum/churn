"""
Train the churn model on the Telco dataset and serialize it.
Run this once to produce churn_model.pkl.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# ── Load dataset (download from Kaggle or provide local CSV) ─────────────
try:
    import kagglehub
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Loaded via kagglehub")
except Exception:
    # Fallback: place the CSV in the same directory
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Loaded local CSV")

# ── Clean ────────────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

ID_COL, TARGET = "customerID", "Churn"
feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
num_cols = [c for c in feature_cols if df[c].dtype != "object"]

X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Preprocessor ─────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols),
])

# ── Best params from Optuna (pre-set to save time) ───────────────────────
best_params = {
    "n_estimators": 180,
    "max_depth": 3,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "min_samples_leaf": 12,
}

pipe = ImbPipeline([
    ("prep",  preprocessor),
    ("smote", SMOTEENN(random_state=42)),
    ("clf",   GradientBoostingClassifier(random_state=42, **best_params)),
])

# ── Calibrated model ──────────────────────────────────────────────────────
print("Fitting calibrated model...")
calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
calibrated.fit(X_train, y_train)

# ── SHAP explainer on base estimator ─────────────────────────────────────
print("Building SHAP explainer...")
# Use one of the internal calibrated estimators
base_pipe = calibrated.calibrated_classifiers_[0].estimator
fitted_prep = base_pipe.named_steps["prep"]
fitted_clf  = base_pipe.named_steps["clf"]

ohe = fitted_prep.named_transformers_["cat"]
cat_feat_names = list(ohe.get_feature_names_out(cat_cols))
all_feat_names = cat_feat_names + num_cols

X_test_proc = fitted_prep.transform(X_test)
explainer = shap.TreeExplainer(fitted_clf)

# ── Bundle everything ─────────────────────────────────────────────────────
bundle = {
    "model":           calibrated,
    "feature_cols":    feature_cols,
    "cat_cols":        cat_cols,
    "num_cols":        num_cols,
    "all_feat_names":  all_feat_names,
    "fitted_prep":     fitted_prep,
    "fitted_clf":      fitted_clf,
    "explainer":       explainer,
    "threshold":       0.38,
}

joblib.dump(bundle, "churn_model.pkl")
print("Saved churn_model.pkl")

# Quick eval
from sklearn.metrics import roc_auc_score, recall_score
y_proba = calibrated.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.38).astype(int)
print(f"Test ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Test Recall  : {recall_score(y_test, y_pred):.4f}")
