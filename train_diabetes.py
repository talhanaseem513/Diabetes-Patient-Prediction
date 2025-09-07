"""
Train diabetes classifier(s) on the Pima Indians Diabetes dataset.

Expected file: data/diabetes.csv  (common columns below)
Columns used:
  Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
  BMI, DiabetesPedigreeFunction, Age, Outcome(0/1)
Outputs (in ./artifacts):
  - preprocessor.pkl (imputer+scaler)
  - model_best.pkl (best by ROC-AUC)
  - model_logreg.pkl, model_rf.pkl, model_xgb.pkl
  - feature_names.json
  - metrics.json (per model + best)
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA_PATH = Path("data/diabetes.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
    "BMI","DiabetesPedigreeFunction","Age"
]
TARGET = "Outcome"

def load_data():
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}. Download a Pima Diabetes CSV to that path."
    df = pd.read_csv(DATA_PATH)
    # Keep only required columns
    use_cols = FEATURES + [TARGET]
    missing = [c for c in use_cols if c not in df.columns]
    assert not missing, f"CSV missing required columns: {missing}"
    return df[use_cols].copy()

def build_preprocessor():
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(transformers=[("num", num_pipe, FEATURES)], remainder="drop")

def fit_with_smote(model, X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    model.fit(X_res, y_res)
    return model

def eval_and_dump(name, model, X_test, y_test, store=True):
    proba = model.predict_proba(X_test)[:,1]
    y_pred = (proba >= 0.5).astype(int)
    roc = roc_auc_score(y_test, proba)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    rpt = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc:.3f}  |  Precision: {pr:.3f}  Recall: {rc:.3f}  F1: {f1:.3f}")
    print(rpt)
    if store:
        joblib.dump(model, ART_DIR / f"model_{name}.pkl")
    return {"roc_auc": roc, "precision_pos": pr, "recall_pos": rc, "f1_pos": f1}

def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preproc = build_preprocessor()
    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)
    joblib.dump(preproc, ART_DIR / "preprocessor.pkl")
    Path(ART_DIR / "feature_names.json").write_text(json.dumps(FEATURES), encoding="utf-8")

    # Models
    results = {}

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg = fit_with_smote(logreg, X_train_t, y_train)
    results["logreg"] = eval_and_dump("logreg", logreg, X_test_t, y_test)

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, random_state=42, class_weight="balanced_subsample"
    )
    rf = fit_with_smote(rf, X_train_t, y_train)
    results["rf"] = eval_and_dump("rf", rf, X_test_t, y_test)

    if HAS_XGB:
        # scale_pos_weight ~ (neg/pos) on original training split
        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = max(1.0, neg / max(pos, 1))
        xgb = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=42, scale_pos_weight=spw
        )
        xgb = fit_with_smote(xgb, X_train_t, y_train)
        results["xgb"] = eval_and_dump("xgb", xgb, X_test_t, y_test)
    else:
        print("\n[!] xgboost not installed; skipping XGBClassifier.")

    # pick best by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = joblib.load(ART_DIR / f"model_{best_name}.pkl")
    joblib.dump(best_model, ART_DIR / "model_best.pkl")

    out = {"by_model": results, "best_model": best_name, "best": results[best_name]}
    Path(ART_DIR / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nBest model: {best_name}  ROC-AUC={results[best_name]['roc_auc']:.3f}")
    print(f"Artifacts saved to: {ART_DIR.resolve()}")

if __name__ == "__main__":
    main()
