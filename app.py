import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Risk Predictor")
st.caption("Binary classification on the Pima Indians Diabetes dataset (Scikit-learn / XGBoost).")

ART_DIR = Path("artifacts")
DEFAULT_MODEL = ART_DIR / "model_best.pkl"
PREPROC_PATH = ART_DIR / "preprocessor.pkl"
FEATS_PATH = ART_DIR / "feature_names.json"
METRICS_PATH = ART_DIR / "metrics.json"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not (DEFAULT_MODEL.exists() and PREPROC_PATH.exists() and FEATS_PATH.exists()):
        return None, None, None
    preproc = joblib.load(PREPROC_PATH)
    model = joblib.load(DEFAULT_MODEL)
    features = json.loads(FEATS_PATH.read_text())
    return preproc, model, features

preproc, model, features = load_artifacts()

left, right = st.columns([2,1])
with right:
    st.subheader("Model status")
    if model is None:
        st.error("Artifacts not found. Run `python train_diabetes.py` first.")
    else:
        st.success("Model loaded")
        if METRICS_PATH.exists():
            m = json.loads(METRICS_PATH.read_text())
            st.metric("ROC-AUC", f"{m['best']['roc_auc']:.3f}")
            st.metric("Recall (pos)", f"{m['best']['recall_pos']:.3f}")
            st.metric("Precision (pos)", f"{m['best']['precision_pos']:.3f}")
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01, help="Score ≥ threshold ⇒ positive (diabetes)")
        st.caption("Tune threshold to trade precision vs recall for screening vs confirmatory use.")

with left:
    st.subheader("Single patient prediction")
    with st.form("single"):
        c1, c2, c3, c4 = st.columns(4)
        Pregnancies = c1.number_input("Pregnancies", 0, 20, 1)
        Glucose = c2.number_input("Glucose", 0, 300, 120)
        BloodPressure = c3.number_input("BloodPressure", 0, 200, 70)
        SkinThickness = c4.number_input("SkinThickness", 0, 100, 20)
        c5, c6, c7, c8 = st.columns(4)
        Insulin = c5.number_input("Insulin", 0, 900, 79)
        BMI = c6.number_input("BMI", 0.0, 80.0, 31.6, step=0.1)
        DiabetesPedigreeFunction = c7.number_input("DPF", 0.0, 3.0, 0.47, step=0.01)
        Age = c8.number_input("Age", 18, 100, 33)
        submitted = st.form_submit_button("Predict")

    if submitted:
        if model is None:
            st.stop()
        row = pd.DataFrame([{
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age,
        }])
        X = preproc.transform(row[features])
        proba = float(model.predict_proba(X)[0,1])
        pred = int(proba >= threshold)
        st.markdown(f"### Risk score: **{proba:.3f}**  →  Prediction: **{'Diabetic' if pred else 'Non-diabetic'}**")
        st.progress(min(max(proba,0),1))

st.markdown("---")
st.subheader("Batch scoring (CSV)")
st.caption("Upload a CSV with columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")
upl = st.file_uploader("Upload CSV", type=["csv"])
if upl is not None and model is not None:
    try:
        df = pd.read_csv(upl)
        miss = [c for c in features if c not in df.columns]
        if miss:
            st.error(f"Missing required columns: {miss}")
        else:
            X = preproc.transform(df[features])
            scores = model.predict_proba(X)[:,1]
            preds = (scores >= threshold).astype(int)
            out = df.copy()
            out["risk_score"] = scores
            out["prediction"] = preds
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"),
                               file_name="diabetes_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to score: {e}")
