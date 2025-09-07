# 🩺 Diabetes Risk Predictor (Healthcare Classification)

Binary classifier that predicts **diabetes risk** (Pima Indians Diabetes dataset).  
Includes training with **SMOTE** for imbalance and deployment via **Streamlit**.

## ✨ Features
- Models: Logistic Regression, Random Forest, (optional) XGBoost
- Imbalance handling: **SMOTE**
- Metrics: ROC-AUC, Precision/Recall/F1 (positive class)
- **Threshold slider** to tune decision policy
- Batch scoring (CSV) + single-patient form

## 📦 Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
