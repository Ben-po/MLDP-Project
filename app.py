import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

st.set_page_config(page_title="Stroke Prediction", layout="wide")
st.title("Stroke Prediction Dashboard")

st.sidebar.header("Inputs")
model_path = st.sidebar.text_input("Model path (.pkl)", value="lda_tuned_model.pkl")

risk_low = st.sidebar.slider("Low Risk Upper Bound", 0.01, 0.49, 0.10, 0.01)
risk_med = st.sidebar.slider("Medium Risk Upper Bound", 0.11, 0.90, 0.30, 0.01)
if risk_low >= risk_med:
    st.sidebar.error("Low risk bound must be lower than medium risk bound.")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def create_risk_bands(probs, low, med):
    return pd.cut(
        probs,
        bins=[0, low, med, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

# -----------------------------
# Manual input form
# -----------------------------
st.subheader("Enter Patient Details")

with st.form("patient_form"):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
    smoking_status = st.selectbox(
        "Smoking Status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"]
    )
    submitted = st.form_submit_button("Predict Risk")

if not submitted:
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Build input row (must match training columns)
X = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}])

try:
    y_prob = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(
        "Prediction failed. Ensure the model expects the same columns. "
        "If you trained with preprocessing, save and load a Pipeline."
    )
    st.exception(e)
    st.stop()

risk_band = create_risk_bands(pd.Series(y_prob), risk_low, risk_med).iloc[0]

st.metric("Stroke Probability", f"{y_prob[0]:.3f}")
st.metric("Risk Band", str(risk_band))
