import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Stroke Prediction", layout="wide")
st.title("Stroke Prediction")

st.sidebar.header("Inputs")
model_path = st.sidebar.text_input("Model path (.pkl)", value="lda_tuned_model.pkl")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

st.subheader("Enter Patient Details")
with st.form("patient_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    hypertension_bool = st.selectbox("Hypertension", [False, True])
    heart_disease_bool = st.selectbox("Heart Disease", [False, True])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Predict")

if not submitted:
    st.stop()

model = load_model(model_path)

X = pd.DataFrame([{
    "age": age,
    "hypertension": 1 if hypertension_bool else 0,
    "heart_disease": 1 if heart_disease_bool else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": gender,
    "smoking_status": smoking_status
}])

y_prob = model.predict_proba(X)[:, 1]
p = float(y_prob[0])
confidence = max(p, 1 - p)

if p < 0.20:
    label = "Less likely"
elif p < 0.50:
    label = "Likely"
else:
    label = "Highly likely"

st.metric("Stroke Probability", f"{p*100:.2f}%")
st.metric("Confidence", f"{confidence*100:.2f}%")
st.success(f"Risk Level: {label}")
