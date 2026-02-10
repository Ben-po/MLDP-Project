import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Stroke Prediction", layout="wide")

# ---- Health/clinical theme ----
st.markdown(
    """
    <style>
    :root {
        --clinic-green: #2E7D32;
        --clinic-light: #E8F5E9;
        --clinic-blue: #0F4C81;
        --clinic-gray: #F5F7FA;
        --card-border: #DDE5EE;
    }
    .app-header {
        background: linear-gradient(90deg, #E8F5E9 0%, #F5F7FA 100%);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .app-header h1 {
        color: var(--clinic-blue);
        font-size: 32px;
        margin: 0;
    }
    .app-header p {
        color: #3E4A59;
        margin: 6px 0 0 0;
    }
    .section-card {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px 18px 8px 18px;
        margin-bottom: 16px;
    }
    .section-title {
        color: var(--clinic-green);
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-card {
        background: var(--clinic-light);
        border: 1px solid #C8E6C9;
        border-radius: 12px;
        padding: 12px;
    }
    .stButton>button {
        background-color: var(--clinic-green);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #256C2A;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-header">
        <h1>Stroke Risk Assessment</h1>
        <p>Clinical screening support with wellness-focused insights</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Inputs")
model_path = st.sidebar.text_input("Model path (.pkl)", value="lda_tuned_model.pkl")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def show_bmi_calc():
    st.session_state["show_bmi_calc"] = True

def calculate_bmi():
    height_cm = st.session_state.get("height_cm", 170.0)
    weight_kg = st.session_state.get("weight_kg", 70.0)
    if height_cm > 0:
        calc_bmi = weight_kg / ((height_cm / 100) ** 2)
        st.session_state["bmi"] = round(calc_bmi, 1)

# Initialize defaults
st.session_state.setdefault("show_bmi_calc", False)
st.session_state.setdefault("bmi", 25.0)
st.session_state.setdefault("height_cm", 170.0)
st.session_state.setdefault("weight_kg", 70.0)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🩺 Patient Details</div>', unsafe_allow_html=True)

with st.form("patient_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    hypertension_bool = st.selectbox("Hypertension", [False, True])
    heart_disease_bool = st.selectbox("Heart Disease", [False, True])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0, step=1.0)

    # BMI label + calculator button side-by-side
    bmi_label_col, bmi_btn_col = st.columns([3, 1])
    with bmi_label_col:
        st.markdown("**BMI**")
    with bmi_btn_col:
        st.form_submit_button("Calculator", on_click=show_bmi_calc)

    # BMI input (label hidden because we already show a label above)
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=st.session_state["bmi"],
        step=0.1,
        label_visibility="collapsed",
        key="bmi"
    )

    if st.session_state["show_bmi_calc"]:
        st.markdown("**BMI Calculator**")
        st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=st.session_state["height_cm"],
            step=1.0,
            key="height_cm"
        )
        st.number_input(
            "Weight (kg)",
            min_value=10.0,
            max_value=300.0,
            value=st.session_state["weight_kg"],
            step=0.5,
            key="weight_kg"
        )
        st.form_submit_button("Calculate BMI", on_click=calculate_bmi)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Predict")

st.markdown("</div>", unsafe_allow_html=True)

if not submitted:
    st.stop()

# ---- Input validation ----
errors = []
if not model_path or not os.path.exists(model_path):
    errors.append("Model file not found. Please provide a valid .pkl path.")
if age <= 0:
    errors.append("Age must be greater than 0.")
if avg_glucose_level <= 1:
    errors.append("Average glucose level must be greater than 1.")
if bmi <= 1:
    errors.append("BMI must be greater than 1.")
if avg_glucose_level > 600:
    errors.append("Average glucose level seems too high. Please verify.")
if bmi > 80:
    errors.append("BMI seems too high. Please verify.")

if errors:
    for e in errors:
        st.error(e)
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

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Risk Summary</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Stroke Probability", f"{p*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.success(f"Risk Level: {label}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
