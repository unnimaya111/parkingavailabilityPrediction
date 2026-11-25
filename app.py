import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Parking Availability Prediction",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

:root {
    --accent1: #0EA5E9;   /* sky blue  */
    --accent2: #6366F1;   /* indigo    */
    --box:     #E0F2FE;   /* box fill (light blue) */
    --border:  #38BDF8;   /* box border (darker blue) */
    --text:    #0F172A;   /* main text  */
}

.stApp {
    background: linear-gradient(135deg, #F0F9FF, #EEF2FF);
    font-family: 'Poppins', sans-serif;
}
footer { visibility: hidden; }
.block-container { max-width: 520px !important; padding-top: 3rem !important; }

h1 {
    text-align: center;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.5px;
    padding-bottom: 0.8rem !important;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #334155;
    font-weight: 500;
    margin-bottom: 3rem;
}

[data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] {
    background: var(--box) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    min-height: 72px !important;
    overflow: hidden;
}
div[data-baseweb="input"] > div {
    background: var(--box) !important;
    display: flex;
    align-items: center;
}
div[data-baseweb="select"] *,
input {
    background: transparent !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}

.stNumberInput button {
    background: transparent !important;
    color: var(--accent2) !important;
    border: none !important;
    align-self: center;
    height: 36px;
    width: 36px;
}

div[data-testid="stElementContainer"]:has(.stButton),
.element-container:has(.stButton) { width: 100% !important; }
.stButton, .stButton > button { width: 100%; }
.stButton > button {
    height: 60px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
}
.stButton > button p {
    color: #FFFFFF !important;
    font-size: 1.15rem;
    font-weight: 700;
}

.result {
    margin-top: 1.5rem;
    padding: 1.5rem;
    text-align: center;
    background: #FFFFFF;
    border: 2px solid var(--accent1);
    border-radius: 20px;
}
.result-title {
    color: var(--text);
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.result-value {
    display: inline-block;
    margin: 0.7rem 0;
    padding: 0.2rem 1.5rem;
    border-radius: 12px;
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
}
.result-sub { color: #334155; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

model_data = joblib.load("parking_model.pkl")
model = model_data["model"]
feature_columns = model_data["feature_columns"]


st.title("Parking Availability Prediction")
st.markdown(
    '<p class="subtitle">Enter the parking details below to predict the parking status.</p>',
    unsafe_allow_html=True
)

parking_ids = [
    col.replace("parking_id_", "")
    for col in feature_columns
    if col.startswith("parking_id_")
]

parking_id = st.selectbox("Parking ID", parking_ids)
capacity = st.number_input("Parking Capacity", min_value=1, value=100)
hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=12, step=1)

predict = st.button("Predict")

if predict:
    input_data = pd.DataFrame({"capacity": [capacity], "hour": [hour]})

    for col in feature_columns:
        if col.startswith("parking_id_"):
            input_data[col] = 1 if col == "parking_id_" + parking_id else 0

    input_data = input_data[feature_columns]
    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result">
        <div class="result-title">Prediction Result</div>
        <div class="result-value">{prediction}</div>
        <div class="result-sub">Parking Status</div>
    </div>
    """, unsafe_allow_html=True)