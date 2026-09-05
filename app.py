import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="PV Fault Predictor", page_icon="☀️", layout="centered")
@st.cache_resource
# ---- Load the trained model and the feature list saved from Colab ----
# Both files must sit in the same folder as this app.py (or be committed to
# the GitHub repo alongside it) so Streamlit Cloud can find them.
model = joblib.load("pv_fault_model.joblib")
FEATURES = joblib.load("pv_fault_model_features.joblib")

st.title("☀️ Solar PV Fault Predictor")
st.write(
    "Enter a system's live sensor readings below. The model predicts whether "
    "the system is operating normally or is likely showing one of four fault "
    "conditions: degradation, dust/soiling, inverter fault, or shading."
)

st.subheader("Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    capacity_kw = st.number_input("Capacity (kW)", min_value=0.0, value=5.0, step=0.5)
    panel_age_days = st.number_input("Panel age (days)", min_value=0, value=800, step=1)
    irradiance_w_m2 = st.number_input("Irradiance (W/m²)", min_value=0.0, value=600.0, step=10.0)
    ambient_temp_c = st.number_input("Ambient temperature (°C)", value=28.0, step=0.5)
    voltage_v = st.number_input("Voltage (V)", min_value=0.0, value=36.5, step=0.1)
    current_a = st.number_input("Current (A)", min_value=0.0, value=30.0, step=0.5)

with col2:
    panel_temp_c = st.number_input("Panel temperature (°C)", value=45.0, step=0.5)
    days_since_cleaning = st.number_input("Days since cleaning", min_value=0, value=10, step=1)
    dc_power_w = st.number_input("DC power output (W)", min_value=0.0, value=1500.0, step=10.0)
    expected_power_w = st.number_input("Expected power (W)", min_value=0.0, value=1550.0, step=10.0)
    performance_ratio = st.number_input(
        "Performance ratio (DC power ÷ expected power)", min_value=0.0, value=0.95, step=0.01
    )

if st.button("Predict Fault Status", type="primary"):
    new_reading = pd.DataFrame([{
        "capacity_kw": capacity_kw,
        "panel_age_days": panel_age_days,
        "irradiance_w_m2": irradiance_w_m2,
        "ambient_temp_c": ambient_temp_c,
        "panel_temp_c": panel_temp_c,
        "days_since_cleaning": days_since_cleaning,
        "voltage_v": voltage_v,
        "current_a": current_a,
        "dc_power_w": dc_power_w,
        "expected_power_w": expected_power_w,
        "performance_ratio": performance_ratio,
    }])[FEATURES]  # enforce the exact column order the model was trained on

    prediction = model.predict(new_reading)[0]
    proba = model.predict_proba(new_reading)[0]
    proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba}).sort_values(
        "Probability", ascending=False
    )

    if prediction == "normal":
        st.success(f"Prediction: **{prediction.upper()}** — system appears to be operating normally.")
    else:
        st.error(f"Prediction: **{prediction.upper()}** — this reading pattern matches a fault condition.")

    st.subheader("Class Probabilities")
    st.bar_chart(proba_df.set_index("Class"))
    st.dataframe(proba_df, hide_index=True)

st.caption(
    "This tool supports an engineer's decision; it does not replace inspection "
    "or maintenance judgement."
)
