import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="PV Fault Predictor", page_icon="☀️", layout="wide")

# ---------------------------------------------------------------------------
# Cached loaders — files load once and stay in memory across reruns/pages
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("pv_fault_model.joblib")
    features = joblib.load("pv_fault_model_features.joblib")
    return model, features


@st.cache_resource
def load_stats():
    return joblib.load("pv_fault_model_stats.joblib")


model, FEATURES = load_model()
stats = load_stats()

FAULT_INFO = {
    "normal": ("✅", "System operating as expected."),
    "degradation": ("📉", "Gradual efficiency loss, typically from aging panels."),
    "dust_soiling": ("🌫️", "Output reduced by dirt or dust build-up on the panels."),
    "inverter_fault": ("⚡", "The inverter is not converting power correctly."),
    "shading": ("🌤️", "Partial shading is cutting into expected output."),
}


# ---------------------------------------------------------------------------
# Page 1 — Dashboard
# ---------------------------------------------------------------------------
def page_dashboard():
    st.title("☀️ Solar PV Fleet Health Dashboard")
    st.caption("Overview of the historical data and model behind the fault predictor.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Systems Monitored", stats["n_systems"])
    col2.metric("Historical Readings", f"{stats['n_readings']:,}")
    col3.metric("Model Accuracy", f"{stats['accuracy'] * 100:.1f}%")
    col4.metric("Avg. Recall on Faults", f"{stats['classification_report']['macro avg']['recall'] * 100:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Reading Distribution by Condition")
        counts = pd.Series(stats["class_counts"]).sort_values(ascending=False)
        st.bar_chart(counts)
        st.caption(
            f"{counts['normal'] / stats['n_readings'] * 100:.1f}% of historical readings are normal — "
            "fault conditions are comparatively rare, which is why recall (not just accuracy) matters most."
        )

    with right:
        st.subheader("What the Model Relies On Most")
        importances = pd.Series(stats["feature_importances"]).sort_values()
        st.bar_chart(importances)
        st.caption("performance_ratio (actual vs. expected power) is typically the strongest single signal.")

    st.divider()
    st.subheader("Confusion Matrix (Test Set)")
    cm = np.array(stats["confusion_matrix"])
    labels = stats["class_labels"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Rows are the true condition, columns are what the model predicted. A clean diagonal means few mix-ups between fault types.")


# ---------------------------------------------------------------------------
# Page 2 — Predict
# ---------------------------------------------------------------------------
def page_predict():
    st.title("🔮 Predict Fault Status")
    st.write(
        "Enter a system's live sensor readings to check whether it's operating "
        "normally or showing signs of a fault."
    )

    with st.form("prediction_form"):
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
        submitted = st.form_submit_button("Predict Fault Status", type="primary", use_container_width=True)

    if submitted:
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
        }])[FEATURES]

        prediction = model.predict(new_reading)[0]
        proba = model.predict_proba(new_reading)[0]
        proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )

        icon, description = FAULT_INFO.get(prediction, ("❓", ""))
        if prediction == "normal":
            st.success(f"{icon} **{prediction.upper()}** — {description}")
        else:
            st.error(f"{icon} **{prediction.upper()}** — {description}")

        st.subheader("Class Probabilities")
        st.bar_chart(proba_df.set_index("Class"))
        st.dataframe(proba_df, hide_index=True, use_container_width=True)

    st.caption("This tool supports an engineer's decision; it does not replace inspection or maintenance judgement.")


# ---------------------------------------------------------------------------
# Page 3 — About
# ---------------------------------------------------------------------------
def page_about():
    st.title("ℹ️ About This Model")
    st.write(
        "This tool predicts the operating condition of a solar PV system from its "
        "live sensor readings, trained on historical data from 8 systems."
    )

    st.subheader("Fault Conditions")
    for label, (icon, desc) in FAULT_INFO.items():
        st.markdown(f"**{icon} {label}** — {desc}")

    st.subheader("Model Details")
    st.markdown(
        f"""
        - **Model type:** Random Forest Classifier (200 trees, class-balanced)
        - **Training data:** {stats['n_readings']:,} hourly readings from {stats['n_systems']} systems
        - **Overall accuracy:** {stats['accuracy'] * 100:.1f}%
        - **Features used:** {', '.join(stats['features'])}
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - Trained on data from a specific set of 8 systems — accuracy on other
          hardware or climates isn't guaranteed.
        - A prediction is a decision-support signal, not a substitute for a
          physical inspection.
        - Rare fault classes (inverter faults, shading) have fewer historical
          examples than "normal" readings.
        """
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("☀️ PV Fault Predictor")
st.sidebar.caption("Solar system health monitoring")

pg = st.navigation([
    st.Page(page_dashboard, title="Dashboard", icon="📊", default=True),
    st.Page(page_predict, title="Predict", icon="🔮"),
    st.Page(page_about, title="About", icon="ℹ️"),
])
pg.run()
