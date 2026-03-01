import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64

# -----------------------------
# Background Image
# -----------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Change image anytime here
set_bg("images/istockphoto-2174551157-612x612.jpg")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/isolation_forest_model.pkl")

st.markdown(
    "<h1 style='text-align: center;'>AI-Based Cyber Threat Detection System</h1>",
    unsafe_allow_html=True
)
st.write("Upload a network traffic CSV file to analyze for anomalies.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# -----------------------------
# Run only after file upload
# -----------------------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Select numeric features
    X = data.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    # Predict anomalies
    preds = model.predict(X)
    scores = model.decision_function(X)

    # Friendly threat labels for UI
    ui_threat_type = []
    for s in scores:
        if s < -0.15:
            ui_threat_type.append("High-Risk Anomaly (Possible Zero-Day Attack)")
        elif s < 0:
            ui_threat_type.append("Mild Suspicious Behavior (Possible Insider Activity)")
        else:
            ui_threat_type.append("Normal Traffic")

    # Create result dataframe
    result = data.loc[X.index].copy()
    result["anomaly_label"] = preds
    result["anomaly_score"] = scores
    result["threat_type"] = ui_threat_type

    st.success("Detection completed successfully")

    st.subheader("Sample Detection Results")
    st.dataframe(result.head(20))

    # -----------------------------
    # Visualization
    # -----------------------------
    st.subheader("Threat Type Distribution")
    st.bar_chart(result["threat_type"].value_counts())

    # -----------------------------
    # Anomaly Summary
    # -----------------------------
    total_records = len(result)
    anomaly_count = (result["anomaly_label"] == -1).sum()
    anomaly_percentage = (anomaly_count / total_records) * 100

    st.subheader("Anomaly Summary")
    st.write(f"Total Records Analyzed: {total_records}")
    st.write(f"Anomalous Records Detected: {anomaly_count}")
    st.write(f"Anomaly Percentage: {anomaly_percentage:.2f}%")

    # -----------------------------
    # Final System Interpretation
    # -----------------------------
    st.subheader("System Interpretation")

    if anomaly_percentage > 30:
        st.error(
            "🚨 High-risk network activity detected. "
            "This traffic shows strong abnormal patterns and requires immediate investigation."
        )
    elif anomaly_percentage > 5:
        st.warning(
            "⚠️ Some suspicious behavior detected. "
            "Continuous monitoring is recommended."
        )
    else:
        st.success(
            "✅ Network traffic appears normal. "
            "No significant threats detected at the moment."
        )