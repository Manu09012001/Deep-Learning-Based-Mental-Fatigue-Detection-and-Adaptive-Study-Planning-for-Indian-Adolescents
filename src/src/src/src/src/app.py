import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

from adaptive_study_planner import generate_study_plan


# Load model
model = tf.keras.models.load_model("fatigue_model.h5")


# Load dataset for scaler fitting (simple approach)
df = pd.read_csv("mental_fatigue_dataset.csv")
X = df.drop("fatigue_label", axis=1)

scaler = StandardScaler()
scaler.fit(X)


def predict_fatigue(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prob = model.predict(input_scaled)[0][0]
    return float(prob)


# UI
st.set_page_config(page_title="Mental Fatigue Detection", layout="centered")

st.title("🧠 Mental Fatigue Detection System")
st.write("AI-based system for Indian adolescents study behavior analysis")

st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0)
sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
screen_time = st.sidebar.slider("Screen Time", 0.0, 12.0, 6.0)
break_time = st.sidebar.slider("Break Time", 0.0, 5.0, 1.5)
stress_level = st.sidebar.slider("Stress Level", 1, 10, 5)

if st.button("Predict Fatigue"):

    input_data = [study_hours, sleep_hours, screen_time, break_time, stress_level]

    fatigue_prob = predict_fatigue(input_data)

    st.subheader("Prediction Result")

    st.write(f"**Fatigue Probability:** {fatigue_prob:.2f}")

    if fatigue_prob > 0.7:
        st.error("High Fatigue Detected ⚠️")
    elif fatigue_prob > 0.4:
        st.warning("Moderate Fatigue Detected ⚡")
    else:
        st.success("Low Fatigue 👍")

    plan = generate_study_plan(fatigue_prob, study_hours, sleep_hours)

    st.subheader("Adaptive Study Plan")

    for k, v in plan.items():
        st.write(f"**{k}:** {v}")
