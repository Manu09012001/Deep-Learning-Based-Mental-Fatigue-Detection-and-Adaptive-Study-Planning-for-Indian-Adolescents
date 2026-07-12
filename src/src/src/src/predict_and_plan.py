import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

from adaptive_study_planner import generate_study_plan


# Load trained model
def load_model(model_path="fatigue_model.h5"):
    model = tf.keras.models.load_model(model_path)
    return model


# Load dataset to fit scaler (simple approach for demo)
def get_scaler(data_path="student_habits_performance.csv"):
    df = pd.read_csv(data_path)

    X = df.drop("fatigue_label", axis=1)

    scaler = StandardScaler()
    scaler.fit(X)

    return scaler


def predict_fatigue(model, scaler, input_data):
    """
    input_data = [study_hours, sleep_hours, screen_time, break_time, stress_level]
    """

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    fatigue_prob = model.predict(input_scaled)[0][0]

    return float(fatigue_prob)


def main():
    # Load model and scaler
    model = load_model()
    scaler = get_scaler()

    print("\n=== Mental Fatigue Detection System ===")

    # Example input (you can replace with user input / app input)
    study_hours = float(input("Enter study hours: "))
    sleep_hours = float(input("Enter sleep hours: "))
    screen_time = float(input("Enter screen time: "))
    break_time = float(input("Enter break time: "))
    stress_level = float(input("Enter stress level (1-10): "))

    input_data = [study_hours, sleep_hours, screen_time, break_time, stress_level]

    # Prediction
    fatigue_prob = predict_fatigue(model, scaler, input_data)

    print(f"\nPredicted Fatigue Probability: {fatigue_prob:.2f}")

    # Generate adaptive plan
    plan = generate_study_plan(fatigue_prob, study_hours, sleep_hours)

    print("\n=== Adaptive Study Plan ===")
    for k, v in plan.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
