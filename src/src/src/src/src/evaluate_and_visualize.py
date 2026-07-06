import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def load_data(path="mental_fatigue_dataset.csv"):
    df = pd.read_csv(path)

    X = df.drop("fatigue_label", axis=1)
    y = df["fatigue_label"]

    return X, y


def main():
    # Load dataset
    X, y = load_data()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model
    model = tf.keras.models.load_model("fatigue_model.h5")

    # Predictions
    y_pred_prob = model.predict(X_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # ---------------- Accuracy Plot ----------------
    print("\nModel Evaluation Completed")

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Mental Fatigue Detection")
    plt.show()

    # ---------------- Fatigue Distribution ----------------
    plt.figure()
    plt.hist(y_pred_prob, bins=20)
    plt.title("Predicted Fatigue Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.show()

    # ---------------- Feature Insight ----------------
    plt.figure()
    plt.bar(X.columns, np.mean(X, axis=0))
    plt.xticks(rotation=45)
    plt.title("Average Feature Contribution")
    plt.show()


if __name__ == "__main__":
    main()
