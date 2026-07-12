import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def load_data():
    df = pd.read_csv("student_habits_performance.csv")

    X = df.drop("fatigue_label", axis=1).values
    y = df["fatigue_label"].values

    return X, y


def reshape_for_lstm(X):
    # Convert to time-series format (samples, timesteps, features)
    return X.reshape((X.shape[0], 1, X.shape[1]))


def build_model(input_shape):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))

    model.add(LSTM(32))
    model.add(Dropout(0.3))

    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    X, y = load_data()

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape for LSTM
    X = reshape_for_lstm(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Train model
    model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Save model
    model.save("lstm_fatigue_model.h5")
    print("LSTM model saved successfully")


if __name__ == "__main__":
    main()
