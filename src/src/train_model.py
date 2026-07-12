import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def load_data(path="student_habits_performance.csv"):
    df = pd.read_csv(path)

    X = df.drop("fatigue_label", axis=1)
    y = df["fatigue_label"]

    return X, y


def build_model(input_dim):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    # Load dataset
    X, y = load_data()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build model
    model = build_model(X_train.shape[1])

    # Train model
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    model.save("fatigue_model.h5")
    print("\nModel saved as fatigue_model.h5")


if __name__ == "__main__":
    main()
