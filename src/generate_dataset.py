import numpy as np
import pandas as pd
import random

def simulate_student_data(num_samples=5000, seed=42):
    """
    Generate synthetic dataset for Mental Fatigue Detection
    for Indian adolescent study behavior.
    """

    np.random.seed(seed)
    random.seed(seed)

    data = []

    for i in range(num_samples):

        # Basic behavioral features
        study_hours = np.random.normal(4, 1.5)          # avg study time
        sleep_hours = np.random.normal(7, 1.2)          # sleep duration
        screen_time = np.random.normal(6, 2)            # mobile/laptop usage

        break_time = np.random.normal(1.5, 0.5)         # breaks during study
        stress_level = np.random.randint(1, 10)         # self-reported stress (1-10)

        # Ensure no negative values
        study_hours = max(0, study_hours)
        sleep_hours = max(0, sleep_hours)
        screen_time = max(0, screen_time)
        break_time = max(0, break_time)

        # Fatigue logic (rule + noise)
        fatigue_score = (
            (study_hours * 0.4) +
            (screen_time * 0.3) +
            (stress_level * 0.2) -
            (sleep_hours * 0.3) -
            (break_time * 0.2)
        )

        # Normalize fatigue score into probability
        fatigue_prob = 1 / (1 + np.exp(-0.5 * (fatigue_score - 5)))

        # Binary label
        fatigue_label = 1 if fatigue_prob > 0.5 else 0

        data.append([
            study_hours,
            sleep_hours,
            screen_time,
            break_time,
            stress_level,
            fatigue_score,
            fatigue_label
        ])

    df = pd.DataFrame(data, columns=[
        "study_hours",
        "sleep_hours",
        "screen_time",
        "break_time",
        "stress_level",
        "fatigue_score",
        "fatigue_label"
    ])

    return df


def save_dataset(path="mental_fatigue_dataset.csv", samples=5000):
    df = simulate_student_data(samples)
    df.to_csv(path, index=False)
    print(f"Dataset saved at: {path}")


if __name__ == "__main__":
    save_dataset()

