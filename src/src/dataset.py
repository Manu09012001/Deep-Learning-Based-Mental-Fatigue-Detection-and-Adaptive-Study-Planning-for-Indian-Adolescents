import pandas as pd
import os

def load_dataset(path="student_habits_performance_with_fatigue.csv"):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please download it from link "
            "and place it in the data folder."
        )

    df = pd.read_csv(path)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset.head())
