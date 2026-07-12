import pandas as pd
import os

def load_dataset(path="student_habits_performance.csv"):
    """
    Load the Kaggle Student Habits vs Academic Performance dataset.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please download it from Kaggle "
            "and place it in the data folder."
        )

    df = pd.read_csv(path)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset.head())
