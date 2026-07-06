import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of samples
num_samples = 5000

# Create synthetic study behaviour dataset
data = {
    "Student_ID": np.arange(1, num_samples + 1),
    "Study_Hours": np.round(np.random.uniform(0.5, 8, num_samples), 2),
    "Break_Frequency": np.random.randint(0, 10, num_samples),
    "Sleep_Hours": np.round(np.random.uniform(3, 10, num_samples), 2),
    "Screen_Time": np.round(np.random.uniform(1, 12, num_samples), 2),
    "Typing_Speed": np.random.randint(20, 90, num_samples),
    "Mouse_Clicks": np.random.randint(100, 2000, num_samples),
    "Stress_Level": np.random.randint(1, 11, num_samples),
    "Mood_Score": np.random.randint(1, 11, num_samples)
}

df = pd.DataFrame(data)

# Rule-based fatigue labels (placeholder for model training)
fatigue = []

for _, row in df.iterrows():
    score = 0

    if row["Sleep_Hours"] < 6:
        score += 2

    if row["Study_Hours"] > 5:
        score += 2

    if row["Stress_Level"] >= 7:
        score += 2

    if row["Screen_Time"] > 7:
        score += 1

    if row["Mood_Score"] <= 4:
        score += 2

    if score <= 2:
        fatigue.append("Low")
    elif score <= 5:
        fatigue.append("Moderate")
    else:
        fatigue.append("High")

df["Fatigue_Level"] = fatigue

# Save dataset
df.to_csv("data/raw/sample_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())
