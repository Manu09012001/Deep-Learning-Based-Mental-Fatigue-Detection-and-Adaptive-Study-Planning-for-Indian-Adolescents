import numpy as np
import pandas as pd

def generate_study_plan(fatigue_prob, study_hours, sleep_hours):
    """
    Generate adaptive study plan based on fatigue level.
    """

    plan = {}

    # Determine fatigue level
    if fatigue_prob < 0.3:
        level = "Low Fatigue"
    elif fatigue_prob < 0.7:
        level = "Moderate Fatigue"
    else:
        level = "High Fatigue"

    plan["fatigue_level"] = level
    plan["fatigue_probability"] = round(fatigue_prob, 2)

    # Adaptive rules
    if level == "Low Fatigue":
        plan["recommended_study_hours"] = min(study_hours + 2, 10)
        plan["break_frequency"] = "After every 90 minutes"
        plan["suggestion"] = "You can focus on deep learning tasks and problem solving."

    elif level == "Moderate Fatigue":
        plan["recommended_study_hours"] = study_hours
        plan["break_frequency"] = "After every 60 minutes"
        plan["suggestion"] = "Balance study with short breaks and light revision."

    else:
        plan["recommended_study_hours"] = max(study_hours - 2, 1)
        plan["break_frequency"] = "After every 30–40 minutes"
        plan["suggestion"] = "Reduce load, focus on revision and rest."

    # Sleep adjustment suggestion
    if sleep_hours < 6:
        plan["sleep_advice"] = "Increase sleep to at least 7–8 hours"
    else:
        plan["sleep_advice"] = "Sleep pattern is healthy"

    return plan


def demo():
    # Example input (you will replace with model output later)
    sample_fatigue_prob = 0.72
    study_hours = 5
    sleep_hours = 6

    plan = generate_study_plan(sample_fatigue_prob, study_hours, sleep_hours)

    print("\n=== Adaptive Study Plan ===")
    for k, v in plan.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    demo()
