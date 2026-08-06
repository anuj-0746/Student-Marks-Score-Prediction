"""
predict.py
----------
Loads the trained model (models/best_model.pkl) and exposes a simple
predict_score() function used by the Streamlit app (app/app.py) and by
this file's command-line interface.

Usage (CLI):
    python -m src.predict --study_hours 6 --attendance 92 \
        --mental_health 8 --sleep_hours 7 --part_time_job No
"""

import argparse

import joblib
import numpy as np

from src.utils import FEATURES, MODEL_PATH, clip_score, encode_part_time_job

_model = None  # Lazily-loaded, cached model instance.


def load_model(path=MODEL_PATH):
    """Load the trained model from disk, caching it after the first call.

    Args:
        path: Path to the .pkl model file.

    Returns:
        The loaded scikit-learn estimator.
    """
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict_score(
    study_hours: float,
    attendance: float,
    mental_health: float,
    sleep_hours: float,
    part_time_job: str,
) -> float:
    """Predict a student's exam score (0-100) from the five input features.

    Args:
        study_hours: Study hours per day (0-12).
        attendance: Attendance percentage (0-100).
        mental_health: Mental health rating (1-10).
        sleep_hours: Sleep hours per day (0-12).
        part_time_job: "Yes" or "No".

    Returns:
        Predicted exam score, clipped to the [0, 100] range.
    """
    model = load_model()

    ptj_encoded = encode_part_time_job(part_time_job)
    input_data = np.array(
        [[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]]
    )

    prediction = model.predict(input_data)[0]
    return clip_score(prediction)


def _parse_args():
    parser = argparse.ArgumentParser(description="Predict a student's exam score.")
    parser.add_argument("--study_hours", type=float, required=True)
    parser.add_argument("--attendance", type=float, required=True)
    parser.add_argument("--mental_health", type=float, required=True)
    parser.add_argument("--sleep_hours", type=float, required=True)
    parser.add_argument("--part_time_job", type=str, choices=["Yes", "No"], required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    score = predict_score(
        study_hours=args.study_hours,
        attendance=args.attendance,
        mental_health=args.mental_health,
        sleep_hours=args.sleep_hours,
        part_time_job=args.part_time_job,
    )
    print(f"Features used: {FEATURES}")
    print(f"Predicted Exam Score: {score:.2f}")


if __name__ == "__main__":
    main()
