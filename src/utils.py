"""
utils.py
--------
Shared constants, paths, and small helper functions used across the
data preprocessing, training, and prediction scripts.

Keeping these in one place means every script (and the Streamlit app)
agrees on the same feature names, feature order, and file locations.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# BASE_DIR points to the project root (the folder that contains src/, data/,
# models/, app/, etc.), regardless of which script/notebook imports this file.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "student_habits_performance.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# These are the five input features the model was trained on, in the exact
# order the model expects them. If this order changes, the model must be
# retrained (or the prediction input must be re-ordered to match).
FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "mental_health_rating",
    "sleep_hours",
    "part_time_job",
]

TARGET = "exam_score"

# The only categorical column that feeds into the model. It is label-encoded
# as: No -> 0, Yes -> 1 (alphabetical order, which is how sklearn's
# LabelEncoder sorts "No"/"Yes" by default).
CATEGORICAL_FEATURES = ["part_time_job"]
PART_TIME_JOB_MAP = {"No": 0, "Yes": 1}


def encode_part_time_job(value: str) -> int:
    """Convert the human-readable Yes/No answer into the 0/1 encoding
    the model was trained on.

    Args:
        value: "Yes" or "No" (case-insensitive).

    Returns:
        1 if the value represents "Yes", otherwise 0.
    """
    return PART_TIME_JOB_MAP.get(str(value).strip().title(), 0)


def clip_score(score: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clip a predicted exam score into the valid [0, 100] range.

    Regression models can occasionally predict values slightly outside the
    real-world bounds of an exam score, so predictions are clipped before
    being shown to the user.
    """
    return max(low, min(high, score))


def ensure_dirs_exist() -> None:
    """Create the data/ and models/ directories if they don't already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
