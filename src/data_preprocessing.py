"""
data_preprocessing.py
----------------------
Loads the raw student habits/performance dataset, cleans it, and prepares
the feature matrix (X) and target vector (y) used for model training.

This mirrors the cleaning + feature-selection steps carried out in the
EDA notebook (notebooks/student_marks_prediction.ipynb), extracted here
as reusable functions so they don't have to be copy-pasted every time
the model is retrained.

Usage:
    from src.data_preprocessing import load_and_prepare_data
    X, y, df = load_and_prepare_data()
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import FEATURES, TARGET, RAW_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV dataset from disk.

    Args:
        path: Path to the CSV file. Defaults to data/student_habits_performance.csv.

    Returns:
        The raw dataset as a pandas DataFrame.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning: drop missing values and duplicate rows.

    Args:
        df: Raw dataframe.

    Returns:
        A cleaned copy of the dataframe.
    """
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns the model actually needs (features + target).

    Args:
        df: Cleaned dataframe.

    Returns:
        A dataframe containing just FEATURES + TARGET.
    """
    return df[FEATURES + [TARGET]].copy()


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the 'part_time_job' column (No -> 0, Yes -> 1).

    Args:
        df: Dataframe containing a 'part_time_job' column.

    Returns:
        Dataframe with 'part_time_job' converted to integers.
    """
    df = df.copy()
    le = LabelEncoder()
    df["part_time_job"] = le.fit_transform(df["part_time_job"])
    return df


def load_and_prepare_data(path=RAW_DATA_PATH):
    """Run the full preprocessing pipeline: load -> clean -> select -> encode.

    Args:
        path: Path to the raw CSV dataset.

    Returns:
        A tuple (X, y, df_model) where:
            X        -- feature matrix (pandas DataFrame)
            y        -- target vector, exam_score (pandas Series)
            df_model -- the cleaned, encoded dataframe used to build X and y
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df = select_features(df)
    df = encode_categorical(df)

    X = df[FEATURES]
    y = df[TARGET]
    return X, y, df


if __name__ == "__main__":
    X, y, df = load_and_prepare_data()
    print(f"Loaded {len(df)} rows after cleaning.")
    print(f"Features: {list(X.columns)}")
    print(df.head())
