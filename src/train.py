"""
train.py
--------
Trains and compares three regression models (Linear Regression, Decision
Tree, Random Forest) using GridSearchCV, evaluates them with RMSE and R²,
and saves the best-performing model to models/best_model.pkl.

This is the script version of the model-training section of
notebooks/student_marks_prediction.ipynb.

Usage:
    python -m src.train
    (run from the project root, with the virtual environment active)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.data_preprocessing import load_and_prepare_data
from src.utils import MODEL_PATH, ensure_dirs_exist

# Candidate models and their hyperparameter search grids.
MODEL_CONFIGS = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {},
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]},
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {"n_estimators": [50, 100], "max_depth": [5, 10]},
    },
}


def train_and_evaluate(X_train, y_train, X_test, y_test, model_configs=MODEL_CONFIGS):
    """Run GridSearchCV for every candidate model and collect its metrics.

    Args:
        X_train, y_train: Training split.
        X_test, y_test: Test split, used for evaluation.
        model_configs: Dict of {model_name: {"model": estimator, "params": grid}}.

    Returns:
        A pandas DataFrame with one row per model: name, best_params, rmse, r2.
    """
    results = []

    for name, config in model_configs.items():
        print(f"Training {name} ...")
        grid = GridSearchCV(
            config["model"], config["params"], cv=5, scoring="neg_mean_squared_error"
        )
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "model": name,
                "best_params": grid.best_params_,
                "rmse": rmse,
                "r2": r2,
            }
        )
        print(f"  -> RMSE: {rmse:.4f} | R2: {r2:.4f}")

    return pd.DataFrame(results)


def select_best_model(results_df: pd.DataFrame) -> str:
    """Pick the model with the lowest RMSE.

    Args:
        results_df: Output of train_and_evaluate().

    Returns:
        The name of the best-performing model.
    """
    best_row = results_df.sort_values(by="rmse").iloc[0]
    return best_row["model"]


def save_model(model, path=MODEL_PATH) -> None:
    """Persist a fitted model to disk with joblib."""
    ensure_dirs_exist()
    joblib.dump(model, path)
    print(f"Saved best model to: {path}")


def main():
    X, y, _ = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    results_df = train_and_evaluate(X_train, y_train, X_test, y_test)
    print("\nModel comparison:")
    print(results_df.sort_values(by="rmse").to_string(index=False))

    best_model_name = select_best_model(results_df)
    print(f"\nBest model: {best_model_name}")

    # Refit the best model's estimator on the FULL dataset before saving,
    # so the deployed model benefits from all available data.
    final_model = MODEL_CONFIGS[best_model_name]["model"]
    final_model.fit(X, y)

    save_model(final_model)


if __name__ == "__main__":
    main()
