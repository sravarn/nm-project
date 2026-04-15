import base64
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

FEATURE_COLUMNS = ["hours", "attendance", "previous_score", "assignment"]
TARGET_COLUMN = "final_score"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the training data and validate the expected columns."""
    dataset = pd.read_csv(csv_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")

    cleaned_dataset = dataset[required_columns].dropna()
    if cleaned_dataset.empty:
        raise ValueError("Dataset does not contain any complete rows.")

    return cleaned_dataset.astype(float)


def split_features_and_target(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the feature matrix X and target vector y."""
    feature_matrix = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    target_values = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    return feature_matrix, target_values


def build_design_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Add an intercept column to the feature matrix."""
    intercept_column = np.ones((feature_matrix.shape[0], 1), dtype=float)
    return np.hstack((intercept_column, feature_matrix))


def solve_least_squares(feature_matrix: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    """Solve theta = (X^T X)^-1 X^T y for the dataset."""
    design_matrix = build_design_matrix(feature_matrix)
    x_transpose = design_matrix.T
    xtx = x_transpose @ design_matrix
    xty = x_transpose @ target_values

    try:
        theta = np.linalg.inv(xtx) @ xty
    except np.linalg.LinAlgError:
        # Pseudo-inverse keeps the app working if the matrix becomes singular.
        theta = np.linalg.pinv(xtx) @ xty

    return theta


def train_model(csv_path: str | Path) -> np.ndarray:
    """Train the custom least squares model and return theta."""
    dataset = load_dataset(csv_path)
    feature_matrix, target_values = split_features_and_target(dataset)
    return solve_least_squares(feature_matrix, target_values)


def load_model_data(csv_path: str | Path) -> dict:
    """Load the dataset and trained least squares model together."""
    dataset = load_dataset(csv_path)
    feature_matrix, target_values = split_features_and_target(dataset)
    return {
        "dataset": dataset,
        "least_squares_theta": solve_least_squares(feature_matrix, target_values),
    }


def predict(feature_values: list[float], theta: np.ndarray) -> float:
    """Predict the final score using the custom least squares model."""
    if len(feature_values) != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} input values.")

    feature_array = np.array([1.0, *feature_values], dtype=float)
    return float(feature_array @ theta)


def calculate_required_study_hours(feature_values: list[float], target_score: float, theta: np.ndarray) -> dict:
    """Estimate the study hours needed to reach the target score."""
    if len(feature_values) != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} input values.")

    current_hours = feature_values[0]
    current_prediction = predict(feature_values, theta)
    hour_coefficient = float(theta[1])

    if current_prediction >= target_score:
        return {
            "required_hours": current_hours,
            "additional_hours": 0.0,
            "message": "You are already on track.",
        }

    if abs(hour_coefficient) < 1e-9:
        return {
            "required_hours": None,
            "additional_hours": None,
            "message": "Study hours cannot be estimated safely from the current model.",
        }

    if hour_coefficient < 0:
        return {
            "required_hours": None,
            "additional_hours": None,
            "message": "The model cannot recommend extra study hours safely because the study-hours coefficient is not positive.",
        }

    other_factors = (
        float(theta[0])
        + float(theta[2]) * feature_values[1]
        + float(theta[3]) * feature_values[2]
        + float(theta[4]) * feature_values[3]
    )
    required_hours = (target_score - other_factors) / hour_coefficient
    required_hours = max(0.0, required_hours)
    additional_hours = max(0.0, required_hours - current_hours)

    if additional_hours <= 0:
        message = "You are already on track."
    else:
        message = f"You need to study {additional_hours:.2f} more hours to reach your target score."

    return {
        "required_hours": required_hours,
        "additional_hours": additional_hours,
        "message": message,
    }


def build_hour_line_features(dataset: pd.DataFrame, hour_values: np.ndarray) -> np.ndarray:
    """Create line points by varying study hours and fixing other inputs at their mean values."""
    mean_values = dataset[FEATURE_COLUMNS].mean()
    line_features = np.column_stack(
        (
            hour_values,
            np.full_like(hour_values, mean_values["attendance"], dtype=float),
            np.full_like(hour_values, mean_values["previous_score"], dtype=float),
            np.full_like(hour_values, mean_values["assignment"], dtype=float),
        )
    )
    return line_features


def create_graph_image(
    dataset: pd.DataFrame,
    least_squares_theta: np.ndarray,
    feature_values: list[float],
    least_squares_prediction: float,
) -> str:
    """Create a base64-encoded graph for the prediction page."""
    hour_values = dataset["hours"].to_numpy(dtype=float)
    final_scores = dataset[TARGET_COLUMN].to_numpy(dtype=float)

    line_hours = np.linspace(hour_values.min(), hour_values.max(), 100)
    line_features = build_hour_line_features(dataset, line_hours)
    least_squares_line = [predict(row.tolist(), least_squares_theta) for row in line_features]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(
        hour_values,
        final_scores,
        color="#1d4ed8",
        alpha=0.75,
        label="Actual Final Scores",
    )
    axis.plot(
        line_hours,
        least_squares_line,
        color="#136f63",
        linewidth=2.5,
        label="Least Squares Line",
    )

    # Highlight the current student's prediction on the graph.
    axis.scatter(
        [feature_values[0]],
        [least_squares_prediction],
        color="#065f46",
        s=90,
        label="Current Prediction",
        zorder=5,
    )

    axis.set_title("Study Hours vs Final Score")
    axis.set_xlabel("Study Hours")
    axis.set_ylabel("Final Score")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    image_buffer = BytesIO()
    figure.savefig(image_buffer, format="png", bbox_inches="tight")
    plt.close(figure)

    image_buffer.seek(0)
    return base64.b64encode(image_buffer.getvalue()).decode("utf-8")
