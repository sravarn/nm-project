import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, render_template, request

from model import calculate_required_study_hours, create_graph_image, load_model_data, predict

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.csv"
PROFILE_FILE = BASE_DIR / "profiles.csv"

FIELD_CONFIGS = [
    {
        "name": "student_name",
        "label": "Student Name",
        "placeholder": "e.g. aatrey",
        "type": "text",
    },
    {
        "name": "hours",
        "label": "Study Hours",
        "placeholder": "e.g. 6.5",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "24",
        "min_value": 0.0,
        "max_value": 24.0,
    },
    {
        "name": "attendance",
        "label": "Attendance (%)",
        "placeholder": "e.g. 82",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "100",
        "min_value": 0.0,
        "max_value": 100.0,
    },
    {
        "name": "previous_score",
        "label": "Previous Score",
        "placeholder": "e.g. 74",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "100",
        "min_value": 0.0,
        "max_value": 100.0,
    },
    {
        "name": "assignment",
        "label": "Assignment Marks",
        "placeholder": "e.g. 78",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "100",
        "min_value": 0.0,
        "max_value": 100.0,
    },
    {
        "name": "target_score",
        "label": "Target Score",
        "placeholder": "e.g. 85",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "100",
        "min_value": 0.0,
        "max_value": 100.0,
    },
]

FEATURE_FIELD_NAMES = ["hours", "attendance", "previous_score", "assignment"]
PROFILE_COLUMNS = [
    "name",
    "hours",
    "attendance",
    "previous_score",
    "assignment",
    "target_score",
    "predicted_score",
]
EMPTY_FORM_VALUES = {field["name"]: "" for field in FIELD_CONFIGS}

COMPARISON_FIELDS = [
    {
        "key": "hours",
        "label": "Study Hours",
        "chart_label": "Hours",
        "metric_name": "study hours",
        "winner_text": "studies more than",
    },
    {
        "key": "attendance",
        "label": "Attendance",
        "chart_label": "Attendance",
        "metric_name": "attendance",
        "winner_text": "has higher attendance than",
    },
    {
        "key": "previous_score",
        "label": "Previous Score",
        "chart_label": "Previous",
        "metric_name": "previous score",
        "winner_text": "has a higher previous score than",
    },
    {
        "key": "assignment",
        "label": "Assignment Marks",
        "chart_label": "Assignment",
        "metric_name": "assignment marks",
        "winner_text": "has higher assignment marks than",
    },
    {
        "key": "predicted_score",
        "label": "Predicted Final Score",
        "chart_label": "Predicted",
        "metric_name": "predicted final score",
        "winner_text": "has a higher predicted final score than",
    },
]


def ensure_profile_file(profile_path: Path):
    """Create profiles.csv with headers if it does not exist yet."""
    if not profile_path.exists():
        pd.DataFrame(columns=PROFILE_COLUMNS).to_csv(profile_path, index=False)


def load_profiles(profile_path: Path) -> pd.DataFrame:
    """Read profiles.csv and return a clean DataFrame."""
    ensure_profile_file(profile_path)

    try:
        profiles = pd.read_csv(profile_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=PROFILE_COLUMNS)

    if profiles.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)

    for column in PROFILE_COLUMNS:
        if column not in profiles.columns:
            profiles[column] = ""

    profiles = profiles[PROFILE_COLUMNS].copy()
    profiles["name"] = profiles["name"].fillna("").astype(str).str.strip()
    return profiles


def safe_float(value, fallback=None):
    """Convert values from CSV safely to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def get_saved_profile_names(profile_path: Path) -> list[str]:
    """Return unique saved student names, keeping the latest entry for each name."""
    profiles = load_profiles(profile_path)
    if profiles.empty:
        return []

    named_profiles = profiles[profiles["name"] != ""].copy()
    if named_profiles.empty:
        return []

    latest_profiles = named_profiles.drop_duplicates(subset="name", keep="last")
    return sorted(latest_profiles["name"].tolist(), key=str.lower)


def get_profile_by_name(profile_path: Path, student_name: str) -> dict | None:
    """Return the latest saved profile for a given student name."""
    cleaned_name = student_name.strip()
    if not cleaned_name:
        return None

    profiles = load_profiles(profile_path)
    if profiles.empty:
        return None

    matching_profiles = profiles[profiles["name"].str.lower() == cleaned_name.lower()]
    if matching_profiles.empty:
        return None

    latest_profile = matching_profiles.iloc[-1]
    profile_data = {
        "student_name": str(latest_profile["name"]),
        "hours": safe_float(latest_profile["hours"], 0.0),
        "attendance": safe_float(latest_profile["attendance"], 0.0),
        "previous_score": safe_float(latest_profile["previous_score"], 0.0),
        "assignment": safe_float(latest_profile["assignment"], 0.0),
        "target_score": safe_float(latest_profile["target_score"], 0.0),
        "predicted_score": safe_float(latest_profile["predicted_score"], None),
    }

    if profile_data["predicted_score"] is None and MODEL_STORE is not None:
        feature_values = [
            profile_data["hours"],
            profile_data["attendance"],
            profile_data["previous_score"],
            profile_data["assignment"],
        ]
        profile_data["predicted_score"] = predict(feature_values, MODEL_STORE["least_squares_theta"])

    if profile_data["predicted_score"] is None:
        profile_data["predicted_score"] = 0.0

    return profile_data


def delete_profile_by_name(profile_path: Path, student_name: str) -> bool:
    """Delete all saved rows for the selected student name."""
    cleaned_name = student_name.strip()
    if not cleaned_name:
        return False

    profiles = load_profiles(profile_path)
    if profiles.empty:
        return False

    remaining_profiles = profiles[profiles["name"].str.lower() != cleaned_name.lower()].copy()
    if len(remaining_profiles) == len(profiles):
        return False

    remaining_profiles.to_csv(profile_path, index=False, columns=PROFILE_COLUMNS)
    return True


def load_trained_model():
    try:
        return load_model_data(DATA_FILE), None
    except Exception as exc:  # pragma: no cover - defensive startup handling
        return None, f"Model training failed: {exc}"


MODEL_STORE, MODEL_ERROR = load_trained_model()
ensure_profile_file(PROFILE_FILE)


def parse_form_values(form_data):
    """Validate user input and return clean values for the app and model."""
    cleaned_values = EMPTY_FORM_VALUES.copy()
    numeric_inputs = {}

    for field in FIELD_CONFIGS:
        raw_value = form_data.get(field["name"], "").strip()
        cleaned_values[field["name"]] = raw_value

        if not raw_value:
            raise ValueError(f"{field['label']} is required.")

        if field["type"] == "text":
            continue

        try:
            numeric_value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field['label']} must be a valid number.") from exc

        if numeric_value < field["min_value"] or numeric_value > field["max_value"]:
            raise ValueError(
                f"{field['label']} must be between {field['min_value']} and {field['max_value']}."
            )

        numeric_inputs[field["name"]] = numeric_value

    feature_values = [numeric_inputs[field_name] for field_name in FEATURE_FIELD_NAMES]
    return cleaned_values, numeric_inputs, feature_values


def save_student_profile(profile_path: Path, student_name: str, numeric_inputs: dict, predicted_score: float):
    """Append the latest student profile and prediction to profiles.csv."""
    profile_row = pd.DataFrame(
        [
            {
                "name": student_name,
                "hours": numeric_inputs["hours"],
                "attendance": numeric_inputs["attendance"],
                "previous_score": numeric_inputs["previous_score"],
                "assignment": numeric_inputs["assignment"],
                "target_score": numeric_inputs["target_score"],
                "predicted_score": round(predicted_score, 2),
            }
        ],
        columns=PROFILE_COLUMNS,
    )
    write_header = not profile_path.exists() or profile_path.stat().st_size == 0
    profile_row.to_csv(profile_path, mode="a", header=write_header, index=False)


def build_results(feature_values, target_score):
    """Build the prediction output shown on the page."""
    least_squares_prediction = predict(feature_values, MODEL_STORE["least_squares_theta"])
    study_plan = calculate_required_study_hours(
        feature_values=feature_values,
        target_score=target_score,
        theta=MODEL_STORE["least_squares_theta"],
    )
    graph_image = create_graph_image(
        dataset=MODEL_STORE["dataset"],
        least_squares_theta=MODEL_STORE["least_squares_theta"],
        feature_values=feature_values,
        least_squares_prediction=least_squares_prediction,
    )

    return {
        "least_squares_prediction": least_squares_prediction,
        "study_plan": study_plan,
        "graph_image": graph_image,
    }


def build_comparison_message(field_config: dict, student_one_name: str, student_one_value: float, student_two_name: str, student_two_value: float) -> tuple[str, str]:
    """Describe which student is stronger in a specific comparison row."""
    if student_one_value > student_two_value:
        return f"{student_one_name} {field_config['winner_text']} {student_two_name}.", "student_one"

    if student_two_value > student_one_value:
        return f"{student_two_name} {field_config['winner_text']} {student_one_name}.", "student_two"

    return (
        f"{student_one_name} and {student_two_name} have the same {field_config['metric_name']}.",
        "tie",
    )


def build_comparison_rows(student_one_profile: dict, student_two_profile: dict) -> list[dict]:
    """Build row-by-row comparison details for the compare page."""
    comparison_rows = []

    for field in COMPARISON_FIELDS:
        student_one_value = float(student_one_profile[field["key"]])
        student_two_value = float(student_two_profile[field["key"]])
        insight, winner = build_comparison_message(
            field_config=field,
            student_one_name=student_one_profile["student_name"],
            student_one_value=student_one_value,
            student_two_name=student_two_profile["student_name"],
            student_two_value=student_two_value,
        )
        comparison_rows.append(
            {
                "label": field["label"],
                "student_one_value": f"{student_one_value:.2f}",
                "student_two_value": f"{student_two_value:.2f}",
                "insight": insight,
                "winner": winner,
            }
        )

    return comparison_rows


def create_comparison_graph(student_one_profile: dict, student_two_profile: dict) -> str:
    """Create a bar chart that compares the two selected students."""
    labels = [field["chart_label"] for field in COMPARISON_FIELDS]
    student_one_values = [float(student_one_profile[field["key"]]) for field in COMPARISON_FIELDS]
    student_two_values = [float(student_two_profile[field["key"]]) for field in COMPARISON_FIELDS]

    base_positions = list(range(len(labels)))
    bar_width = 0.36
    student_one_positions = [position - (bar_width / 2) for position in base_positions]
    student_two_positions = [position + (bar_width / 2) for position in base_positions]

    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(
        student_one_positions,
        student_one_values,
        width=bar_width,
        color="#136f63",
        label=student_one_profile["student_name"],
    )
    axis.bar(
        student_two_positions,
        student_two_values,
        width=bar_width,
        color="#1d4ed8",
        label=student_two_profile["student_name"],
    )

    axis.set_xticks(base_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Values")
    axis.set_title("Student Score Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()

    image_buffer = BytesIO()
    figure.savefig(image_buffer, format="png", bbox_inches="tight")
    plt.close(figure)

    image_buffer.seek(0)
    return base64.b64encode(image_buffer.getvalue()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["POST"], endpoint="predict_route")
def index():
    least_squares_prediction = None
    additional_hours_needed = None
    required_total_hours = None
    target_message = None
    graph_image = None
    error = MODEL_ERROR
    form_values = EMPTY_FORM_VALUES.copy()

    if request.method == "POST" and MODEL_STORE is not None:
        try:
            form_values, numeric_inputs, feature_values = parse_form_values(request.form)
            results = build_results(feature_values, numeric_inputs["target_score"])
            least_squares_prediction = results["least_squares_prediction"]
            additional_hours_needed = results["study_plan"]["additional_hours"]
            required_total_hours = results["study_plan"]["required_hours"]
            target_message = results["study_plan"]["message"]
            graph_image = results["graph_image"]

            save_student_profile(
                profile_path=PROFILE_FILE,
                student_name=form_values["student_name"],
                numeric_inputs=numeric_inputs,
                predicted_score=least_squares_prediction,
            )
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            error = str(exc)

    return render_template(
        "index.html",
        field_configs=FIELD_CONFIGS,
        form_values=form_values,
        least_squares_prediction=least_squares_prediction,
        additional_hours_needed=additional_hours_needed,
        required_total_hours=required_total_hours,
        target_message=target_message,
        graph_image=graph_image,
        saved_profile_names=get_saved_profile_names(PROFILE_FILE),
        error=error,
    )


@app.route("/load_profile", methods=["GET"])
def load_profile():
    """Load a saved profile and return it as JSON for the form."""
    student_name = request.args.get("name", "").strip()
    if not student_name:
        return jsonify({"success": False, "message": "Please select a saved profile first."}), 400

    profile = get_profile_by_name(PROFILE_FILE, student_name)
    if profile is None:
        return jsonify({"success": False, "message": "Profile not found."}), 404

    return jsonify(
        {
            "success": True,
            "message": "Profile loaded.",
            "profile": profile,
        }
    )


@app.route("/delete_profile", methods=["POST"])
def delete_profile():
    """Delete the selected saved profile from profiles.csv."""
    request_data = request.get_json(silent=True) or request.form
    student_name = str(request_data.get("name", "")).strip()

    if not student_name:
        return jsonify({"success": False, "message": "Please select a saved profile first."}), 400

    profile_deleted = delete_profile_by_name(PROFILE_FILE, student_name)
    if not profile_deleted:
        return jsonify({"success": False, "message": "Profile not found."}), 404

    return jsonify(
        {
            "success": True,
            "message": "Profile deleted.",
            "saved_profile_names": get_saved_profile_names(PROFILE_FILE),
        }
    )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """Compare two saved student profiles and visualize the difference."""
    comparison_rows = []
    comparison_graph = None
    student_one_profile = None
    student_two_profile = None
    selected_student_one = ""
    selected_student_two = ""
    error = None

    if request.method == "POST":
        selected_student_one = request.form.get("student_one", "").strip()
        selected_student_two = request.form.get("student_two", "").strip()

        if not selected_student_one or not selected_student_two:
            error = "Please select both students to compare."
        elif selected_student_one.lower() == selected_student_two.lower():
            error = "Please select two different students."
        else:
            student_one_profile = get_profile_by_name(PROFILE_FILE, selected_student_one)
            student_two_profile = get_profile_by_name(PROFILE_FILE, selected_student_two)

            if student_one_profile is None or student_two_profile is None:
                error = "One or both selected profiles could not be found."
                student_one_profile = None
                student_two_profile = None
            else:
                comparison_rows = build_comparison_rows(student_one_profile, student_two_profile)
                comparison_graph = create_comparison_graph(student_one_profile, student_two_profile)

    return render_template(
        "compare.html",
        saved_profile_names=get_saved_profile_names(PROFILE_FILE),
        selected_student_one=selected_student_one,
        selected_student_two=selected_student_two,
        student_one_profile=student_one_profile,
        student_two_profile=student_two_profile,
        comparison_rows=comparison_rows,
        comparison_graph=comparison_graph,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
