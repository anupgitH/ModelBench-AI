import io
import json
import re
from pathlib import Path
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ModelBench AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ModelBench AI")
st.markdown(
    """
    ### Automated Machine Learning Training & Evaluation

    Upload a dataset, train classification models, compare their
    performance, make predictions, and save trained models and reports.
    """
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    """Create a filename-safe string."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name))
    return name.strip("_")[:80] or "model"


def make_preprocessor(X):
    """Create preprocessing for numerical and categorical columns."""

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def make_models(seed=42):
    """Return supported classification algorithms."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def create_pipeline(model, X):
    """Combine preprocessing and the estimator."""

    return Pipeline([
        ("preprocessing", make_preprocessor(X)),
        ("model", model),
    ])


def save_model_artifact(
    pipeline,
    model_name,
    target_column,
    feature_columns,
    metrics,
):
    """Save the trained model and its metadata to disk."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_filename(model_name)}_{timestamp}"

    model_path = MODELS_DIR / f"{filename}.joblib"
    metadata_path = MODELS_DIR / f"{filename}.json"

    # Save the complete fitted preprocessing + model pipeline.
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": model_name,
        "target_column": str(target_column),
        "feature_columns": list(feature_columns),
        "metrics": metrics,
        "model_file": model_path.name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, default=str)

    # Verify both files were written successfully.
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise OSError("Model file was not created correctly.")

    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise OSError("Metadata file was not created correctly.")

    return model_path, metadata_path


def save_report_artifacts(
    results,
    selected_model,
    classification_report_data,
    dataset_name,
    target_column,
):
    """Save comparison CSV and detailed evaluation JSON."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comparison_path = (
        REPORTS_DIR / f"model_comparison_{timestamp}.csv"
    )
    evaluation_path = (
        REPORTS_DIR / f"evaluation_report_{timestamp}.json"
    )

    results.to_csv(comparison_path, index=False)

    report_data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "target_column": str(target_column),
        "selected_model": selected_model,
        "model_comparison": results.to_dict(orient="records"),
        "classification_report": classification_report_data,
    }

    with evaluation_path.open("w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=4, default=str)

    if not comparison_path.exists() or comparison_path.stat().st_size == 0:
        raise OSError("Comparison CSV was not created correctly.")

    if not evaluation_path.exists() or evaluation_path.stat().st_size == 0:
        raise OSError("Evaluation report was not created correctly.")

    return comparison_path, evaluation_path


def read_json(path):
    """Read JSON safely."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("ModelBench AI")
st.sidebar.caption("Machine Learning Workspace")

st.sidebar.markdown(
    """
    **Features**
    - Dataset exploration
    - Model training
    - Performance comparison
    - Confusion matrix
    - Feature importance
    - Manual predictions
    - Persistent model storage
    - Evaluation report exports
    """
)

st.sidebar.write(f"Models folder: `{MODELS_DIR.name}/`")
st.sidebar.write(f"Reports folder: `{REPORTS_DIR.name}/`")


# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("1. 📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV dataset to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if df.columns.duplicated().any():
    st.error("Your CSV contains duplicate column names.")
    st.stop()

if len(df.columns) < 2:
    st.error(
        "The dataset must contain at least one feature "
        "and one target column."
    )
    st.stop()

st.session_state["mb_dataset_name"] = uploaded_file.name


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("2. 📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
col4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("View dataset statistics"):
    st.subheader("Summary Statistics")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True,
    )

    st.subheader("Missing Values")
    missing_df = df.isna().sum().to_frame("Missing Values")
    missing_df["Missing Percentage"] = (
        missing_df["Missing Values"] / len(df) * 100
    ).round(2)

    st.dataframe(missing_df, use_container_width=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

st.header("3. ⚙️ Configure Experiment")

default_target_index = (
    df.columns.tolist().index("target")
    if "target" in df.columns
    else len(df.columns) - 1
)

target = st.selectbox(
    "Select target column",
    options=df.columns.tolist(),
    index=default_target_index,
)

col1, col2 = st.columns(2)

with col1:
    test_percentage = st.slider(
        "Test dataset percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

with col2:
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

available_models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]

selected_models = st.multiselect(
    "Select models to train",
    options=available_models,
    default=available_models,
)


# ============================================================
# TRAIN MODELS
# ============================================================

if st.button(
    "🚀 Train and Evaluate Models",
    type="primary",
    use_container_width=True,
):
    if not selected_models:
        st.error("Select at least one model.")
        st.stop()

    clean_df = df.dropna(subset=[target]).copy()

    if clean_df.empty:
        st.error("The target column contains no usable values.")
        st.stop()

    if clean_df[target].nunique() < 2:
        st.error("The target column must contain at least two classes.")
        st.stop()

    if clean_df[target].nunique() > 50:
        st.error(
            "This application supports classification with at most "
            "50 target classes."
        )
        st.stop()

    X = clean_df.drop(columns=[target]).copy()
    y = clean_df[target].copy()

    usable_columns = [
        column
        for column in X.columns
        if X[column].notna().any()
        and X[column].nunique(dropna=True) > 1
    ]

    X = X[usable_columns]

    if X.shape[1] == 0:
        st.error("No usable feature columns were found.")
        st.stop()

    stratify_data = y if y.value_counts().min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percentage / 100,
            random_state=int(random_seed),
            stratify=stratify_data,
        )
    except ValueError as exc:
        st.error(f"Dataset split failed: {exc}")
        st.stop()

    fitted_models = {}
    metric_rows = []

    progress = st.progress(0)
    status = st.empty()

    for index, model_name in enumerate(selected_models):
        status.write(f"Training {model_name}...")

        try:
            estimator = make_models(int(random_seed))[model_name]
            pipeline = create_pipeline(estimator, X_train)

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            metrics = {
                "Accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "Precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "Recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "F1 Score": float(
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }

            fitted_models[model_name] = pipeline

            metric_rows.append({
                "Model": model_name,
                **metrics,
            })

        except Exception as exc:
            st.error(f"{model_name} failed: {exc}")

        progress.progress((index + 1) / len(selected_models))

    status.empty()
    progress.empty()

    if not metric_rows:
        st.error("No models trained successfully.")
        st.stop()

    results_df = pd.DataFrame(metric_rows).sort_values(
        "F1 Score",
        ascending=False,
    ).reset_index(drop=True)

    st.session_state["mb_results"] = results_df
    st.session_state["mb_models"] = fitted_models
    st.session_state["mb_X_train"] = X_train
    st.session_state["mb_X_test"] = X_test
    st.session_state["mb_y_test"] = y_test
    st.session_state["mb_target"] = target
    st.session_state["mb_feature_columns"] = X_train.columns.tolist()
    st.session_state["mb_selected_model"] = results_df.iloc[0]["Model"]
    st.session_state["mb_dataset_name"] = uploaded_file.name

    st.success("Training completed successfully!")


# ============================================================
# RESTORE SESSION STATE
# ============================================================

required_keys = [
    "mb_results",
    "mb_models",
    "mb_X_train",
    "mb_X_test",
    "mb_y_test",
    "mb_target",
    "mb_feature_columns",
]

if not all(key in st.session_state for key in required_keys):
    st.info("Train at least one model to see evaluation results.")
    st.stop()

results_df = st.session_state["mb_results"]
fitted_models = st.session_state["mb_models"]
X_train = st.session_state["mb_X_train"]
X_test = st.session_state["mb_X_test"]
y_test = st.session_state["mb_y_test"]
target = st.session_state["mb_target"]
feature_columns = st.session_state["mb_feature_columns"]

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

best_row = results_df.iloc[0]
default_model = st.session_state.get(
    "mb_selected_model",
    best_row["Model"],
)

if default_model not in fitted_models:
    default_model = best_row["Model"]

st.header("4. 📈 Model Performance")

metric_columns = st.columns(4)

for column, metric_name in zip(metric_columns, metric_names):
    column.metric(
        metric_name,
        f"{best_row[metric_name]:.2%}",
    )

st.subheader("Model Comparison Table")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}",
    }),
    use_container_width=True,
)

st.subheader("Model Comparison Chart")
st.bar_chart(results_df.set_index("Model")[metric_names])

st.subheader("Performance Heatmap")

heatmap_data = results_df.set_index("Model")[metric_names]

fig, ax = plt.subplots(
    figsize=(10, max(3, len(heatmap_data) * 0.8))
)

heatmap_image = ax.imshow(
    heatmap_data.values,
    aspect="auto",
    vmin=0,
    vmax=1,
)

ax.set_xticks(range(len(metric_names)))
ax.set_xticklabels(metric_names, rotation=30, ha="right")
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

for row_index in range(heatmap_data.shape[0]):
    for column_index in range(heatmap_data.shape[1]):
        ax.text(
            column_index,
            row_index,
            f"{heatmap_data.iloc[row_index, column_index]:.2f}",
            ha="center",
            va="center",
        )

ax.set_title("Model Performance Heatmap")
fig.colorbar(heatmap_image, ax=ax)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ============================================================
# DETAILED MODEL EVALUATION
# ============================================================

st.header("5. 🔍 Detailed Evaluation")

selected_model = st.selectbox(
    "Choose a model for detailed evaluation",
    options=list(fitted_models.keys()),
    index=list(fitted_models.keys()).index(default_model),
    key="evaluation_model",
)

selected_pipeline = fitted_models[selected_model]
predictions = selected_pipeline.predict(X_test)

st.subheader("Classification Report")

classification_data = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0,
)

classification_df = pd.DataFrame(classification_data).transpose()

st.dataframe(
    classification_df.style.format("{:.3f}"),
    use_container_width=True,
)

st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
)

ax.set_title(f"Confusion Matrix — {selected_model}")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("6. 🌳 Feature Importance")

estimator = selected_pipeline.named_steps["model"]
preprocessor = selected_pipeline.named_steps["preprocessing"]

if hasattr(estimator, "feature_importances_"):
    try:
        transformed_names = preprocessor.get_feature_names_out()
        importances = estimator.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": transformed_names,
            "Importance": importances,
        }).sort_values(
            "Importance",
            ascending=False,
        ).head(20)

        st.dataframe(importance_df, use_container_width=True)

        fig, ax = plt.subplots(
            figsize=(10, max(4, len(importance_df) * 0.35))
        )

        ax.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1],
        )

        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    except Exception as exc:
        st.warning(f"Could not calculate feature importance: {exc}")
else:
    st.info(
        "Feature importance is available for tree-based models, "
        "such as Decision Tree and Random Forest."
    )


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

st.header("7. 💾 Save Models")

st.write(
    "Models are saved permanently in the project's `models/` folder."
)

model_to_save = st.selectbox(
    "Choose a trained model to save",
    options=list(fitted_models.keys()),
    index=list(fitted_models.keys()).index(
        selected_model if selected_model in fitted_models else default_model
    ),
    key="model_to_save",
)

model_save_name = st.text_input(
    "Model filename",
    value=safe_filename(model_to_save),
)

if st.button(
    "💾 Save Model and Metadata",
    use_container_width=True,
):
    try:
        model_row = results_df.loc[
            results_df["Model"] == model_to_save
        ].iloc[0]

        metrics_data = {
            metric: float(model_row[metric])
            for metric in metric_names
        }

        model_path, metadata_path = save_model_artifact(
            pipeline=fitted_models[model_to_save],
            model_name=model_save_name,
            target_column=target,
            feature_columns=feature_columns,
            metrics=metrics_data,
        )

        st.success("Model and metadata saved successfully!")
        st.write(f"Model file: `{model_path}`")
        st.write(f"Metadata file: `{metadata_path}`")

    except Exception as exc:
        st.error(f"Model saving failed: {exc}")


# ============================================================
# SAVE REPORTS
# ============================================================

st.header("8. 📄 Save Evaluation Reports")

if st.button(
    "📑 Save Comparison CSV and Evaluation JSON",
    use_container_width=True,
):
    try:
        current_report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        comparison_path, evaluation_path = save_report_artifacts(
            results=results_df,
            selected_model=selected_model,
            classification_report_data=current_report,
            dataset_name=st.session_state.get(
                "mb_dataset_name",
                uploaded_file.name,
            ),
            target_column=target,
        )

        st.success("Reports saved successfully!")
        st.write(f"Comparison CSV: `{comparison_path}`")
        st.write(f"Evaluation JSON: `{evaluation_path}`")

    except Exception as exc:
        st.error(f"Report saving failed: {exc}")


# ============================================================
# DOWNLOAD CURRENT ARTIFACTS
# ============================================================

st.header("9. ⬇️ Download Artifacts")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="Download Comparison CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    model_buffer = io.BytesIO()
    joblib.dump(fitted_models[selected_model], model_buffer)

    st.download_button(
        label="Download Selected Model",
        data=model_buffer.getvalue(),
        file_name=f"{safe_filename(selected_model)}.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )

metadata_download = {
    "dataset": st.session_state.get("mb_dataset_name", ""),
    "target_column": str(target),
    "selected_model": selected_model,
    "feature_columns": feature_columns,
    "model_comparison": results_df.to_dict(orient="records"),
}

st.download_button(
    label="Download Experiment Metadata JSON",
    data=json.dumps(
        metadata_download,
        indent=4,
        default=str,
    ).encode("utf-8"),
    file_name="experiment_metadata.json",
    mime="application/json",
    use_container_width=True,
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

st.header("10. 📦 Load Saved Model")

saved_model_paths = sorted(
    MODELS_DIR.glob("*.joblib"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if saved_model_paths:
    saved_model_names = [path.name for path in saved_model_paths]

    selected_saved_model_name = st.selectbox(
        "Choose a saved model",
        options=saved_model_names,
    )

    selected_saved_model_path = MODELS_DIR / selected_saved_model_name
    selected_metadata_path = selected_saved_model_path.with_suffix(".json")

    metadata = read_json(selected_metadata_path)

    if metadata:
        st.write("Model metadata")
        st.json(metadata)

    if st.button(
        "Load Model for Prediction",
        use_container_width=True,
    ):
        try:
            loaded_pipeline = joblib.load(selected_saved_model_path)

            st.session_state["mb_loaded_model"] = loaded_pipeline
            st.session_state["mb_loaded_model_name"] = (
                selected_saved_model_path.name
            )
            st.session_state["mb_loaded_model_features"] = (
                metadata.get("feature_columns", feature_columns)
            )
            st.session_state["mb_loaded_model_target"] = (
                metadata.get("target_column", target)
            )

            st.success(
                f"Loaded saved model: {selected_saved_model_path.name}"
            )

        except Exception as exc:
            st.error(f"Could not load saved model: {exc}")

else:
    st.info("No saved models found. Train and save a model first.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

st.header("11. 🔮 Make Predictions")

loaded_pipeline = st.session_state.get("mb_loaded_model")

prediction_source = st.radio(
    "Choose prediction model",
    options=["Currently trained model", "Saved model"],
    horizontal=True,
)

if prediction_source == "Saved model" and loaded_pipeline is None:
    st.warning(
        "Load a saved model in the section above before selecting "
        "the saved-model prediction option."
    )
    prediction_pipeline = selected_pipeline
    prediction_features = feature_columns
    prediction_reference = X_train
else:
    if prediction_source == "Saved model":
        prediction_pipeline = loaded_pipeline
        prediction_features = st.session_state.get(
            "mb_loaded_model_features",
            feature_columns,
        )
        prediction_reference = X_train
    else:
        prediction_pipeline = selected_pipeline
        prediction_features = feature_columns
        prediction_reference = X_train

numeric_features = set()

preprocessor = prediction_pipeline.named_steps["preprocessing"]

if hasattr(preprocessor, "transformers_"):
    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            numeric_features.update(columns)

with st.form("prediction_form"):
    st.subheader("Enter feature values")

    inputs = {}
    input_columns = st.columns(2)

    for index, feature in enumerate(prediction_features):
        with input_columns[index % 2]:
            if (
                feature in prediction_reference.columns
                and pd.api.types.is_numeric_dtype(
                    prediction_reference[feature]
                )
            ) or feature in numeric_features:
                values = (
                    prediction_reference[feature].dropna()
                    if feature in prediction_reference.columns
                    else pd.Series(dtype=float)
                )

                try:
                    default_value = (
                        float(values.iloc[0])
                        if not values.empty
                        else 0.0
                    )
                except (ValueError, TypeError):
                    default_value = 0.0

                inputs[feature] = st.number_input(
                    str(feature),
                    value=default_value,
                    key=f"prediction_{prediction_source}_{feature}",
                )
            else:
                values = (
                    prediction_reference[feature].dropna()
                    if feature in prediction_reference.columns
                    else pd.Series(dtype=object)
                )

                default_value = (
                    str(values.iloc[0])
                    if not values.empty
                    else ""
                )

                inputs[feature] = st.text_input(
                    str(feature),
                    value=default_value,
                    key=f"prediction_{prediction_source}_{feature}",
                )

    predict_clicked = st.form_submit_button(
        "Predict",
        use_container_width=True,
    )

if predict_clicked:
    try:
        input_df = pd.DataFrame(
            [inputs],
            columns=prediction_features,
        )

        for feature in numeric_features:
            if feature in input_df.columns:
                input_df[feature] = pd.to_numeric(
                    input_df[feature],
                    errors="coerce",
                )

        predicted_class = prediction_pipeline.predict(input_df)[0]

        st.success(f"Predicted class: {predicted_class}")

        if hasattr(prediction_pipeline, "predict_proba"):
            probabilities = prediction_pipeline.predict_proba(input_df)[0]
            class_labels = prediction_pipeline.named_steps["model"].classes_

            probabilities_df = pd.DataFrame({
                "Class": [str(label) for label in class_labels],
                "Probability": probabilities,
            }).sort_values(
                "Probability",
                ascending=False,
            )

            st.subheader("Class Probabilities")

            st.dataframe(
                probabilities_df.style.format({
                    "Probability": "{:.2%}",
                }),
                use_container_width=True,
            )

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.bar(
                probabilities_df["Class"],
                probabilities_df["Probability"] * 100,
            )

            ax.set_title("Prediction Probabilities")
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 105)
            ax.grid(axis="y", alpha=0.25)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ============================================================
# SAVED MODELS LIST
# ============================================================

st.header("12. 📚 Saved Models Library")

saved_model_paths = sorted(
    MODELS_DIR.glob("*.joblib"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if saved_model_paths:
    for path in saved_model_paths:
        with st.expander(f"📦 {path.name}"):
            metadata_path = path.with_suffix(".json")
            metadata = read_json(metadata_path)

            st.write(f"Model location: `{path}`")
            st.write(f"File size: {path.stat().st_size:,} bytes")

            if metadata:
                st.json(metadata)

            try:
                st.download_button(
                    label=f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/octet-stream",
                    key=f"download_model_{path.name}",
                )
            except OSError as exc:
                st.error(f"Could not read model file: {exc}")

else:
    st.info("No models have been saved yet.")


# ============================================================
# SAVED REPORTS LIST
# ============================================================

st.header("13. 🗂️ Saved Reports Library")

report_paths = sorted(
    list(REPORTS_DIR.glob("*.csv"))
    + list(REPORTS_DIR.glob("*.json")),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if report_paths:
    for path in report_paths:
        with st.expander(f"📄 {path.name}"):
            st.write(f"Location: `{path}`")
            st.write(f"File size: {path.stat().st_size:,} bytes")

            try:
                if path.suffix == ".csv":
                    st.dataframe(
                        pd.read_csv(path).head(100),
                        use_container_width=True,
                    )
                    mime = "text/csv"
                else:
                    st.json(read_json(path))
                    mime = "application/json"

                st.download_button(
                    label=f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=f"download_report_{path.name}",
                )

            except Exception as exc:
                st.error(f"Could not read report: {exc}")

else:
    st.info("No reports have been saved yet.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ModelBench AI | Automated Machine Learning Training & Evaluation"
)