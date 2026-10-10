import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

MODEL_DIR = os.path.join("models")
TEST_PATH = os.path.join("split_data", "test", "test_data.csv")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
XGBOOST_CONFIG_PATH = os.path.join(MODEL_DIR, "xgboost_config.pkl")
REPORTS_DIR = "reports"


def evaluate_model(model, model_name, feature_names, label_encoder, test_df):
    X_test = test_df[feature_names]
    y_true_raw = test_df["Label"].astype(str).str.strip()
    y_true = label_encoder.inverse_transform(label_encoder.transform(y_true_raw))

    start_time = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_duration = time.perf_counter() - start_time

    if model_name == "XGBoost":
        predictions = label_encoder.inverse_transform(predictions)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_true, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "prediction_time_seconds": prediction_duration,
        "test_rows": len(test_df),
        "feature_count": len(feature_names),
        "classes": list(model.classes_) if model_name == "Random Forest" else list(label_encoder.classes_),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }


def create_comparison_plot(comparison_df):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1-Score", "F1-Score"),
    ]
    rf_values = comparison_df.loc[comparison_df["Model"] == "Random Forest", [column for _, column in metrics]].to_numpy()[0]
    xgb_values = comparison_df.loc[comparison_df["Model"] == "XGBoost", [column for _, column in metrics]].to_numpy()[0]

    plt.figure(figsize=(11, 7))
    x = np.arange(len(metrics))
    width = 0.38
    bars_rf = plt.bar(x - width / 2, rf_values, width=width, label="Random Forest")
    bars_xgb = plt.bar(x + width / 2, xgb_values, width=width, label="XGBoost")

    for index, (metric_name, _) in enumerate(metrics):
        plt.text(x[index] - width / 2, rf_values[index] + 0.002, f"{rf_values[index]:.4f}", ha="center", va="bottom")
        plt.text(x[index] + width / 2, xgb_values[index] + 0.002, f"{xgb_values[index]:.4f}", ha="center", va="bottom")

    plt.xticks(x, [metric_name for metric_name, _ in metrics])
    plt.ylim(0, 1.03)
    plt.title("Random Forest vs XGBoost Performance")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def main():
    feature_names = joblib.load(FEATURES_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    test_df = pd.read_csv(TEST_PATH)

    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGBOOST_MODEL_PATH)
    xgb_config = joblib.load(XGBOOST_CONFIG_PATH)

    rf_result = evaluate_model(rf_model, "Random Forest", feature_names, label_encoder, test_df)
    xgb_result = evaluate_model(xgb_model, "XGBoost", feature_names, label_encoder, test_df)

    comparison_rows = []
    for result in [rf_result, xgb_result]:
        comparison_rows.append({
            "Model": result["model"],
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-Score": result["f1_score"],
            "Training Time (seconds)": xgb_config["training_duration_seconds"] if result["model"] == "XGBoost" else np.nan,
            "Prediction Time (seconds)": result["prediction_time_seconds"],
            "Feature Count": result["feature_count"],
            "Test Rows": result["test_rows"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    comparison_plot_path = create_comparison_plot(comparison_df)

    best_model = comparison_df.loc[comparison_df["F1-Score"].idxmax(), "Model"]
    print("MODEL COMPARISON")
    print(comparison_df.to_string(index=False))
    print(f"\nBest model by F1-score: {best_model}")
    print(f"Comparison CSV saved to: {os.path.abspath(comparison_path)}")
    print(f"Comparison image saved to: {os.path.abspath(comparison_plot_path)}")


if __name__ == "__main__":
    main()
