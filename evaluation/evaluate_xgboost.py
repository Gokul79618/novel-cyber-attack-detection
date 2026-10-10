import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_PATH = os.path.join("models", "xgboost.pkl")
FEATURES_PATH = os.path.join("models", "feature_names.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
TEST_PATH = os.path.join("split_data", "test", "test_data.csv")
REPORTS_DIR = "reports"


def summarize_metrics(y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.reset_index().rename(columns={"index": "Class"})

    per_class = []
    for label in labels:
        if label in report:
            value = report[label]
            per_class.append(
                {
                    "Class": label,
                    "Precision": value["precision"],
                    "Recall": value["recall"],
                    "F1-Score": value["f1-score"],
                    "Support": value["support"],
                }
            )

    per_class_df = pd.DataFrame(per_class)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report_df,
        "per_class": per_class_df,
    }


def create_confusion_matrix_plot(cm, labels):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    path = os.path.join(REPORTS_DIR, "xgboost_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def create_metric_plot(metrics):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value * 100:.2f}%", ha="center", va="bottom")
    plt.ylim(0, 1.1)
    plt.title("XGBoost Performance Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    path = os.path.join(REPORTS_DIR, "xgboost_performance_metrics.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def main():
    print("PHASE 4 - XGBOOST EVALUATION")
    print("-" * 30)

    for required_path in [MODEL_PATH, FEATURES_PATH, LABEL_ENCODER_PATH, TEST_PATH]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Missing required file: {required_path}")

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    test_df = pd.read_csv(TEST_PATH)
    if "Label" not in test_df.columns:
        raise ValueError("Label column is missing from the testing CSV.")

    missing_features = [feature for feature in feature_names if feature not in test_df.columns]
    if missing_features:
        raise ValueError(f"Test data is missing required features: {missing_features}")

    X_test = test_df[feature_names]
    y_true_raw = test_df["Label"].astype(str).str.strip()
    y_true = label_encoder.transform(y_true_raw)

    if not all(pd.api.types.is_numeric_dtype(X_test[col]) for col in X_test.columns):
        raise ValueError("Test features contain non-numeric columns.")
    if X_test.isna().any().any():
        raise ValueError("Test features contain NaN values.")
    if np.isinf(X_test.to_numpy(dtype=float)).any():
        raise ValueError("Test features contain infinity values.")

    print("TEST FEATURE VALIDATION: PASSED")

    start_time = time.time()
    y_pred_encoded = model.predict(X_test)
    prediction_duration_seconds = time.time() - start_time
    print(f"Prediction duration (seconds): {prediction_duration_seconds:.2f}")

    labels = list(label_encoder.classes_)
    metrics = summarize_metrics(y_true, y_pred_encoded, list(range(len(labels))))

    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    cm = confusion_matrix(y_true, y_pred_encoded, labels=list(range(len(labels))))

    metric_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            ],
        }
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metric_df.to_csv(os.path.join(REPORTS_DIR, "xgboost_metrics.csv"), index=False)
    metrics["report"].to_csv(os.path.join(REPORTS_DIR, "xgboost_classification_report.csv"), index=False)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(REPORTS_DIR, "xgboost_confusion_matrix.csv"), index=True)

    confusion_path = create_confusion_matrix_plot(cm, labels)
    metric_plot_path = create_metric_plot(metrics)

    print("\nXGBOOST PERFORMANCE METRICS")
    print(metric_df)

    print("\nXGBOOST CLASSIFICATION REPORT")
    print(metrics["report"].to_string(index=False))

    print(f"\nMetrics saved to: {os.path.abspath(os.path.join(REPORTS_DIR, 'xgboost_metrics.csv'))}")
    print(f"Classification report saved to: {os.path.abspath(os.path.join(REPORTS_DIR, 'xgboost_classification_report.csv'))}")
    print(f"Confusion matrix CSV saved to: {os.path.abspath(os.path.join(REPORTS_DIR, 'xgboost_confusion_matrix.csv'))}")
    print(f"Confusion matrix image saved to: {os.path.abspath(confusion_path)}")
    print(f"Performance metric image saved to: {os.path.abspath(metric_plot_path)}")

    print("\nPHASE 4 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
