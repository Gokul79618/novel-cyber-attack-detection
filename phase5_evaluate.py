import gc
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def summarize_confidence(values):
    if len(values) == 0:
        return {
            "Minimum": np.nan,
            "1st Percentile": np.nan,
            "5th Percentile": np.nan,
            "10th Percentile": np.nan,
            "25th Percentile": np.nan,
            "Median": np.nan,
            "Mean": np.nan,
            "75th Percentile": np.nan,
            "90th Percentile": np.nan,
            "95th Percentile": np.nan,
            "99th Percentile": np.nan,
            "Maximum": np.nan,
            "Standard Deviation": np.nan,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "Minimum": float(np.min(arr)),
        "1st Percentile": float(np.percentile(arr, 1)),
        "5th Percentile": float(np.percentile(arr, 5)),
        "10th Percentile": float(np.percentile(arr, 10)),
        "25th Percentile": float(np.percentile(arr, 25)),
        "Median": float(np.median(arr)),
        "Mean": float(np.mean(arr)),
        "75th Percentile": float(np.percentile(arr, 75)),
        "90th Percentile": float(np.percentile(arr, 90)),
        "95th Percentile": float(np.percentile(arr, 95)),
        "99th Percentile": float(np.percentile(arr, 99)),
        "Maximum": float(np.max(arr)),
        "Standard Deviation": float(np.std(arr)),
    }


def main():
    print("PHASE 5B - MODEL EVALUATION")
    print("-" * 28)

    required_files = [
        os.path.join("models", "random_forest.pkl"),
        os.path.join("models", "feature_names.pkl"),
        os.path.join("split_data", "test", "test_data.csv"),
        os.path.join("test_files", "novel_attack_labeled.csv"),
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Missing required file -> {file_path}")
            return

    os.makedirs("evaluation_results", exist_ok=True)
    print("Required files found.")

    model_path = os.path.join("models", "random_forest.pkl")
    feature_names_path = os.path.join("models", "feature_names.pkl")

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    print("MODEL AND FEATURE METADATA LOADED")
    print(f"Total saved features: {len(feature_names)}")
    print(f"Model expected feature count: {model.n_features_in_}")
    print(f"Total trained classes: {len(model.classes_)}")
    print("Exact trained classes:")
    print(list(model.classes_))

    if model.n_features_in_ != len(feature_names):
        raise ValueError("Loaded model feature count does not match the saved feature metadata.")

    if "Label" in feature_names:
        raise ValueError("Label should not appear in feature_names.")

    known_test_path = os.path.join("split_data", "test", "test_data.csv")
    print(f"\nLoading known test data from: {known_test_path}")
    test_df = pd.read_csv(known_test_path)
    print("KNOWN TEST DATA LOADED")
    print(f"Dataset shape: {test_df.shape}")
    print(f"Total rows: {test_df.shape[0]}")
    print(f"Total columns: {test_df.shape[1]}")

    if "Label" not in test_df.columns:
        raise ValueError("Label column is missing from the known test CSV.")

    missing_features = [feature for feature in feature_names if feature not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing known test features: {missing_features}")

    X_test = test_df[feature_names]
    y_test = test_df["Label"]

    if not all(pd.api.types.is_numeric_dtype(X_test[col]) for col in X_test.columns):
        raise ValueError("Known test features contain non-numeric columns.")
    if X_test.isna().any().any():
        raise ValueError("Known test features contain NaN values.")
    if np.isinf(X_test.to_numpy(dtype=float)).any():
        raise ValueError("Known test features contain infinity values.")

    print("KNOWN TEST FEATURE VALIDATION: PASSED")

    print("\nPredicting known test data...")
    start_time = time.time()
    known_predictions = model.predict(X_test)
    known_probabilities = model.predict_proba(X_test)
    known_max_confidence = known_probabilities.max(axis=1)
    prediction_duration_seconds = time.time() - start_time
    print("KNOWN TEST PREDICTION COMPLETED")
    print(f"Prediction duration (seconds): {prediction_duration_seconds:.2f}")

    if len(known_predictions) != len(y_test):
        raise ValueError("Known prediction length does not match the number of known test rows.")
    if len(known_max_confidence) != len(y_test):
        raise ValueError("Known maximum confidence length does not match the number of known test rows.")

    accuracy = accuracy_score(y_test, known_predictions)
    weighted_precision = precision_score(y_test, known_predictions, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_test, known_predictions, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, known_predictions, average="weighted", zero_division=0)

    print("\nKNOWN-CLASS MODEL PERFORMANCE")
    print(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Weighted Precision: {weighted_precision:.6f} ({weighted_precision * 100:.2f}%)")
    print(f"Weighted Recall: {weighted_recall:.6f} ({weighted_recall * 100:.2f}%)")
    print(f"Weighted F1-Score: {weighted_f1:.6f} ({weighted_f1 * 100:.2f}%)")

    report_string = classification_report(
        y_test,
        known_predictions,
        labels=model.classes_,
        zero_division=0,
    )
    print("\nKNOWN-CLASS CLASSIFICATION REPORT")
    print(report_string)
    report_path = os.path.join("evaluation_results", "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_string)
    print(f"Classification report saved to: {os.path.abspath(report_path)}")

    cm = confusion_matrix(y_test, known_predictions, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{cls}" for cls in model.classes_], columns=[f"Predicted_{cls}" for cls in model.classes_])
    cm_path = os.path.join("evaluation_results", "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"CONFUSION MATRIX SAVED: {os.path.abspath(cm_path)}")
    print(f"Confusion matrix shape: {cm_df.shape}")

    confidence_stats = summarize_confidence(known_max_confidence)
    print("\nKNOWN TEST CONFIDENCE STATISTICS")
    for stat_name, value in confidence_stats.items():
        print(f"{stat_name}: {value:.6f}")

    novel_path = os.path.join("test_files", "novel_attack_labeled.csv")
    print(f"\nLoading held-out Web Attack data from: {novel_path}")
    novel_df = pd.read_csv(novel_path)
    print("HELD-OUT WEB ATTACK DATA LOADED")
    print(f"Dataset shape: {novel_df.shape}")
    print(f"Total rows: {novel_df.shape[0]}")
    print(f"Total columns: {novel_df.shape[1]}")

    if "Label" not in novel_df.columns:
        raise ValueError("Label column is missing from the held-out Web Attack CSV.")
    missing_novel_features = [feature for feature in feature_names if feature not in novel_df.columns]
    if missing_novel_features:
        raise ValueError(f"Missing held-out Web Attack features: {missing_novel_features}")

    X_novel = novel_df[feature_names]
    y_novel = novel_df["Label"]

    if not all(pd.api.types.is_numeric_dtype(X_novel[col]) for col in X_novel.columns):
        raise ValueError("Held-out Web Attack features contain non-numeric columns.")
    if X_novel.isna().any().any():
        raise ValueError("Held-out Web Attack features contain NaN values.")
    if np.isinf(X_novel.to_numpy(dtype=float)).any():
        raise ValueError("Held-out Web Attack features contain infinity values.")

    non_web_labels = [label for label in y_novel.astype(str).unique() if "web attack" not in label.lower()]
    if non_web_labels:
        raise ValueError(f"Held-out labels validation failed: {non_web_labels}")
    if any(label in set(model.classes_) for label in y_novel.astype(str).unique()):
        raise ValueError("Held-out Web Attack labels unexpectedly match trained model classes.")

    print("HELD-OUT WEB ATTACK EXCLUSION VALIDATION: PASSED")
    print("HELD-OUT FEATURE VALIDATION: PASSED")

    print("\nPredicting held-out Web Attack data...")
    novel_raw_predictions = model.predict(X_novel)
    novel_probabilities = model.predict_proba(X_novel)
    novel_max_confidence = novel_probabilities.max(axis=1)
    print("HELD-OUT WEB ATTACK PREDICTION COMPLETED")

    print("\nRAW MODEL PREDICTIONS FOR HELD-OUT WEB ATTACKS")
    print(pd.Series(novel_raw_predictions).value_counts())

    novel_confidence_stats = summarize_confidence(novel_max_confidence)
    print("\nHELD-OUT WEB ATTACK CONFIDENCE STATISTICS")
    for stat_name, value in novel_confidence_stats.items():
        print(f"{stat_name}: {value:.6f}")

    confidence_comparison = pd.DataFrame(
        {
            "Statistic": list(confidence_stats.keys()),
            "Known_Test": [confidence_stats[name] for name in confidence_stats.keys()],
            "Heldout_Web_Attack": [novel_confidence_stats[name] for name in novel_confidence_stats.keys()],
        }
    )
    confidence_comparison_path = os.path.join("evaluation_results", "confidence_comparison.csv")
    confidence_comparison.to_csv(confidence_comparison_path, index=False)
    print(f"CONFIDENCE COMPARISON SAVED: {os.path.abspath(confidence_comparison_path)}")

    confidence_analysis = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        confidence_analysis.append({
            "Threshold": threshold,
            "Known_Low_Confidence_Count": int((known_max_confidence < threshold).sum()),
            "Known_Low_Confidence_Percentage": float((known_max_confidence < threshold).mean() * 100),
            "Heldout_Low_Confidence_Count": int((novel_max_confidence < threshold).sum()),
            "Heldout_Low_Confidence_Percentage": float((novel_max_confidence < threshold).mean() * 100),
        })

    threshold_df = pd.DataFrame(confidence_analysis)
    threshold_path = os.path.join("evaluation_results", "threshold_analysis.csv")
    threshold_df.to_csv(threshold_path, index=False)
    print("\nCANDIDATE THRESHOLD ANALYSIS")
    print(threshold_df)
    print(f"Threshold analysis saved to: {os.path.abspath(threshold_path)}")

    novel_analysis = pd.DataFrame({
        "Actual_Heldout_Label": y_novel.astype(str),
        "Raw_Model_Prediction": pd.Series(novel_raw_predictions).astype(str),
        "Maximum_Confidence": novel_max_confidence,
    })
    novel_analysis_path = os.path.join("evaluation_results", "novel_attack_confidence_analysis.csv")
    novel_analysis.to_csv(novel_analysis_path, index=False)
    print(f"HELD-OUT CONFIDENCE ANALYSIS SAVED: {os.path.abspath(novel_analysis_path)}")

    output_files = [
        os.path.join("evaluation_results", "classification_report.txt"),
        os.path.join("evaluation_results", "confusion_matrix.csv"),
        os.path.join("evaluation_results", "confidence_comparison.csv"),
        os.path.join("evaluation_results", "novel_attack_confidence_analysis.csv"),
        os.path.join("evaluation_results", "threshold_analysis.csv"),
    ]

    for file_path in output_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing evaluation output file: {file_path}")
        if os.path.getsize(file_path) <= 0:
            raise ValueError(f"Evaluation output file is empty: {file_path}")

    print("EVALUATION OUTPUT FILE VERIFICATION: PASSED")

    del test_df, X_test, y_test, known_predictions, known_probabilities, known_max_confidence
    del novel_df, X_novel, y_novel, novel_raw_predictions, novel_probabilities, novel_max_confidence
    gc.collect()

    print("\nPhase 5B completed successfully.")


if __name__ == "__main__":
    main()
