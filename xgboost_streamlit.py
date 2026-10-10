import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from predict import get_defense_recommendation

MODEL_PATH = os.path.join("models", "xgboost.pkl")
FEATURES_PATH = os.path.join("models", "feature_names.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
SUSPICIOUS_THRESHOLD = 0.50


def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if model.n_features_in_ != len(feature_names):
        raise ValueError("XGBoost model feature count does not match saved feature metadata.")
    if "Label" in feature_names:
        raise ValueError("Label should not be present in feature_names.")
    return model, list(feature_names), label_encoder


def validate_uploaded_dataframe(df, feature_names):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Uploaded content must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip()
    missing_features = [feature for feature in feature_names if feature not in normalized.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    selected = normalized[feature_names].copy()
    if not all(pd.api.types.is_numeric_dtype(selected[column]) for column in selected.columns):
        raise ValueError("All required features must be numeric.")
    if selected.isna().any().any() or np.isinf(selected.to_numpy(dtype=float)).any():
        raise ValueError("Input data contains missing or infinite feature values.")
    return selected


def predict_dataframe(df):
    model, feature_names, label_encoder = load_model_artifacts()
    features = validate_uploaded_dataframe(df, feature_names)
    encoded_predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    max_confidence = probabilities.max(axis=1)
    raw_predictions = label_encoder.inverse_transform(encoded_predictions)

    if len(raw_predictions) != len(features):
        raise ValueError("Prediction count does not match the number of input rows.")
    if probabilities.shape[1] != len(label_encoder.classes_):
        raise ValueError("Probability matrix shape does not match the label encoder classes.")

    final_categories = []
    defense_recommendations = []
    for raw_prediction, confidence in zip(raw_predictions, max_confidence):
        if confidence < SUSPICIOUS_THRESHOLD:
            category = "SUSPICIOUS"
        elif raw_prediction == "BENIGN":
            category = "BENIGN"
        else:
            category = "KNOWN ATTACK"
        final_categories.append(category)
        defense_recommendations.append(
            get_defense_recommendation(category, raw_prediction)
        )

    return pd.DataFrame({
        "Raw_Prediction": raw_predictions,
        "Confidence": np.round(max_confidence, 4),
        "Confidence_Percentage": np.round(max_confidence * 100, 2),
        "Final_Category": final_categories,
        "Defense_Recommendation": defense_recommendations,
    })


def create_prediction_summary(results_df):
    benign_count = int((results_df["Final_Category"] == "BENIGN").sum())
    known_attack_count = int((results_df["Final_Category"] == "KNOWN ATTACK").sum())
    suspicious_count = int((results_df["Final_Category"] == "SUSPICIOUS").sum())
    total_flows = len(results_df)
    if benign_count + known_attack_count + suspicious_count != total_flows:
        raise ValueError("Prediction summary counts do not match the total number of flows.")
    return {
        "Total_Flows": total_flows,
        "Benign_Count": benign_count,
        "Known_Attack_Count": known_attack_count,
        "Suspicious_Count": suspicious_count,
        "Benign_Percentage": round(benign_count / total_flows * 100, 2) if total_flows else 0.0,
        "Known_Attack_Percentage": round(known_attack_count / total_flows * 100, 2) if total_flows else 0.0,
        "Suspicious_Percentage": round(suspicious_count / total_flows * 100, 2) if total_flows else 0.0,
    }


def evaluate_held_out_test(test_csv_path):
    model, feature_names, label_encoder = load_model_artifacts()
    test_df = pd.read_csv(test_csv_path)
    test_df.columns = test_df.columns.str.strip()
    if "Label" not in test_df.columns:
        raise ValueError("The held-out test CSV must contain a Label column.")

    features = validate_uploaded_dataframe(test_df.drop(columns=["Label"]), feature_names)
    y_test = test_df["Label"].astype(str).str.strip()
    encoded_predictions = model.predict(features)
    predictions = label_encoder.inverse_transform(encoded_predictions)
    classes = list(label_encoder.classes_)
    probabilities = model.predict_proba(features)
    confidence = probabilities.max(axis=1)

    matrix = confusion_matrix(y_test, predictions, labels=classes)
    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for name, values in report.items():
        if isinstance(values, dict):
            report_rows.append({
                "Class": name,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-Score": values["f1-score"],
                "Support": int(values["support"]),
            })

    return {
        "metrics": {
            "Accuracy": float(np.trace(matrix) / matrix.sum()),
            "Precision (Macro Average)": float(precision_score(y_test, predictions, labels=classes, average="macro", zero_division=0)),
            "Recall (Macro Average)": float(recall_score(y_test, predictions, labels=classes, average="macro", zero_division=0)),
            "F1-Score (Macro Average)": float(f1_score(y_test, predictions, labels=classes, average="macro", zero_division=0)),
            "Balanced Accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "ROC-AUC (OvR, Macro Average)": float(roc_auc_score(y_test, probabilities, labels=classes, multi_class="ovr", average="macro")),
            "MCC": float(matthews_corrcoef(y_test, predictions)),
            "Precision (Weighted)": float(precision_score(y_test, predictions, labels=classes, average="weighted", zero_division=0)),
            "Recall (Weighted)": float(recall_score(y_test, predictions, labels=classes, average="weighted", zero_division=0)),
            "F1-Score (Weighted)": float(f1_score(y_test, predictions, labels=classes, average="weighted", zero_division=0)),
        },
        "labels": classes,
        "confusion_matrix": matrix,
        "classification_report": pd.DataFrame(report_rows),
        "class_distribution": pd.DataFrame({
            "Class": classes,
            "Count": [int(matrix[i, i]) for i in range(len(classes))],
        }),
        "test_samples": len(y_test),
        "feature_count": len(feature_names),
        "class_count": len(classes),
        "confidence": {
            "high_count": int((confidence >= SUSPICIOUS_THRESHOLD).sum()),
            "low_count": int((confidence < SUSPICIOUS_THRESHOLD).sum()),
            "average": float(confidence.mean() * 100),
            "minimum": float(confidence.min() * 100),
            "maximum": float(confidence.max() * 100),
            "suspicious_percentage": float((confidence < SUSPICIOUS_THRESHOLD).mean() * 100),
            "threshold": SUSPICIOUS_THRESHOLD,
        },
    }
