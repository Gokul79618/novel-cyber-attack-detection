import argparse
import os

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("models", "xgboost.pkl")
FEATURES_PATH = os.path.join("models", "feature_names.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
SUSPICIOUS_THRESHOLD = 0.50


def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if model.n_features_in_ != len(feature_names):
        raise ValueError("Model feature count does not match the saved feature metadata.")
    if "Label" in feature_names:
        raise ValueError("Label should not be present in feature_names.")
    return model, feature_names, label_encoder


def validate_sample_frame(df, feature_names):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input data is empty.")

    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip()
    missing_features = [feature for feature in feature_names if feature not in normalized.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    selected = normalized[feature_names].copy()
    if not all(pd.api.types.is_numeric_dtype(selected[col]) for col in selected.columns):
        raise ValueError("All required features must be numeric.")
    if selected.isna().any().any():
        raise ValueError("Input data contains missing feature values.")
    if np.isinf(selected.to_numpy(dtype=float)).any():
        raise ValueError("Input data contains infinite feature values.")
    return selected


def predict_sample(input_path):
    model, feature_names, label_encoder = load_model_artifacts()
    input_df = pd.read_csv(input_path)
    features = validate_sample_frame(input_df, feature_names)

    encoded_predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = probabilities.max(axis=1)
    predicted_classes = label_encoder.inverse_transform(encoded_predictions)
    class_names = list(label_encoder.classes_)

    if len(class_names) != probabilities.shape[1]:
        raise ValueError("Probability column count does not match the label encoder classes.")

    rows = []
    for index, (predicted_class, maximum_confidence) in enumerate(
        zip(predicted_classes, confidence)
    ):
        if maximum_confidence < SUSPICIOUS_THRESHOLD:
            category = "SUSPICIOUS - Potential Novel Attack"
        elif predicted_class == "BENIGN":
            category = "BENIGN"
        else:
            category = "KNOWN ATTACK"

        rows.append({
            "Row": index,
            "Predicted_Class": predicted_class,
            "Maximum_Confidence": float(maximum_confidence),
            "Confidence_Percentage": float(maximum_confidence * 100),
            "Category": category,
        })

    return pd.DataFrame(rows), model, feature_names, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Predict XGBoost intrusion categories from a CSV file.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=os.path.join("test_files", "novel_attack_upload.csv"),
        help="CSV containing the exact 70 model features without the Label column.",
    )
    args = parser.parse_args()

    results, model, feature_names, label_encoder = predict_sample(args.input_csv)
    category_counts = results["Category"].value_counts()

    print("XGBOOST PREDICTION")
    print(f"Input file: {os.path.abspath(args.input_csv)}")
    print(f"Rows predicted: {len(results)}")
    print(f"Model: {type(model).__name__}")
    print(f"Features used: {len(feature_names)}")
    print(f"Expected feature order: exact models/feature_names.pkl order")
    print(f"Maximum confidence: {results['Maximum_Confidence'].max():.6f}")
    print(f"Mean confidence: {results['Maximum_Confidence'].mean():.6f}")
    print(f"Minimum confidence: {results['Maximum_Confidence'].min():.6f}")
    print("\nCategory counts:")
    print(category_counts.to_string())
    print("\nFirst 5 predictions:")
    print(results.head(5).to_string(index=False))

    if len(results) == 0:
        raise ValueError("No predictions were generated.")
    if not results["Category"].isin([
        "BENIGN",
        "KNOWN ATTACK",
        "SUSPICIOUS - Potential Novel Attack",
    ]).all():
        raise ValueError("One or more prediction categories are invalid.")

    print("XGBOOST PREDICTION VALIDATION: PASSED")


if __name__ == "__main__":
    main()
