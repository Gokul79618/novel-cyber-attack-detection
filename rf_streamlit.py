import os

import joblib
import numpy as np
import pandas as pd

from predict import get_defense_recommendation

MODEL_PATH = os.path.join("models", "random_forest.pkl")
FEATURES_PATH = os.path.join("models", "feature_names.pkl")
SUSPICIOUS_THRESHOLD = 0.95


def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    if model.n_features_in_ != len(feature_names):
        raise ValueError("Random Forest model feature count does not match saved feature metadata.")
    if "Label" in feature_names:
        raise ValueError("Label should not be present in feature_names.")
    return model, list(feature_names)


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
    model, feature_names = load_model_artifacts()
    features = validate_uploaded_dataframe(df, feature_names)
    predicted_classes = model.predict(features)
    probabilities = model.predict_proba(features)
    class_names = list(model.classes_)
    confidence = probabilities.max(axis=1)

    if len(class_names) != probabilities.shape[1]:
        raise ValueError("Probability matrix shape does not match the Random Forest classes.")

    rows = []
    for index, (predicted_class, row_probabilities, maximum_confidence) in enumerate(
        zip(predicted_classes, probabilities, confidence)
    ):
        predicted_class_name = str(predicted_class)
        if maximum_confidence < SUSPICIOUS_THRESHOLD:
            final_category = "SUSPICIOUS"
        else:
            final_category = "BENIGN" if predicted_class_name == "BENIGN" else "KNOWN ATTACK"

        rows.append({
            "Row": index,
            "Raw_Prediction": predicted_class_name,
            "Predicted_Class": predicted_class_name,
            "Probability": float(maximum_confidence),
            "Confidence": float(maximum_confidence),
            "Confidence_Percentage": float(maximum_confidence * 100),
            "Final_Category": final_category,
            "Defense_Recommendation": get_defense_recommendation(final_category, predicted_class_name),
        })

    return pd.DataFrame(rows)
