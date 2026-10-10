import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("models", "xgboost.pkl")
FEATURES_PATH = os.path.join("models", "feature_names.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
NOVEL_PATH = os.path.join(
    "test_files",
    "heldout_suspicious",
    "web_attack_suspicious_labeled.csv",
)
REPORTS_DIR = "reports"
CONFIDENCE_THRESHOLD = 0.50


def summarize_confidence(values):
    values = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(values)),
        "1st_percentile": float(np.percentile(values, 1)),
        "5th_percentile": float(np.percentile(values, 5)),
        "10th_percentile": float(np.percentile(values, 10)),
        "25th_percentile": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "75th_percentile": float(np.percentile(values, 75)),
        "90th_percentile": float(np.percentile(values, 90)),
        "95th_percentile": float(np.percentile(values, 95)),
        "99th_percentile": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
        "standard_deviation": float(np.std(values)),
    }


def main():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    novel_df = pd.read_csv(NOVEL_PATH)

    if list(novel_df.columns[: len(feature_names)]) != feature_names:
        raise ValueError("Held-out Web Attack feature order does not match feature_names.pkl.")
    if "Label" not in novel_df.columns:
        raise ValueError("Held-out Web Attack data is missing the Label column.")

    novel_features = novel_df[feature_names]
    novel_labels = novel_df["Label"].astype(str).str.strip()
    if novel_labels.str.contains("web attack", case=False, na=False).sum() != len(novel_df):
        raise ValueError("Held-out Web Attack labels are not uniformly Web Attack.")
    if novel_features.isna().any().any():
        raise ValueError("Held-out Web Attack data contains missing feature values.")

    probabilities = model.predict_proba(novel_features)
    predicted_encoded = model.predict(novel_features)
    max_confidence = probabilities.max(axis=1)
    predicted_labels = pd.Series(label_encoder.inverse_transform(predicted_encoded))
    suspicious_mask = max_confidence < CONFIDENCE_THRESHOLD
    known_mask = ~suspicious_mask

    analysis = {
        "threshold": CONFIDENCE_THRESHOLD,
        "sample_count": len(novel_df),
        "flagged_suspicious": int(suspicious_mask.sum()),
        "classified_as_known": int(known_mask.sum()),
        "confidence_minimum": float(np.min(max_confidence)),
        "confidence_mean": float(np.mean(max_confidence)),
        "confidence_median": float(np.median(max_confidence)),
        "confidence_maximum": float(np.max(max_confidence)),
        "prediction_distribution": predicted_labels.value_counts().sort_index().to_dict(),
        "suspicious_distribution": predicted_labels[suspicious_mask].value_counts().sort_index().to_dict(),
    }

    result_df = pd.DataFrame([analysis])
    os.makedirs(REPORTS_DIR, exist_ok=True)
    result_path = os.path.join(REPORTS_DIR, "xgboost_novel_attack_analysis.csv")
    result_df.to_csv(result_path, index=False)

    confidence_stats = summarize_confidence(max_confidence)
    confidence_df = pd.DataFrame([confidence_stats])
    confidence_path = os.path.join(REPORTS_DIR, "xgboost_novel_attack_confidence.csv")
    confidence_df.to_csv(confidence_path, index=False)

    plt.figure(figsize=(9, 6))
    plt.hist(max_confidence, bins=20, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.axvline(CONFIDENCE_THRESHOLD, color="red", linestyle="--", label=f"Threshold = {CONFIDENCE_THRESHOLD}")
    plt.title("XGBoost Held-Out Web Attack Confidence Distribution")
    plt.xlabel("Maximum Prediction Confidence")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "xgboost_novel_attack_confidence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("XGBOOST HELD-OUT WEB ATTACK ANALYSIS")
    print(result_df.to_string(index=False))
    print("\nConfidence statistics:")
    print(confidence_df.to_string(index=False))
    print(f"\nAnalysis CSV: {os.path.abspath(result_path)}")
    print(f"Confidence CSV: {os.path.abspath(confidence_path)}")
    print(f"Confidence plot: {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    main()
