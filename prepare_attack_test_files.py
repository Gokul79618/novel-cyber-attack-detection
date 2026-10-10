import os
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from predict import create_prediction_summary, predict_dataframe

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset", "MachineLearningCVE")
TEST_DATA_PATH = os.path.join(BASE_DIR, "split_data", "test", "test_data.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
NOVEL_UPLOAD_PATH = os.path.join(BASE_DIR, "test_files", "novel_attack_upload.csv")
NOVEL_LABELED_PATH = os.path.join(BASE_DIR, "test_files", "novel_attack_labeled.csv")
OUTPUT_ROOT = os.path.join(BASE_DIR, "test_files")
REPORTS_DIR = os.path.join(OUTPUT_ROOT, "reports")


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def normalize_label(value):
    return str(value).strip() if value is not None else ""


def discover_original_files():
    csv_files = sorted(
        [os.path.join(DATASET_ROOT, name) for name in os.listdir(DATASET_ROOT) if name.lower().endswith(".csv")]
    )
    if not csv_files:
        raise FileNotFoundError(f"No original CIC-IDS2017 CSV files found in {DATASET_ROOT}")
    return csv_files


def inspect_original_labels(csv_files):
    print("ORIGINAL CIC-IDS2017 FILES")
    print(f"Number of original CSV files found: {len(csv_files)}")
    print("Exact filenames:")
    for csv_file in csv_files:
        print(os.path.basename(csv_file))
    print()

    label_counts_by_file = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        if "Label" not in df.columns:
            raise ValueError(f"Label column missing from {os.path.basename(csv_file)}")
        df["Label"] = df["Label"].astype(str).str.strip()
        label_counts = Counter(df["Label"].tolist())
        label_counts_by_file[os.path.basename(csv_file)] = label_counts
        print(f"FILE: {os.path.basename(csv_file)}")
        print("Labels found:")
        for label, count in sorted(label_counts.items()):
            print(f"{label}: {count}")
        print()
    return label_counts_by_file


def print_dataset_label_summary(label_counts_by_file):
    combined_counter = Counter()
    for counts in label_counts_by_file.values():
        combined_counter.update(counts)

    print("ALL DATASET LABELS")
    for label, count in sorted(combined_counter.items()):
        print(f"{label}: {count}")
    print(f"Total unique labels: {len(combined_counter)}")
    print()
    return combined_counter


def load_test_split_and_features():
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Missing test split: {TEST_DATA_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature metadata: {FEATURES_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df.columns = test_df.columns.str.strip()
    if "Label" not in test_df.columns:
        raise ValueError("test_data.csv is missing the Label column.")
    test_df["Label"] = test_df["Label"].astype(str).str.strip()

    feature_names = joblib.load(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)

    print("TEST DATASET")
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Test row count: {len(test_df)}")
    print(f"Test column count: {len(test_df.columns)}")
    print("Unique test labels:")
    print(sorted(test_df["Label"].unique().tolist()))
    print("Test class distribution:")
    print(dict(Counter(test_df["Label"].tolist())))
    print()

    print("MODEL FEATURES")
    print(f"Total feature count: {len(feature_names)}")
    print("Exact feature list in saved order:")
    for feature in feature_names:
        print(feature)
    print()

    missing_features = [feature for feature in feature_names if feature not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in test_data.csv: {missing_features}")

    print("TRAINED MODEL CLASSES")
    print(list(model.classes_))
    print()
    trained_classes = [str(label).strip() for label in model.classes_]
    known_attack_classes = [label for label in trained_classes if label != "BENIGN"]
    print(f"Total known attack classes: {len(known_attack_classes)}")
    print("Known attack classes from model:")
    print(known_attack_classes)
    print()

    return test_df, feature_names, known_attack_classes


def validate_feature_frame(df, feature_names):
    if df.empty:
        raise ValueError("Generated upload file is empty.")
    if len(df.columns) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} columns but found {len(df.columns)}.")
    if list(df.columns) != feature_names:
        raise ValueError("Column order does not match feature_names.pkl.")
    if "Label" in df.columns:
        raise ValueError("Label column should not be present in upload file.")
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError("Encountered non-numeric columns.")
    if df.isna().any().any():
        raise ValueError("Encountered NaN values.")
    if np.isinf(df.to_numpy(dtype=float)).any():
        raise ValueError("Encountered infinity values.")
    return True


def save_upload_csv(output_path, df, feature_names):
    df = df.copy()
    df = df[feature_names]
    validate_feature_frame(df, feature_names)
    df.to_csv(output_path, index=False)
    return output_path


def sample_rows(df, sample_size, random_state=42):
    if len(df) <= sample_size:
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def run_prediction_and_summary(file_path, source_type, expected_label):
    upload_df = pd.read_csv(file_path)
    results_df = predict_dataframe(upload_df)
    summary = create_prediction_summary(results_df)
    if len(results_df) != len(upload_df):
        raise ValueError(f"Prediction row-count validation failed for {os.path.basename(file_path)}")

    print(f"FILE: {os.path.basename(file_path)}")
    print(f"Input rows: {len(upload_df)}")
    print(f"Benign: {summary['Benign_Count']}")
    print(f"Known Attack: {summary['Known_Attack_Count']}")
    print(f"Suspicious: {summary['Suspicious_Count']}")
    print("Raw prediction distribution:")
    print(results_df["Raw_Prediction"].value_counts().to_dict())
    print("Final category distribution:")
    print(results_df["Final_Category"].value_counts().to_dict())
    print()

    return {
        "File_Name": os.path.basename(file_path),
        "File_Path": file_path,
        "Source_Type": source_type,
        "Expected_Attack_Label": expected_label,
        "Input_Row_Count": len(upload_df),
        "Feature_Count": len(upload_df.columns),
        "Benign_Count": summary["Benign_Count"],
        "Known_Attack_Count": summary["Known_Attack_Count"],
        "Suspicious_Count": summary["Suspicious_Count"],
        "Validation_Status": "PASSED",
    }


def write_validation_report(report_rows):
    report_path = os.path.join(REPORTS_DIR, "prediction_validation_report.csv")
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"Validation report saved to: {report_path}")
    return report_path


def main():
    print("PREPARING ATTACK-SPECIFIC TEST FILES")
    print("-" * 32)

    ensure_directory(OUTPUT_ROOT)
    ensure_directory(os.path.join(OUTPUT_ROOT, "known_attacks"))
    ensure_directory(os.path.join(OUTPUT_ROOT, "benign"))
    ensure_directory(os.path.join(OUTPUT_ROOT, "suspicious"))
    ensure_directory(os.path.join(OUTPUT_ROOT, "mixed"))
    ensure_directory(REPORTS_DIR)

    csv_files = discover_original_files()
    label_counts_by_file = inspect_original_labels(csv_files)
    combined_counter = print_dataset_label_summary(label_counts_by_file)

    test_df, feature_names, known_attack_classes = load_test_split_and_features()
    test_df = test_df.copy()
    test_df["Label"] = test_df["Label"].astype(str).str.strip()

    if not os.path.exists(NOVEL_UPLOAD_PATH):
        raise FileNotFoundError(f"Missing held-out upload file: {NOVEL_UPLOAD_PATH}")
    if not os.path.exists(NOVEL_LABELED_PATH):
        raise FileNotFoundError(f"Missing held-out labeled file: {NOVEL_LABELED_PATH}")

    novel_upload_df = pd.read_csv(NOVEL_UPLOAD_PATH)
    novel_labeled_df = pd.read_csv(NOVEL_LABELED_PATH)
    novel_labeled_df.columns = novel_labeled_df.columns.str.strip()
    if "Label" not in novel_labeled_df.columns:
        raise ValueError("novel_attack_labeled.csv is missing the Label column.")
    if len(novel_upload_df.columns) != len(feature_names):
        raise ValueError("novel_attack_upload.csv does not contain the expected feature count.")
    if list(novel_upload_df.columns) != feature_names:
        raise ValueError("novel_attack_upload.csv feature order does not match feature_names.pkl.")

    print("HELD-OUT WEB ATTACK LABELS")
    print(sorted(novel_labeled_df["Label"].astype(str).str.strip().unique().tolist()))
    print()

    created_files = []
    report_rows = []

    print("KNOWN ATTACK FILE LINEAGE")
    print("Original 8 CIC-IDS2017 CSV files → preprocessing → 80/20 stratified split → test_data.csv → Label-based attack filtering → exact 70 feature selection → Label removed → attack-specific upload CSV")
    print()

    for attack_class in known_attack_classes:
        attack_rows = test_df.loc[test_df["Label"] == attack_class].copy()
        if attack_rows.empty:
            print(f"WARNING: {attack_class} has only 0 independent test rows.")
            continue
        if len(attack_rows) < 10:
            print(f"WARNING: {attack_class} has only {len(attack_rows)} independent test rows.")
        output_path = os.path.join(OUTPUT_ROOT, "known_attacks", f"{attack_class.lower().replace(' ', '_').replace('-', '_')}_attack.csv")
        output_path = output_path.replace("__", "_")
        save_upload_csv(output_path, attack_rows, feature_names)
        created_files.append(output_path)
        report_rows.append(run_prediction_and_summary(output_path, "KNOWN_ATTACK_TEST_SPLIT", attack_class))

    benign_rows = test_df.loc[test_df["Label"] == "BENIGN"].copy()
    benign_sample = sample_rows(benign_rows, 1000)
    benign_output = os.path.join(OUTPUT_ROOT, "benign", "benign_traffic.csv")
    save_upload_csv(benign_output, benign_sample, feature_names)
    created_files.append(benign_output)
    report_rows.append(run_prediction_and_summary(benign_output, "BENIGN_TEST_SPLIT", "BENIGN"))

    suspicious_output = os.path.join(OUTPUT_ROOT, "suspicious", "web_attack_suspicious.csv")
    save_upload_csv(suspicious_output, novel_upload_df, feature_names)
    created_files.append(suspicious_output)
    report_rows.append(run_prediction_and_summary(suspicious_output, "HELD_OUT_WEB_ATTACK", "HELD_OUT_WEB_ATTACK"))

    mixed_specs = [
        ("mixed_ddos_portscan.csv", ["DDoS", "PortScan"], 2000),
        ("mixed_dos_attacks.csv", ["DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"], 2000),
        ("mixed_bruteforce_attacks.csv", ["FTP-Patator", "SSH-Patator"], 2000),
        ("mixed_known_attacks.csv", known_attack_classes, 2000),
        ("mixed_known_suspicious.csv", known_attack_classes + ["HELD_OUT_WEB_ATTACK"], 2000),
        ("complete_network_demo.csv", ["BENIGN"] + known_attack_classes + ["HELD_OUT_WEB_ATTACK"], 2000),
    ]

    for file_name, source_labels, max_rows_per_class in mixed_specs:
        selected_frames = []
        for label in source_labels:
            if label == "HELD_OUT_WEB_ATTACK":
                selected_frames.append(novel_upload_df.copy())
            elif label == "BENIGN":
                selected_frames.append(benign_sample.copy())
            else:
                frame = test_df.loc[test_df["Label"] == label].copy()
                if len(frame) > max_rows_per_class:
                    frame = frame.sample(n=max_rows_per_class, random_state=42).reset_index(drop=True)
                selected_frames.append(frame)
        mixed_rows = pd.concat(selected_frames, ignore_index=True)
        mixed_rows = mixed_rows.sample(frac=1, random_state=42).reset_index(drop=True)
        output_path = os.path.join(OUTPUT_ROOT, "mixed", file_name)
        save_upload_csv(output_path, mixed_rows, feature_names)
        created_files.append(output_path)
        expected_label = "MULTIPLE" if len(source_labels) > 1 else source_labels[0]
        report_rows.append(run_prediction_and_summary(output_path, "MIXED_TEST_FILE", expected_label))

    report_path = write_validation_report(report_rows)

    print("KNOWN ATTACK FILE LINEAGE")
    print("Original 8 CIC-IDS2017 CSV files → preprocessing → Web Attack exclusion → 80/20 stratified split → test_data.csv → Label-based attack filtering → exact 70 model features selected → Label removed → attack-specific upload CSV")
    print()
    print("HELD-OUT WEB ATTACK LINEAGE")
    print("Thursday Web Attack CIC-IDS2017 traffic → Web Attack rows extracted before model training → duplicate removal → excluded from Random Forest training → novel_attack_labeled.csv → Label removed → novel_attack_upload.csv → web_attack_suspicious.csv")
    print()
    print("Created files:")
    for path in created_files:
        print(path)
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
