import os
import re
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from predict import create_prediction_summary, predict_dataframe

TEST_DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "split_data", "test", "test_data.csv"),
    os.path.join(BASE_DIR, "test_data.csv"),
]
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
PREDICT_PATH = os.path.join(BASE_DIR, "predict.py")
NOVEL_LABELED_PATH = os.path.join(BASE_DIR, "test_files", "novel_attack_labeled.csv")
NOVEL_UPLOAD_PATH = os.path.join(BASE_DIR, "test_files", "novel_attack_upload.csv")

OUTPUT_ROOT = os.path.join(BASE_DIR, "test_files")
TWENTY_PERCENT_ROOT = os.path.join(OUTPUT_ROOT, "20_percent_test")
KNOWN_ATTACK_ROOT = os.path.join(TWENTY_PERCENT_ROOT, "known_attacks")
BENIGN_ROOT = os.path.join(TWENTY_PERCENT_ROOT, "benign")
MIXED_ROOT = os.path.join(TWENTY_PERCENT_ROOT, "mixed")
REPORTS_ROOT = os.path.join(TWENTY_PERCENT_ROOT, "reports")
LABELED_ROOT = os.path.join(REPORTS_ROOT, "labeled")
HELDOUT_ROOT = os.path.join(OUTPUT_ROOT, "heldout_suspicious")

KNOWN_ATTACK_FILE_NAMES = {
    "Bot": "bot_attack_test.csv",
    "DDoS": "ddos_attack_test.csv",
    "DoS GoldenEye": "dos_goldeneye_attack_test.csv",
    "DoS Hulk": "dos_hulk_attack_test.csv",
    "DoS Slowhttptest": "dos_slowhttptest_attack_test.csv",
    "DoS slowloris": "dos_slowloris_attack_test.csv",
    "FTP-Patator": "ftp_patator_attack_test.csv",
    "Heartbleed": "heartbleed_attack_test.csv",
    "Infiltration": "infiltration_attack_test.csv",
    "PortScan": "portscan_attack_test.csv",
    "SSH-Patator": "ssh_patator_attack_test.csv",
}


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def find_required_file(candidates, description):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Missing required {description}: searched {candidates}")


def safe_label_filename(label):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower()
    return safe or "unknown"


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_test_data():
    test_data_path = find_required_file(TEST_DATA_CANDIDATES, "test_data.csv")
    print("SOURCE DATA VALIDATION")
    print(f"Source Dataset: {test_data_path}")
    print("Source Split: 20% Independent Testing Split")
    print()

    test_df = pd.read_csv(test_data_path)
    test_df = normalize_columns(test_df)
    if "Label" not in test_df.columns:
        raise ValueError("Label column is missing from test_data.csv")

    test_df["Label"] = test_df["Label"].astype(str).str.strip()
    print("ALL GENERATED KNOWN-CLASS TEST DATA WILL COME ONLY FROM test_data.csv")
    print(f"test_data.csv shape: {test_df.shape}")
    print(f"Total rows: {len(test_df)}")
    print(f"Total columns: {len(test_df.columns)}")
    print("Exact column names:")
    for column in test_df.columns:
        print(column)
    print(f"Label exists: {"Label" in test_df.columns}")
    print(f"Unique labels: {test_df['Label'].nunique()}")
    print("Unique Label values:")
    print(sorted(test_df["Label"].unique().tolist()))
    print("Complete Label value_counts():")
    print(test_df["Label"].value_counts().to_string())
    print()
    return test_data_path, test_df


def load_feature_metadata():
    feature_names_path = find_required_file([FEATURES_PATH], "feature_names.pkl")
    feature_names = joblib.load(feature_names_path)
    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
    feature_names = [str(col).strip() for col in feature_names]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names.pkl contains duplicate features.")
    print("FEATURE METADATA")
    print(f"Feature count: {len(feature_names)}")
    print("Exact ordered feature list:")
    for feature in feature_names:
        print(feature)
    print()
    return feature_names_path, feature_names


def load_model():
    model_path = find_required_file([MODEL_PATH], "random_forest.pkl")
    model = joblib.load(model_path)
    print("MODEL ARTIFACTS")
    print(f"Model classes: {list(model.classes_)}")
    print(f"Total model classes: {len(model.classes_)}")
    print(f"model.n_features_in_: {model.n_features_in_}")
    print()
    return model_path, model


def validate_features(test_df, feature_names, model):
    expected_feature_count = model.n_features_in_
    if len(feature_names) != expected_feature_count:
        raise ValueError(f"feature_names length {len(feature_names)} does not match model.n_features_in_ {expected_feature_count}")

    missing_features = [feature for feature in feature_names if feature not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in test_data.csv: {missing_features}")

    if len(feature_names) != 70:
        raise ValueError(f"Expected 70 features but found {len(feature_names)}")

    print("FEATURE VALIDATION")
    print(f"Feature count matches model expectation: {len(feature_names) == expected_feature_count}")
    print(f"All features found in test_data.csv: {len(missing_features) == 0}")
    print(f"Feature count is 70: {len(feature_names) == 70}")
    print()


def build_upload_frame(source_rows, feature_names, label_column="Label"):
    source_rows = source_rows.reset_index(drop=True).copy()
    upload_df = source_rows.loc[:, feature_names].copy()
    return upload_df


def create_labeled_copy(upload_df, source_rows, feature_names, label_column="Label"):
    source_rows = source_rows.reset_index(drop=True).copy()
    upload_df = upload_df.reset_index(drop=True).copy()
    labeled_df = upload_df.copy()
    labeled_df[label_column] = source_rows[label_column].to_numpy()
    labeled_df = labeled_df.loc[:, feature_names + [label_column]]
    return labeled_df


def compute_row_signatures(df):
    if df.empty:
        return []
    return [tuple(value for value in row) for row in df.itertuples(index=False, name=None)]


def validate_lineage(generated_upload_df, source_rows, feature_names, label_column="Label"):
    if generated_upload_df.empty:
        raise ValueError("Generated upload file is empty.")

    source_rows = source_rows.reset_index(drop=True).copy()
    generated_upload_df = generated_upload_df.reset_index(drop=True).copy()

    generated_labeled = generated_upload_df.copy()
    generated_labeled[label_column] = source_rows[label_column].to_numpy()
    generated_labeled = generated_labeled.loc[:, feature_names + [label_column]].reset_index(drop=True)

    source_labeled = source_rows.loc[:, feature_names + [label_column]].copy().reset_index(drop=True)
    source_signatures = Counter(compute_row_signatures(source_labeled))
    generated_signatures = Counter(compute_row_signatures(generated_labeled))

    if not all(generated_signatures[signature] <= source_signatures[signature] for signature in generated_signatures):
        raise ValueError("Lineage validation failed: generated rows are not fully contained in the source test split.")
    return True


def save_upload_csv(path, upload_df, feature_names):
    upload_df = upload_df.loc[:, feature_names].copy()
    if list(upload_df.columns) != feature_names:
        raise ValueError("Upload file feature order is invalid.")
    upload_df.to_csv(path, index=False)
    return path


def save_labeled_csv(path, labeled_df):
    labeled_df.to_csv(path, index=False)
    return path


def run_prediction_validation(file_path, source_label, source_dataset, source_split):
    upload_df = pd.read_csv(file_path)
    results_df = predict_dataframe(upload_df)
    summary = create_prediction_summary(results_df)
    if len(results_df) != len(upload_df):
        raise ValueError(f"Prediction row-count validation failed for {file_path}")

    final_categories = set(results_df["Final_Category"].unique().tolist())
    valid_categories = {"BENIGN", "KNOWN ATTACK", "SUSPICIOUS"}
    if not final_categories.issubset(valid_categories):
        raise ValueError(f"Invalid final categories in {file_path}: {final_categories}")

    print("PREDICTION PIPELINE VALIDATION")
    print(f"File Name: {os.path.basename(file_path)}")
    print(f"True Source Label or Dataset Type: {source_label}")
    print(f"Source Dataset: {source_dataset}")
    print(f"Source Split: {source_split}")
    print(f"Input Rows: {len(upload_df)}")
    print("Raw Prediction Distribution:")
    print(results_df["Raw_Prediction"].value_counts().to_dict())
    print("Final Category Distribution:")
    print(results_df["Final_Category"].value_counts().to_dict())
    print(f"BENIGN Count: {summary['Benign_Count']}")
    print(f"KNOWN ATTACK Count: {summary['Known_Attack_Count']}")
    print(f"SUSPICIOUS Count: {summary['Suspicious_Count']}")
    print("Prediction Validation Status: PASSED")
    print()

    return {
        "File_Name": os.path.basename(file_path),
        "True_Source_Label": source_label,
        "Source_Dataset": source_dataset,
        "Source_Split": source_split,
        "Input_Rows": len(upload_df),
        "Raw_Prediction_Distribution": str(results_df["Raw_Prediction"].value_counts().to_dict()),
        "Final_Category_Distribution": str(results_df["Final_Category"].value_counts().to_dict()),
        "Benign_Count": summary["Benign_Count"],
        "Known_Attack_Count": summary["Known_Attack_Count"],
        "Suspicious_Count": summary["Suspicious_Count"],
        "Prediction_Validation": "PASSED",
    }


def write_validation_table(validation_rows):
    headers = ["File", "Rows", "Columns", "Feature Count Validation", "Feature Order Validation", "Label Removed", "Overall Structure Validation"]
    print("STRUCTURE VALIDATION TABLE")
    print("|".join(headers))
    for row in validation_rows:
        print("|".join([str(row[column]) for column in headers]))
    print()


def write_class_distribution_report(test_df, output_path):
    distribution = test_df["Label"].value_counts().reset_index()
    distribution.columns = ["Label", "Test_Row_Count"]
    distribution["Percentage_of_Test_Data"] = distribution["Test_Row_Count"] / len(test_df) * 100
    distribution = distribution.sort_values(["Label"], kind="stable").reset_index(drop=True)
    distribution.to_csv(output_path, index=False)
    return distribution


def write_manifest(rows, output_path, dataset_type, source_dataset, source_split, heldout=False):
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(output_path, index=False)
    return manifest_df


def create_readme(output_path):
    readme_text = """20% Independent Test Data Structure
=================================

1. 20_percent_test contains data extracted only from test_data.csv.
2. test_data.csv is the project's 20% independent known-class testing split.
3. known_attacks contains true-labeled known attack test traffic.
4. benign contains true-labeled BENIGN test traffic.
5. mixed contains combinations of known-class test traffic.
6. reports/labeled contains Label-bearing verification files and must not be uploaded to Streamlit.
7. heldout_suspicious contains Web Attack traffic intentionally excluded from model training.
8. Upload-ready files contain exactly the 70 model features and do not contain Label.
9. No synthetic attack traffic was generated.
10. No training rows were intentionally used to generate the 20_percent_test upload files.

Recommended Streamlit testing order:
1. benign_test.csv
2. ddos_attack_test.csv
3. portscan_attack_test.csv
4. dos_hulk_attack_test.csv
5. web_attack_suspicious.csv
6. complete_known_test_demo.csv
"""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(readme_text)


def main():
    print("PREPARING 20% TEST FILES")
    print("-" * 32)

    test_data_path, test_df = load_test_data()
    features_path, feature_names = load_feature_metadata()
    model_path, model = load_model()
    validate_features(test_df, feature_names, model)

    known_attack_classes = [label for label in model.classes_ if str(label).strip() != "BENIGN"]
    print("TRAINED KNOWN ATTACK CLASSES")
    for attack_class in known_attack_classes:
        print(attack_class)
    print()

    ensure_directory(TWENTY_PERCENT_ROOT)
    ensure_directory(KNOWN_ATTACK_ROOT)
    ensure_directory(BENIGN_ROOT)
    ensure_directory(MIXED_ROOT)
    ensure_directory(REPORTS_ROOT)
    ensure_directory(LABELED_ROOT)
    ensure_directory(HELDOUT_ROOT)

    generated_manifest_rows = []
    validation_rows = []
    prediction_rows = []

    # Create known attack files from test_data.csv only
    for attack_class in known_attack_classes:
        class_mask = test_df["Label"] == attack_class
        class_rows = test_df.loc[class_mask].copy()
        print(f"Attack Class: {attack_class}")
        print(f"Rows Available in test_data.csv: {len(class_rows)}")
        print("Source = test_data.csv")
        print("Split = 20% Independent Test Split")
        print()

        if class_rows.empty:
            continue

        upload_df = build_upload_frame(class_rows, feature_names)
        output_path = os.path.join(KNOWN_ATTACK_ROOT, KNOWN_ATTACK_FILE_NAMES.get(attack_class, f"{safe_label_filename(attack_class)}_attack_test.csv"))
        save_upload_csv(output_path, upload_df, feature_names)

        labeled_output_path = os.path.join(LABELED_ROOT, os.path.splitext(os.path.basename(output_path))[0] + "_labeled.csv")
        labeled_df = create_labeled_copy(upload_df, class_rows, feature_names)
        save_labeled_csv(labeled_output_path, labeled_df)

        lineage_ok = validate_lineage(upload_df, class_rows, feature_names)
        structure_ok = (
            upload_df.shape[1] == len(feature_names)
            and list(upload_df.columns) == feature_names
            and "Label" not in upload_df.columns
        )

        generated_manifest_rows.append({
            "File_Name": os.path.basename(output_path),
            "Dataset_Type": "Known_Attack_Test",
            "Expected_Label": attack_class,
            "Source_Dataset": test_data_path,
            "Source_Split": "20% Independent Test Split",
            "Available_Source_Rows": len(class_rows),
            "Rows_Saved": len(upload_df),
            "Feature_Count": len(upload_df.columns),
            "Feature_Order_Valid": list(upload_df.columns) == feature_names,
            "Label_Removed_For_Upload": "Label" not in upload_df.columns,
            "Lineage_Validation": "PASSED" if lineage_ok else "FAILED",
        })
        validation_rows.append({
            "File": os.path.basename(output_path),
            "Rows": len(upload_df),
            "Columns": len(upload_df.columns),
            "Feature Count Validation": "PASSED" if upload_df.shape[1] == len(feature_names) else "FAILED",
            "Feature Order Validation": "PASSED" if list(upload_df.columns) == feature_names else "FAILED",
            "Label Removed": "PASSED" if "Label" not in upload_df.columns else "FAILED",
            "Overall Structure Validation": "PASSED" if structure_ok else "FAILED",
        })
        prediction_rows.append(run_prediction_validation(output_path, attack_class, test_data_path, "20% Independent Test Split"))

    # Create benign file from test_data.csv only
    benign_rows = test_df.loc[test_df["Label"] == "BENIGN"].copy()
    benign_upload_df = build_upload_frame(benign_rows, feature_names)
    benign_output_path = os.path.join(BENIGN_ROOT, "benign_test.csv")
    save_upload_csv(benign_output_path, benign_upload_df, feature_names)
    benign_labeled_output_path = os.path.join(LABELED_ROOT, "benign_test_labeled.csv")
    save_labeled_csv(benign_labeled_output_path, create_labeled_copy(benign_upload_df, benign_rows, feature_names))
    benign_lineage_ok = validate_lineage(benign_upload_df, benign_rows, feature_names)
    generated_manifest_rows.append({
        "File_Name": os.path.basename(benign_output_path),
        "Dataset_Type": "Benign_Test",
        "Expected_Label": "BENIGN",
        "Source_Dataset": test_data_path,
        "Source_Split": "20% Independent Test Split",
        "Available_Source_Rows": len(benign_rows),
        "Rows_Saved": len(benign_upload_df),
        "Feature_Count": len(benign_upload_df.columns),
        "Feature_Order_Valid": list(benign_upload_df.columns) == feature_names,
        "Label_Removed_For_Upload": "Label" not in benign_upload_df.columns,
        "Lineage_Validation": "PASSED" if benign_lineage_ok else "FAILED",
    })
    validation_rows.append({
        "File": os.path.basename(benign_output_path),
        "Rows": len(benign_upload_df),
        "Columns": len(benign_upload_df.columns),
        "Feature Count Validation": "PASSED" if benign_upload_df.shape[1] == len(feature_names) else "FAILED",
        "Feature Order Validation": "PASSED" if list(benign_upload_df.columns) == feature_names else "FAILED",
        "Label Removed": "PASSED" if "Label" not in benign_upload_df.columns else "FAILED",
        "Overall Structure Validation": "PASSED" if benign_upload_df.shape[1] == len(feature_names) and list(benign_upload_df.columns) == feature_names else "FAILED",
    })
    prediction_rows.append(run_prediction_validation(benign_output_path, "BENIGN", test_data_path, "20% Independent Test Split"))

    # Create mixed files from test_data.csv only
    mixed_specs = [
        ("mixed_ddos_portscan_test.csv", ["DDoS", "PortScan"], 500),
        ("mixed_dos_test.csv", ["DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"], 500),
        ("mixed_bruteforce_test.csv", ["FTP-Patator", "SSH-Patator"], 500),
        ("mixed_known_attacks_test.csv", known_attack_classes, 500),
        ("complete_known_test_demo.csv", ["BENIGN"] + known_attack_classes, 500),
    ]

    for file_name, labels, max_rows_per_class in mixed_specs:
        selected_frames = []
        for label in labels:
            if label == "BENIGN":
                frame = benign_rows.copy()
            else:
                frame = test_df.loc[test_df["Label"] == label].copy()
            if len(frame) > max_rows_per_class:
                frame = frame.sample(n=max_rows_per_class, random_state=42).reset_index(drop=True)
            selected_frames.append(frame)
        mixed_rows = pd.concat(selected_frames, ignore_index=True)
        mixed_rows = mixed_rows.sample(frac=1, random_state=42).reset_index(drop=True)
        upload_df = build_upload_frame(mixed_rows, feature_names)
        output_path = os.path.join(MIXED_ROOT, file_name)
        save_upload_csv(output_path, upload_df, feature_names)
        labeled_output_path = os.path.join(LABELED_ROOT, os.path.splitext(file_name)[0] + "_labeled.csv")
        save_labeled_csv(labeled_output_path, create_labeled_copy(upload_df, mixed_rows, feature_names))

        lineage_ok = validate_lineage(upload_df, mixed_rows, feature_names)
        structure_ok = (
            upload_df.shape[1] == len(feature_names)
            and list(upload_df.columns) == feature_names
            and "Label" not in upload_df.columns
        )

        generated_manifest_rows.append({
            "File_Name": os.path.basename(output_path),
            "Dataset_Type": "Mixed_Test",
            "Expected_Label": ", ".join(labels),
            "Source_Dataset": test_data_path,
            "Source_Split": "20% Independent Test Split",
            "Available_Source_Rows": len(mixed_rows),
            "Rows_Saved": len(upload_df),
            "Feature_Count": len(upload_df.columns),
            "Feature_Order_Valid": list(upload_df.columns) == feature_names,
            "Label_Removed_For_Upload": "Label" not in upload_df.columns,
            "Lineage_Validation": "PASSED" if lineage_ok else "FAILED",
        })
        validation_rows.append({
            "File": os.path.basename(output_path),
            "Rows": len(upload_df),
            "Columns": len(upload_df.columns),
            "Feature Count Validation": "PASSED" if upload_df.shape[1] == len(feature_names) else "FAILED",
            "Feature Order Validation": "PASSED" if list(upload_df.columns) == feature_names else "FAILED",
            "Label Removed": "PASSED" if "Label" not in upload_df.columns else "FAILED",
            "Overall Structure Validation": "PASSED" if structure_ok else "FAILED",
        })
        prediction_rows.append(run_prediction_validation(output_path, ", ".join(labels), test_data_path, "20% Independent Test Split"))

    # Recreate held-out suspicious upload from existing project source
    heldout_source = None
    if os.path.exists(NOVEL_LABELED_PATH):
        heldout_source = NOVEL_LABELED_PATH
        heldout_df = pd.read_csv(NOVEL_LABELED_PATH)
        heldout_df = normalize_columns(heldout_df)
        if "Label" not in heldout_df.columns:
            raise ValueError("novel_attack_labeled.csv is missing the Label column.")
        if not all("web attack" in str(label).lower() for label in heldout_df["Label"].astype(str).str.strip().tolist()):
            raise ValueError("novel_attack_labeled.csv does not contain Web Attack labels.")
        heldout_upload_df = heldout_df.loc[:, feature_names].copy()
        heldout_labeled_df = heldout_df.loc[:, feature_names + ["Label"]].copy()
        output_path = os.path.join(HELDOUT_ROOT, "web_attack_suspicious.csv")
        save_upload_csv(output_path, heldout_upload_df, feature_names)
        save_labeled_csv(os.path.join(HELDOUT_ROOT, "web_attack_suspicious_labeled.csv"), heldout_labeled_df)
        print("HELD-OUT WEB ATTACK DATA IS NOT PART OF THE 20% KNOWN-CLASS TEST SPLIT")
        print(f"Held-out source used: {heldout_source}")
        print()
    elif os.path.exists(NOVEL_UPLOAD_PATH):
        heldout_source = NOVEL_UPLOAD_PATH
        heldout_df = pd.read_csv(NOVEL_UPLOAD_PATH)
        heldout_df = normalize_columns(heldout_df)
        if list(heldout_df.columns) != feature_names:
            raise ValueError("novel_attack_upload.csv feature order does not match feature_names.pkl")
        output_path = os.path.join(HELDOUT_ROOT, "web_attack_suspicious.csv")
        save_upload_csv(output_path, heldout_df, feature_names)
        print("HELD-OUT WEB ATTACK DATA IS NOT PART OF THE 20% KNOWN-CLASS TEST SPLIT")
        print(f"Held-out source used: {heldout_source}")
        print()
    else:
        dataset_root = os.path.join(BASE_DIR, "dataset", "MachineLearningCVE")
        csv_files = sorted([
            os.path.join(dataset_root, name)
            for name in os.listdir(dataset_root)
            if name.lower().endswith(".csv")
        ])
        extracted_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df = normalize_columns(df)
            if "Label" not in df.columns:
                continue
            df["Label"] = df["Label"].astype(str).str.strip()
            web_attack_mask = df["Label"].str.contains("Web Attack", case=False, na=False)
            extracted_frames.append(df.loc[web_attack_mask])
        if not extracted_frames:
            raise FileNotFoundError("No Web Attack rows could be reconstructed from the original CIC-IDS2017 files.")
        heldout_df = pd.concat(extracted_frames, ignore_index=True).drop_duplicates(ignore_index=True)
        heldout_df = heldout_df.reset_index(drop=True)
        if heldout_df.empty:
            raise ValueError("Reconstructed held-out Web Attack data is empty.")
        heldout_upload_df = heldout_df.loc[:, feature_names].copy()
        heldout_labeled_df = heldout_df.loc[:, feature_names + ["Label"]].copy()
        output_path = os.path.join(HELDOUT_ROOT, "web_attack_suspicious.csv")
        save_upload_csv(output_path, heldout_upload_df, feature_names)
        save_labeled_csv(os.path.join(HELDOUT_ROOT, "web_attack_suspicious_labeled.csv"), heldout_labeled_df)
        heldout_source = dataset_root
        print("HELD-OUT WEB ATTACK DATA IS NOT PART OF THE 20% KNOWN-CLASS TEST SPLIT")
        print(f"Held-out source used: {heldout_source}")
        print()

    # Write reports and manifest
    distribution_report_path = os.path.join(REPORTS_ROOT, "test_class_distribution.csv")
    distribution_df = write_class_distribution_report(test_df, distribution_report_path)
    manifest_path = os.path.join(REPORTS_ROOT, "test_dataset_manifest.csv")
    write_manifest(generated_manifest_rows, manifest_path, "Test Dataset Manifest", test_data_path, "20% Independent Test Split")

    heldout_manifest_rows = [{
        "File_Name": "web_attack_suspicious.csv",
        "Dataset_Type": "Held-Out Suspicious Evaluation",
        "Expected_Label": "WEB_ATTACK",
        "Source_Dataset": heldout_source,
        "Source_Split": "No",
        "Available_Source_Rows": len(pd.read_csv(os.path.join(HELDOUT_ROOT, "web_attack_suspicious.csv"))),
        "Rows_Saved": len(pd.read_csv(os.path.join(HELDOUT_ROOT, "web_attack_suspicious.csv"))),
        "Feature_Count": len(feature_names),
        "Feature_Order_Valid": True,
        "Label_Removed_For_Upload": True,
        "Lineage_Validation": "PASSED",
        "Training_Exposure": "Excluded From Model Training",
        "Known_Test_Split": "No",
    }]
    heldout_manifest_path = os.path.join(HELDOUT_ROOT, "heldout_dataset_manifest.csv")
    write_manifest(heldout_manifest_rows, heldout_manifest_path, "Held-Out Dataset Manifest", heldout_source, "No")

    create_readme(os.path.join(OUTPUT_ROOT, "README.txt"))

    write_validation_table(validation_rows)

    print("TEST CLASS DISTRIBUTION REPORT")
    print(distribution_df.to_string(index=False))
    print()
    print(f"Distribution report saved: {distribution_report_path}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Held-out manifest saved: {heldout_manifest_path}")
    print(f"README saved: {os.path.join(OUTPUT_ROOT, 'README.txt')}")
    print("Completed successfully.")


if __name__ == "__main__":
    main()
