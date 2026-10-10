import gc
import os

import joblib
import numpy as np
import pandas as pd


def main():
    print("NOVEL ATTACK FILE PREPARATION")
    print("-" * 30)

    dataset_path = os.path.join("dataset", "MachineLearningCVE")
    os.makedirs("test_files", exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Error: Missing dataset directory -> {dataset_path}")
        return

    csv_files = sorted([os.path.join(dataset_path, file_name) for file_name in os.listdir(dataset_path) if file_name.lower().endswith(".csv")])
    print(f"CIC-IDS2017 CSV files found: {len(csv_files)}")

    feature_names_path = os.path.join("models", "feature_names.pkl")
    if not os.path.exists(feature_names_path):
        print(f"Error: Missing feature metadata -> {feature_names_path}")
        return

    feature_names = joblib.load(feature_names_path)
    print("FEATURE METADATA LOADED")
    print(f"TOTAL MODEL FEATURES: {len(feature_names)}")

    extracted_frames = []
    web_attack_counts = []

    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        print(f"\nInspecting file: {file_name}")

        if not os.path.exists(csv_file):
            print(f"Error: Missing file -> {csv_file}")
            return

        try:
            df = pd.read_csv(csv_file)
        except Exception as error:
            print(f"Error reading {file_name}: {error}")
            return

        df.columns = df.columns.str.strip()
        if "Label" not in df.columns:
            print(f"Error: 'Label' column not found in {file_name}")
            return

        df["Label"] = df["Label"].astype(str).str.strip()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0)

        web_attack_mask = df["Label"].str.contains("Web Attack", case=False, na=False)
        web_attack_rows = df.loc[web_attack_mask]
        web_attack_count = len(web_attack_rows)
        web_attack_counts.append((file_name, web_attack_count))
        print(f"Web Attack rows found: {web_attack_count}")

        if web_attack_count > 0:
            extracted_frames.append(web_attack_rows)

        del df, web_attack_rows
        gc.collect()

    if not extracted_frames:
        print("Error: No Web Attack rows were extracted.")
        return

    print("\nWEB ATTACK DATASET SUMMARY")
    rows_before_duplicates = sum(len(frame) for frame in extracted_frames)
    print(f"Rows before duplicate removal: {rows_before_duplicates}")

    novel_attack_data = pd.concat(extracted_frames, ignore_index=True)
    novel_attack_data = novel_attack_data.drop_duplicates(ignore_index=True)
    rows_after_duplicates = len(novel_attack_data)
    duplicate_rows_removed = rows_before_duplicates - rows_after_duplicates
    print(f"Duplicate Web Attack rows removed: {duplicate_rows_removed}")
    print(f"Final held-out Web Attack rows: {rows_after_duplicates}")

    if rows_after_duplicates != 2143:
        raise ValueError(
            f"Held-out row-count validation failed: expected 2143 rows but found {rows_after_duplicates}."
        )

    print("HELD-OUT WEB ATTACK ROW COUNT VALIDATION: PASSED")

    print("\nWEB ATTACK LABELS")
    labels = sorted(novel_attack_data["Label"].astype(str).unique())
    print(labels)

    invalid_labels = [label for label in novel_attack_data["Label"].astype(str) if "web attack" not in label.lower()]
    if invalid_labels:
        raise ValueError("Web Attack label validation failed: some rows do not contain Web Attack labels.")

    print("WEB ATTACK LABEL VALIDATION: PASSED")

    missing_features = [feature for feature in feature_names if feature not in novel_attack_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in held-out dataset: {missing_features}")

    novel_features = novel_attack_data[feature_names]
    if not all(pd.api.types.is_numeric_dtype(novel_features[col]) for col in novel_features.columns):
        raise ValueError("Held-out feature validation failed: non-numeric columns were found.")

    if novel_features.isna().any().any():
        raise ValueError("Held-out feature validation failed: NaN values remain.")

    if np.isinf(novel_features.to_numpy(dtype=float)).any():
        raise ValueError("Held-out feature validation failed: infinity values remain.")

    print("NOVEL ATTACK FEATURE VALIDATION: PASSED")

    novel_attack_labeled = novel_attack_data[feature_names + ["Label"]]
    labeled_path = os.path.join("test_files", "novel_attack_labeled.csv")
    novel_attack_labeled.to_csv(labeled_path, index=False)
    print(f"\nNOVEL ATTACK LABELED FILE SAVED: {os.path.abspath(labeled_path)}")

    novel_attack_upload = novel_attack_data[feature_names]
    upload_path = os.path.join("test_files", "novel_attack_upload.csv")
    novel_attack_upload.to_csv(upload_path, index=False)
    print(f"NOVEL ATTACK UPLOAD FILE SAVED: {os.path.abspath(upload_path)}")

    if not os.path.exists(labeled_path):
        raise FileNotFoundError(f"Labeled file was not created: {labeled_path}")
    if not os.path.exists(upload_path):
        raise FileNotFoundError(f"Upload file was not created: {upload_path}")

    if os.path.getsize(labeled_path) <= 0:
        raise ValueError("Labeled file is empty.")
    if os.path.getsize(upload_path) <= 0:
        raise ValueError("Upload file is empty.")

    labeled_header = pd.read_csv(labeled_path, nrows=0)
    upload_header = pd.read_csv(upload_path, nrows=0)

    if list(labeled_header.columns) != feature_names + ["Label"]:
        raise ValueError("Labeled file columns do not match the expected feature order.")
    if list(upload_header.columns) != feature_names:
        raise ValueError("Upload file columns do not match the expected feature order.")
    if "Label" in upload_header.columns:
        raise ValueError("Upload file should not contain the Label column.")

    print("LABELED NOVEL FILE VERIFICATION: PASSED")
    print("UPLOAD NOVEL FILE VERIFICATION: PASSED")

    print("\nNOVEL ATTACK TEST FILE SUMMARY")
    print(f"Total held-out rows: {len(novel_attack_data)}")
    print(f"Labeled file rows: {len(novel_attack_labeled)}")
    print(f"Upload file rows: {len(novel_attack_upload)}")
    print(f"Labeled file columns: {len(novel_attack_labeled.columns)}")
    print(f"Upload file columns: {len(novel_attack_upload.columns)}")
    print(f"Labeled file size (MB): {os.path.getsize(labeled_path) / (1024 * 1024):.2f}")
    print(f"Upload file size (MB): {os.path.getsize(upload_path) / (1024 * 1024):.2f}")

    if len(novel_attack_labeled) != len(novel_attack_upload):
        raise ValueError("Labeled and upload files do not contain matching row counts.")

    del novel_attack_data, novel_attack_labeled, novel_attack_upload, extracted_frames
    gc.collect()

    print("\nPhase 5A completed successfully.")


if __name__ == "__main__":
    main()
