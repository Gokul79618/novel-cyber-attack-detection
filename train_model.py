import gc
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_cic_ids_dataset():
    print("CIC-IDS2017 DATASET PREPROCESSING")
    print("-" * 36)

    dataset_path = os.path.join("dataset", "MachineLearningCVE")
    csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))

    print(f"Dataset path: {dataset_path}")
    print(f"Total CSV files found: {len(csv_files)}")

    if not csv_files:
        print("No CSV files found. Exiting safely.")
        return

    print("Detected CSV files:")
    for csv_file in csv_files:
        print(os.path.basename(csv_file))

    cleaned_frames = []
    rows_removed_per_file = []

    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        print(f"\nProcessing file: {file_name}")

        if not os.path.exists(csv_file):
            print(f"Error: File not found -> {file_name}")
            return

        if os.path.getsize(csv_file) == 0:
            print(f"Warning: File is empty -> {file_name}")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as error:
            print(f"Error reading {file_name}: {error}")
            return

        # Clean column names by removing surrounding whitespace.
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            print(f"Error: 'Label' column not found in {file_name}")
            return

        # Clean label values by removing surrounding whitespace.
        df["Label"] = df["Label"].astype(str).str.strip()

        # Replace positive/negative infinity with NaN.
        df = df.replace([np.inf, -np.inf], np.nan)

        rows_before_cleaning = len(df)
        df = df.dropna(axis=0)
        rows_removed = rows_before_cleaning - len(df)
        rows_removed_per_file.append((file_name, rows_removed))

        print(f"Rows before NaN/infinity removal: {rows_before_cleaning}")
        print(f"Rows removed due to NaN/infinity: {rows_removed}")
        print(f"Rows after NaN/infinity removal: {len(df)}")

        cleaned_frames.append(df)
        del df
        gc.collect()

    if not cleaned_frames:
        print("No valid cleaned dataframes were created. Exiting safely.")
        return

    print("\nMerging cleaned files...")
    combined_df = pd.concat(cleaned_frames, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Total rows: {combined_df.shape[0]}")
    print(f"Total columns: {combined_df.shape[1]}")

    del cleaned_frames
    gc.collect()

    print("\nRemoving duplicate rows...")
    rows_before_duplicates = len(combined_df)
    combined_df = combined_df.drop_duplicates(ignore_index=True)
    rows_after_duplicates = len(combined_df)
    duplicate_rows_removed = rows_before_duplicates - rows_after_duplicates

    print(f"Rows before duplicate removal: {rows_before_duplicates}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Final shape after duplicate removal: {combined_df.shape}")

    print("\nATTACK CATEGORIES")
    print("-" * 18)
    print(combined_df["Label"].unique())

    print("\nCLASS DISTRIBUTION")
    print("-" * 18)
    print(combined_df["Label"].value_counts())

    # Dynamically detect Web Attack labels using a case-insensitive search.
    unique_labels = sorted(combined_df["Label"].dropna().astype(str).unique())
    web_attack_labels = [label for label in unique_labels if "web attack" in label.lower()]

    print("\nWEB ATTACK LABELS IDENTIFIED")
    print("-" * 28)
    if not web_attack_labels:
        print("Error: No Web Attack labels were found in the dataset.")
        return

    print(web_attack_labels)

    novel_attack_mask = combined_df["Label"].isin(web_attack_labels)
    novel_attack_count = int(novel_attack_mask.sum())
    known_count = int(len(combined_df) - novel_attack_count)

    # Create the held-out Web Attack subset first, then remove it from the main frame.
    # This avoids creating two full copies of the large dataset at the same time.
    novel_attack_data = combined_df.loc[novel_attack_mask]
    known_data = combined_df.loc[~novel_attack_mask]

    print("\nHeld-out Web Attack separation")
    print(f"Total rows before separation: {len(combined_df)}")
    print(f"Known-data rows: {known_count}")
    print(f"Held-out Web Attack rows: {novel_attack_count}")
    print(f"Number of Web Attack labels held out: {len(web_attack_labels)}")

    row_sum_check = known_count + novel_attack_count
    if row_sum_check != len(combined_df):
        raise ValueError(
            f"Row count validation failed: known_data + novel_attack_data = {row_sum_check}, expected {len(combined_df)}"
        )

    print("Row-count validation: PASSED")

    remaining_web_attack_labels = [label for label in known_data["Label"].astype(str).unique() if "web attack" in label.lower()]
    if remaining_web_attack_labels:
        raise ValueError(f"Web Attack holdout validation failed. Remaining labels: {remaining_web_attack_labels}")

    print("Web Attack holdout validation: PASSED")
    print("\nRemaining known-data class distribution")
    print(known_data["Label"].value_counts())

    print("\nRows removed for NaN/infinite values by file:")
    for file_name, removed in rows_removed_per_file:
        print(f"- {file_name}: {removed}")

    print("\nPreparing usable model features...")
    if "Label" not in known_data.columns:
        raise ValueError("Label column is missing from known_data before model dataset creation.")

    known_data.columns = known_data.columns.str.strip()
    known_data["Label"] = known_data["Label"].astype(str).str.strip()

    candidate_features = [col for col in known_data.columns if col != "Label"]
    dropped_features = []
    feature_names = []

    for col in candidate_features:
        if not pd.api.types.is_numeric_dtype(known_data[col]):
            dropped_features.append((col, "non-numeric"))
            continue

        if known_data[col].nunique(dropna=False) <= 1:
            dropped_features.append((col, "constant"))
            continue

        feature_names.append(col)

    print(f"Total original feature columns excluding Label: {len(candidate_features)}")
    print("Non-numeric columns detected:")
    for col, reason in dropped_features:
        if reason == "non-numeric":
            print(f"- {col}: {reason}")

    print("Dropped features and reasons:")
    for col, reason in dropped_features:
        print(f"- {col}: {reason}")

    print(f"Final feature count: {len(feature_names)}")
    print("Final feature list in order:")
    for feature_name in feature_names:
        print(f"- {feature_name}")

    # Random Forest does not require feature scaling because tree-based splits are not distance-based.
    for feature_name in feature_names:
        if not pd.api.types.is_numeric_dtype(known_data[feature_name]):
            raise ValueError(f"Final feature validation failed: {feature_name} is not numeric.")

        if known_data[feature_name].isna().any():
            raise ValueError(f"Final feature validation failed: {feature_name} contains NaN values.")

        if np.isinf(known_data[feature_name].to_numpy(dtype=float)).any():
            raise ValueError(f"Final feature validation failed: {feature_name} contains infinity values.")

    print("FINAL FEATURE VALIDATION: PASSED")

    print("\nCLASS DISTRIBUTION BEFORE SAMPLING")
    print(known_data["Label"].value_counts())

    sampled_frames = []
    for label_value in known_data["Label"].dropna().unique():
        class_rows = known_data.loc[known_data["Label"] == label_value]
        sample_size = min(100000, len(class_rows))
        sampled_rows = class_rows.sample(n=sample_size, random_state=42)
        sampled_frames.append(sampled_rows)
        del class_rows
        gc.collect()

    model_data = pd.concat(sampled_frames, ignore_index=True)
    del sampled_frames
    gc.collect()

    print("\nCLASS DISTRIBUTION AFTER SAMPLING")
    print(model_data["Label"].value_counts())
    print(f"\nTOTAL SAMPLED MODEL DATA ROWS: {len(model_data)}")

    os.makedirs(os.path.join("split_data", "train"), exist_ok=True)
    os.makedirs(os.path.join("split_data", "test"), exist_ok=True)

    output_columns = feature_names + ["Label"]
    model_data = model_data[output_columns]

    class_counts = model_data["Label"].value_counts()
    stratify_valid = all(count >= 2 for count in class_counts.values)
    if stratify_valid:
        train_data, test_data = train_test_split(
            model_data,
            test_size=0.20,
            random_state=42,
            stratify=model_data["Label"],
        )
        print("STRATIFIED 80/20 SPLIT: ENABLED")
    else:
        invalid_classes = [f"{label} ({count})" for label, count in class_counts.items() if count < 2]
        print("Warning: Stratification is not mathematically valid because some classes have fewer than 2 rows:")
        for invalid_class in invalid_classes:
            print(f"- {invalid_class}")
        train_data, test_data = train_test_split(
            model_data,
            test_size=0.20,
            random_state=42,
        )

    train_data = train_data[output_columns]
    test_data = test_data[output_columns]

    print("\n80/20 SPLIT SUMMARY")
    print(f"Total sampled model rows: {len(model_data)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Testing rows: {len(test_data)}")
    print(f"Training percentage: {len(train_data) / len(model_data) * 100:.2f}%")
    print(f"Testing percentage: {len(test_data) / len(model_data) * 100:.2f}%")

    if len(train_data) + len(test_data) != len(model_data):
        raise ValueError("80/20 row-count validation failed.")

    print("80/20 ROW COUNT VALIDATION: PASSED")

    web_attack_labels_set = set(web_attack_labels)
    if any(label in web_attack_labels_set for label in train_data["Label"].astype(str).str.strip().unique()):
        raise ValueError("Web Attack labels were found in the training split.")
    if any(label in web_attack_labels_set for label in test_data["Label"].astype(str).str.strip().unique()):
        raise ValueError("Web Attack labels were found in the testing split.")

    print("WEB ATTACK EXCLUSION FROM TRAIN FILE: PASSED")
    print("WEB ATTACK EXCLUSION FROM TEST FILE: PASSED")

    print("\nTRAINING FILE CLASS DISTRIBUTION")
    print(train_data["Label"].value_counts())
    print("\nTESTING FILE CLASS DISTRIBUTION")
    print(test_data["Label"].value_counts())

    train_csv_path = os.path.join("split_data", "train", "train_data.csv")
    test_csv_path = os.path.join("split_data", "test", "test_data.csv")

    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)

    print(f"\nTRAINING CSV SAVED: {os.path.abspath(train_csv_path)}")
    print(f"TESTING CSV SAVED: {os.path.abspath(test_csv_path)}")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV was not created: {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Testing CSV was not created: {test_csv_path}")
    if os.path.getsize(train_csv_path) <= 0:
        raise ValueError("Training CSV file is empty.")
    if os.path.getsize(test_csv_path) <= 0:
        raise ValueError("Testing CSV file is empty.")

    train_header = pd.read_csv(train_csv_path, nrows=0)
    test_header = pd.read_csv(test_csv_path, nrows=0)

    if list(train_header.columns) != output_columns:
        raise ValueError("Training CSV columns do not match the expected feature order.")
    if list(test_header.columns) != output_columns:
        raise ValueError("Testing CSV columns do not match the expected feature order.")
    if "Label" not in train_header.columns or "Label" not in test_header.columns:
        raise ValueError("Label column is missing from one or both saved CSV files.")

    print("TRAINING CSV VERIFICATION: PASSED")
    print("TESTING CSV VERIFICATION: PASSED")

    os.makedirs("models", exist_ok=True)
    feature_names_path = os.path.join("models", "feature_names.pkl")
    dropped_features_path = os.path.join("models", "dropped_features.pkl")
    joblib.dump(feature_names, feature_names_path)
    joblib.dump(dropped_features, dropped_features_path)

    print(f"Saved feature metadata: {os.path.abspath(feature_names_path)}")
    print(f"Saved dropped feature metadata: {os.path.abspath(dropped_features_path)}")

    if not os.path.exists(feature_names_path):
        raise FileNotFoundError("feature_names.pkl was not created.")
    if not os.path.exists(dropped_features_path):
        raise FileNotFoundError("dropped_features.pkl was not created.")

    del known_data, novel_attack_data, combined_df, model_data, train_data, test_data
    gc.collect()

    print("\nSplit data creation completed successfully.")


if __name__ == "__main__":
    preprocess_cic_ids_dataset()
