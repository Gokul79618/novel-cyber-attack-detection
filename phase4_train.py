import gc
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    print("PHASE 4 - RANDOM FOREST TRAINING")
    print("-" * 32)

    required_files = [
        os.path.join("split_data", "train", "train_data.csv"),
        os.path.join("split_data", "test", "test_data.csv"),
        os.path.join("models", "feature_names.pkl"),
    ]

    os.makedirs("models", exist_ok=True)

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Missing required file -> {file_path}")
            return

    print("Required files found.")

    feature_names_path = os.path.join("models", "feature_names.pkl")
    feature_names = joblib.load(feature_names_path)
    print("FEATURE METADATA LOADED")
    print(f"TOTAL MODEL FEATURES: {len(feature_names)}")
    print("Features:")
    for index, feature_name in enumerate(feature_names):
        print(f"{index}: {feature_name}")

    train_csv_path = os.path.join("split_data", "train", "train_data.csv")
    print(f"\nLoading training data from: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    print("TRAINING DATA LOADED")
    print(f"Training dataset shape: {train_df.shape}")
    print(f"Training rows: {train_df.shape[0]}")
    print(f"Training columns: {train_df.shape[1]}")

    if "Label" not in train_df.columns:
        raise ValueError("Label column is missing from the training CSV.")

    missing_features = [feature for feature in feature_names if feature not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing training features: {missing_features}")

    if "Label" in feature_names:
        raise ValueError("Label should not appear in feature_names.")

    try:
        train_df[feature_names]
    except Exception as error:
        raise ValueError(f"Feature order validation failed: {error}")

    print("Training feature columns verified.")

    X_train = train_df[feature_names]
    y_train = train_df["Label"]

    print("\nX_TRAIN SHAPE")
    print(X_train.shape)
    print("Y_TRAIN SHAPE")
    print(y_train.shape)
    print("\nTRAINING CLASS DISTRIBUTION")
    print(y_train.value_counts())

    if not all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns):
        raise ValueError("Training features contain non-numeric columns.")

    if X_train.isna().any().any():
        raise ValueError("Training features contain NaN values.")

    if np.isinf(X_train.to_numpy(dtype=float)).any():
        raise ValueError("Training features contain infinity values.")

    print("TRAINING FEATURE VALIDATION: PASSED")

    # Random Forest is a tree-based model and does not require feature scaling because
    # its decision splits are not distance-based.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    print("\nRANDOM FOREST CONFIGURATION")
    print({
        "n_estimators": model.n_estimators,
        "random_state": model.random_state,
        "n_jobs": model.n_jobs,
        "class_weight": model.class_weight,
    })

    start_time = time.time()
    print("\nRANDOM FOREST TRAINING STARTED")
    model.fit(X_train, y_train)
    training_duration_seconds = time.time() - start_time
    training_duration_minutes = training_duration_seconds / 60.0

    print("RANDOM FOREST TRAINING COMPLETED")
    print(f"Training duration (seconds): {training_duration_seconds:.2f}")
    print(f"Training duration (minutes): {training_duration_minutes:.2f}")

    print("\nTRAINED MODEL CLASSES")
    print(list(model.classes_))
    print(f"TOTAL TRAINED CLASSES: {len(model.classes_)}")

    web_attack_found = [cls for cls in model.classes_ if isinstance(cls, str) and "web attack" in cls.lower()]
    if web_attack_found:
        raise ValueError(f"Web Attack classes were found in the trained model: {web_attack_found}")

    print("WEB ATTACK EXCLUSION FROM TRAINED MODEL: PASSED")

    model_path = os.path.join("models", "random_forest.pkl")
    joblib.dump(model, model_path)
    print(f"\nRANDOM FOREST MODEL SAVED: {os.path.abspath(model_path)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file was not created: {model_path}")

    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    loaded_model = joblib.load(model_path)
    if list(loaded_model.classes_) != list(model.classes_):
        raise ValueError("Loaded model classes do not match the trained model classes.")

    if loaded_model.n_features_in_ != len(feature_names):
        raise ValueError("Loaded model feature count does not match the expected feature count.")

    print("SAVED MODEL VERIFICATION: PASSED")
    print(f"MODEL EXPECTED FEATURE COUNT: {loaded_model.n_features_in_}")
    print(f"Model file size (MB): {model_size_mb:.2f}")

    del train_df, X_train, y_train
    gc.collect()

    print("\nPhase 4 completed successfully.")


if __name__ == "__main__":
    main()
