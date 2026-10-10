import gc
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


MODEL_PATH = os.path.join("models", "xgboost.pkl")
FEATURES_PATH = os.path.join("models", "feature_names.pkl")
SELECTED_FEATURES_PATH = os.path.join("models", "selected_features.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
TRAIN_PATH = os.path.join("split_data", "train", "train_data.csv")
TEST_PATH = os.path.join("split_data", "test", "test_data.csv")
CONFIG_PATH = os.path.join("models", "xgboost_config.pkl")


def validate_project_files():
    required_files = [FEATURES_PATH, SELECTED_FEATURES_PATH, LABEL_ENCODER_PATH, TRAIN_PATH, TEST_PATH]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def load_feature_metadata():
    feature_names = joblib.load(FEATURES_PATH)
    selected_features = joblib.load(SELECTED_FEATURES_PATH)

    if list(feature_names) != list(selected_features):
        raise ValueError("Selected feature list does not match the saved feature metadata.")
    if "Label" in feature_names:
        raise ValueError("Label should not appear in the saved feature metadata.")
    return list(feature_names)


def ensure_label_encoder(train_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    class_names = list(label_encoder.classes_)
    expected_labels = sorted(set(train_labels.astype(str).str.strip().tolist()))
    actual_labels = sorted(set(class_names))
    if actual_labels != expected_labels:
        raise ValueError(
            "Label encoder classes do not match the training labels. "
            f"Expected: {expected_labels}. Actual: {actual_labels}."
        )

    return label_encoder


def main():
    print("PHASE 3 - XGBOOST TRAINING")
    print("-" * 30)

    validate_project_files()

    feature_names = load_feature_metadata()
    print("FEATURE METADATA LOADED")
    print(f"TOTAL MODEL FEATURES: {len(feature_names)}")
    print("Feature order:")
    for index, feature_name in enumerate(feature_names):
        print(f"{index}: {feature_name}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"\nLoading training data from: {TRAIN_PATH}")
    print(f"Training dataset shape: {train_df.shape}")
    print(f"Testing dataset shape: {test_df.shape}")

    if "Label" not in train_df.columns:
        raise ValueError("Label column is missing from the training CSV.")
    if "Label" not in test_df.columns:
        raise ValueError("Label column is missing from the testing CSV.")

    expected_train_shape = (338190, 71)
    expected_test_shape = (84548, 71)
    if train_df.shape != expected_train_shape:
        raise ValueError(
            f"Unexpected training split shape: {train_df.shape}; expected {expected_train_shape}."
        )
    if test_df.shape != expected_test_shape:
        raise ValueError(
            f"Unexpected test split shape: {test_df.shape}; expected {expected_test_shape}."
        )

    missing_features = [feature for feature in feature_names if feature not in train_df.columns]
    if missing_features:
        raise ValueError(f"Training data is missing required feature columns: {missing_features}")

    X_train = train_df[feature_names].copy()
    y_train_raw = train_df["Label"].astype(str).str.strip()

    if not all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns):
        raise ValueError("Training features contain non-numeric columns.")
    if X_train.isna().any().any():
        raise ValueError("Training features contain NaN values.")
    if np.isinf(X_train.to_numpy(dtype=float)).any():
        raise ValueError("Training features contain infinity values.")

    print("TRAINING FEATURE VALIDATION: PASSED")

    label_encoder = ensure_label_encoder(y_train_raw)
    y_train = label_encoder.transform(y_train_raw)

    classes = list(label_encoder.classes_)
    print("\nTRAINING CLASS DISTRIBUTION")
    print(train_df["Label"].value_counts())
    print(f"TOTAL TRAINING CLASSES: {len(classes)}")
    print("CLASS NAMES:")
    for class_name in classes:
        print(f"- {class_name}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        learning_rate=0.05,
        n_estimators=500,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
    )

    print("\nXGBOOST CONFIGURATION")
    print({
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "random_state": 42,
        "n_jobs": -1,
    })

    start_time = time.time()
    print("\nXGBOOST TRAINING STARTED")
    model.fit(X_train, y_train, verbose=False)
    training_duration_seconds = time.time() - start_time
    training_duration_minutes = training_duration_seconds / 60.0

    print("XGBOOST TRAINING COMPLETED")
    print(f"Training duration (seconds): {training_duration_seconds:.2f}")
    print(f"Training duration (minutes): {training_duration_minutes:.2f}")

    model_path = MODEL_PATH
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nXGBOOST MODEL SAVED: {os.path.abspath(model_path)}")

    config_payload = {
        "feature_names": feature_names,
        "class_names": classes,
        "num_features": len(feature_names),
        "num_classes": len(classes),
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "eval_metric": "mlogloss",
        "random_state": 42,
        "training_duration_seconds": training_duration_seconds,
    }
    joblib.dump(config_payload, CONFIG_PATH)
    print(f"XGBOOST CONFIG SAVED: {os.path.abspath(CONFIG_PATH)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file was not created: {model_path}")

    loaded_model = joblib.load(model_path)
    if loaded_model.n_features_in_ != len(feature_names):
        raise ValueError("Loaded model feature count does not match the expected feature count.")

    print("\nMODEL VERIFICATION: PASSED")
    print(f"MODEL EXPECTED FEATURE COUNT: {loaded_model.n_features_in_}")
    print(f"MODEL CLASSES: {list(loaded_model.classes_)}")
    print(f"TOTAL MODEL CLASSES: {len(loaded_model.classes_)}")
    print(f"TRAINING DATASET SHAPE: {train_df.shape}")
    print(f"TESTING DATASET SHAPE: {test_df.shape}")
    print(f"NUMBER OF FEATURES: {len(feature_names)}")
    print(f"NUMBER OF CLASSES: {len(classes)}")
    print(f"CLASS NAMES: {classes}")
    print(f"TRAINING TIME (SECONDS): {training_duration_seconds:.2f}")
    print(f"MODEL SAVE PATH: {os.path.abspath(model_path)}")

    del train_df, test_df, X_train, y_train_raw, y_train
    gc.collect()

    print("\nPhase 3 completed successfully.")


if __name__ == "__main__":
    main()
