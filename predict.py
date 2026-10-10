import os

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/random_forest.pkl"
FEATURES_PATH = "models/feature_names.pkl"
SUSPICIOUS_THRESHOLD = 0.95

# Threshold selected experimentally from the saved evaluation results:
# approximately 92.25% of held-out Web Attack flows were below the threshold,
# while only about 0.94% of known-test flows were below the threshold.

ATTACK_INFORMATION = {
    "Bot": {
        "threat_type": "Botnet / Malware Activity",
        "severity": "High",
        "prevention": "Use endpoint protection, restrict suspicious outbound connections, monitor command-and-control traffic, and keep systems patched.",
        "recommendation": "Isolate the affected host, block malicious command-and-control traffic, and scan the endpoint for malware.",
    },
    "DDoS": {
        "threat_type": "Distributed Denial of Service",
        "severity": "Critical",
        "prevention": "Use rate limiting, traffic filtering, DDoS mitigation services, load balancing, and upstream network protection.",
        "recommendation": "Apply rate limiting, activate DDoS filtering or upstream mitigation, and block identified malicious sources.",
    },
    "DoS GoldenEye": {
        "threat_type": "HTTP Denial of Service",
        "severity": "High",
        "prevention": "Apply HTTP request rate limits, configure a web application firewall, monitor abnormal request patterns, and restrict repeated clients.",
        "recommendation": "Limit abnormal HTTP request rates, configure a web application firewall, and block repeated malicious clients.",
    },
    "DoS Hulk": {
        "threat_type": "HTTP Flood Denial of Service",
        "severity": "High",
        "prevention": "Use HTTP rate limiting, DDoS protection, reverse proxies, and request burst monitoring.",
        "recommendation": "Apply HTTP rate limiting, enable DDoS protection, and restrict excessive request bursts.",
    },
    "DoS Slowhttptest": {
        "threat_type": "Slow HTTP Denial of Service",
        "severity": "High",
        "prevention": "Configure strict HTTP connection timeouts, limit concurrent connections, use reverse proxies, and monitor slow client behavior.",
        "recommendation": "Configure HTTP connection timeouts, limit concurrent connections, and use reverse-proxy protection.",
    },
    "DoS slowloris": {
        "threat_type": "Slow Connection Denial of Service",
        "severity": "High",
        "prevention": "Reduce idle connection timeouts, limit connections per client, use reverse proxies, and monitor incomplete HTTP requests.",
        "recommendation": "Reduce idle connection timeouts, limit connections per client, and use a reverse proxy or web application firewall.",
    },
    "FTP-Patator": {
        "threat_type": "FTP Brute-Force Attack",
        "severity": "High",
        "prevention": "Enforce strong credentials, disable unnecessary FTP access, apply login rate limiting, and use account lockout policies.",
        "recommendation": "Block repeated failed FTP login sources, enforce strong credentials, and apply account lockout or rate limiting.",
    },
    "Heartbleed": {
        "threat_type": "OpenSSL Information Disclosure Vulnerability",
        "severity": "Critical",
        "prevention": "Patch vulnerable OpenSSL versions, maintain vulnerability management, replace affected certificates, and regularly scan exposed services.",
        "recommendation": "Patch vulnerable OpenSSL versions immediately, rotate affected certificates and credentials, and inspect systems for compromise.",
    },
    "Infiltration": {
        "threat_type": "Unauthorized Network Infiltration",
        "severity": "Critical",
        "prevention": "Apply network segmentation, strong access controls, endpoint monitoring, least privilege, and continuous log analysis.",
        "recommendation": "Isolate the affected system, inspect unauthorized access activity, review logs, and reset potentially compromised credentials.",
    },
    "PortScan": {
        "threat_type": "Network Reconnaissance",
        "severity": "Medium",
        "prevention": "Close unnecessary ports, restrict exposed services, configure firewall rules, and monitor repeated connection attempts.",
        "recommendation": "Block or rate-limit scanning sources, close unnecessary ports, and review firewall and intrusion detection rules.",
    },
    "SSH-Patator": {
        "threat_type": "SSH Brute-Force Attack",
        "severity": "High",
        "prevention": "Use SSH key authentication, disable password login where possible, apply login rate limiting, and restrict SSH access.",
        "recommendation": "Block repeated SSH brute-force sources, disable password authentication where possible, and enforce key-based access.",
    },
}

EXPECTED_ATTACK_CLASSES = [
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
]

DEFENSE_RECOMMENDATIONS = {
    attack_name: metadata["recommendation"]
    for attack_name, metadata in ATTACK_INFORMATION.items()
}

SUSPICIOUS_RECOMMENDATION = "Investigate the traffic manually, isolate the source if risk is suspected, review related logs, and apply temporary monitoring or blocking until the activity is verified."


def validate_attack_information(expected_attack_classes=None):
    expected_classes = list(expected_attack_classes or EXPECTED_ATTACK_CLASSES)
    metadata_keys = set(ATTACK_INFORMATION.keys())
    missing = [attack_name for attack_name in expected_classes if attack_name not in metadata_keys]
    return {
        "total": len(metadata_keys),
        "expected": len(expected_classes),
        "missing": missing,
        "passed": len(missing) == 0,
    }


def get_attack_metadata(attack_name):
    return ATTACK_INFORMATION.get(attack_name)


def load_model_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature metadata: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    if model.n_features_in_ != len(feature_names):
        raise ValueError("Loaded model feature count does not match the saved feature metadata.")
    if "Label" in feature_names:
        raise ValueError("Label should not be present in feature_names.")

    return model, feature_names


def validate_uploaded_dataframe(df, feature_names):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Uploaded content must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    df = df.copy()
    df.columns = df.columns.str.strip()

    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    selected_features = df[feature_names]

    if not all(pd.api.types.is_numeric_dtype(selected_features[col]) for col in selected_features.columns):
        non_numeric = [col for col in selected_features.columns if not pd.api.types.is_numeric_dtype(selected_features[col])]
        raise ValueError(f"Non-numeric required features found: {non_numeric}")

    selected_features = selected_features.replace([np.inf, -np.inf], np.nan)
    invalid_row_count = int(selected_features.isna().any(axis=1).sum())
    if invalid_row_count > 0:
        raise ValueError(f"Invalid rows detected after feature validation: {invalid_row_count}")

    if len(selected_features.columns) != len(feature_names):
        raise ValueError("Validated feature count does not match the expected feature count.")

    return selected_features


def get_defense_recommendation(category, raw_prediction):
    if category == "BENIGN":
        return "No defensive action required. Continue normal network monitoring."
    if category == "SUSPICIOUS":
        return SUSPICIOUS_RECOMMENDATION
    if category == "KNOWN ATTACK":
        if raw_prediction in DEFENSE_RECOMMENDATIONS:
            return DEFENSE_RECOMMENDATIONS[raw_prediction]
        return "Review the detected attack traffic, inspect related logs, isolate affected systems if necessary, and apply appropriate network controls."
    return "Review the detected traffic, inspect related logs, isolate affected systems if necessary, and apply appropriate network controls."


def predict_dataframe(df):
    model, feature_names = load_model_artifacts()
    validated_features = validate_uploaded_dataframe(df, feature_names)

    raw_predictions = model.predict(validated_features)
    probabilities = model.predict_proba(validated_features)
    max_confidence = probabilities.max(axis=1)

    if len(raw_predictions) != len(validated_features):
        raise ValueError("Prediction count does not match the number of input rows.")
    if len(max_confidence) != len(validated_features):
        raise ValueError("Confidence count does not match the number of input rows.")
    if probabilities.shape[1] != len(model.classes_):
        raise ValueError("Probability matrix shape does not match the trained model classes.")

    final_categories = []
    defense_recommendations = []
    for raw_prediction, confidence in zip(raw_predictions, max_confidence):
        if confidence < SUSPICIOUS_THRESHOLD:
            final_category = "SUSPICIOUS"
        elif raw_prediction == "BENIGN":
            final_category = "BENIGN"
        else:
            final_category = "KNOWN ATTACK"

        final_categories.append(final_category)
        defense_recommendations.append(get_defense_recommendation(final_category, raw_prediction))

    results_df = pd.DataFrame({
        "Raw_Prediction": raw_predictions,
        "Confidence": np.round(max_confidence, 4),
        "Confidence_Percentage": np.round(max_confidence * 100, 2),
        "Final_Category": final_categories,
        "Defense_Recommendation": defense_recommendations,
    })

    return results_df


def create_prediction_summary(results_df):
    total_flows = len(results_df)
    benign_count = int((results_df["Final_Category"] == "BENIGN").sum())
    known_attack_count = int((results_df["Final_Category"] == "KNOWN ATTACK").sum())
    suspicious_count = int((results_df["Final_Category"] == "SUSPICIOUS").sum())

    if benign_count + known_attack_count + suspicious_count != total_flows:
        raise ValueError("Prediction summary counts do not match the total number of flows.")

    summary = {
        "Total_Flows": total_flows,
        "Benign_Count": benign_count,
        "Known_Attack_Count": known_attack_count,
        "Suspicious_Count": suspicious_count,
        "Benign_Percentage": round((benign_count / total_flows) * 100 if total_flows else 0.0, 2),
        "Known_Attack_Percentage": round((known_attack_count / total_flows) * 100 if total_flows else 0.0, 2),
        "Suspicious_Percentage": round((suspicious_count / total_flows) * 100 if total_flows else 0.0, 2),
    }
    return summary


def create_known_attack_summary(results_df):
    if not {"Raw_Prediction", "Final_Category"}.issubset(results_df.columns):
        raise ValueError("results_df must contain Raw_Prediction and Final_Category columns.")

    known_attack_rows = results_df.loc[results_df["Final_Category"] == "KNOWN ATTACK"].copy()
    if known_attack_rows.empty:
        return pd.DataFrame(columns=[
            "Attack_Name",
            "Detected_Flows",
            "Percentage_of_Total_Flows",
            "Threat_Type",
            "Severity",
            "Prevention",
            "Defense_Recommendation",
        ])

    attack_counts = known_attack_rows["Raw_Prediction"].value_counts()
    total_flows = len(results_df)
    summary_rows = []

    for attack_name, detected_flows in attack_counts.items():
        metadata = get_attack_metadata(attack_name)
        summary_rows.append({
            "Attack_Name": attack_name,
            "Detected_Flows": int(detected_flows),
            "Percentage_of_Total_Flows": round((int(detected_flows) / total_flows) * 100 if total_flows else 0.0, 2),
            "Threat_Type": metadata["threat_type"] if metadata else "Not Available",
            "Severity": metadata["severity"] if metadata else "Not Available",
            "Prevention": metadata["prevention"] if metadata else "No metadata available.",
            "Defense_Recommendation": metadata["recommendation"] if metadata else "No metadata available.",
        })

    return pd.DataFrame(summary_rows, columns=[
        "Attack_Name",
        "Detected_Flows",
        "Percentage_of_Total_Flows",
        "Threat_Type",
        "Severity",
        "Prevention",
        "Defense_Recommendation",
    ]).sort_values("Detected_Flows", ascending=False, ignore_index=True)


def create_suspicious_pattern_summary(results_df):
    if not {"Raw_Prediction", "Final_Category"}.issubset(results_df.columns):
        raise ValueError("results_df must contain Raw_Prediction and Final_Category columns.")

    suspicious_rows = results_df.loc[results_df["Final_Category"] == "SUSPICIOUS"].copy()
    if suspicious_rows.empty:
        return pd.DataFrame(columns=[
            "Possible_Model_Pattern",
            "Suspicious_Flow_Count",
            "Percentage_of_Suspicious_Flows",
            "Percentage_of_Total_Flows",
        ])

    pattern_counts = suspicious_rows["Raw_Prediction"].value_counts()
    total_suspicious_flows = len(suspicious_rows)
    total_flows = len(results_df)
    summary_rows = []

    for pattern_name, suspicious_flow_count in pattern_counts.items():
        summary_rows.append({
            "Possible_Model_Pattern": pattern_name,
            "Suspicious_Flow_Count": int(suspicious_flow_count),
            "Percentage_of_Suspicious_Flows": round((int(suspicious_flow_count) / total_suspicious_flows) * 100 if total_suspicious_flows else 0.0, 2),
            "Percentage_of_Total_Flows": round((int(suspicious_flow_count) / total_flows) * 100 if total_flows else 0.0, 2),
        })

    return pd.DataFrame(summary_rows, columns=[
        "Possible_Model_Pattern",
        "Suspicious_Flow_Count",
        "Percentage_of_Suspicious_Flows",
        "Percentage_of_Total_Flows",
    ]).sort_values("Suspicious_Flow_Count", ascending=False, ignore_index=True)


def main():
    test_file_path = os.path.join("test_files", "novel_attack_upload.csv")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Missing test input file: {test_file_path}")

    print("PREDICTION PIPELINE TEST")
    print(f"Uploaded test file: {os.path.abspath(test_file_path)}")
    test_df = pd.read_csv(test_file_path)
    print(f"Input shape: {test_df.shape}")

    results_df = predict_dataframe(test_df)
    summary = create_prediction_summary(results_df)

    print("\nPREDICTION SUMMARY")
    print(f"Total flows: {summary['Total_Flows']}")
    print(f"Benign count: {summary['Benign_Count']}")
    print(f"Known Attack count: {summary['Known_Attack_Count']}")
    print(f"Suspicious count: {summary['Suspicious_Count']}")
    print(f"Benign percentage: {summary['Benign_Percentage']}%")
    print(f"Known Attack percentage: {summary['Known_Attack_Percentage']}%")
    print(f"Suspicious percentage: {summary['Suspicious_Percentage']}%")

    print("\nRAW PREDICTION DISTRIBUTION")
    print(results_df["Raw_Prediction"].value_counts())

    print("\nFINAL CATEGORY DISTRIBUTION")
    print(results_df["Final_Category"].value_counts())

    print("\nFIRST 10 PREDICTION RESULTS")
    print(results_df.head(10).to_string(index=False))

    if len(results_df) != len(test_df):
        raise ValueError("Result row count does not match the input row count.")

    valid_categories = {"BENIGN", "KNOWN ATTACK", "SUSPICIOUS"}
    if not set(results_df["Final_Category"].unique()).issubset(valid_categories):
        raise ValueError("One or more final categories are invalid.")

    attack_metadata_validation = validate_attack_information()
    print("\nATTACK METADATA VALIDATION")
    print(f"Total attack metadata entries: {attack_metadata_validation['total']}")
    print(f"Expected attack metadata entries: {attack_metadata_validation['expected']}")
    print(f"Missing attack metadata entries: {attack_metadata_validation['missing']}")
    print(f"Metadata validation status: {'PASSED' if attack_metadata_validation['passed'] else 'FAILED'}")

    print("\nPREDICTION PIPELINE VALIDATION: PASSED")


if __name__ == "__main__":
    main()
