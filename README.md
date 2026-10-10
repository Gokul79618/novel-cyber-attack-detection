# Machine Learning-Based Novel Cyber Attack Detection and Prevention Recommendation System

## Overview

This project is a Streamlit application for offline analysis of network-flow CSV files. It uses saved Random Forest or XGBoost classifiers to predict traffic classes, displays prediction confidence, highlights low-confidence results as suspicious, and provides defense recommendations. It is an analysis and decision-support tool: it does not monitor live traffic or block attacks.

## Key features

- Upload a CSV containing network-flow features and validate it against the selected model's expected features.
- Choose Random Forest or XGBoost for batch predictions.
- Review predicted classes, confidence, suspicious classifications, and defense recommendations.
- Explore summary metrics and filtered prediction results in the dashboard.
- Download prediction results and generated security reports.
- Prepare CIC-IDS2017 training data and run model evaluation scripts locally.

## Technology stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn and XGBoost
- joblib
- Plotly and Matplotlib/Seaborn
- Pillow and ReportLab

## Project structure

```text
attack_prediction/
├── .streamlit/
│   └── config.toml
├── evaluation/
│   ├── compare_models.py
│   ├── evaluate_xgboost.py
│   └── evaluate_xgboost_novel.py
├── models/                         # Inference artifacts and feature metadata
├── training/
│   └── train_xgboost.py
├── app.py                          # Streamlit application entry point
├── phase4_train.py                 # Random Forest training from prepared splits
├── phase5_evaluate.py
├── phase5_prepare_novel.py
├── predict.py
├── predict_xgboost.py
├── prepare_20percent_test_files.py
├── prepare_attack_test_files.py
├── rf_streamlit.py
├── streamlit_app.py                # Compatibility wrapper for app.py
├── train_model.py                  # CIC-IDS2017 preparation and train/test split
├── xgboost_streamlit.py
├── requirements.txt
├── .gitignore
└── run_streamlit.cmd                # Windows launcher (port 8502)
```

The `.gitignore` excludes the local `dataset/`, `split_data/`, `test_files/`, `reports/`, and `evaluation_results/` directories, along with environments, caches, secrets, logs, and temporary files. These generated or local data files are not distributed with the source. The local report documents and the error screenshot are not part of the recommended public source tree.

## Windows setup

Use a supported Python installation and run the following commands in PowerShell from the project directory:

```powershell
Set-Location "C:\Users\kgoku\Desktop\attack_prediction"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual-environment interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Launch the dashboard

The application entry point is `app.py`. After installing the dependencies, start it with:

```powershell
python -m streamlit run app.py
```

Alternatively, run the included Windows launcher, which starts `app.py` on port 8502 using `.venv`:

```powershell
.\run_streamlit.cmd
```

## Using the models

1. Start the dashboard and upload a CSV of network-flow records.
2. Select **Random Forest** or **XGBoost** in the model selector.
3. Review the predictions, confidence, suspicious/known/benign categories, summaries, and recommendations.

The uploaded CSV must contain the numeric feature columns expected by the selected model. The saved feature metadata determines which columns are used and their order; extra columns, including a label column, are not used as model features. Random Forest and XGBoost use different confidence thresholds for categorizing low-confidence predictions as suspicious.

The files in `models/` are included only where needed for runtime inference. The dashboard's inference functions load the corresponding model and feature metadata from that directory. If those artifacts are omitted, inference will not work until compatible models and metadata have been generated.

## Preparing CIC-IDS2017 data

CIC-IDS2017 CSVs are not included. Obtain the dataset from its official source and observe its applicable terms. To prepare data locally:

1. Place the CICFlowMeter CSV files under `dataset/MachineLearningCVE/`. Keep the original CSVs out of Git; the directory is ignored.
2. From the project root, run:

   ```powershell
   python train_model.py
   ```

   This script cleans and combines the source CSVs, holds out labels containing “Web Attack,” and writes prepared training/test CSVs under `split_data/` plus feature metadata under `models/`.
3. Use `phase4_train.py` for the Random Forest training workflow and `training/train_xgboost.py` for XGBoost training when their required prepared files are available.

Generated splits and test files are local artifacts and are ignored by Git. Model training can require substantial memory, disk space, and processing time.

## Limitations

- This is an offline, batch CSV analysis application, not a live network detector or automated prevention system.
- The held-out Web Attack category and confidence threshold simulate potential novel-attack identification. They do **not** prove true zero-day detection or generalization to previously unseen real-world attacks.
- A confidence threshold is a heuristic: an unseen attack may still receive a high-confidence known-class prediction, and a low-confidence prediction is not necessarily malicious.
- Predictions depend on input feature names, numeric values, and compatibility with the feature metadata used to train the saved model.
- Model performance depends on the dataset and evaluation procedure; saved evaluation results should not be interpreted as a guarantee of production performance.
