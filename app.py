import os
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as ReportLabImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from predict import (
    SUSPICIOUS_RECOMMENDATION,
    create_known_attack_summary,
    create_prediction_summary,
    create_suspicious_pattern_summary,
    get_attack_metadata,
)
from rf_streamlit import (
    SUSPICIOUS_THRESHOLD as RANDOM_FOREST_THRESHOLD,
    load_model_artifacts as load_random_forest_artifacts,
    predict_dataframe as predict_random_forest,
    validate_uploaded_dataframe as validate_random_forest_dataframe,
)
from xgboost_streamlit import (
    SUSPICIOUS_THRESHOLD as XGBOOST_THRESHOLD,
    load_model_artifacts as load_xgboost_artifacts,
    predict_dataframe as predict_xgboost,
    validate_uploaded_dataframe as validate_xgboost_dataframe,
)


st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #FFFFFF;
        --surface: #FFFFFF;
        --text: #0F172A;
        --muted: #64748B;
        --border: #E2E8F0;
        --primary: #16A34A;
        --success: #16A34A;
        --warning: #F59E0B;
        --danger: #DC2626;
        --critical: #B91C1C;
        --high: #EA580C;
        --medium: #F59E0B;
        --low: #16A34A;
        --suspicious: #F59E0B;
        --shadow: 0 4px 14px rgba(17, 24, 39, 0.06);
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    .main .block-container {
        max-width: 1440px;
        padding: 1.5rem 2rem 3rem;
    }
    .hero-shell {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
    }
    .hero-banner {
        height: 4px;
        background: #16A34A;
    }
    .hero-panel {
        background: transparent;
        padding: 1.5rem 1.8rem 1.6rem;
    }
    .hero-eyebrow {
        color: #15803D !important;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .hero-title {
        font-size: clamp(1.75rem, 3.2vw, 2.5rem);
        font-weight: 800;
        color: #111827 !important;
        margin-bottom: 0.4rem;
        line-height: 1.12;
        letter-spacing: -0.025em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #64748B !important;
        margin-bottom: 0.8rem;
        max-width: 900px;
    }
    .hero-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.1rem;
        color: #64748B;
        font-size: 0.78rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .system-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.42rem 0.72rem;
        border: 1px solid rgba(22, 163, 74, 0.24);
        border-radius: 999px;
        color: #15803D;
        background: #F0FDF4;
        letter-spacing: 0.08em;
        white-space: nowrap;
    }
    .system-status::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #16A34A;
        box-shadow: 0 0 8px rgba(22, 163, 74, 0.28);
    }
    .hero-points {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.7rem;
    }
    .hero-point {
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        color: #15803D !important;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .summary-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: var(--shadow);
        height: 100%;
    }
    .summary-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.3rem;
    }
    .summary-value {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--text);
    }
    .summary-chip {
        display: inline-block;
        padding: 0.24rem 0.6rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        margin-top: 0.4rem;
        background: #DCFCE7;
        color: #15803D;
    }
    .workflow-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        box-shadow: var(--shadow);
    }
    .workflow-step {
        display: inline-block;
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        color: #15803D;
        padding: 0.36rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-weight: 600;
    }
    .workflow-arrow {
        color: #15803D;
        margin: 0 0.25rem;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.8rem;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 14px;
        padding: 0.8rem 1.15rem;
        color: #111827 !important;
        font-weight: 700;
        min-width: 200px;
        transition: border-color 160ms ease, background 160ms ease;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #F0FDF4;
        border-color: #16A34A;
        box-shadow: 0 0 0 1px rgba(22, 163, 74, 0.18), 0 4px 12px rgba(22, 163, 74, 0.10);
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        color: #15803D !important;
        font-weight: 700;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #16A34A;
    }
    div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {
        background-color: #16A34A !important;
    }
    .ranking-table th, .ranking-table td {
        padding: 0.55rem 0.6rem;
    }
    .ranking-table th {
        color: #475569;
        font-weight: 700;
    }
    .ranking-table td {
        color: #0f172a;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"] {
        background: #16A34A;
        color: #ffffff;
        border: 1px solid #16A34A;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.16);
    }
    div.stButton > button[kind="primary"]:hover {
        background: #15803D;
        color: #ffffff;
        border-color: #15803D;
    }
    .dashboard-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.2rem 1.25rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text);
        margin-bottom: 0.7rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem;
        box-shadow: var(--shadow);
        min-height: 130px;
    }
    .metric-icon {
        font-size: 1.3rem;
        margin-bottom: 0.35rem;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.2rem;
    }
    .metric-subtitle {
        font-size: 0.9rem;
        color: var(--muted);
    }
    .status-card {
        border-radius: 20px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        background: var(--surface);
        box-shadow: var(--shadow);
        min-height: 240px;
    }
    .status-card.low { border-left: 6px solid var(--success); }
    .status-card.medium { border-left: 6px solid var(--warning); }
    .status-card.high { border-left: 6px solid var(--high); }
    .status-card.critical { border-left: 6px solid var(--critical); }
    .status-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }
    .status-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.4rem;
    }
    .status-detail {
        font-size: 0.92rem;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }
    .detail-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.8rem;
    }
    .detail-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.25rem;
    }
    .detail-subtitle {
        font-size: 0.92rem;
        color: var(--muted);
        margin-bottom: 0.6rem;
    }
    .detail-grid {
        display: grid;
        gap: 0.5rem;
        margin-top: 0.45rem;
    }
    .detail-grid .row {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #eef2f7;
        padding-bottom: 0.35rem;
    }
    .detail-grid .label { color: var(--muted); }
    .detail-grid .value { font-weight: 700; color: var(--text); }
    .detected-attacks-section {
        margin-bottom: 0;
    }
    .detected-attacks-section .stMarkdown {
        margin-bottom: 0 !important;
    }
    .detected-attacks-card {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 12px;
        padding: 0.7rem 0.8rem;
        overflow: hidden;
        margin-top: 0.2rem;
        margin-bottom: 0;
    }
    .detected-attacks-card table,
    .detected-attacks-card thead,
    .detected-attacks-card tbody,
    .detected-attacks-card tr,
    .detected-attacks-card th,
    .detected-attacks-card td,
    .detected-attacks-card p,
    .detected-attacks-card span,
    .detected-attacks-card div {
        color: #111827 !important;
    }
    .detected-attacks-card table {
        width: 100%;
        border-collapse: collapse;
    }
    .detected-attacks-card th {
        color: #64748B !important;
        font-weight: 600;
        text-align: left;
        padding: 0.7rem 0.75rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .detected-attacks-card td {
        color: #111827 !important;
        font-weight: 600;
        padding: 0.7rem 0.75rem;
        border-bottom: 1px solid #F3F4F6;
    }
    .detected-attacks-card tr:last-child td {
        border-bottom: none;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.24rem 0.6rem;
        border-radius: 999px;
        font-size: 0.77rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .severity-critical { background: #FEE2E2; color: #B91C1C; }
    .severity-high { background: #FFEDD5; color: #C2410C; }
    .severity-medium { background: #FEF3C7; color: #B45309; }
    .severity-low { background: #DCFCE7; color: #15803D; }
    .detected-attacks-card .severity-badge.severity-critical { background: #FEE2E2; color: #B91C1C !important; }
    .detected-attacks-card .severity-badge.severity-high { background: #FFEDD5; color: #C2410C !important; }
    .detected-attacks-card .severity-badge.severity-medium { background: #FEF3C7; color: #B45309 !important; }
    .detected-attacks-card .severity-badge.severity-low { background: #DCFCE7; color: #15803D !important; }
    .notice-box {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        color: #92400E;
        margin-bottom: 1rem;
    }
    .thin-divider { border-top: 1px solid #E5E7EB; margin: 1.25rem 0; }

    h1, h2, h3, h4, p, label, small, [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }
    .section-title {
        color: #111827;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }
    .summary-label, .metric-label, .status-label { color: #64748B !important; }
    .summary-value, .metric-value, .status-value { color: #111827 !important; }
    .summary-card strong { color: #111827; }
    .metric-icon { display: none; }
    .metric-card, .status-card, .detail-card, .workflow-card {
        background: #FFFFFF;
        border-color: #E5E7EB;
        box-shadow: 0 4px 14px rgba(17, 24, 39, 0.06);
    }
    .status-detail, .detail-subtitle {
        color: #64748B !important;
    }
    .detail-title { color: #111827; }
    .detail-grid .row { border-bottom-color: #E5E7EB; }
    .detail-grid .label { color: #64748B; }
    .detail-grid .value { color: #111827; }
    .workflow-arrow { color: #15803D; }
    .stTextInput input, .stSelectbox [data-baseweb="select"] > div,
    .stSlider [data-baseweb="slider"] {
        background-color: #FFFFFF;
        color: #111827;
        border-color: #D1D5DB;
    }
    .stTextInput input::placeholder { color: #64748B; }
    [data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 1px dashed #D1D5DB;
        border-radius: 14px;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #16A34A; }
    [data-testid="stFileUploader"] button,
    [data-testid="stDownloadButton"] button,
    div.stButton > button:not([kind="primary"]) {
        background: #FFFFFF;
        border: 1px solid #16A34A;
        color: #15803D;
        border-radius: 10px;
    }
    [data-testid="stDownloadButton"] button:hover,
    div.stButton > button:not([kind="primary"]):hover {
        color: #15803D;
        border-color: #15803D;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #D1D5DB;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 12px;
    }
    [data-testid="stExpander"] summary { color: #111827; }
    [data-testid="stAlert"] { background: #F0FDF4; color: #111827; }
    [data-testid="stStatusWidget"] { background: #FFFFFF; }
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.96);
        border-bottom: 1px solid #E5E7EB;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        background: #FFFFFF;
        color: #111827;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {
        color: #111827;
    }
    [data-testid="stFileUploader"] section small {
        color: #64748B;
        opacity: 1;
    }
    .upload-note {
        border: 1px solid #D1D5DB;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        background: #FFFFFF;
        color: #64748B;
    }
    .selected-model-card {
        border: 1px solid #16A34A;
        border-radius: 12px;
        background: #F0FDF4;
        padding: 0.7rem 0.9rem;
        margin-top: 0.65rem;
    }
    .selected-model-label {
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .selected-model-value {
        color: #15803D;
        font-size: 1.05rem;
        font-weight: 800;
        margin-top: 0.15rem;
    }
    .model-description {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: -0.2rem;
        padding-left: 0.25rem;
    }
    .model-used-card {
        border-left: 3px solid #16A34A;
    }
    .model-info-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
    }
    .model-info-item {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 11px;
        padding: 0.8rem 0.9rem;
    }
    .model-info-label {
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .model-info-value {
        color: #111827;
        font-weight: 700;
        margin-top: 0.28rem;
    }
    .risk-explanation-card { border-left: 5px solid #16A34A; }
    .risk-explanation-card.risk-medium { border-left-color: #F59E0B; }
    .risk-explanation-card.risk-high { border-left-color: #EA580C; }
    .risk-explanation-card.risk-critical { border-left-color: #B91C1C; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #F7F9F8 !important;
        color: #111827 !important;
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1, h2, h3, h4, p, label, small,
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
        color: #111827;
    }
    .summary-card, .metric-card, .status-card, .detail-card, .workflow-card,
    .dashboard-card, .model-info-item, .risk-explanation-card {
        background: #FFFFFF;
        border-color: #E5E7EB;
        box-shadow: 0 3px 12px rgba(17, 24, 39, 0.055);
    }
    .section-title { color: #111827 !important; }
    .hero-shell {
        background: #FFFFFF;
        border-color: #E5E7EB;
        box-shadow: 0 5px 16px rgba(17, 24, 39, 0.07);
    }
    .hero-banner { background: #16A34A; }
    .hero-eyebrow, .selected-model-value { color: #15803D !important; }
    .hero-title, .summary-value, .metric-value, .status-value,
    .detail-title, .detail-grid .value, .model-info-value {
        color: #111827 !important;
    }
    .hero-subtitle, .hero-topline, .summary-label, .metric-label, .status-label,
    .status-detail, .detail-subtitle, .detail-grid .label, .model-description,
    .model-info-label, .metric-subtitle {
        color: #64748B !important;
    }
    .system-status {
        color: #15803D;
        background: #F0FDF4;
        border-color: #BBF7D0;
    }
    .system-status::before { background: #16A34A; box-shadow: 0 0 7px rgba(22, 163, 74, 0.3); }
    .summary-chip {
        background: #DCFCE7;
        color: #15803D;
    }
    .workflow-step {
        background: #F0FDF4;
        border-color: #DCFCE7;
        color: #15803D;
    }
    .workflow-arrow { color: #16A34A; }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        color: #111827 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p { color: #111827 !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #F0FDF4;
        border-color: #16A34A;
        box-shadow: 0 0 0 1px rgba(22, 163, 74, 0.16), 0 4px 12px rgba(22, 163, 74, 0.09);
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-selected="true"] p { color: #15803D !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover { border-color: #16A34A; }
    div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {
        background-color: #16A34A !important;
    }
    .selected-model-card {
        background: #F0FDF4;
        border-color: #86EFAC;
    }
    .selected-model-label { color: #64748B; }
    .model-used-card { border-left: 4px solid #16A34A; }
    .metric-card { border-color: #E5E7EB; }
    .status-card.low { border-left-color: #16A34A; }
    .status-card.medium { border-left-color: #F59E0B; }
    .status-card.high { border-left-color: #EA580C; }
    .status-card.critical { border-left-color: #B91C1C; }
    .detail-grid .row { border-bottom-color: #E5E7EB; }
    .detected-attacks-card {
        background: #FFFFFF;
        border-color: #E5E7EB;
    }
    .detected-attacks-card table,
    .detected-attacks-card thead,
    .detected-attacks-card tbody,
    .detected-attacks-card tr,
    .detected-attacks-card th,
    .detected-attacks-card td,
    .detected-attacks-card p,
    .detected-attacks-card span,
    .detected-attacks-card div { color: #111827 !important; }
    .detected-attacks-card th {
        color: #374151 !important;
        background: #F3F4F6;
        border-bottom-color: #E5E7EB;
    }
    .detected-attacks-card td { border-bottom-color: #F3F4F6; }
    .severity-critical { background: #FEE2E2; color: #B91C1C; }
    .severity-high { background: #FFEDD5; color: #C2410C; }
    .severity-medium { background: #FEF3C7; color: #B45309; }
    .severity-low { background: #DCFCE7; color: #15803D; }
    .detected-attacks-card .severity-badge.severity-critical { background: #FEE2E2; color: #B91C1C !important; }
    .detected-attacks-card .severity-badge.severity-high { background: #FFEDD5; color: #C2410C !important; }
    .detected-attacks-card .severity-badge.severity-medium { background: #FEF3C7; color: #B45309 !important; }
    .detected-attacks-card .severity-badge.severity-low { background: #DCFCE7; color: #15803D !important; }
    .notice-box {
        background: #FFFBEB;
        border-color: #FDE68A;
        color: #92400E;
    }
    .stTextInput input, .stSelectbox [data-baseweb="select"] > div {
        background: #FFFFFF;
        color: #111827;
        border-color: #D1D5DB;
    }
    .stTextInput input:focus, .stSelectbox [data-baseweb="select"] > div:focus-within {
        border-color: #16A34A;
        box-shadow: 0 0 0 1px rgba(22, 163, 74, 0.16);
    }
    .stTextInput input::placeholder { color: #64748B; }
    [data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border-color: #D1D5DB;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #16A34A; }
    [data-testid="stFileUploader"] button,
    [data-testid="stDownloadButton"] button,
    div.stButton > button:not([kind="primary"]) {
        background: #FFFFFF;
        border: 1px solid #16A34A;
        color: #15803D;
    }
    [data-testid="stDownloadButton"] button:hover,
    div.stButton > button:not([kind="primary"]):hover {
        background: #F0FDF4;
        border-color: #15803D;
        color: #15803D;
    }
    div.stButton > button[kind="primary"] {
        background: #16A34A;
        border-color: #16A34A;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.16);
    }
    div.stButton > button[kind="primary"]:hover {
        background: #15803D;
        border-color: #15803D;
        color: #FFFFFF;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-color: #E5E7EB;
        border-radius: 12px;
    }
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border-color: #E5E7EB;
    }
    [data-testid="stExpander"] summary { color: #111827; }
    [data-testid="stAlert"] { background: #F0FDF4; color: #111827; }
    [data-testid="stStatusWidget"] { background: #FFFFFF; }
    .thin-divider { border-top-color: #E5E7EB; }
    @media (max-width: 760px) {
        .main .block-container { padding: 1rem 0.8rem 2rem; }
        .hero-panel { padding: 1.2rem; }
        .hero-topline { align-items: flex-start; flex-direction: column; }
        .model-info-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        div[data-testid="stRadio"] div[role="radiogroup"] { flex-wrap: wrap; }
        div[data-testid="stRadio"] div[role="radiogroup"] label { min-width: 0; }
    }
    @media (max-width: 1100px) {
        .metric-label { font-size: 0.68rem; letter-spacing: 0.05em; }
        .metric-subtitle { font-size: 0.8rem; }
    }
    :root {
        --bg: #FFFFFF;
        --text: #0F172A;
        --muted: #64748B;
        --border: #E2E8F0;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .main .block-container {
        max-width: 1380px;
        padding: 1.1rem 1.6rem 2.4rem;
    }
    .hero-shell {
        border-color: #E2E8F0;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.045);
        margin-bottom: 1rem;
    }
    .hero-banner {
        height: 4px;
        background: linear-gradient(90deg, #15803D 0%, #16A34A 68%, #BBF7D0 100%);
    }
    .hero-panel { padding: 1.2rem 1.55rem 1.35rem; }
    .hero-topline { margin-bottom: 0.8rem; font-size: 0.72rem; }
    .hero-eyebrow { color: #15803D !important; font-size: 0.7rem; margin-bottom: 0.35rem; }
    .hero-title {
        color: #0F172A !important;
        font-size: clamp(1.65rem, 2.8vw, 2.25rem);
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle { font-size: 0.94rem; margin-bottom: 0; }
    .section-title {
        color: #0F172A !important;
        font-size: 0.88rem;
        letter-spacing: 0.09em;
        margin: 0.1rem 0 0.55rem;
    }
    .thin-divider { border-color: #E2E8F0; margin: 0.9rem 0; }
    .summary-card, .metric-card, .status-card, .detail-card, .workflow-card,
    .dashboard-card, .model-info-item, .risk-explanation-card {
        border-color: #E2E8F0;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.045);
    }
    .summary-card { padding: 0.85rem 0.9rem; }
    .summary-label, .metric-label, .status-label, .model-info-label { color: #64748B !important; }
    .summary-value, .metric-value, .status-value, .detail-title,
    .detail-grid .value, .model-info-value { color: #0F172A !important; }
    .metric-card {
        min-height: 112px;
        padding: 0.8rem 0.85rem 0.72rem;
        border-left: 3px solid #16A34A;
    }
    .metric-card.metric-accent-warning { border-left-color: #D97706; }
    .metric-card.metric-accent-danger,
    .metric-card.metric-accent-critical { border-left-color: #DC2626; }
    .metric-card .metric-label {
        min-height: 2.2em;
        font-size: 0.71rem;
        line-height: 1.35;
        letter-spacing: 0.055em;
    }
    .metric-card .metric-value {
        font-size: clamp(1.1rem, 1.55vw, 1.45rem);
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .metric-card .metric-subtitle { font-size: 0.78rem; line-height: 1.35; }
    .risk-explanation-card {
        border-left-width: 4px;
        background: #F5FBF7;
        margin-top: 0.75rem;
        padding: 0.9rem 1rem;
    }
    .status-card { min-height: 0; padding: 1rem 1.05rem; }
    .status-value { font-size: 1.12rem; }
    .detail-card { padding: 0.9rem 1rem; }
    .detail-card .section-title { font-size: 0.75rem; }
    .workflow-card.workflow-timeline {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.55rem;
        padding: 0.8rem;
        background: #F5FBF7;
        border-color: #E2E8F0;
    }
    .workflow-timeline .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-height: 56px;
        margin: 0;
        padding: 0.5rem 0.6rem;
        border-radius: 10px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        color: #334155;
        font-size: 0.73rem;
        line-height: 1.3;
    }
    .workflow-timeline .workflow-step::before {
        content: counter(workflow-step, decimal-leading-zero);
        display: inline-grid;
        flex: 0 0 1.65rem;
        width: 1.65rem;
        height: 1.65rem;
        place-items: center;
        border-radius: 50%;
        background: #DCFCE7;
        color: #15803D;
        font-size: 0.65rem;
        font-weight: 800;
    }
    .workflow-timeline { counter-reset: workflow-step; }
    .workflow-timeline .workflow-step { counter-increment: workflow-step; }
    .workflow-timeline .workflow-step:nth-child(4) {
        background: #F0FDF4;
        border-color: #86EFAC;
        color: #15803D;
        font-weight: 700;
    }
    .workflow-arrow { display: none; }
    .system-status { background: #F0FDF4; border-color: #BBF7D0; color: #15803D; }
    .selected-model-card {
        background: #F0FDF4;
        border-color: #86EFAC;
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
        margin-top: 0.6rem;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        align-items: stretch;
        gap: 0.75rem;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        flex: 1 1 0;
        min-width: 0;
        min-height: 78px;
        padding: 0.75rem 0.95rem;
        border-color: #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.035);
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        margin: 0;
        color: #0F172A !important;
        font-size: 0.92rem;
        font-weight: 750;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p::after {
        display: block;
        margin-top: 0.28rem;
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 400;
        line-height: 1.35;
        white-space: normal;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(1) p::after {
        content: "Fast & reliable ensemble classifier";
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(2) p::after {
        content: "Gradient boosting classifier";
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #F0FDF4;
        border-color: #16A34A;
        box-shadow: 0 0 0 1px rgba(22, 163, 74, 0.12);
    }
    .model-description { display: none; }
    [data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 1px dashed #86EFAC;
        border-radius: 14px;
        min-height: 116px;
        transition: border-color 140ms ease, background 140ms ease;
    }
    [data-testid="stFileUploader"] section:hover,
    [data-testid="stFileUploader"] section:focus-within {
        border-color: #16A34A;
        background: #F5FBF7;
    }
    .upload-note {
        background: #F5FBF7;
        border-color: #DCFCE7;
        padding: 0.72rem 0.9rem;
        border-radius: 11px;
    }
    div.stButton > button[kind="primary"] {
        min-height: 2.9rem;
        border-radius: 11px;
        font-size: 0.82rem;
        letter-spacing: 0.055em;
        background: #16A34A;
        border-color: #16A34A;
        box-shadow: 0 3px 9px rgba(22, 163, 74, 0.15);
    }
    div.stButton > button[kind="primary"]:hover {
        background: #15803D;
        border-color: #15803D;
        transform: translateY(-1px);
    }
    [data-testid="stDownloadButton"] button,
    div.stButton > button:not([kind="primary"]) {
        min-height: 2.75rem;
        border-radius: 10px;
        border-color: #16A34A;
        color: #15803D;
        font-weight: 700;
    }
    [data-testid="stDownloadButton"] button { text-transform: uppercase; }
    [data-testid="stPlotlyChart"] {
        padding: 0.25rem;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background: #FFFFFF;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background: #FFFFFF;
    }
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 11px;
    }
    [data-testid="stExpander"] summary { color: #0F172A; }
    .detected-attacks-card { border-color: #E2E8F0; border-radius: 12px; padding: 0.45rem 0.55rem; }
    .detected-attacks-card th { background: #F5FBF7; font-size: 0.77rem; padding: 0.6rem 0.65rem; }
    .detected-attacks-card td { font-size: 0.82rem; padding: 0.58rem 0.65rem; }
    .detected-attacks-card tr:hover td { background: #F8FAFC; }
    .severity-badge { white-space: nowrap; }
    .model-info-grid { gap: 0.55rem; }
    .model-info-item { padding: 0.7rem 0.78rem; }
    .notice-box {
        border-radius: 11px;
        border-color: #FDE68A;
        padding: 0.8rem 0.9rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.96);
        border-bottom: 1px solid #E2E8F0;
    }
    [data-testid="stCaptionContainer"] { color: #64748B !important; }
    .dashboard-footer {
        background: #F5FBF7;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    @media (max-width: 900px) {
        .main .block-container { padding: 0.9rem 1rem 1.8rem; }
        .workflow-card.workflow-timeline { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
        .hero-panel { padding: 1rem; }
        .hero-topline { align-items: center; flex-direction: row; }
        .workflow-card.workflow-timeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        div[data-testid="stRadio"] div[role="radiogroup"] { flex-direction: column; }
        div[data-testid="stRadio"] div[role="radiogroup"] label { min-height: 0; }
        .model-info-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model_artifacts():
    return load_random_forest_artifacts()


def predict_selected_model(df, selected_model):
    if selected_model == "Random Forest":
        return predict_random_forest(df)
    if selected_model == "XGBoost":
        return predict_xgboost(df)
    raise ValueError(f"Unsupported model selection: {selected_model}")


@st.cache_data(show_spinner=False)
def create_performance_metrics_image(metrics, test_samples, feature_count):
    width, height = 1800, 1380
    image = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(image)

    font_candidates = [
        ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    font_regular_path = next((regular for regular, _ in font_candidates if os.path.exists(regular)), None)
    font_bold_path = next((bold for _, bold in font_candidates if os.path.exists(bold)), None)

    def get_font(size, bold=False):
        font_path = font_bold_path if bold else font_regular_path
        try:
            return ImageFont.truetype(font_path, size=size) if font_path else ImageFont.load_default(size=size)
        except (OSError, TypeError):
            return ImageFont.load_default()

    title_font = get_font(56, bold=True)
    subtitle_font = get_font(27)
    card_label_font = get_font(22, bold=True)
    card_value_font = get_font(43, bold=True)
    chart_title_font = get_font(31, bold=True)
    row_label_font = get_font(24, bold=True)
    value_font = get_font(23, bold=True)
    small_font = get_font(19)

    navy = "#0F172A"
    muted = "#64748B"
    blue = "#2563EB"
    border = "#DCE4EF"

    draw.text((100, 52), "MODEL PERFORMANCE METRICS", fill=navy, font=title_font)
    draw.text(
        (102, 132),
        "Random Forest Evaluation on Held-Out CIC-IDS2017 Test Dataset",
        fill=muted,
        font=subtitle_font,
    )

    kpis = [
        ("ACCURACY", metrics.get("Accuracy"), False),
        ("PRECISION (MACRO AVERAGE)", metrics.get("Precision (Macro Average)"), False),
        ("RECALL (MACRO AVERAGE)", metrics.get("Recall (Macro Average)"), False),
        ("F1-SCORE (MACRO AVERAGE)", metrics.get("F1-Score (Macro Average)"), False),
    ]
    card_y, card_height, card_gap = 210, 140, 20
    card_width = (width - 200 - card_gap * 3) // 4
    for index, (label, value, _) in enumerate(kpis):
        left = 100 + index * (card_width + card_gap)
        draw.rounded_rectangle(
            (left, card_y, left + card_width, card_y + card_height),
            radius=12,
            fill="#FFFFFF",
            outline=border,
            width=2,
        )
        draw.text((left + 24, card_y + 24), label, fill=muted, font=card_label_font)
        display_value = "N/A" if value is None else f"{value * 100:.2f}%"
        draw.text((left + 24, card_y + 67), display_value, fill=blue, font=card_value_font)

    draw.text((100, 395), "Held-Out Test Metrics", fill=navy, font=chart_title_font)
    rows = [
        ("Accuracy", "Accuracy", "percentage"),
        ("Precision (Macro Average)", "Precision (Macro Average)", "percentage"),
        ("Recall (Macro Average)", "Recall (Macro Average)", "percentage"),
        ("F1-Score (Macro Average)", "F1-Score (Macro Average)", "percentage"),
        ("Specificity (Macro Average)", "Specificity (Macro Average)", "percentage"),
        ("Balanced Accuracy", "Balanced Accuracy", "percentage"),
        ("ROC-AUC (Macro Average)", "ROC-AUC (OvR, Macro Average)", "percentage"),
        ("MCC", "MCC", "mcc"),
    ]
    bar_left, bar_width = 680, 780
    first_row_y, row_step = 475, 78
    for index, (label, metric_key, metric_type) in enumerate(rows):
        y = first_row_y + index * row_step
        value = metrics.get(metric_key)
        draw.text((100, y + 3), label, fill=navy, font=row_label_font)
        draw.rounded_rectangle(
            (bar_left, y, bar_left + bar_width, y + 36),
            radius=8,
            fill="#E7EDF5",
        )
        if value is None:
            display_value = "N/A"
        elif metric_type == "mcc":
            zero_x = bar_left + bar_width // 2
            draw.line((zero_x, y - 5, zero_x, y + 41), fill="#94A3B8", width=2)
            bounded_mcc = max(-1.0, min(1.0, float(value)))
            mcc_width = int(abs(bounded_mcc) * (bar_width // 2))
            if mcc_width:
                if bounded_mcc >= 0:
                    draw.rounded_rectangle((zero_x, y, zero_x + mcc_width, y + 36), radius=8, fill=blue)
                else:
                    draw.rounded_rectangle((zero_x - mcc_width, y, zero_x, y + 36), radius=8, fill=blue)
            display_value = f"{value:.4f} (range -1 to 1)"
        else:
            percentage = max(0.0, min(1.0, float(value)))
            fill_width = int(percentage * bar_width)
            if fill_width:
                draw.rounded_rectangle((bar_left, y, bar_left + fill_width, y + 36), radius=8, fill=blue)
            display_value = f"{value * 100:.2f}%"
        draw.text((1490, y + 4), display_value, fill=navy, font=value_font)

    draw.text((bar_left, 1110), "-1", fill=muted, font=small_font)
    draw.text((bar_left + bar_width // 2 - 15, 1110), "0", fill=muted, font=small_font)
    draw.text((bar_left + bar_width - 35, 1110), "1", fill=muted, font=small_font)
    draw.rounded_rectangle((100, 1200, width - 100, 1310), radius=10, fill="#FFFFFF", outline=border, width=2)
    draw.text(
        (125, 1235),
        f"Evaluation Dataset: CIC-IDS2017 Held-Out Test Set  |  Samples: {test_samples:,}  |  Features: {feature_count}  |  Model: Random Forest",
        fill=navy,
        font=small_font,
    )

    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG", optimize=True)
    return image_buffer.getvalue()


@st.cache_data(show_spinner="Evaluating held-out test data...")
def evaluate_held_out_test(test_csv_path, file_modified_time):
    model, feature_names = get_cached_model_artifacts()
    test_df = pd.read_csv(test_csv_path)
    test_df.columns = test_df.columns.str.strip()

    if "Label" not in test_df.columns:
        raise ValueError("The held-out test CSV must contain a Label column.")

    X_test = validate_uploaded_dataframe(test_df.drop(columns=["Label"]), feature_names)
    y_test = test_df["Label"].astype(str).str.strip()
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test contain different numbers of rows.")
    if X_test.shape[1] != model.n_features_in_ or X_test.shape[1] != 70:
        raise ValueError("The test data does not contain the model's required 70 ordered features.")
    if y_test.isna().any():
        raise ValueError("The held-out test labels contain missing values.")

    classes = list(model.classes_)
    unsupported_labels = sorted(set(y_test.unique()) - set(classes))
    if unsupported_labels:
        raise ValueError(f"Test labels are not supported by the saved model: {unsupported_labels}")

    predictions = model.predict(X_test)
    if len(predictions) != len(y_test):
        raise ValueError("Prediction count does not match the held-out test rows.")

    matrix = confusion_matrix(y_test, predictions, labels=classes)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else float("nan")
    macro_precision = precision_score(y_test, predictions, labels=classes, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, predictions, labels=classes, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, predictions, labels=classes, average="macro", zero_division=0)
    weighted_precision = precision_score(y_test, predictions, labels=classes, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_test, predictions, labels=classes, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, labels=classes, average="weighted", zero_division=0)

    class_stats = []
    for index, class_name in enumerate(classes):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else np.nan
        class_stats.append({
            "Class": str(class_name),
            "True Positives": true_positive,
            "True Negatives": int(true_negative),
            "False Positives": false_positive,
            "False Negatives": false_negative,
            "Specificity": float(specificity) if not np.isnan(specificity) else np.nan,
        })
    macro_specificity = float(np.nanmean([item["Specificity"] for item in class_stats]))

    probabilities = None
    roc_auc = None
    roc_auc_note = "N/A"
    if hasattr(model, "predict_proba"):
        try:
            candidate_probabilities = model.predict_proba(X_test)
            if candidate_probabilities.shape == (len(y_test), len(classes)):
                probabilities = candidate_probabilities
            if probabilities is not None and y_test.nunique() == len(classes):
                roc_auc = float(roc_auc_score(
                    y_test,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                ))
                roc_auc_note = ""
        except (AttributeError, NotImplementedError, ValueError):
            probabilities = None

    report_values = classification_report(
        y_test,
        predictions,
        labels=classes,
        target_names=[str(class_name) for class_name in classes],
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for name, values in report_values.items():
        if isinstance(values, dict):
            report_rows.append({
                "Class": name,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-Score": values["f1-score"],
                "Support": int(values["support"]),
            })
    class_report = pd.DataFrame(report_rows)

    confidence = probabilities.max(axis=1) if probabilities is not None else None
    threshold = float(RANDOM_FOREST_THRESHOLD)
    metrics = {
        "Accuracy": accuracy,
        "Precision (Macro Average)": float(macro_precision),
        "Recall (Macro Average)": float(macro_recall),
        "F1-Score (Macro Average)": float(macro_f1),
        "Specificity (Macro Average)": macro_specificity,
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "ROC-AUC (OvR, Macro Average)": roc_auc,
        "MCC": float(matthews_corrcoef(y_test, predictions)),
        "Precision (Weighted)": float(weighted_precision),
        "Recall (Weighted)": float(weighted_recall),
        "F1-Score (Weighted)": float(weighted_f1),
    }
    class_distribution = y_test.value_counts().reindex(classes, fill_value=0).rename_axis("Class").reset_index(name="Count")
    class_distribution["Percentage"] = class_distribution["Count"] / total * 100

    training_path = os.path.join("split_data", "train", "train_data.csv")
    validation_path = os.path.join("split_data", "validation", "validation_data.csv")
    return {
        "metrics": metrics,
        "roc_auc_note": roc_auc_note,
        "labels": [str(class_name) for class_name in classes],
        "confusion_matrix": matrix,
        "class_stats": pd.DataFrame(class_stats),
        "classification_report": class_report,
        "class_distribution": class_distribution,
        "dataset_name": os.path.basename(test_csv_path),
        "test_samples": len(y_test),
        "feature_count": X_test.shape[1],
        "class_count": len(classes),
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(pd.read_csv(training_path)) if os.path.exists(training_path) else "Not available",
        "validation_samples": len(pd.read_csv(validation_path)) if os.path.exists(validation_path) else None,
        "confidence": {
            "high_count": int((confidence >= threshold).sum()) if confidence is not None else None,
            "low_count": int((confidence < threshold).sum()) if confidence is not None else None,
            "average": float(confidence.mean() * 100) if confidence is not None else None,
            "minimum": float(confidence.min() * 100) if confidence is not None else None,
            "maximum": float(confidence.max() * 100) if confidence is not None else None,
            "suspicious_percentage": float((confidence < threshold).mean() * 100) if confidence is not None else None,
            "threshold": threshold,
        },
    }


def get_model_info():
    selected_model = st.session_state.get("selected_model", "Random Forest")
    if "model_info" not in st.session_state or st.session_state.model_info.get("model") != selected_model:
        try:
            if selected_model == "XGBoost":
                model, feature_names, label_encoder = load_xgboost_artifacts()
                model_name = "xgboost.pkl"
                model_file = os.path.join("models", "xgboost.pkl")
                model_label = "XGBoost"
                class_count = len(label_encoder.classes_)
                threshold = XGBOOST_THRESHOLD
            else:
                model, feature_names = load_random_forest_artifacts()
                model_name = "random_forest.pkl"
                model_file = os.path.join("models", "random_forest.pkl")
                model_label = "Random Forest"
                class_count = len(model.classes_)
                threshold = RANDOM_FOREST_THRESHOLD

            train_path = os.path.join("split_data", "train", "train_data.csv")
            test_path = os.path.join("split_data", "test", "test_data.csv")
            training_samples = len(pd.read_csv(train_path)) if os.path.exists(train_path) else 0
            testing_samples = len(pd.read_csv(test_path)) if os.path.exists(test_path) else 0

            st.session_state.model_info = {
                "model": model_label,
                "model_name": model_name,
                "algorithm": model_label,
                "dataset": "CIC-IDS2017",
                "training_samples": int(training_samples),
                "testing_samples": int(testing_samples),
                "number_of_features": int(model.n_features_in_),
                "number_of_classes": int(class_count),
                "confidence_threshold": float(threshold),
                "feature_names": list(feature_names),
                "model_file": model_file,
                "accuracy": None,
            }
        except Exception as exc:
            st.session_state.model_info = {"error": str(exc)}
    return st.session_state.model_info


def format_metric(value, decimals=2):
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}" if decimals is not None else f"{value:,.0f}"
    return str(value)


def get_attack_description(attack_name):
    descriptions = {
        "Bot": "A bot attack uses compromised hosts to carry out coordinated malicious activity such as credential abuse or malware control.",
        "DDoS": "A DDoS attack overwhelms a service with excessive traffic so legitimate access becomes unavailable.",
        "DoS GoldenEye": "A DoS GoldenEye attack floods a service with repeated HTTP requests that exhaust server resources.",
        "DoS Hulk": "A DoS Hulk attack sends a high volume of HTTP requests to overwhelm web applications.",
        "DoS Slowhttptest": "A Slowhttptest attack holds connections open to exhaust server capacity and delay legitimate traffic.",
        "DoS slowloris": "A slowloris attack keeps many partial connections open to consume server resources over time.",
        "FTP-Patator": "An FTP-Patator attack repeatedly attempts password guesses against FTP services.",
        "Heartbleed": "Heartbleed is a vulnerability that exposes sensitive memory contents through vulnerable OpenSSL implementations.",
        "Infiltration": "An infiltration attack attempts unauthorized access into internal networks or systems.",
        "PortScan": "A port scan probes services to discover open ports and exploitable entry points.",
        "SSH-Patator": "An SSH-Patator attack repeatedly attempts password guesses against SSH services.",
    }
    return descriptions.get(attack_name, "This activity is analyzed and classified by the trained machine learning model.")


def calculate_security_score(summary, known_attack_summary):
    known_attack_pct = float(summary.get("Known_Attack_Percentage", 0.0))
    suspicious_pct = float(summary.get("Suspicious_Percentage", 0.0))
    total_flows = max(int(summary.get("Total_Flows", 0)), 1)
    known_attack_count = int(summary.get("Known_Attack_Count", 0))
    suspicious_count = int(summary.get("Suspicious_Count", 0))
    attack_type_count = len(known_attack_summary) if known_attack_summary is not None else 0

    severity_penalty = 0.0
    if known_attack_summary is not None and not known_attack_summary.empty:
        severity_rank = {"CRITICAL": 18.0, "HIGH": 10.0, "MEDIUM": 5.0, "LOW": 2.0}
        severities = [str(row.get("Severity", "LOW")).upper() for _, row in known_attack_summary.iterrows()]
        severity_penalty = max([severity_rank.get(severity, 2.0) for severity in severities], default=0.0)

    attack_density_penalty = (known_attack_pct * 0.45) + (suspicious_pct * 0.25)
    spread_penalty = min(attack_type_count * 1.5, 12.0)
    volume_penalty = min(((known_attack_count + suspicious_count) / total_flows) * 20.0, 20.0)
    score = 100 - attack_density_penalty - severity_penalty - spread_penalty - volume_penalty
    return round(max(0.0, min(100.0, score)), 1)


def determine_risk_level(score):
    if score >= 80:
        return "low"
    if score >= 60:
        return "medium"
    if score >= 30:
        return "high"
    return "critical"


def calculate_risk_level(summary, known_attack_summary):
    score = calculate_security_score(summary, known_attack_summary)
    level = determine_risk_level(score)
    return level, score


def generate_risk_explanation(summary, known_attack_summary, risk_level):
    known_attack_pct = float(summary.get("Known_Attack_Percentage", 0.0))
    suspicious_pct = float(summary.get("Suspicious_Percentage", 0.0))
    known_attack_count = int(summary.get("Known_Attack_Count", 0))
    suspicious_count = int(summary.get("Suspicious_Count", 0))
    benign_pct = float(summary.get("Benign_Percentage", 0.0))
    benign_count = int(summary.get("Benign_Count", 0))
    dominant_attack = known_attack_summary.iloc[0]["Attack_Name"] if known_attack_summary is not None and not known_attack_summary.empty else "None"

    if benign_count > known_attack_count + suspicious_count:
        return (
            f"{benign_pct:.2f}% of analyzed traffic was classified as benign. "
            f"{known_attack_pct:.2f}% was classified as known malicious traffic and "
            f"{suspicious_pct:.2f}% as potential novel / suspicious traffic requiring investigation. "
            f"This traffic mix results in a {risk_level.upper()} overall network risk."
        )

    if known_attack_count > 0 and dominant_attack != "None":
        return (
            f"{known_attack_pct:.2f}% of analyzed traffic was classified as known malicious traffic, primarily {dominant_attack}. "
            f"{suspicious_pct:.2f}% was classified as potential novel / suspicious traffic requiring investigation. "
            f"This traffic mix results in a {risk_level.upper()} overall network risk."
        )

    if suspicious_count > 0:
        return (
            f"{suspicious_pct:.2f}% of analyzed traffic was classified as potential novel / suspicious because model confidence fell below the threshold. "
            f"{benign_pct:.2f}% was classified as benign. This traffic mix results in a "
            f"{risk_level.upper()} overall network risk."
        )

    return (
        f"{benign_pct:.2f}% of analyzed traffic was classified as benign, resulting in a "
        f"{risk_level.upper()} overall network risk."
    )


def calculate_confidence_statistics(results_df):
    confidence_values = results_df["Confidence_Percentage"].astype(float)
    return {
        "average": round(float(confidence_values.mean()), 2),
        "median": round(float(confidence_values.median()), 2),
        "minimum": round(float(confidence_values.min()), 2),
        "maximum": round(float(confidence_values.max()), 2),
        "high_confidence_count": int((confidence_values >= 95.0).sum()),
        "low_confidence_count": int((confidence_values < 95.0).sum()),
    }


def get_display_severity(row):
    final_category = str(row["Final_Category"]).upper()
    if final_category == "BENIGN":
        return "Safe/Benign"
    if final_category == "SUSPICIOUS":
        return "Medium"
    if final_category == "KNOWN ATTACK":
        metadata = get_attack_metadata(row["Raw_Prediction"])
        severity = str(metadata.get("severity", "Medium") if metadata else "Medium").upper()
        severity_map = {
            "CRITICAL": "Critical",
            "HIGH": "High",
            "MEDIUM": "Medium",
            "LOW": "Low",
        }
        return severity_map.get(severity, "Medium")
    return "Medium"


def build_flow_display_dataframe(uploaded_df, results_df):
    display_df = results_df.copy().reset_index(drop=True)

    if uploaded_df is not None and not uploaded_df.empty:
        flow_ids = [int(index) for index in uploaded_df.index.tolist()]
    else:
        flow_ids = list(range(len(display_df)))

    display_df.insert(0, "Flow_ID", flow_ids)
    display_df["Severity"] = display_df.apply(get_display_severity, axis=1)
    display_df["Confidence_Value"] = display_df["Confidence_Percentage"].astype(float)

    display_df = display_df[[
        "Flow_ID",
        "Raw_Prediction",
        "Confidence_Value",
        "Final_Category",
        "Severity",
        "Defense_Recommendation",
    ]]
    display_df.columns = [
        "Flow ID",
        "Prediction",
        "Confidence (%)",
        "Final Category",
        "Severity",
        "Recommended Action",
    ]
    display_df["Final Category"] = display_df["Final Category"].astype(str).str.upper()
    display_df["Severity"] = display_df["Severity"].astype(str)
    display_df["Recommended Action"] = display_df["Recommended Action"].astype(str)
    return display_df


def generate_recommended_action(known_attack_summary):
    if known_attack_summary.empty:
        return "Continue routine monitoring and review any low-confidence traffic that requires investigation."
    dominant_attack = known_attack_summary.iloc[0]["Attack_Name"]
    return f"Prioritize monitoring and defensive controls for {dominant_attack}, then review related logs and isolate any confirmed malicious activity."


def render_metric_card(title, value, subtitle, icon, accent="primary"):
    accent_map = {
        "primary": "#16A34A",
        "success": "#16A34A",
        "warning": "#F59E0B",
        "danger": "#DC2626",
        "critical": "#B91C1C",
        "purple": "#15803D",
    }
    color = accent_map.get(accent, accent_map["primary"])
    st.markdown(
        f"""
        <div class="metric-card metric-accent-{accent}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle" style="color:{color};">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(status_label, detail_lines, risk_level):
    status_class = "low"
    if risk_level == "medium":
        status_class = "medium"
    elif risk_level == "high":
        status_class = "high"
    elif risk_level == "critical":
        status_class = "critical"

    details_html = "".join(f"<div class='status-detail'>{line}</div>" for line in detail_lines)
    st.markdown(
        f"""
        <div class='status-card {status_class}'>
            <div class='status-label'>OVERALL STATUS</div>
            <div class='status-value'>{status_label}</div>
            {details_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_attack_defense_card(attack_row):
    attack_name = attack_row["Attack_Name"]
    metadata = get_attack_metadata(attack_name)
    severity = metadata["severity"] if metadata else "N/A"
    severity_key = severity.upper()
    severity_class = severity_key.lower() if severity_key in {"CRITICAL", "HIGH", "MEDIUM", "LOW"} else "low"
    traffic_share = f"{attack_row['Percentage_of_Total_Flows']}%"
    detected_flows = int(attack_row["Detected_Flows"])

    st.markdown(
        f"""
        <div class='detail-card'>
            <div class='detail-title'>{attack_name}</div>
            <div class='detail-subtitle'>{metadata['threat_type'] if metadata else 'Threat metadata unavailable'}</div>
            <div style='margin-bottom: 0.5rem;'><span class='severity-badge severity-{severity_class}'>{severity}</span></div>
            <div class='detail-grid'>
                <div class='row'><div class='label'>DETECTED FLOWS</div><div class='value'>{detected_flows}</div></div>
                <div class='row'><div class='label'>TRAFFIC SHARE</div><div class='value'>{traffic_share}</div></div>
            </div>
            <div class='section-title' style='margin-top: 1rem;'>PREVENTIVE MEASURES</div>
            <div class='status-detail'>{metadata['prevention'] if metadata else 'No prevention metadata available.'}</div>
            <div class='section-title' style='margin-top: 0.8rem;'>RECOMMENDED DEFENSE</div>
            <div class='status-detail'>{metadata['recommendation'] if metadata else 'No recommendation metadata available.'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_risk_level(summary, known_attack_summary):
    if not known_attack_summary.empty and any(str(row["Severity"]).upper() == "CRITICAL" for _, row in known_attack_summary.iterrows()):
        return "critical", 4
    if int(summary["Known_Attack_Count"]) > 0:
        return "high", 3
    if int(summary["Suspicious_Count"]) > 0:
        return "medium", 2
    return "low", 1


def create_pdf_report(summary, known_attack_summary, suspicious_pattern_summary, results_df, uploaded_file_name, model_info, evaluation=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="Security Report", author="NIDS Dashboard")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("SECURITY REPORT", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Paragraph(
        f"The uploaded file {uploaded_file_name} produced {summary['Total_Flows']} traffic flows. "
        f"{summary['Known_Attack_Count']} flows were classified as known attacks, {summary['Suspicious_Count']} as suspicious, "
        f"and {summary['Benign_Count']} as benign.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Traffic Statistics", styles["Heading2"]))
    traffic_data = [
        ["Category", "Count", "Percentage"],
        ["Benign", summary["Benign_Count"], f"{summary['Benign_Percentage']}%"],
        ["Known Attack", summary["Known_Attack_Count"], f"{summary['Known_Attack_Percentage']}%"],
        ["Suspicious", summary["Suspicious_Count"], f"{summary['Suspicious_Percentage']}%"],
    ]
    table = Table(traffic_data, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ])
    )
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detected Attacks", styles["Heading2"]))
    if known_attack_summary.empty:
        story.append(Paragraph("No high-confidence known attacks were detected.", styles["BodyText"]))
    else:
        attack_rows = []
        for _, row in known_attack_summary.iterrows():
            attack_rows.append([row["Attack_Name"], int(row["Detected_Flows"]), f"{row['Percentage_of_Total_Flows']}%", row["Severity"]])
        attack_table = Table([ ["Attack", "Detected Flows", "Traffic Share", "Severity"], *attack_rows ], repeatRows=1)
        attack_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ])
        )
        story.append(attack_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Suspicious Traffic", styles["Heading2"]))
    if suspicious_pattern_summary.empty:
        story.append(Paragraph("No suspicious patterns were detected.", styles["BodyText"]))
    else:
        story.append(Paragraph(SUSPICIOUS_RECOMMENDATION, styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recommendations", styles["Heading2"]))
    if not known_attack_summary.empty:
        for _, row in known_attack_summary.iterrows():
            story.append(Paragraph(f"- {row['Attack_Name']}: {row['Defense_Recommendation']}", styles["BodyText"]))
    else:
        story.append(Paragraph("- Continue normal monitoring and review suspicious flows manually.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Information", styles["Heading2"]))
    story.append(Paragraph(f"Model: {model_info.get('model', 'Random Forest')}", styles["BodyText"]))
    story.append(Paragraph(f"Training Samples: {model_info.get('training_samples', 'N/A')}", styles["BodyText"]))
    story.append(Paragraph(f"Testing Samples: {model_info.get('testing_samples', 'N/A')}", styles["BodyText"]))
    story.append(Paragraph(f"Number of Features: {model_info.get('number_of_features', 'N/A')}", styles["BodyText"]))
    story.append(Paragraph(f"Number of Classes: {model_info.get('number_of_classes', 'N/A')}", styles["BodyText"]))
    story.append(Paragraph(f"Confidence Threshold: {model_info.get('confidence_threshold', '0.50')}", styles["BodyText"]))

    if evaluation is not None:
        story.append(Spacer(1, 14))
        story.append(Paragraph("MODEL PERFORMANCE METRICS", styles["Heading2"]))
        story.append(Paragraph("Random Forest Evaluation on Held-Out CIC-IDS2017 Test Dataset", styles["BodyText"]))
        story.append(Paragraph(
            f"Evaluation type: Held-out Test Evaluation. Dataset: {evaluation['dataset_name']}. "
            f"Test samples: {evaluation['test_samples']}. Features: {evaluation['feature_count']}. "
            f"Classes: {evaluation['class_count']}. Evaluated: {evaluation['evaluation_timestamp']}. "
            "Training samples were not used to calculate these metrics.",
            styles["BodyText"],
        ))

        evaluation_rows = [["Metric", "Value"]]
        for metric_name, value in evaluation["metrics"].items():
            if value is None:
                display_value = evaluation["roc_auc_note"]
            elif metric_name == "MCC":
                display_value = f"{value:.4f}"
            else:
                display_value = f"{value * 100:.2f}%"
            evaluation_rows.append([metric_name, display_value])
        evaluation_table = Table(evaluation_rows, colWidths=[180, 100], repeatRows=1)
        evaluation_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(Spacer(1, 8))
        story.append(evaluation_table)
        story.append(Spacer(1, 10))
        if evaluation.get("performance_image"):
            image_reader = ReportLabImage(
                BytesIO(evaluation["performance_image"]),
                width=doc.width,
                height=doc.width * 1380 / 1800,
            )
            story.append(image_reader)
            story.append(Spacer(1, 10))

        story.append(Paragraph("Confusion Matrix - Held-Out Test Dataset", styles["Heading3"]))
        labels = evaluation["labels"]
        matrix_rows = [["Actual / Predicted", *labels]]
        matrix_rows.extend([[label, *[str(int(value)) for value in row]] for label, row in zip(labels, evaluation["confusion_matrix"])])
        matrix_widths = [74] + [(doc.width - 74) / max(len(labels), 1)] * len(labels)
        matrix_table = Table(matrix_rows, colWidths=matrix_widths, repeatRows=1)
        matrix_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(matrix_table)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Classification Report", styles["Heading3"]))
        report = evaluation["classification_report"]
        report_rows = [list(report.columns)]
        report_rows.extend(report.astype(str).values.tolist())
        report_table = Table(report_rows, repeatRows=1)
        report_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(report_table)
        story.append(Spacer(1, 8))
        story.append(Paragraph("Test Class Distribution", styles["Heading3"]))
        distribution = evaluation["class_distribution"]
        distribution_rows = [["Class", "Count", "Percentage"]]
        distribution_rows.extend([
            [str(row["Class"]), str(int(row["Count"])), f"{row['Percentage']:.2f}%"]
            for _, row in distribution.iterrows()
        ])
        distribution_table = Table(distribution_rows, repeatRows=1)
        distribution_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(distribution_table)

    doc.build(story)
    return buffer.getvalue()


st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-banner"></div>
        <div class="hero-panel">
            <div class="hero-topline">
                <span>SOC THREAT ANALYTICS</span>
                <span class="system-status">System Online</span>
            </div>
            <div class="hero-eyebrow">NETWORK INTRUSION DETECTION</div>
            <div class="hero-title">NETWORK INTRUSION DETECTION</div>
            <div class="hero-subtitle">Machine Learning Based Network Security &amp; Defense Recommendation System</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_file_name = None
selected_model = st.session_state.get("selected_model", "Random Forest")

with st.container():
    st.markdown('<div class="section-title">UPLOAD NETWORK TRAFFIC</div>', unsafe_allow_html=True)
    st.caption("Upload a CIC-IDS2017-compatible CSV dataset.")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="uploaded_csv",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class='upload-note'>
                Select a CIC-IDS2017-compatible CSV file to prepare a network traffic analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        uploaded_file_name = uploaded_file.name
        file_size_bytes = uploaded_file.size if uploaded_file.size is not None else 0
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes else 0.0

        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"CSV reading failed: {error}")
            st.stop()

        if uploaded_df.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        file_info_cols = st.columns(5)
        with file_info_cols[0]:
            render_metric_card("Uploaded File", uploaded_file_name, "Source dataset", "🗂️", "primary")
        with file_info_cols[1]:
            render_metric_card("Total Flows", f"{len(uploaded_df):,}", "Rows in upload", "📊", "primary")
        with file_info_cols[2]:
            render_metric_card("Number of Features", f"{len(uploaded_df.columns):,}", "Columns in upload", "🧩", "primary")
        with file_info_cols[3]:
            render_metric_card("File Size", f"{file_size_mb} MB", "Upload size", "📦", "primary")
        with file_info_cols[4]:
            render_metric_card("Status", "READY FOR ANALYSIS", "Ready to process", "🛡️", "success")

        st.markdown('<div class="section-title">SELECT MACHINE LEARNING MODEL</div>', unsafe_allow_html=True)
        model_options = ["Random Forest", "XGBoost"]
        selected_model = st.radio(
            "Select the detection model",
            model_options,
            index=model_options.index(selected_model) if selected_model in model_options else 0,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.selected_model = selected_model
        st.markdown(
            f"<div class='selected-model-card'><div class='selected-model-label'>Selected Model</div>"
            f"<div class='selected-model-value'>{selected_model.upper()}</div></div>",
            unsafe_allow_html=True,
        )

        if selected_model != st.session_state.get("last_selected_model", selected_model):
            st.session_state.analysis_ready = False
            st.session_state.results_df = None
            st.session_state.summary = None
            st.session_state.last_selected_model = selected_model

        if st.button("Analyze Network Traffic", width="stretch", type="primary"):
            with st.spinner("Validating dataset, preprocessing traffic, running the selected model, analyzing confidence, and preparing recommendations..."):
                start_time = time.perf_counter()
                try:
                    results_df = predict_selected_model(uploaded_df, selected_model)
                    summary = create_prediction_summary(results_df)
                except ValueError as error:
                    st.error(f"CSV validation failed: {error}")
                    st.stop()
                except Exception as error:
                    st.error(f"Analysis failed: {error}")
                    st.stop()
                finally:
                    elapsed_time = time.perf_counter() - start_time
                    st.session_state.analysis_processing_time = elapsed_time

            st.session_state.results_df = results_df
            st.session_state.summary = summary
            st.session_state.last_file_name = uploaded_file_name
            st.session_state.last_selected_model = selected_model
            st.session_state.analysis_ready = True

        with st.expander("Uploaded Data Preview", expanded=False):
            st.dataframe(uploaded_df.head(10), width="stretch", height=250)

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


if st.session_state.analysis_ready and st.session_state.last_file_name == uploaded_file_name and st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    summary = st.session_state.summary
    known_attack_summary = create_known_attack_summary(results_df)
    suspicious_pattern_summary = create_suspicious_pattern_summary(results_df)
    risk_level, security_score = calculate_risk_level(summary, known_attack_summary)
    confidence_stats = calculate_confidence_statistics(results_df)
    selected_model = st.session_state.get("selected_model", "Random Forest")
    risk_explanation = generate_risk_explanation(summary, known_attack_summary, risk_level)

    st.markdown(
        f"<div class='summary-card model-used-card'><div class='summary-label'>MODEL USED</div>"
        f"<div class='summary-value'>{selected_model.upper()}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">SECURITY RISK OVERVIEW</div>', unsafe_allow_html=True)
    overview_cols = st.columns(5)
    risk_accent = {
        "low": "success",
        "medium": "warning",
        "high": "danger",
        "critical": "critical",
    }.get(risk_level, "primary")
    overview_items = [
        ("TOTAL FLOWS", f"{summary['Total_Flows']:,}", "Analyzed network traffic", "primary"),
        ("BENIGN", f"{summary['Benign_Count']:,}", f"{summary['Benign_Percentage']}% of traffic", "success"),
        ("KNOWN ATTACKS", f"{summary['Known_Attack_Count']:,}", f"{summary['Known_Attack_Percentage']}% of traffic", "danger"),
        ("POTENTIAL NOVEL / SUSPICIOUS", f"{summary['Suspicious_Count']:,}", f"{summary['Suspicious_Percentage']}% of traffic", "warning"),
        ("OVERALL RISK", risk_level.upper(), "Calculated from analysis results", risk_accent),
    ]
    for column, (label, value, subtitle, accent) in zip(overview_cols, overview_items):
        with column:
            render_metric_card(label, value, subtitle, "", accent)

    risk_detail_cols = st.columns(4)
    risk_detail_items = [
        ("THREAT TRAFFIC", f"{summary['Known_Attack_Percentage']}%", "Classified known attacks"),
        ("SUSPICIOUS TRAFFIC", f"{summary['Suspicious_Percentage']}%", "Low-confidence flows"),
        ("MODEL CONFIDENCE", f"{confidence_stats['average']:.2f}%", "Average across analyzed flows"),
        ("DETECTED ATTACK TYPES", f"{len(known_attack_summary)}", "Distinct known attack classes"),
    ]
    for column, (label, value, subtitle) in zip(risk_detail_cols, risk_detail_items):
        with column:
            st.markdown(
                f"<div class='summary-card'><div class='summary-label'>{label}</div>"
                f"<div class='summary-value'>{value}</div><div class='metric-subtitle'>{subtitle}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div class='summary-card risk-explanation-card risk-{risk_level}'><div class='summary-label'>RISK ASSESSMENT</div>"
        f"<div class='summary-value'>{risk_level.upper()} NETWORK RISK</div>"
        f"<div class='status-detail'><strong>Reason:</strong> {risk_explanation}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    score_cols = st.columns([1.2, 1.0])
    with score_cols[0]:
        st.markdown('<div class="section-title">SECURITY SCORE</div>', unsafe_allow_html=True)
    with score_cols[0]:
        st.markdown(f"<div class='summary-card'><div class='summary-label'>NETWORK SAFETY INDICATOR</div><div class='summary-value'>{security_score:.1f} / 100</div><div class='summary-chip'>{risk_level.upper()}</div><div style='font-size:0.8rem; color:#64748b; margin-top:0.4rem;'>Higher score = safer network; lower score = higher risk.</div></div>", unsafe_allow_html=True)
    with score_cols[1]:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=security_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Security Score", 'font': {'color': '#111827'}},
            number={'font': {'size': 28, 'color': '#111827'}, 'valueformat': '.1f'},
            delta={'reference': 50, 'increasing': {'color': '#16A34A'}, 'decreasing': {'color': '#DC2626'}},
            gauge={
                'axis': {'range': [0, 100], 'tickmode': 'array', 'tickvals': [0, 25, 50, 75, 100], 'ticktext': ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'SAFE'], 'tickfont': {'color': '#64748B'}},
                'bar': {'color': '#16A34A'},
                'steps': [
                    {'range': [0, 29], 'color': '#FEE2E2'},
                    {'range': [30, 59], 'color': '#FFEDD5'},
                    {'range': [60, 79], 'color': '#FEF3C7'},
                    {'range': [80, 100], 'color': '#DCFCE7'},
                ],
                'threshold': {'line': {'color': '#111827', 'width': 2}, 'thickness': 0.75, 'value': 50},
            },
        ))
        gauge_fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#111827",
        )
        st.plotly_chart(gauge_fig, width="stretch")

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">MODEL CONFIDENCE</div>', unsafe_allow_html=True)
    conf_cols = st.columns([1, 1, 1, 1, 1, 1])
    with conf_cols[0]:
        st.markdown(f"<div class='summary-card'><div class='summary-label'>AVERAGE</div><div class='summary-value'>{confidence_stats['average']:.2f}%</div></div>", unsafe_allow_html=True)
    with conf_cols[1]:
        st.markdown(f"<div class='summary-card'><div class='summary-label'>MEDIAN</div><div class='summary-value'>{confidence_stats['median']:.2f}%</div></div>", unsafe_allow_html=True)
    with conf_cols[2]:
        st.markdown(f"<div class='summary-card'><div class='summary-label'>MINIMUM</div><div class='summary-value'>{confidence_stats['minimum']:.2f}%</div></div>", unsafe_allow_html=True)
    with conf_cols[3]:
        st.markdown(f"<div class='summary-card'><div class='summary-label'>MAXIMUM</div><div class='summary-value'>{confidence_stats['maximum']:.2f}%</div></div>", unsafe_allow_html=True)
    with conf_cols[4]:
        st.markdown(f"<div class='summary-card'><div class='summary-label'>HIGH CONFIDENCE FLOWS</div><div class='summary-value'>{confidence_stats['high_confidence_count']:,}</div></div>", unsafe_allow_html=True)
    with conf_cols[5]:
        st.markdown(f"<div class='summary-card'><div class='summary-label'>LOW CONFIDENCE FLOWS</div><div class='summary-value'>{confidence_stats['low_confidence_count']:,}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">PROJECT WORKFLOW</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='workflow-card workflow-timeline'>
            <div class='workflow-step'>NETWORK TRAFFIC CSV</div>
            <div class='workflow-step'>FEATURE VALIDATION</div>
            <div class='workflow-step'>PREPROCESSING</div>
            <div class='workflow-step'>SELECTED ML MODEL</div>
            <div class='workflow-step'>CLASS PREDICTION</div>
            <div class='workflow-step'>CONFIDENCE ANALYSIS</div>
            <div class='workflow-step'>BENIGN / KNOWN ATTACK / SUSPICIOUS</div>
            <div class='workflow-step'>RISK &amp; SEVERITY ANALYSIS</div>
            <div class='workflow-step'>DEFENSE RECOMMENDATION</div>
            <div class='workflow-step'>SECURITY REPORT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    visual_cols = st.columns([1.1, 1.1, 1.1])
    with visual_cols[0]:
        st.markdown('<div class="section-title">TRAFFIC DISTRIBUTION</div>', unsafe_allow_html=True)
        pie_df = pd.DataFrame(
            {
                "Category": ["Benign", "Known Attack", "Potential Novel / Suspicious"],
                "Count": [summary["Benign_Count"], summary["Known_Attack_Count"], summary["Suspicious_Count"]],
                "Percentage": [summary["Benign_Percentage"], summary["Known_Attack_Percentage"], summary["Suspicious_Percentage"]],
            }
        )
        pie_fig = px.pie(
            pie_df,
            values="Count",
            names="Category",
            hover_data={"Count": True, "Percentage": True},
            color_discrete_map={"Benign": "#16A34A", "Known Attack": "#DC2626", "Potential Novel / Suspicious": "#F59E0B"},
            hole=0.45,
        )
        pie_fig.update_traces(text=[f"{row['Category']}<br>{row['Count']:,}<br>{row['Percentage']:.2f}%" for _, row in pie_df.iterrows()], textposition="inside")
        pie_fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#111827",
            legend=dict(bgcolor="#FFFFFF", font=dict(color="#374151")),
        )
        st.plotly_chart(pie_fig, width="stretch")

    with visual_cols[1]:
        st.markdown('<div class="section-title">DETECTED ATTACK TYPES</div>', unsafe_allow_html=True)
        attack_bar_df = known_attack_summary.copy()
        if not attack_bar_df.empty:
            attack_bar_df = attack_bar_df.sort_values("Detected_Flows", ascending=False).reset_index(drop=True)
            attack_bar_df["Attack_Name"] = attack_bar_df["Attack_Name"].astype(str)
            attack_bar_fig = px.bar(
                attack_bar_df,
                x="Attack_Name",
                y="Detected_Flows",
                text="Detected_Flows",
                color="Attack_Name",
                color_discrete_sequence=["#DC2626", "#B91C1C", "#EF4444", "#F87171", "#991B1B"],
            )
            attack_bar_fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            attack_bar_fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                xaxis_title="Attack",
                yaxis_title="Detected Flows",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font_color="#111827",
                xaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#374151")),
                yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#374151")),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(attack_bar_fig, width="stretch")
        else:
            st.info("No attack distribution available yet.")

    with visual_cols[2]:
        st.markdown('<div class="section-title">SEVERITY DISTRIBUTION</div>', unsafe_allow_html=True)
        severity_df = known_attack_summary.groupby("Severity", dropna=False).agg(Detected_Flows=("Detected_Flows", "sum")).reset_index()
        if not severity_df.empty:
            severity_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            severity_df["Severity"] = severity_df["Severity"].astype(str).str.upper()
            severity_df = severity_df[severity_df["Severity"].isin(severity_order)]
            severity_df = severity_df.set_index("Severity").reindex(severity_order).reset_index().fillna(0)
            severity_fig = px.bar(
                severity_df,
                x="Severity",
                y="Detected_Flows",
                color="Severity",
                color_discrete_map={"CRITICAL": "#DC2626", "HIGH": "#EA580C", "MEDIUM": "#F59E0B", "LOW": "#16A34A"},
            )
            severity_fig.update_traces(texttemplate='%{y:,}', textposition='outside')
            severity_fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                xaxis_title="Severity",
                yaxis_title="Detected Flows",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font_color="#111827",
                xaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#374151")),
                yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#374151")),
            )
            st.plotly_chart(severity_fig, width="stretch")
        else:
            st.info("No severity distribution available yet.")

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    status_cols = st.columns([1.1, 0.9])
    with status_cols[0]:
        st.markdown('<div class="section-title">SECURITY STATUS PANEL</div>', unsafe_allow_html=True)
        known_attack_count = int(summary["Known_Attack_Count"])
        suspicious_count = int(summary["Suspicious_Count"])
        risk_status = risk_level.upper()
        detail_lines = [
            f"Critical-Severity Attack Detected: {'Yes' if any(str(row['Severity']).upper() == 'CRITICAL' for _, row in known_attack_summary.iterrows()) else 'No'}",
            f"Known Attack Types: {len(known_attack_summary) if not known_attack_summary.empty else 0}",
            f"Known Attack Flows: {known_attack_count:,}",
            f"Suspicious Flows: {suspicious_count:,}",
            f"Overall Network Risk: {risk_status}",
        ]
        render_status_card(f"{risk_status} OVERALL NETWORK RISK", detail_lines, risk_level)

    with status_cols[1]:
        st.markdown('<div class="section-title">DETECTED ATTACKS</div>', unsafe_allow_html=True)
        if known_attack_summary.empty:
            st.info("No high-confidence known attack traffic was detected.")
        else:
            threat_table = known_attack_summary.copy()
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            threat_table["Severity_Priority"] = threat_table["Severity"].map(severity_order).fillna(99)
            threat_table = threat_table.sort_values(["Severity_Priority", "Detected_Flows"], ascending=[True, False]).reset_index(drop=True)
            display_table = threat_table[[
                "Attack_Name",
                "Detected_Flows",
                "Percentage_of_Total_Flows",
                "Threat_Type",
                "Severity",
            ]].copy()
            display_table.columns = ["Attack", "Detected Flows", "Traffic Share", "Threat Type", "Severity"]
            display_table["Detected Flows"] = display_table["Detected Flows"].map(lambda value: f"{value:,}")
            display_table["Traffic Share"] = display_table["Traffic Share"].map(lambda value: f"{value:.2f}%")
            display_table["Severity"] = display_table["Severity"].astype(str).str.upper()
            display_table["Severity"] = display_table["Severity"].map(
                lambda value: f"<span class='severity-badge severity-{value.lower() if value.lower() in {'critical','high','medium','low'} else 'low'}'>{value}</span>"
            )
            table_html = display_table.to_html(index=False, escape=False, classes=["detected-attacks-table"])
            st.markdown(
                f"""
                <div class="detected-attacks-section">
                    <div class="detected-attacks-card">
                        {table_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">THREAT DEFENSE CENTER</div>', unsafe_allow_html=True)
    if known_attack_summary.empty:
        st.info("No known attack detected. The system is currently operating in a monitoring-only state for this dataset.")
    else:
        threat_cards = known_attack_summary.copy()
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        threat_cards["Severity_Priority"] = threat_cards["Severity"].map(severity_order).fillna(99)
        threat_cards = threat_cards.sort_values(["Severity_Priority", "Detected_Flows"], ascending=[True, False]).reset_index(drop=True)
        for _, attack_row in threat_cards.iterrows():
            render_attack_defense_card(attack_row)

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ATTACK RANKING</div>', unsafe_allow_html=True)
    if not known_attack_summary.empty:
        ranking_df = known_attack_summary.copy()
        ranking_df = ranking_df.sort_values(["Detected_Flows"], ascending=False).reset_index(drop=True)
        ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))
        ranking_df["Detected_Flows"] = ranking_df["Detected_Flows"].map(lambda value: f"{value:,}")
        ranking_df["Percentage_of_Total_Flows"] = ranking_df["Percentage_of_Total_Flows"].map(lambda value: f"{value:.2f}%")
        ranking_df = ranking_df.rename(columns={
            "Attack_Name": "Attack",
            "Detected_Flows": "Detected Flows",
            "Percentage_of_Total_Flows": "Traffic Share",
            "Severity": "Severity",
        })
        st.dataframe(ranking_df[["Rank", "Attack", "Detected Flows", "Traffic Share", "Severity"]], width="stretch", hide_index=True)
    else:
        st.info("No known attack ranking available yet.")

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">POTENTIAL NOVEL / SUSPICIOUS TRAFFIC</div>', unsafe_allow_html=True)
    if int(summary["Suspicious_Count"]) > 0:
        suspicious_cols = st.columns([1, 2])
        with suspicious_cols[0]:
            st.markdown(
                f"""
                <div class='detail-card'>
                    <div class='status-label'>SUSPICIOUS FLOW COUNT</div>
                    <div class='metric-value'>{summary['Suspicious_Count']:,}</div>
                    <div class='status-label'>TRAFFIC SHARE</div>
                    <div class='metric-value' style='font-size:1rem;'>{summary['Suspicious_Percentage']}%</div>
                    <div class='status-label'>AVERAGE SUSPICIOUS CONFIDENCE</div>
                    <div class='metric-value' style='font-size:1rem;'>{results_df.loc[results_df['Final_Category'] == 'SUSPICIOUS', 'Confidence_Percentage'].mean():.2f}%</div>
                    <div class='status-label'>REASON</div>
                    <div class='status-value'>LOW MODEL CONFIDENCE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with suspicious_cols[1]:
            st.markdown(
                f"""
                <div class='detail-card'>
                    <div class='section-title'>RECOMMENDED INVESTIGATION</div>
                    <div class='status-detail'>{SUSPICIOUS_RECOMMENDATION}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div class='notice-box'>Suspicious traffic indicates low model confidence and may represent previously unseen traffic. Further investigation is recommended. It is not automatically treated as a confirmed novel attack.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Suspicious Pattern Details", expanded=False):
            st.dataframe(suspicious_pattern_summary, width="stretch", hide_index=True)
    else:
        st.info("No suspicious flows were detected in the current dataset.")

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">FLOW INVESTIGATION</div>', unsafe_allow_html=True)
    filter_cols = st.columns([1.35, 1.0, 1.0, 1.0, 1.25])
    with filter_cols[0]:
        search_query = st.text_input("Search attack or recommendation", placeholder="Search attack or recommendation")
    with filter_cols[1]:
        raw_prediction_options = ["ALL"] + sorted([str(value) for value in pd.unique(results_df["Raw_Prediction"])])
        raw_prediction_filter = st.selectbox("Attack Type", raw_prediction_options)
    with filter_cols[2]:
        category_filter = st.selectbox("Category", ["ALL", "BENIGN", "KNOWN ATTACK", "SUSPICIOUS"])
    with filter_cols[3]:
        severity_filter = st.selectbox("Severity", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE"])
    with filter_cols[4]:
        min_confidence = st.slider("Confidence (%)", min_value=0, max_value=100, value=0, step=1, format="%d")

    filtered_results = results_df.copy()
    if category_filter != "ALL":
        filtered_results = filtered_results[filtered_results["Final_Category"] == category_filter]
    if raw_prediction_filter != "ALL":
        filtered_results = filtered_results[filtered_results["Raw_Prediction"] == raw_prediction_filter]
    if severity_filter != "ALL":
        filtered_results = filtered_results.copy()
        filtered_results["_display_severity"] = filtered_results.apply(get_display_severity, axis=1)
        severity_lookup = {
            "CRITICAL": "Critical",
            "HIGH": "High",
            "MEDIUM": "Medium",
            "LOW": "Low",
            "SAFE": "Safe/Benign",
        }.get(severity_filter, "Medium")

        if severity_lookup == "Safe/Benign":
            filtered_results = filtered_results[filtered_results["_display_severity"] == "Safe/Benign"]
        else:
            filtered_results = filtered_results[filtered_results["_display_severity"] == severity_lookup]

    confidence_column = "Confidence_Percentage"
    if confidence_column not in filtered_results.columns:
        confidence_column = "Confidence"
    filtered_results["_display_confidence"] = filtered_results[confidence_column].astype(float)
    if filtered_results["_display_confidence"].max() <= 1.0:
        filtered_results["_display_confidence"] = filtered_results["_display_confidence"] * 100.0
    filtered_results = filtered_results[filtered_results["_display_confidence"] >= min_confidence]

    if search_query:
        search_query = search_query.lower()
        filtered_results = filtered_results[
            filtered_results["Raw_Prediction"].astype(str).str.lower().str.contains(search_query)
            | filtered_results["Defense_Recommendation"].astype(str).str.lower().str.contains(search_query)
            | filtered_results["Final_Category"].astype(str).str.lower().str.contains(search_query)
        ]

    visible_results = build_flow_display_dataframe(uploaded_df, filtered_results)
    st.dataframe(
        visible_results,
        width="stretch",
        height=420,
        hide_index=True,
        column_config={
            "Flow ID": st.column_config.NumberColumn(width="small"),
            "Prediction": st.column_config.TextColumn(width="small"),
            "Confidence (%)": st.column_config.NumberColumn(width="small"),
            "Final Category": st.column_config.TextColumn(width="small"),
            "Severity": st.column_config.TextColumn(width="small"),
            "Recommended Action": st.column_config.TextColumn(
                width="large",
                help="Long recommendations are abbreviated in this table; the CSV export retains full text.",
                max_chars=120,
            ),
        },
    )

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">SECURITY REPORTS</div>', unsafe_allow_html=True)
    report_col, csv_col = st.columns([1, 1])
    with report_col:
        if st.button("Generate Security Report", width="stretch", type="primary"):
            model_info = get_model_info()
            pdf_bytes = create_pdf_report(
                summary,
                known_attack_summary,
                suspicious_pattern_summary,
                results_df,
                uploaded_file_name,
                model_info,
            )
            st.session_state.generated_pdf = pdf_bytes
            st.success("Security report generated successfully.")
    with csv_col:
        st.download_button(
            label="Download CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{os.path.splitext(uploaded_file_name)[0]}_analysis_results.csv",
            mime="text/csv",
            width="stretch",
        )

    if "generated_pdf" in st.session_state and st.session_state.generated_pdf:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state.generated_pdf,
            file_name=f"{os.path.splitext(uploaded_file_name)[0]}_security_report.pdf",
            mime="application/pdf",
            width="stretch",
        )

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">MODEL INFORMATION</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="model-info-grid">
            <div class="model-info-item"><div class="model-info-label">Selected Model</div><div class="model-info-value">{selected_model}</div></div>
            <div class="model-info-item"><div class="model-info-label">Features</div><div class="model-info-value">70</div></div>
            <div class="model-info-item"><div class="model-info-label">Dataset</div><div class="model-info-value">CIC-IDS2017</div></div>
            <div class="model-info-item"><div class="model-info-label">Classification</div><div class="model-info-value">Benign / Known Attack / Potential Novel-Suspicious</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("ADDITIONAL MODEL DETAILS", expanded=False):
        model_info = get_model_info()
        if "error" in model_info:
            st.warning(model_info["error"])
        else:
            model_detail_cols = st.columns(4)
            model_details = [
                ("Algorithm", model_info.get("algorithm", selected_model)),
                ("Model File", model_info.get("model_name", "N/A")),
                ("Trained Classes", str(model_info.get("number_of_classes", "N/A"))),
                ("Training Samples", f"{model_info.get('training_samples', 'N/A'):,}" if isinstance(model_info.get("training_samples"), int) else str(model_info.get("training_samples", "N/A"))),
                ("Test Samples", f"{model_info.get('testing_samples', 'N/A'):,}" if isinstance(model_info.get("testing_samples"), int) else str(model_info.get("testing_samples", "N/A"))),
                ("Confidence Threshold", f"{model_info.get('confidence_threshold', 0.95) * 100:.0f}%"),
            ]
            for column, (label, value) in zip(model_detail_cols * 2, model_details):
                with column:
                    st.markdown(
                        f"<div class='model-info-item'><div class='model-info-label'>{label}</div>"
                        f"<div class='model-info-value'>{value}</div></div>",
                        unsafe_allow_html=True,
                    )

    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class='workflow-card dashboard-footer'>
            <div style='font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.45rem;'>NETWORK INTRUSION DETECTION AND DEFENSE RECOMMENDATION SYSTEM</div>
            <div style='font-size:1rem; font-weight:700; color:#0f172a;'>Machine Learning Based Network Security Analysis</div>
            <div style='font-size:0.9rem; color:#64748b; margin-top:0.25rem;'>CIC-IDS2017 · Version 1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
