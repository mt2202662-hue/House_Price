"""Streamlit interface for the trained Ames House Price regression model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
MODEL_PATH = ROOT / "models" / "best_model.joblib"
CONFIG_PATH = ROOT / "models" / "app_config.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Ames House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction")
st.caption("Kaggle Ames Housing dataset · Supervised Learning · Regression")

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.error("Trained model files were not found. First run: python train.py")
    st.stop()

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return model, config, metadata

model, config, metadata = load_assets()
defaults = config["defaults"]

with st.sidebar:
    st.header("Model information")
    st.write("**Selected algorithm:**", metadata.get("selected_model", "Not recorded"))
    st.write("**ML type:** Supervised Learning")
    st.write("**Task:** Regression")
    st.write("**Target transformation:** log1p / expm1")
    st.divider()
    st.info("This is an educational model trained on historical Ames, Iowa house-sale data. It is not a live property valuation service.")

st.write(
    "Adjust the selected property features below. For fields not shown in this form, "
    "the app uses median or most-frequent values from the training dataset."
)

with st.form("prediction_form"):
    st.subheader("Property information")
    cols = st.columns(2)
    row = dict(defaults)
    for index, field in enumerate(config["ui_fields"]):
        name = field["name"]
        default = defaults.get(name)
        with cols[index % 2]:
            if field["kind"] == "category":
                options = config["categorical_options"].get(name, [])
                if not options:
                    row[name] = default
                else:
                    default_text = str(default)
                    default_index = options.index(default_text) if default_text in options else 0
                    row[name] = st.selectbox(field["label"], options, index=default_index)
            else:
                lo, hi = field["min"], field["max"]
                value = int(round(float(default))) if default is not None else lo
                value = min(max(value, lo), hi)
                row[name] = st.number_input(
                    field["label"], min_value=lo, max_value=hi,
                    value=value, step=field["step"],
                )
    submitted = st.form_submit_button("Predict estimated sale price", type="primary", use_container_width=True)

if submitted:
    input_frame = pd.DataFrame([row], columns=config["input_columns"])
    predicted_log = float(model.predict(input_frame)[0])
    predicted_price = float(np.expm1(max(predicted_log, 0.0)))
    st.success("Prediction completed")
    st.metric("Predicted historical sale price", f"${predicted_price:,.0f}")
    st.caption(
        "This is an educational estimate for an Ames-style property. Most dataset "
        "features are filled with training-set defaults, so the estimate may not "
        "reflect a real individual property's value."
    )

with st.expander("View model validation metrics"):
    metrics = metadata.get("validation_metrics", {})
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Holdout RMSLE", f"{metrics.get('holdout_rmsle', float('nan')):.4f}")
        c2.metric("Holdout MAE", f"${metrics.get('holdout_mae_usd', float('nan')):,.0f}")
        c3.metric("Holdout RMSE", f"${metrics.get('holdout_rmse_usd', float('nan')):,.0f}")
        c4.metric("Holdout R²", f"{metrics.get('holdout_r2', float('nan')):.3f}")
        st.caption("These are validation results from a held-out part of train.csv, not scores on Kaggle's hidden test labels.")
    else:
        st.info("Metrics will appear after train.py completes.")
