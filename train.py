"""Train, compare, evaluate and export regression models for Ames house prices.

Run from the project root:
    python train.py

The target is trained on log1p(SalePrice) because the Kaggle competition is
scored with RMSLE. Cross-validation therefore estimates RMSLE on the log scale.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from house_price_project.features import add_engineered_features  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    engineered = add_engineered_features(X)
    numeric_columns = engineered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = engineered.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, X: pd.DataFrame) -> Pipeline:
    return Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
        ("preprocessing", make_preprocessor(X)),
        ("model", model),
    ])


def model_zoo() -> dict:
    """A baseline plus five complementary regression algorithms."""
    return {
        "Dummy Median Baseline": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=20.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=70, max_features=0.85, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=2,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=90, max_features=0.85, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.035, max_depth=2,
            loss="huber", random_state=RANDOM_STATE,
        ),
    }


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def get_default_values(X: pd.DataFrame) -> dict:
    defaults: dict = {}
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            value = X[column].median()
            defaults[column] = 0.0 if pd.isna(value) else float(value)
        else:
            mode = X[column].mode(dropna=True)
            defaults[column] = "Missing" if mode.empty else str(mode.iloc[0])
    return defaults


def build_ui_config(X: pd.DataFrame) -> dict:
    defaults = get_default_values(X)
    categorical_options = {}
    for col in ["Neighborhood", "KitchenQual", "ExterQual"]:
        if col in X.columns:
            categorical_options[col] = sorted(X[col].dropna().astype(str).unique().tolist())

    ui_fields = [
        {"name": "OverallQual", "label": "Overall quality (1–10)", "kind": "integer", "min": 1, "max": 10, "step": 1},
        {"name": "GrLivArea", "label": "Above-ground living area (sq ft)", "kind": "integer", "min": 200, "max": 6000, "step": 50},
        {"name": "GarageCars", "label": "Garage capacity (cars)", "kind": "integer", "min": 0, "max": 5, "step": 1},
        {"name": "TotalBsmtSF", "label": "Basement area (sq ft)", "kind": "integer", "min": 0, "max": 6000, "step": 50},
        {"name": "FullBath", "label": "Full bathrooms", "kind": "integer", "min": 0, "max": 5, "step": 1},
        {"name": "LotArea", "label": "Lot area (sq ft)", "kind": "integer", "min": 500, "max": 100000, "step": 100},
        {"name": "YearBuilt", "label": "Year built", "kind": "integer", "min": 1870, "max": 2026, "step": 1},
        {"name": "OverallCond", "label": "Overall condition (1–10)", "kind": "integer", "min": 1, "max": 10, "step": 1},
        {"name": "GarageArea", "label": "Garage area (sq ft)", "kind": "integer", "min": 0, "max": 1500, "step": 25},
        {"name": "TotRmsAbvGrd", "label": "Above-ground rooms", "kind": "integer", "min": 2, "max": 15, "step": 1},
        {"name": "Neighborhood", "label": "Neighborhood", "kind": "category"},
        {"name": "KitchenQual", "label": "Kitchen quality", "kind": "category"},
        {"name": "ExterQual", "label": "Exterior quality", "kind": "category"},
    ]
    # Keep only UI fields present in this version of the input dataset.
    ui_fields = [field for field in ui_fields if field["name"] in X.columns]
    return {
        "defaults": defaults,
        "categorical_options": categorical_options,
        "ui_fields": ui_fields,
        "input_columns": X.columns.tolist(),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_path = DATA_DIR / "sample_submission.csv"
    for path in [train_path, test_path, sample_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Check data/raw/.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)

    target = "SalePrice"
    if target not in train.columns:
        raise ValueError("SalePrice target column is missing from train.csv.")

    y = train[target].astype(float)
    X = train.drop(columns=[target, "Id"], errors="ignore")
    X_test = test.drop(columns=["Id"], errors="ignore")
    test_ids = test["Id"].copy() if "Id" in test.columns else sample_submission["Id"].copy()

    if y.isna().any() or (y < 0).any():
        raise ValueError("SalePrice must be present and non-negative for log1p training.")

    # Untouched holdout validation split. Test.csv is never used for fitting.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    y_fit_log = np.log1p(y_fit)
    y_valid_log = np.log1p(y_valid)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted_models = {}
    print("\n========== MODEL TRAINING & COMPARISON ==========")
    print(f"Training rows: {len(X_fit):,} | Validation rows: {len(X_valid):,}")
    print("Target transform: log1p(SalePrice) to align training with RMSLE.\n")

    for name, estimator in model_zoo().items():
        print(f"Training: {name} ...", flush=True)
        pipeline = make_pipeline(estimator, X_fit)
        start = time.perf_counter()
        cv_scores = -cross_val_score(
            pipeline, X_fit, y_fit_log,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
        pipeline.fit(X_fit, y_fit_log)
        fit_seconds = time.perf_counter() - start

        pred_log = pipeline.predict(X_valid)
        pred_price = np.expm1(np.maximum(pred_log, 0.0))
        valid_rmsle = rmsle(y_valid.to_numpy(), pred_price)
        valid_mae = float(mean_absolute_error(y_valid, pred_price))
        valid_rmse = float(np.sqrt(mean_squared_error(y_valid, pred_price)))
        valid_r2 = float(r2_score(y_valid, pred_price))

        rows.append({
            "model": name,
            "cv_rmsle_mean": float(cv_scores.mean()),
            "cv_rmsle_std": float(cv_scores.std()),
            "validation_rmsle": valid_rmsle,
            "validation_mae_usd": valid_mae,
            "validation_rmse_usd": valid_rmse,
            "validation_r2": valid_r2,
            "fit_and_cv_seconds": round(fit_seconds, 2),
        })
        fitted_models[name] = pipeline
        print(
            f"  CV RMSLE={cv_scores.mean():.5f} (+/- {cv_scores.std():.5f}) | "
            f"Validation RMSLE={valid_rmsle:.5f} | R2={valid_r2:.4f}"
        )

    leaderboard = pd.DataFrame(rows).sort_values("cv_rmsle_mean", ascending=True)
    leaderboard.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = str(leaderboard.iloc[0]["model"])
    print("\n========== VALIDATION LEADERBOARD ==========")
    print(leaderboard.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nSelected model by lowest 5-fold CV RMSLE: {best_name}")

    # Save honest holdout results for the report.
    best_valid_pipeline = fitted_models[best_name]
    best_valid_pred_log = best_valid_pipeline.predict(X_valid)
    best_valid_pred = np.expm1(np.maximum(best_valid_pred_log, 0.0))
    validation_metrics = {
        "selected_model": best_name,
        "selection_method": "lowest mean 5-fold CV RMSLE on the training portion",
        "holdout_rows": int(len(X_valid)),
        "holdout_rmsle": rmsle(y_valid.to_numpy(), best_valid_pred),
        "holdout_mae_usd": float(mean_absolute_error(y_valid, best_valid_pred)),
        "holdout_rmse_usd": float(np.sqrt(mean_squared_error(y_valid, best_valid_pred))),
        "holdout_r2": float(r2_score(y_valid, best_valid_pred)),
        "note": "Holdout metrics are from a random 20% split of train.csv. Kaggle test.csv has no labels, so its score is unknown until submission is scored by Kaggle.",
    }
    (OUTPUT_DIR / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "actual_SalePrice": y_valid.to_numpy(),
        "predicted_SalePrice": best_valid_pred,
        "absolute_error": np.abs(y_valid.to_numpy() - best_valid_pred),
    }).to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    # Refit the selected pipeline on all labelled training rows for the Kaggle test.
    print(f"\nRefitting {best_name} on all {len(X):,} labelled rows ...")
    final_model = make_pipeline(model_zoo()[best_name], X)
    final_model.fit(X, np.log1p(y))
    joblib.dump(final_model, MODEL_DIR / "best_model.joblib")

    test_pred_log = final_model.predict(X_test)
    test_pred_price = np.expm1(np.maximum(test_pred_log, 0.0))
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_pred_price})
    # Match the competition's sample submission columns and order.
    submission = submission[sample_submission.columns]
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    config = build_ui_config(X)
    (MODEL_DIR / "app_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    model_metadata = {
        "project": "Kaggle Ames House Prices",
        "selected_model": best_name,
        "target": "SalePrice",
        "target_transform": "log1p; predictions converted back with expm1",
        "training_rows": int(len(X)),
        "input_feature_count_before_engineering": int(X.shape[1]),
        "validation_metrics": validation_metrics,
        "kaggle_metric": "RMSLE",
        "warning": "Educational prediction only. The Streamlit form exposes selected features; other features use training-set median/mode defaults.",
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )

    print("\n========== COMPLETE ==========")
    print("Best model:", best_name)
    print("Saved model:", MODEL_DIR / "best_model.joblib")
    print("Saved Kaggle submission:", OUTPUT_DIR / "submission.csv")
    print("Saved validation metrics:", OUTPUT_DIR / "validation_metrics.json")
    print("Next: inspect outputs/model_comparison.csv and outputs/validation_metrics.json")


if __name__ == "__main__":
    main()
