"""Model training pipeline: load data, build features, train, calibrate, evaluate.

Usage:
    python -m model.train
"""

import joblib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from config.settings import (
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS, ELO_BURN_IN_YEARS,
    MODEL_DIR, EVAL_DIR,
)
from model.features import load_matches, build_feature_matrix, FEATURE_NAMES
from model.evaluate import (
    evaluate_model, plot_calibration, plot_feature_importance, compare_models,
)


class CalibratedModel:
    """Wraps a classifier with isotonic regression calibration."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        cal = np.clip(cal, 0.001, 0.999)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _split_by_year(X, y, meta, year_range):
    """Select rows where meta.year is within year_range (inclusive)."""
    mask = (meta["year"] >= year_range[0]) & (meta["year"] <= year_range[1])
    return X[mask].copy(), y[mask].copy(), meta[mask].copy()


def train_baseline(X_train, y_train):
    """Logistic Regression baseline (already well-calibrated)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost + isotonic calibration on validation set."""
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        early_stopping_rounds=30,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  XGBoost best iteration: {model.best_iteration}")

    # Calibrate on validation set using isotonic regression
    raw_probs = model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(raw_probs, y_val)
    return CalibratedModel(model, iso), model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM + isotonic calibration on validation set."""
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    print(f"  LightGBM best iteration: {model.best_iteration_}")

    raw_probs = model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(raw_probs, y_val)
    return CalibratedModel(model, iso), model


def save_model(model, name: str, metrics: dict) -> Path:
    """Save model + metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name}.joblib"
    meta_path = MODEL_DIR / f"{name}_meta.json"

    joblib.dump(model, model_path)

    meta = {
        "name": name,
        "features": FEATURE_NAMES,
        "train_years": list(TRAIN_YEARS),
        "val_years": list(VAL_YEARS),
        "test_years": list(TEST_YEARS),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {name} to {model_path}")
    return model_path


def main():
    print("=" * 60)
    print("PHASE 1: MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading match data...")
    matches = load_matches()
    print(f"  Loaded {len(matches)} matches ({matches['year'].min()}-{matches['year'].max()})")

    # Step 2: Build features
    print("\n[2/5] Building feature matrix...")
    X, y, meta = build_feature_matrix(matches)

    # Step 3: Split
    print("\n[3/5] Splitting data...")
    X_train, y_train, meta_train = _split_by_year(X, y, meta, TRAIN_YEARS)
    X_val, y_val, meta_val = _split_by_year(X, y, meta, VAL_YEARS)
    X_test, y_test, meta_test = _split_by_year(X, y, meta, TEST_YEARS)

    # Exclude burn-in period from training
    print(f"  Train: {len(X_train)} matches ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]})")
    print(f"  Val:   {len(X_val)} matches ({VAL_YEARS[0]}-{VAL_YEARS[1]})")
    print(f"  Test:  {len(X_test)} matches ({TEST_YEARS[0]}-{TEST_YEARS[1]})")
    print(f"  Target balance (train): {y_train.mean():.3f}")

    # Fill NaN for logistic regression (tree models handle NaN natively)
    X_train_filled = X_train.fillna(0)
    X_val_filled = X_val.fillna(0)
    X_test_filled = X_test.fillna(0)

    # Step 4: Train models
    print("\n[4/5] Training models...")

    print("\n  --- Logistic Regression (baseline) ---")
    lr = train_baseline(X_train_filled, y_train)

    print("\n  --- XGBoost + Isotonic Calibration ---")
    xgb_cal, xgb_raw = train_xgboost(X_train, y_train, X_val, y_val)

    print("\n  --- LightGBM + Isotonic Calibration ---")
    lgb_cal, lgb_raw = train_lightgbm(X_train, y_train, X_val, y_val)

    # Step 5: Evaluate
    print("\n[5/5] Evaluating on TEST set...")

    results = {}

    print("\n  Logistic Regression:")
    results["LogisticRegression"] = evaluate_model(lr, X_test_filled, y_test, "LR")

    print("\n  XGBoost (raw):")
    results["XGBoost_raw"] = evaluate_model(xgb_raw, X_test, y_test, "XGB_raw")

    print("\n  XGBoost (calibrated):")
    results["XGBoost_calibrated"] = evaluate_model(xgb_cal, X_test, y_test, "XGB_cal")

    print("\n  LightGBM (raw):")
    results["LightGBM_raw"] = evaluate_model(lgb_raw, X_test, y_test, "LGB_raw")

    print("\n  LightGBM (calibrated):")
    results["LightGBM_calibrated"] = evaluate_model(lgb_cal, X_test, y_test, "LGB_cal")

    compare_models(results)

    # Plots
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    y_probs = {
        "Logistic Regression": lr.predict_proba(X_test_filled)[:, 1],
        "XGBoost (raw)": xgb_raw.predict_proba(X_test)[:, 1],
        "XGBoost (calibrated)": xgb_cal.predict_proba(X_test)[:, 1],
        "LightGBM (raw)": lgb_raw.predict_proba(X_test)[:, 1],
        "LightGBM (calibrated)": lgb_cal.predict_proba(X_test)[:, 1],
    }

    plot_calibration(np.array(y_test), y_probs, EVAL_DIR / "calibration_curves.png")
    plot_feature_importance(xgb_raw, FEATURE_NAMES, EVAL_DIR / "feature_importance_xgb.png")
    plot_feature_importance(lgb_raw, FEATURE_NAMES, EVAL_DIR / "feature_importance_lgb.png")

    # Save best model
    # Pick the calibrated model with lowest ECE
    best_name = min(
        ["XGBoost_calibrated", "LightGBM_calibrated"],
        key=lambda n: results[n]["ece"],
    )
    best_model = xgb_cal if "XGBoost" in best_name else lgb_cal
    save_model(best_model, "best_model", results[best_name])
    save_model(lr, "baseline_lr", results["LogisticRegression"])

    print(f"\nBest model: {best_name} (ECE={results[best_name]['ece']:.4f})")
    print("Done.")


if __name__ == "__main__":
    main()
