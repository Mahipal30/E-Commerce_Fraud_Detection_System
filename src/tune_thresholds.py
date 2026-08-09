# src/tune_thresholds.py
"""
Choose thresholds for operational risk bands using validation probabilities.

Reads:
- models/xgb_model.joblib (bundle)
- reports/metrics.json (time_cutoff)

Writes:
- reports/thresholds.json
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, TARGET_COL, TIME_COL
from src.features import engineer_features
from src.utils import save_json, load_json

HIGH_PCT = 0.01
MED_PCT = 0.05  # cumulative


def apply_encoder_and_imputer(df: pd.DataFrame, cat_maps: dict, medians: dict):
    df = df.copy()

    # encode categoricals with stored maps
    for c, mapping in cat_maps.items():
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("<<NA>>").map(mapping).fillna(-1).astype("int32")

    df = df.replace([np.inf, -np.inf], np.nan)

    # impute numeric with stored medians
    for c, med in medians.items():
        if c in df.columns and df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(med)

    # any remaining NaNs
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna(-1).astype("int32")

    return df


def main():
    bundle = joblib.load(MODELS_DIR / "xgb_model.joblib")
    booster = bundle["booster"]
    feature_cols = bundle["feature_cols"]
    drop_cols = bundle["drop_cols"]
    cat_maps = bundle["cat_maps"]
    medians = bundle["medians"]

    cutoff = load_json(REPORTS_DIR / "metrics.json")["time_cutoff"]

    X = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    y = pd.read_parquet(DATA_PROCESSED / "y_train.parquet")[TARGET_COL].values.astype(int)

    X = engineer_features(X)

    val_idx = X[TIME_COL] > cutoff
    X_val = X.loc[val_idx].drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    y_val = y[val_idx.values]

    X_val = apply_encoder_and_imputer(X_val, cat_maps, medians)

    # align columns
    for c in feature_cols:
        if c not in X_val.columns:
            X_val[c] = 0
    X_val = X_val[feature_cols]

    dval = xgb.DMatrix(X_val, feature_names=feature_cols)
    proba = booster.predict(dval)

    t_high = float(np.quantile(proba, 1 - HIGH_PCT))
    t_med = float(np.quantile(proba, 1 - MED_PCT))

    # best-F1 threshold (report only)
    prec, rec, thr = precision_recall_curve(y_val, proba)
    f1s = (2 * prec[1:] * rec[1:]) / np.maximum(prec[1:] + rec[1:], 1e-12)
    best_i = int(np.argmax(f1s)) if len(f1s) else 0
    t_f1 = float(thr[best_i]) if len(thr) else 0.5

    flagged = (proba >= t_med).astype(int)

    summary = {
        "threshold_high": t_high,
        "threshold_medium": t_med,
        "threshold_best_f1": t_f1,
        "val_flagged_rate": float(flagged.mean()),
        "val_precision_flagged": float(precision_score(y_val, flagged, zero_division=0)),
        "val_recall_flagged": float(recall_score(y_val, flagged, zero_division=0)),
        "val_f1_flagged": float(f1_score(y_val, flagged, zero_division=0)),
        "val_n": int(len(y_val)),
        "val_fraud_rate": float(y_val.mean()),
        "policy": {
            "high_pct": HIGH_PCT,
            "med_pct_cumulative": MED_PCT,
            "meaning": {"high": "manual_review", "medium": "step_up_verification", "low": "approve"},
        },
    }

    save_json(REPORTS_DIR / "thresholds.json", summary)
    print("Saved: reports/thresholds.json")
    print(summary)


if __name__ == "__main__":
    main()