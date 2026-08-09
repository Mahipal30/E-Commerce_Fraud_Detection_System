# src/train_xgb_time_split.py
"""
FAST trainer using XGBoost core API (works even when sklearn wrapper lacks early stopping/callbacks).

Outputs:
- models/xgb_model.joblib  (bundle: booster, feature_cols, drop_cols, cat_maps, medians)
- reports/metrics.json
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, TARGET_COL, TIME_COL, RANDOM_SEED
from src.features import engineer_features
from src.utils import save_json


def time_split(X: pd.DataFrame, y: np.ndarray, time_col: str, split_q: float = 0.8):
    if time_col not in X.columns:
        raise ValueError(f"{time_col} not found in X. Cannot time-split.")
    cutoff = float(X[time_col].quantile(split_q))
    train_idx = X[time_col] <= cutoff
    val_idx = X[time_col] > cutoff
    return X.loc[train_idx], X.loc[val_idx], y[train_idx.values], y[val_idx.values], cutoff


def fit_encoder_and_imputer(train: pd.DataFrame, val: pd.DataFrame):
    """
    Encodes object/string columns using a mapping learned from TRAIN+VAL categories.
    Unseen categories -> -1. Numeric NaNs filled with TRAIN median.
    Returns encoded train/val + cat_maps + medians.
    """
    train = train.copy()
    val = val.copy()

    cat_maps = {}
    medians = {}

    obj_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()

    # build mappings from train+val (not test)
    for c in obj_cols:
        both = pd.concat([train[c].astype("string"), val[c].astype("string")], axis=0)
        uniq = pd.Index(both.fillna("<<NA>>").unique())
        mapping = {k: i for i, k in enumerate(uniq.tolist())}
        cat_maps[c] = mapping

        train[c] = train[c].astype("string").fillna("<<NA>>").map(mapping).fillna(-1).astype("int32")
        val[c] = val[c].astype("string").fillna("<<NA>>").map(mapping).fillna(-1).astype("int32")

    # numeric cleaning + medians
    train = train.replace([np.inf, -np.inf], np.nan)
    val = val.replace([np.inf, -np.inf], np.nan)

    for c in train.columns:
        if train[c].dtype.kind in "biufc":
            med = float(train[c].median())
            medians[c] = med
            train[c] = train[c].fillna(med)
            val[c] = val[c].fillna(med)
        else:
            # should not happen after encoding
            train[c] = train[c].fillna(-1).astype("int32")
            val[c] = val[c].fillna(-1).astype("int32")

    return train, val, cat_maps, medians


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    y = pd.read_parquet(DATA_PROCESSED / "y_train.parquet")[TARGET_COL].values.astype(int)

    # feature engineering
    X = engineer_features(X)

    # time split
    X_train, X_val, y_train, y_val, cutoff = time_split(X, y, TIME_COL, split_q=0.8)

    drop_cols = [c for c in ["TransactionID", TIME_COL] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")

    # encode + impute
    X_train, X_val, cat_maps, medians = fit_encoder_and_imputer(X_train, X_val)

    feature_cols = list(X_train.columns)
    X_train = X_train[feature_cols]
    X_val = X_val[feature_cols]

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = float(neg / max(pos, 1))

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "tree_method": "hist",
        "scale_pos_weight": spw,
        "seed": RANDOM_SEED,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=50,
    )

    proba = booster.predict(dval)
    pr_auc = float(average_precision_score(y_val, proba))
    roc_auc = float(roc_auc_score(y_val, proba))

    metrics = {
        "val_pr_auc": pr_auc,
        "val_roc_auc": roc_auc,
        "time_split_quantile": 0.8,
        "time_cutoff": cutoff,
        "scale_pos_weight": spw,
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
        "best_score": float(getattr(booster, "best_score", np.nan)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_val": float(y_val.mean()),
        "n_features": int(len(feature_cols)),
    }

    bundle = {
        "booster": booster,
        "feature_cols": feature_cols,
        "drop_cols": drop_cols,
        "cat_maps": cat_maps,
        "medians": medians,
    }

    joblib.dump(bundle, MODELS_DIR / "xgb_model.joblib")
    save_json(REPORTS_DIR / "metrics.json", metrics)

    print("Saved model bundle: models/xgb_model.joblib")
    print("Saved metrics: reports/metrics.json")
    print(f"VAL PR-AUC={pr_auc:.5f} ROC-AUC={roc_auc:.5f}")


if __name__ == "__main__":
    main()