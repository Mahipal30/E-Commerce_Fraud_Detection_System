"""Generate Kaggle submission + risk score outputs for test set (core XGBoost bundle)."""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, ID_COL
from src.features import engineer_features
from src.utils import load_json


def apply_encoder_and_imputer(df: pd.DataFrame, cat_maps: dict, medians: dict) -> pd.DataFrame:
    df = df.copy()

    # 1) Encode any columns we have mappings for (learned from train/val)
    for c, mapping in cat_maps.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype("string")
                .fillna("<<NA>>")
                .map(mapping)
                .fillna(-1)
                .astype("int32")
            )

    # 2) For any remaining object/string columns (e.g., values like "NotFound"),
    # encode them safely via factorize (test-only encoding is OK for inference)
    remaining_obj = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in remaining_obj:
        df[c] = pd.factorize(df[c].astype("string").fillna("<<NA>>"), sort=True)[0].astype("int32")

    # 3) Replace inf and impute numeric using training medians
    df = df.replace([np.inf, -np.inf], np.nan)

    for c, med in medians.items():
        if c in df.columns and df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(med)

    # 4) Final fallback for any remaining NaNs
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(0)
        else:
            # should not happen now, but safe
            df[c] = pd.factorize(df[c].astype("string").fillna("<<NA>>"), sort=True)[0].astype("int32")

    return df


def main():
    bundle = joblib.load(MODELS_DIR / "xgb_model.joblib")
    booster = bundle["booster"]
    feature_cols = bundle["feature_cols"]
    drop_cols = bundle["drop_cols"]
    cat_maps = bundle["cat_maps"]
    medians = bundle["medians"]

    thr = load_json(REPORTS_DIR / "thresholds.json")
    t_high = thr["threshold_high"]
    t_med = thr["threshold_medium"]

    X_test = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    X_test = engineer_features(X_test)
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors="ignore")

    X_test = apply_encoder_and_imputer(X_test, cat_maps, medians)

    # Align columns exactly as training
    for c in feature_cols:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[feature_cols]

    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    proba = booster.predict(dtest)

    risk_score = np.rint(100 * proba).astype(int)
    band = np.where(proba >= t_high, "HIGH", np.where(proba >= t_med, "MEDIUM", "LOW"))

    sub = pd.read_parquet(DATA_INTERIM / "sample_submission.parquet")
    sub["isFraud"] = proba
    out_sub = REPORTS_DIR / "submission.csv"
    sub.to_csv(out_sub, index=False)

    out_scored = REPORTS_DIR / "test_scored.csv"
    scored = pd.DataFrame(
        {
            ID_COL: sub[ID_COL].values,
            "fraud_proba": proba,
            "risk_score": risk_score,
            "risk_band": band,
        }
    )
    scored.to_csv(out_scored, index=False)

    print(f"Saved: {out_sub}")
    print(f"Saved: {out_scored}")


if __name__ == "__main__":
    main()