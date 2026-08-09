import json
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.config import MODELS_DIR, REPORTS_DIR
from src.db import insert_score, fetch_recent
from src.features import engineer_features
from src.utils import load_json

app = FastAPI(title="Fraud Risk Scoring API", version="2.0")

# ----------------------------
# Load full model bundle + thresholds at startup
# ----------------------------
full_bundle = joblib.load(MODELS_DIR / "xgb_model.joblib")
full_booster = full_bundle["booster"]
full_feature_cols = full_bundle["feature_cols"]
full_drop_cols = full_bundle["drop_cols"]
full_cat_maps = full_bundle["cat_maps"]
full_medians = full_bundle["medians"]

thr = load_json(REPORTS_DIR / "thresholds.json")
full_t_high = float(thr["threshold_high"])
full_t_med = float(thr["threshold_medium"])

COMMON_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "aol.com", "live.com", "msn.com", "proton.me", "protonmail.com"
}

INTERACTIVE_FIELDS = {
    "TransactionDT", "TransactionAmt", "ProductCD", "card4", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo", "id_30", "id_31"
}

INTERACTIVE_T_MED = 40
INTERACTIVE_T_HIGH = 70


# ----------------------------
# Request Schemas + Validation
# ----------------------------
class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")
    payload: Dict[str, Any] = Field(..., description="Transaction fields (raw input)")


class TransactionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payloads: List[Dict[str, Any]] = Field(..., min_length=1, max_length=2000)


def _validate_single_payload(payload: Dict[str, Any]) -> None:
    if "TransactionDT" not in payload:
        raise HTTPException(status_code=422, detail="Missing required field: TransactionDT")
    if "TransactionAmt" not in payload:
        raise HTTPException(status_code=422, detail="Missing required field: TransactionAmt")

    try:
        float(payload["TransactionDT"])
    except Exception as e:
        raise HTTPException(status_code=422, detail="TransactionDT must be numeric") from e

    try:
        float(payload["TransactionAmt"])
    except Exception as e:
        raise HTTPException(status_code=422, detail="TransactionAmt must be numeric") from e


def _validate_batch_payloads(payloads: List[Dict[str, Any]]) -> None:
    if len(payloads) == 0:
        raise HTTPException(status_code=422, detail="payloads cannot be empty")
    if len(payloads) > 2000:
        raise HTTPException(status_code=413, detail="payloads too large (max 2000 rows)")

    for i, p in enumerate(payloads):
        try:
            _validate_single_payload(p)
        except HTTPException as e:
            raise HTTPException(status_code=e.status_code, detail=f"Row {i}: {e.detail}") from None


# ----------------------------
# Shared helpers
# ----------------------------
def _norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _is_missing_like(s: str) -> bool:
    return s in {"", "none", "nan", "notfound", "<<na>>", "null", "unknown"}


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    v = _safe_float(x)
    if v is None:
        return None
    return int(v)


def is_interactive_payload(payload: Dict[str, Any]) -> bool:
    meaningful_keys = {k for k, v in payload.items() if v not in (None, "", [])}
    return meaningful_keys.issubset(INTERACTIVE_FIELDS) and len(meaningful_keys) <= 15


# ----------------------------
# Full-model inference helpers
# ----------------------------
def apply_encoder_and_imputer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c, mapping in full_cat_maps.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype("string")
                .fillna("<<NA>>")
                .map(mapping)
                .fillna(-1)
                .astype("int32")
            )

    remaining_obj = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in remaining_obj:
        df[c] = pd.factorize(df[c].astype("string").fillna("<<NA>>"), sort=True)[0].astype("int32")

    df = df.replace([np.inf, -np.inf], np.nan)

    for c, med in full_medians.items():
        if c in df.columns and df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(med)

    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(0)

    return df


def prepare_features_from_payloads(payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(payloads)
    df = engineer_features(df)
    df = df.drop(columns=[c for c in full_drop_cols if c in df.columns], errors="ignore")
    df = apply_encoder_and_imputer(df)

    for c in full_feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df[full_feature_cols]


def score_full_payloads(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    X = prepare_features_from_payloads(payloads)
    d = xgb.DMatrix(X, feature_names=full_feature_cols)
    proba = full_booster.predict(d)

    risk_score = np.rint(100 * proba).astype(int)
    band = np.where(proba >= full_t_high, "HIGH", np.where(proba >= full_t_med, "MEDIUM", "LOW"))

    results = []
    for p, rs, b in zip(proba, risk_score, band):
        results.append(
            {
                "fraud_proba": float(p),
                "risk_score": int(rs),
                "risk_band": str(b),
                "reasons": [],
                "model_used": "xgb_full_bundle",
                "thresholds": {"medium": full_t_med, "high": full_t_high},
            }
        )
    return results


# ----------------------------
# Interactive scoring helpers
# ----------------------------
def score_interactive_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    reasons: List[str] = []

    amt = _safe_float(payload.get("TransactionAmt"))
    tx_dt = _safe_float(payload.get("TransactionDT"))
    product = _norm(payload.get("ProductCD"))
    card_brand = _norm(payload.get("card4"))
    card_type = _norm(payload.get("card6"))
    p_email = _norm(payload.get("P_emaildomain"))
    r_email = _norm(payload.get("R_emaildomain"))
    device_type = _norm(payload.get("DeviceType"))
    device_info = _norm(payload.get("DeviceInfo"))
    os_name = _norm(payload.get("id_30"))
    browser = _norm(payload.get("id_31"))
    addr1 = _safe_int(payload.get("addr1"))
    addr2 = _safe_int(payload.get("addr2"))

    if amt is None:
        score += 12
        reasons.append("Amount missing or invalid")
    elif amt < 5:
        score += 3
        reasons.append("Very small amount")
    elif amt < 50:
        score += 8
        reasons.append("Low amount")
    elif amt < 200:
        score += 16
        reasons.append("Regular amount")
    elif amt < 500:
        score += 30
        reasons.append("Moderately high amount")
    elif amt < 1500:
        score += 48
        reasons.append("High amount")
    else:
        score += 62
        reasons.append("Very high amount")

    if tx_dt is None:
        score += 6
        reasons.append("Transaction time missing")
    else:
        dt_hour = int((tx_dt / 3600) % 24)
        if dt_hour in {0, 1, 2, 3, 4, 5}:
            score += 10
            reasons.append("Night-time transaction pattern")
        elif dt_hour in {22, 23}:
            score += 6
            reasons.append("Late-hour transaction pattern")

    product_weights = {"w": 4, "c": 12, "r": 14, "h": 18, "s": 8}
    if product:
        score += product_weights.get(product, 9)
        if product in {"c", "r", "h"}:
            reasons.append(f"Higher-risk product type: {product.upper()}")
    else:
        score += 8
        reasons.append("Product type missing")

    if card_brand in {"visa", "mastercard", "american express", "discover"}:
        score += 4
    elif _is_missing_like(card_brand):
        score += 9
        reasons.append("Card brand missing")
    else:
        score += 7
        reasons.append("Uncommon card brand")

    if card_type == "debit":
        score += 2
    elif card_type == "credit":
        score += 7
    elif _is_missing_like(card_type):
        score += 10
        reasons.append("Card type missing")
    else:
        score += 9
        reasons.append("Uncommon card type")

    if _is_missing_like(p_email):
        score += 12
        reasons.append("Purchaser email domain missing")
    elif p_email not in COMMON_EMAIL_DOMAINS:
        score += 7
        reasons.append("Uncommon purchaser email domain")
    else:
        score -= 3

    if _is_missing_like(r_email):
        score += 7
        reasons.append("Recipient email domain missing")
    elif r_email not in COMMON_EMAIL_DOMAINS:
        score += 5
        reasons.append("Uncommon recipient email domain")

    if p_email and r_email and not _is_missing_like(p_email) and not _is_missing_like(r_email) and p_email != r_email:
        score += 14
        reasons.append("Purchaser and recipient email domains differ")
    elif p_email and r_email and p_email == r_email:
        score -= 4

    if device_type == "mobile":
        score += 8
        reasons.append("Mobile device transaction")
    elif device_type == "desktop":
        score += 3
    else:
        score += 7
        reasons.append("Device type missing")

    suspicious_tokens = ["headless", "selenium", "bot", "crawl", "python", "curl", "vm", "emulator"]
    if _is_missing_like(device_info):
        score += 10
        reasons.append("Device info missing")
    elif any(tok in device_info for tok in suspicious_tokens):
        score += 20
        reasons.append("Suspicious device fingerprint")
    elif any(tok in device_info for tok in ["iphone", "android", "windows", "mac", "samsung"]):
        score -= 2

    if os_name and browser:
        if "ios" in os_name and "chrome" in browser:
            score += 4
            reasons.append("OS/browser combination worth step-up verification")
        if "windows" in os_name and device_type == "mobile":
            score += 10
            reasons.append("Mobile device type conflicts with OS")
    else:
        score += 6
        reasons.append("OS or browser missing")

    if addr1 is None and addr2 is None:
        score += 7
        reasons.append("Address identifiers missing")
    elif addr1 is None or addr2 is None:
        score += 4
        reasons.append("One address identifier missing")

    score = max(0, min(int(round(score)), 100))
    band = "HIGH" if score >= INTERACTIVE_T_HIGH else ("MEDIUM" if score >= INTERACTIVE_T_MED else "LOW")
    proba = round(score / 100.0, 6)

    return {
        "fraud_proba": proba,
        "risk_score": score,
        "risk_band": band,
        "reasons": reasons[:8],
        "model_used": "interactive_rules_v1",
        "thresholds": {"medium": INTERACTIVE_T_MED / 100.0, "high": INTERACTIVE_T_HIGH / 100.0},
    }


def score_interactive_payloads(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [score_interactive_payload(p) for p in payloads]


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/score")
def score(tx: Transaction):
    _validate_single_payload(tx.payload)

    if is_interactive_payload(tx.payload):
        out = score_interactive_payload(tx.payload)
    else:
        out = score_full_payloads([tx.payload])[0]

    insert_score(tx.payload, out["fraud_proba"], out["risk_score"], out["risk_band"])
    return out


@app.post("/score_batch")
def score_batch(req: TransactionBatch):
    _validate_batch_payloads(req.payloads)

    if all(is_interactive_payload(p) for p in req.payloads):
        results = score_interactive_payloads(req.payloads)
    else:
        results = score_full_payloads(req.payloads)

    bands = np.array([r["risk_band"] for r in results], dtype=object)
    probas = np.array([r["fraud_proba"] for r in results], dtype=float)

    counts = {
        "HIGH": int((bands == "HIGH").sum()),
        "MEDIUM": int((bands == "MEDIUM").sum()),
        "LOW": int((bands == "LOW").sum()),
        "total": int(len(bands)),
    }

    top_idx = np.argsort(-probas)[: min(10, len(probas))]
    top10 = [results[i] for i in top_idx]

    return {"counts": counts, "top10": top10, "results": results}


@app.get("/admin/recent")
def admin_recent(limit: int = 50):
    limit = max(1, min(int(limit), 500))
    rows = fetch_recent(limit=limit)
    out = []
    for created_at, payload_json, fraud_proba, risk_score, risk_band in rows:
        out.append(
            {
                "created_at": created_at,
                "payload": json.loads(payload_json),
                "fraud_proba": float(fraud_proba),
                "risk_score": int(risk_score),
                "risk_band": str(risk_band),
            }
        )
    return {"items": out}
