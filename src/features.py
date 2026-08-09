"""Feature engineering: minimal, defensible, and leakage-aware.

We start with safe temporal features derived from TransactionDT, and
simple frequency/velocity features based on stable identifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from src.config import TIME_COL

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TIME_COL not in out.columns:
        return out

    # TransactionDT is seconds from a reference time (competition-specific).
    # We can still derive relative time buckets safely.
    dt = out[TIME_COL].astype("float64")
    out["dt_hour"] = ((dt / 3600) % 24).astype("int16")
    out["dt_day"] = (dt / 86400).astype("int32")
    out["dt_dayofweek"] = (out["dt_day"] % 7).astype("int8")

    # Night flag (heuristic)
    out["is_night"] = out["dt_hour"].isin([0,1,2,3,4,5]).astype("int8")
    return out

def add_group_frequency(df: pd.DataFrame, group_cols: List[str], prefix: str) -> pd.DataFrame:
    out = df.copy()
    for col in group_cols:
        if col not in out.columns:
            continue
        s = out[col].astype("string")  # stable hashing of mixed types
        vc = s.value_counts(dropna=False)
        out[f"{prefix}_{col}_freq"] = s.map(vc).astype("int32")
    return out

def add_velocity_features(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """Compute per-identifier time since previous transaction (seconds) on the dataset ordering.
    This is leakage-safe only if you compute it *within* each split and preserve temporal order.
    """
    out = df.copy()
    if id_col not in out.columns or time_col not in out.columns:
        return out

    # sort by time within id
    out = out.sort_values([id_col, time_col])
    out[f"{id_col}_dt_since_prev"] = out.groupby(id_col)[time_col].diff().fillna(-1).astype("float32")
    # restore original order not necessary if pipeline keeps aligned indices; caller can reset
    return out

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)

    # Common identifier-like columns in IEEE-CIS (may not all exist in both train/test after merge)
    id_like = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    out = add_group_frequency(out, id_like, prefix="g")
    return out
