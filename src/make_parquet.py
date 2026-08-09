"""Convert the Kaggle CSVs to Parquet for faster subsequent runs."""

import pandas as pd
from src.config import DATA_RAW, DATA_INTERIM

FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]

def _read_csv(path):
    df = pd.read_csv(path, low_memory=False)
    # downcast numerics to reduce memory footprint
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df

def main():
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        p = DATA_RAW / f
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Unzip ieee-fraud-detection.zip into data/raw.")
        print(f"Reading {p.name} ...")
        df = _read_csv(p)
        out = DATA_INTERIM / f.replace(".csv", ".parquet")
        print(f"Writing {out.name} ... rows={len(df):,} cols={df.shape[1]:,}")
        df.to_parquet(out, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
