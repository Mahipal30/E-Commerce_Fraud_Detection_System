"""Join transaction + identity into unified train/test datasets."""

import pandas as pd
from src.config import DATA_INTERIM, DATA_PROCESSED, ID_COL, TARGET_COL

def load_join(split: str) -> pd.DataFrame:
    tx = pd.read_parquet(DATA_INTERIM / f"{split}_transaction.parquet")
    idn = pd.read_parquet(DATA_INTERIM / f"{split}_identity.parquet")
    df = tx.merge(idn, on=ID_COL, how="left")
    return df

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    train = load_join("train")
    test = load_join("test")

    y = train[TARGET_COL].astype("int8")
    X = train.drop(columns=[TARGET_COL])

    X.to_parquet(DATA_PROCESSED / "X_train.parquet", index=False)
    y.to_frame(TARGET_COL).to_parquet(DATA_PROCESSED / "y_train.parquet", index=False)
    test.to_parquet(DATA_PROCESSED / "X_test.parquet", index=False)

    print("Saved:")
    print(" - data/processed/X_train.parquet")
    print(" - data/processed/y_train.parquet")
    print(" - data/processed/X_test.parquet")

if __name__ == "__main__":
    main()
