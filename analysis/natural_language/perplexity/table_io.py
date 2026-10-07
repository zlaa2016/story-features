from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists() and path.suffix.lower() == ".parquet":
        alt = path.with_suffix(".csv.gz")
        if alt.exists():
            path = alt
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            alt = path.with_suffix(".csv.gz")
            if alt.exists():
                return pd.read_csv(alt, low_memory=False)
            raise
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv.gz")
    compression = "gzip" if path.name.endswith(".gz") else None
    df.to_csv(path, index=False, compression=compression)
    return path

