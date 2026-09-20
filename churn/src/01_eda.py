"""Day 1, Step 1: load the raw Telco churn CSV and inspect it before touching it.

Nothing is cleaned or dropped here on purpose. The point of this step is to see
the data exactly as it arrives, including the defects, so later cleaning steps
are justified by something observed rather than something assumed.
"""

from pathlib import Path

import pandas as pd

# Paths are derived from this file's location so the script runs from anywhere.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

pd.set_option("display.width", 120)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("SHAPE")
    print("=" * 70)
    print(f"rows: {df.shape[0]}   columns: {df.shape[1]}")

    print("\n" + "=" * 70)
    print("DTYPES")
    print("=" * 70)
    print(df.dtypes.to_string())

    print("\n" + "=" * 70)
    print("MISSING VALUES (pandas' own definition: NaN / None)")
    print("=" * 70)
    na_counts = df.isna().sum()
    print(na_counts[na_counts > 0].to_string() if na_counts.sum() else "none")

    # pandas only counts NaN as missing. A column read as text can hide missing
    # values as whitespace strings, which isna() will never flag.
    print("\n" + "=" * 70)
    print("BLANK / WHITESPACE-ONLY STRINGS IN TEXT COLUMNS")
    print("=" * 70)
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    blanks = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in text_cols}
    blanks = {c: n for c, n in blanks.items() if n > 0}
    print(pd.Series(blanks).to_string() if blanks else "none")

    print("\n" + "=" * 70)
    print("TARGET: Churn")
    print("=" * 70)
    counts = df["Churn"].value_counts()
    rates = df["Churn"].value_counts(normalize=True) * 100
    for label in counts.index:
        print(f"{label:>4}: {counts[label]:>5}  ({rates[label]:.2f}%)")
    print(f"\nmajority-class accuracy if we always predict 'No': {rates.max():.2f}%")


if __name__ == "__main__":
    main()
