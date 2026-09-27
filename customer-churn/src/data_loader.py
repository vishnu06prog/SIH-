"""
data_loader.py
Loads the IBM Telco Customer Churn CSV and does the *basic cleaning* that is
safe to do BEFORE the train/test split (no statistics are learned here).

What happens here:
  1. Read the CSV with Pandas.
  2. Convert TotalCharges from text to a number (11 rows contain a blank " ").
  3. Drop customerID (an ID carries no predictive information).
  4. Convert the target Churn from "Yes"/"No" to 1/0.

Missing values are NOT filled here. They are filled inside the Scikit-learn
pipeline (preprocessing.py) so the fill value is learned from training data only.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "Churn"

# Numerical columns -> will be imputed (median) and scaled (StandardScaler)
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Categorical columns -> will be imputed (most frequent) and one-hot encoded
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_raw_data(path=DATA_PATH):
    """Read the CSV exactly as it is on disk (used for EDA / inspection)."""
    return pd.read_csv(path)


def clean_data(df):
    """Basic, leakage-free cleaning. Returns a new DataFrame."""
    df = df.copy()

    # TotalCharges is stored as text because 11 customers have " " (blank).
    # errors="coerce" turns anything that is not a number into NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # customerID is unique per row -> useless for prediction, drop it.
    df = df.drop(columns=["customerID"])

    # Target: "Yes" -> 1 (churned), "No" -> 0 (stayed)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


def load_clean_data(path=DATA_PATH):
    return clean_data(load_raw_data(path))


def get_features_and_target(df):
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def describe_dataset(df):
    """Print the basic facts an interviewer will ask about."""
    print("Shape (rows, columns):", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column (NaN):\n", df.isna().sum()[df.isna().sum() > 0])
    if df["TotalCharges"].dtype == object or str(df["TotalCharges"].dtype) == "str":
        blanks = (df["TotalCharges"].astype(str).str.strip() == "").sum()
        print("Blank strings in TotalCharges:", blanks)
    print("\nDuplicate rows (all columns):", df.duplicated().sum())
    if "customerID" in df.columns:
        print("Duplicate rows ignoring customerID:",
              df.drop(columns="customerID").duplicated().sum())
    print("\nClass distribution:\n", df[TARGET].value_counts())
    print("\nClass distribution (%):\n", (df[TARGET].value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    raw = load_raw_data()
    print("=== RAW DATA ===")
    describe_dataset(raw)
    clean = clean_data(raw)
    print("\n=== AFTER CLEANING ===")
    print("Shape:", clean.shape)
    print("Missing values:\n", clean.isna().sum()[clean.isna().sum() > 0])
    print("Rows with missing TotalCharges have tenure =",
          clean.loc[clean["TotalCharges"].isna(), "tenure"].unique())
