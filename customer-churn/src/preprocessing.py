"""
preprocessing.py
Builds the Scikit-learn preprocessing pipeline.

    raw customer columns
          |
    ColumnTransformer
      |-- numerical   : SimpleImputer(median)        -> StandardScaler
      |-- categorical : SimpleImputer(most_frequent) -> OneHotEncoder
          |
    one numeric matrix (all numbers, similar scale) -> neural network

IMPORTANT: this object is fit ONLY on the training set (see train.py).
Validation, test and new customers are only *transformed* with it.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        # Fills the 11 missing TotalCharges with the TRAINING median
        ("imputer", SimpleImputer(strategy="median")),
        # (value - mean) / std  -> mean 0, std 1, using TRAINING mean/std
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        # No categorical NaNs in this dataset, but keeps prediction safe
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # "Month-to-month" -> [1, 0, 0]; unseen categories become all zeros
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_feature_names(fitted_preprocessor):
    """Names of the columns produced after one-hot encoding (useful to show)."""
    return list(fitted_preprocessor.get_feature_names_out())
