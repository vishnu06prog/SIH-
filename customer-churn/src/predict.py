"""
predict.py
Predict churn for ONE new customer.

Flow of a prediction:
  dict of raw values (e.g. Contract="Month-to-month", tenure=2)
    -> pandas DataFrame with 1 row
    -> clean TotalCharges (same rule as training)
    -> preprocessor.transform  (training median / mean / std / categories)
    -> model.predict           (Dense -> ReLU -> ... -> Sigmoid)
    -> probability in [0, 1]  -> compare with threshold -> class

Run:  python -m src.predict
"""
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.data_loader import CATEGORICAL_FEATURES, MODELS_DIR, NUMERICAL_FEATURES

_preprocessor = None
_model = None


def load_artifacts():
    global _preprocessor, _model
    if _model is None:
        _preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        _model = keras.models.load_model(MODELS_DIR / "churn_dnn.keras")
    return _preprocessor, _model


def predict_churn(customer, threshold=0.5):
    """customer: dict with the 19 raw feature columns. Returns (probability, class)."""
    preprocessor, model = load_artifacts()

    row = pd.DataFrame([customer])
    missing = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) - set(row.columns)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")

    X = np.asarray(preprocessor.transform(row), dtype="float32")   # transform ONLY, never fit
    probability = float(model.predict(X, verbose=0)[0, 0])
    predicted_class = int(probability >= threshold)
    return probability, predicted_class


def explain(customer, threshold=0.5):
    p, c = predict_churn(customer, threshold)
    label = "WILL CHURN" if c == 1 else "WILL STAY"
    print(f"Customer has {p:.0%} probability of churn. Predicted class: {c} ({label})")
    return p, c


if __name__ == "__main__":
    high_risk = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.70, "TotalCharges": 191.40,
    }
    low_risk = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.00, "TotalCharges": 3900.00,
    }
    print("High-risk example:")
    explain(high_risk)
    print("\nLow-risk example:")
    explain(low_risk)
