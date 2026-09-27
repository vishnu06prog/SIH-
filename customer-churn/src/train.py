"""
train.py
End-to-end training script.

Steps:
  1. Load + clean data                         (data_loader.py)
  2. Split: 64% train / 16% validation / 20% test  (stratified by Churn)
  3. Fit the preprocessor on TRAIN only, transform val and test
  4. Compute class weights from TRAIN labels   (class imbalance)
  5. Train the DNN with validation data + EarlyStopping
  6. Train a Logistic Regression baseline on the same features
  7. Evaluate both on the untouched TEST set, save plots
  8. Save the fitted preprocessor + model to models/

Run:  python -m src.train
"""
import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.data_loader import FIGURES_DIR, MODELS_DIR, get_features_and_target, load_clean_data
from src.evaluate import (
    plot_confusion_matrix, plot_roc_curve, plot_training_history, print_report, threshold_table,
)
from src.model import build_model
from src.preprocessing import build_preprocessor, get_feature_names

SEED = 42
EPOCHS = 100          # upper limit; EarlyStopping usually stops much earlier
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def main():
    tf.keras.utils.set_random_seed(SEED)  # sets Python, NumPy and TF seeds
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Data ----------
    df = load_clean_data()
    X, y = get_features_and_target(df)

    # ---------- 2. Split (BEFORE any fitting) ----------
    # stratify=y keeps ~26.5% churners in every split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=SEED)
    print(f"Train: {X_train.shape}  Validation: {X_val.shape}  Test: {X_test.shape}")
    print(f"Churn rate -> train {y_train.mean():.3f} | val {y_val.mean():.3f} | test {y_test.mean():.3f}")

    # ---------- 3. Preprocessing: fit on TRAIN only ----------
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)   # learns median, mean/std, categories
    X_val_p = preprocessor.transform(X_val)           # only applies what was learned
    X_test_p = preprocessor.transform(X_test)
    X_train_p, X_val_p, X_test_p = (np.asarray(a, dtype="float32") for a in (X_train_p, X_val_p, X_test_p))
    print(f"Features after preprocessing: {X_train_p.shape[1]}")
    print("First 10 feature names:", get_feature_names(preprocessor)[:10])

    # ---------- 4. Class imbalance -> class weights ----------
    # balanced weight = n_samples / (n_classes * count_of_class)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Class weights: {class_weight}")

    # ---------- 5. Train the DNN ----------
    model = build_model(n_features=X_train_p.shape[1], learning_rate=LEARNING_RATE)
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",          # watch loss on validation data
        patience=10,                 # stop if no improvement for 10 epochs
        restore_best_weights=True,   # go back to the best epoch's weights
    )
    history = model.fit(
        X_train_p, y_train.values,
        validation_data=(X_val_p, y_val.values),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=2,
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nTrained for {len(history.history['loss'])} epochs; best epoch (lowest val_loss) = {best_epoch}")
    plot_training_history(history)

    # ---------- 6. Baseline: Logistic Regression ----------
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_train_p, y_train)

    # ---------- 7. Evaluate ----------
    val_prob = model.predict(X_val_p, verbose=0).ravel()
    print("\nPrecision/recall trade-off on the VALIDATION set:")
    threshold_table(y_val.values, val_prob)

    dnn_test_prob = model.predict(X_test_p, verbose=0).ravel()
    base_test_prob = baseline.predict_proba(X_test_p)[:, 1]

    print_report("DNN on TEST set", y_test.values, dnn_test_prob)
    print_report("Logistic Regression baseline on TEST set", y_test.values, base_test_prob)

    # Reference point: a "model" that always says "no churn"
    print(f"Always-predict-'Stayed' accuracy on test: {1 - y_test.mean():.3f} (recall = 0.000)")

    plot_confusion_matrix(y_test.values, dnn_test_prob)
    plot_roc_curve(y_test.values, dnn_test_prob, base_test_prob)

    # ---------- 8. Save ----------
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    model.save(MODELS_DIR / "churn_dnn.keras")
    print(f"\nSaved preprocessor and model to {MODELS_DIR}")


if __name__ == "__main__":
    main()
