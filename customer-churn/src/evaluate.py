"""
evaluate.py
Metrics and plots used to judge the model on the TEST set.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.data_loader import FIGURES_DIR


def compute_metrics(y_true, y_prob, threshold=0.5):
    """y_prob = predicted churn probabilities; threshold turns them into 0/1."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),  # uses probabilities, not 0/1
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_report(name, y_true, y_prob, threshold=0.5):
    m = compute_metrics(y_true, y_prob, threshold)
    tn, fp, fn, tp = m["confusion_matrix"].ravel()
    print(f"\n===== {name} (threshold = {threshold}) =====")
    print(f"Accuracy : {m['accuracy']:.3f}")
    print(f"Precision: {m['precision']:.3f}   (of customers we flagged, how many really churned)")
    print(f"Recall   : {m['recall']:.3f}   (of customers who really churned, how many we caught)")
    print(f"F1-score : {m['f1']:.3f}")
    print(f"ROC-AUC  : {m['roc_auc']:.3f}")
    print(f"Confusion matrix -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(classification_report(y_true, (y_prob >= threshold).astype(int),
                                target_names=["Stayed", "Churned"], digits=3))
    return m


def threshold_table(y_true, y_prob, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Shows the precision/recall trade-off. Computed on VALIDATION data."""
    print("\nThreshold | Precision | Recall | F1    | Customers flagged")
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, t)
        flagged = int((y_prob >= t).sum())
        print(f"   {t:.1f}    |  {m['precision']:.3f}    | {m['recall']:.3f}  | {m['f1']:.3f} | {flagged}")


def plot_confusion_matrix(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_confusion_matrix.png", dpi=110)
    plt.close(fig)


def plot_roc_curve(y_true, dnn_prob, baseline_prob=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, dnn_prob, name="DNN", ax=ax)
    if baseline_prob is not None:
        RocCurveDisplay.from_predictions(y_true, baseline_prob, name="Logistic Regression", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
    ax.legend(loc="lower right")
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_roc_curve.png", dpi=110)
    plt.close(fig)


def plot_training_history(history):
    h = history.history
    epochs = np.arange(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, h["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, h["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, h["loss"], label="Training loss")
    axes[1].plot(epochs, h["val_loss"], label="Validation loss")
    best = int(np.argmin(h["val_loss"])) + 1
    axes[1].axvline(best, color="grey", linestyle="--", label=f"Best epoch ({best})")
    axes[1].set_title("Loss (binary cross-entropy) per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_training_history.png", dpi=110)
    plt.close(fig)
