"""
eda.py
Exploratory Data Analysis with Pandas, Seaborn and Matplotlib.
Every function saves one figure to reports/figures/ and returns the numbers
behind it, so the conclusion is based on data and not only on the picture.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import FIGURES_DIR, load_clean_data

sns.set_theme(style="whitegrid")
CHURN_LABELS = {0: "Stayed", 1: "Churned"}


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=110)
    plt.close(fig)


def _with_labels(df):
    out = df.copy()
    out["ChurnLabel"] = out["Churn"].map(CHURN_LABELS)
    return out


def plot_churn_distribution(df):
    counts = df["Churn"].map(CHURN_LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=df["Churn"].map(CHURN_LABELS), order=["Stayed", "Churned"], ax=ax)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}\n({p.get_height() / len(df):.1%})",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Churn distribution")
    ax.set_xlabel("")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, "01_churn_distribution.png")
    return counts


def churn_rate_by(df, column):
    """% of customers who churned inside each category of `column`."""
    return (df.groupby(column)["Churn"].mean() * 100).round(1).sort_values(ascending=False)


def plot_churn_rate_by_category(df, column, filename):
    rates = churn_rate_by(df, column)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=rates.index.astype(str), y=rates.values, ax=ax, color="steelblue")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center")
    ax.set_title(f"Churn rate by {column}")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(column)
    ax.set_ylim(0, max(rates.values) * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, filename)
    return rates


def plot_tenure(df):
    d = _with_labels(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=d, x="tenure", hue="ChurnLabel", bins=36, multiple="stack", ax=axes[0])
    axes[0].set_title("Tenure (months) by churn")
    sns.boxplot(data=d, x="ChurnLabel", y="tenure", order=["Stayed", "Churned"], ax=axes[1])
    axes[1].set_title("Tenure boxplot")
    axes[1].set_xlabel("")
    _save(fig, "03_churn_vs_tenure.png")

    bands = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                   labels=["0-12", "13-24", "25-48", "49-72"])
    return {
        "median_tenure": df.groupby("Churn")["tenure"].median().rename(CHURN_LABELS),
        "churn_rate_by_tenure_band_%": (df.groupby(bands, observed=True)["Churn"].mean() * 100).round(1),
    }


def plot_monthly_charges(df):
    d = _with_labels(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(data=d, x="MonthlyCharges", hue="ChurnLabel", common_norm=False, fill=True, ax=axes[0])
    axes[0].set_title("Monthly charges density by churn")
    sns.boxplot(data=d, x="ChurnLabel", y="MonthlyCharges", order=["Stayed", "Churned"], ax=axes[1])
    axes[1].set_title("Monthly charges boxplot")
    axes[1].set_xlabel("")
    _save(fig, "04_churn_vs_monthly_charges.png")
    return df.groupby("Churn")["MonthlyCharges"].median().rename(CHURN_LABELS)


def plot_total_charges(df):
    d = _with_labels(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=d, x="ChurnLabel", y="TotalCharges", order=["Stayed", "Churned"], ax=ax)
    ax.set_title("Total charges by churn")
    ax.set_xlabel("")
    _save(fig, "06_churn_vs_total_charges.png")
    return df.groupby("Churn")["TotalCharges"].median().rename(CHURN_LABELS)


def plot_correlation_heatmap(df):
    corr = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation (numerical features + Churn)")
    _save(fig, "07_correlation_heatmap.png")
    return corr.round(2)


def plot_services(df):
    services = ["InternetService", "OnlineSecurity", "TechSupport", "OnlineBackup",
                "DeviceProtection", "StreamingTV"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    results = {}
    for ax, col in zip(axes.ravel(), services):
        rates = churn_rate_by(df, col)
        results[col] = rates
        sns.barplot(x=rates.index.astype(str), y=rates.values, ax=ax, color="indianred")
        ax.set_title(col)
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=10, fontsize=8)
    _save(fig, "08_churn_vs_services.png")
    return results


def run_all():
    df = load_clean_data()
    results = {
        "churn_distribution": plot_churn_distribution(df),
        "contract": plot_churn_rate_by_category(df, "Contract", "02_churn_vs_contract.png"),
        "tenure": plot_tenure(df),
        "monthly_charges_median": plot_monthly_charges(df),
        "payment_method": plot_churn_rate_by_category(df, "PaymentMethod", "05_churn_vs_payment_method.png"),
        "total_charges_median": plot_total_charges(df),
        "correlation": plot_correlation_heatmap(df),
        "services": plot_services(df),
    }
    return results


if __name__ == "__main__":
    for name, value in run_all().items():
        print(f"\n===== {name} =====")
        print(value)
    print(f"\nFigures saved to {FIGURES_DIR}")
