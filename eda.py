"""Exploratory Data Analysis for Kaggle's Ames House Prices dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Kaggle files are missing. Put train.csv and test.csv in data/raw/."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("\n========== AMES HOUSE PRICES: EDA ==========")
    print(f"Training shape: {train.shape} (rows, columns)")
    print(f"Testing shape:  {test.shape} (rows, columns)")
    print("\nFirst five training rows:")
    print(train.head().to_string(index=False))

    print("\nData types and non-null counts:")
    train.info()

    print("\nTarget variable (SalePrice) statistics:")
    print(train["SalePrice"].describe())

    print("\nMissing-value counts (largest first):")
    missing = train.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(30).to_string())

    missing_report = pd.DataFrame({
        "missing_count": train.isna().sum(),
        "missing_percent": (train.isna().mean() * 100).round(2),
        "dtype": train.dtypes.astype(str),
    }).sort_values("missing_percent", ascending=False)
    missing_report.to_csv(OUTPUT_DIR / "missing_values_report.csv")
    train.describe(include="all").transpose().to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv"
    )

    print("\nDuplicate ID count:", int(train["Id"].duplicated().sum()))
    print("Duplicate complete-row count:", int(train.duplicated().sum()))
    print("SalePrice is missing:", bool(train["SalePrice"].isna().any()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(train["SalePrice"], bins=35, edgecolor="black", alpha=0.8)
    axes[0].set_title("SalePrice distribution")
    axes[0].set_xlabel("SalePrice (USD)")
    axes[0].set_ylabel("Number of houses")
    log_target = __import__("numpy").log1p(train["SalePrice"])
    axes[1].hist(log_target, bins=35, edgecolor="black", alpha=0.8)
    axes[1].set_title("log1p(SalePrice) distribution")
    axes[1].set_xlabel("log1p(SalePrice)")
    axes[1].set_ylabel("Number of houses")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "saleprice_distributions.png", dpi=160)
    plt.close(fig)

    numeric = train.select_dtypes(include="number").drop(columns=["Id"], errors="ignore")
    corr = numeric.corr(numeric_only=True)
    top_corr = corr["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(15)
    top_corr.to_csv(OUTPUT_DIR / "top_numeric_correlations.csv", header=["absolute_correlation"])

    selected_corr = corr.loc[top_corr.index.tolist() + ["SalePrice"], top_corr.index.tolist() + ["SalePrice"]]
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(selected_corr.to_numpy(), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(selected_corr.columns)), selected_corr.columns, rotation=90)
    ax.set_yticks(range(len(selected_corr.index)), selected_corr.index)
    ax.set_title("Correlation of top numerical features with SalePrice")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    top_missing = missing_report[missing_report["missing_count"] > 0].head(20)
    if not top_missing.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_missing.index[::-1], top_missing["missing_percent"].iloc[::-1])
        ax.set_title("Top columns by percentage of missing values")
        ax.set_xlabel("Missing values (%)")
        ax.set_ylabel("Column")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=160)
        plt.close(fig)

    print("\nSaved EDA reports and charts in:", OUTPUT_DIR)
    print("EDA completed successfully.")


if __name__ == "__main__":
    main()
