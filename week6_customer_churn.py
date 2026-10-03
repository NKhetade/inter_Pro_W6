"""Week 6 capstone runner. Full workflow including deep learning is in the notebook."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Inspect the Telco Churn dataset.")
    parser.add_argument("--data", default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(15))
    print("\nTarget distribution:\n", df["Churn"].value_counts(dropna=False))
    print("\nRun Week_6_Customer_Churn_Capstone.ipynb for the full analysis, clustering, model training, and evaluation.")

if __name__ == "__main__":
    main()
