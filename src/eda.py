import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRAIN_PATH, TEST_PATH, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def perform_eda():
    logging.info("Starting Exploratory Data Analysis (EDA)...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Training set: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"Test set:     {test.shape[0]} rows, {test.shape[1]} columns")
    print(f"Missing values in train: {train.isnull().sum().sum()}")
    print(f"Missing values in test:  {test.isnull().sum().sum()}")

    print("\n" + "=" * 60)
    print("TARGET CLASS DISTRIBUTION (TRAIN)")
    print("=" * 60)
    counts = train[TARGET_COL].value_counts()
    percentages = train[TARGET_COL].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})
    print(summary.to_string())

    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURE COMPARISON")
    print("=" * 60)
    for col in ["proto", "service"]:
        train_unique = set(train[col].dropna())
        test_unique = set(test[col].dropna())
        print(f"\nFeature: '{col}'")
        print(f"  Train categories ({len(train_unique)}): {sorted(list(train_unique))}")
        print(f"  Test categories  ({len(test_unique)}): {sorted(list(test_unique))}")
        unseen = test_unique - train_unique
        if unseen:
            print(f"  [!] Unseen categories in test: {unseen}")
        else:
            print("  All test categories exist in train.")

    print("\n" + "=" * 60)
    print("BEHAVIORAL PROFILES ACROSS ATTACK CLASSES (MEDIAN VALUES)")
    print("=" * 60)
    behavior_cols = [
        "flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "flow_pkts_per_sec",
        "flow_SYN_flag_count", "flow_RST_flag_count", "flow_ACK_flag_count",
        "fwd_pkts_payload.tot", "bwd_pkts_payload.tot", "fwd_init_window_size"
    ]
    medians = train.groupby(TARGET_COL)[behavior_cols].median().T
    print(medians.to_string())

    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)

if __name__ == "__main__":
    perform_eda()
