import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def evaluate_oof():
    oof_path = MODELS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        logging.error(f"OOF predictions file not found at {oof_path}. Run training first.")
        return

    df = pd.read_csv(oof_path)
    y_true = df["target_true"]
    y_pred = df["target_pred"]

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 60)
    print(f"Overall Out-of-Fold Macro F1:    {macro_f1:.4f}")
    print(f"Overall Out-of-Fold Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in CLASSES], columns=[f"Pred: {c}" for c in CLASSES])
    print(cm_df.to_string())

if __name__ == "__main__":
    evaluate_oof()
