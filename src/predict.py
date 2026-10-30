import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CLASSES,
    IDX_TO_LABEL,
    MODELS_DIR,
    OUTPUT_SUBMISSION_PATH,
    TEST_PATH,
)
from src.features import engineer_features
from src.preprocess import clean_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def predict_pipeline(
    test_csv_path: Path = TEST_PATH,
    output_path: Path = OUTPUT_SUBMISSION_PATH,
) -> pd.DataFrame:
    """
    Loads trained fold models, processes test data, and generates predictions.
    """
    logging.info(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    test_ids = test_df["id"].values

    # Preprocessing & Feature Engineering
    logging.info("Preprocessing and extracting features...")
    test_clean = clean_data(test_df)
    test_feats = engineer_features(test_clean)

    # Load feature columns and label mapping
    feat_path = MODELS_DIR / "feature_columns.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature columns metadata not found at {feat_path}. Train the model first.")
    
    with open(feat_path, "r") as f:
        feature_cols = json.load(f)

    with open(MODELS_DIR / "label_mapping.json", "r") as f:
        mapping = json.load(f)
        idx_to_label = {int(k): v for k, v in mapping["idx_to_label"].items()}

    # Encode categoricals
    cat_cols = ["proto", "service"]
    for col in cat_cols:
        test_feats[f"{col}_enc"] = test_feats[col].astype("category").cat.codes

    # Ensure all expected feature columns exist
    for col in feature_cols:
        if col not in test_feats.columns:
            test_feats[col] = 0.0

    X_test = test_feats[feature_cols].copy()

    # Discover and load all fold models
    model_files = sorted(list(MODELS_DIR.glob("xgb_fold_*.json")))
    if not model_files:
        raise FileNotFoundError(f"No fold models found in {MODELS_DIR}. Please run training first.")

    logging.info(f"Found {len(model_files)} fold models. Performing ensemble inference...")
    test_probs = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)

    for mf in model_files:
        logging.info(f"Inference with {mf.name}...")
        booster = xgb.Booster()
        booster.load_model(str(mf))
        dtest = xgb.DMatrix(X_test)
        fold_probs = booster.predict(dtest)
        test_probs += fold_probs / len(model_files)

    # Generate predictions
    pred_indices = np.argmax(test_probs, axis=1)
    pred_labels = [idx_to_label[i] for i in pred_indices]

    submission_df = pd.DataFrame({
        "id": test_ids,
        "Target": pred_labels,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    logging.info(f"Submission successfully written to {output_path}")
    logging.info(f"Summary of predicted classes:\n{submission_df['Target'].value_counts()}")

    return submission_df


if __name__ == "__main__":
    predict_pipeline()
