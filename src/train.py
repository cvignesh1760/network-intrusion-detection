import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CLASSES,
    DROP_COLS,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    MODELS_DIR,
    N_SPLITS,
    OUTPUT_SUBMISSION_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
)
from src.features import engineer_features
from src.preprocess import clean_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def train_pipeline():
    logging.info("Starting Network Intrusion Detection Training Pipeline...")

    # 1. Load Data
    logging.info(f"Loading training data from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    logging.info(f"Train data shape: {train_df.shape}")

    logging.info(f"Loading test data from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)
    logging.info(f"Test data shape: {test_df.shape}")

    # 2. Preprocess & Feature Engineering
    logging.info("Preprocessing and engineering features...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    train_feats = engineer_features(train_df)
    test_feats = engineer_features(test_df)

    # 3. Categorical Encoding (Label encoding across train+test)
    cat_cols = ["proto", "service"]
    for col in cat_cols:
        all_cats = sorted(
            list(set(train_feats[col].astype(str)).union(set(test_feats[col].astype(str))))
        )
        cat_map = {cat: idx for idx, cat in enumerate(all_cats)}
        train_feats[f"{col}_enc"] = train_feats[col].astype(str).map(cat_map).fillna(-1)
        test_feats[f"{col}_enc"] = test_feats[col].astype(str).map(cat_map).fillna(-1)

    # Prepare X, y, and X_test
    feature_cols = [
        col
        for col in train_feats.columns
        if col not in DROP_COLS and col not in cat_cols
    ]
    logging.info(f"Total features utilized for modeling: {len(feature_cols)}")

    X = train_feats[feature_cols].copy()
    y = train_feats[TARGET_COL].map(LABEL_TO_IDX).values
    X_test = test_feats[feature_cols].copy()

    # Save feature names and label mappings
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "label_mapping.json", "w") as f:
        json.dump(
            {"label_to_idx": LABEL_TO_IDX, "idx_to_label": IDX_TO_LABEL}, f, indent=2
        )

    # Compute balanced class weights for Macro F1 optimization
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([weight_dict[c] for c in y])

    logging.info("Computed class weights for balanced Macro-F1 optimization:")
    for c in classes:
        logging.info(f"  Class {c} ({IDX_TO_LABEL[c]}): {weight_dict[c]:.4f}")

    # 4. Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
    test_probs = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)

    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logging.info(f"--- Training Fold {fold + 1}/{N_SPLITS} ---")
        X_train, y_train, w_train = (
            X.iloc[train_idx],
            y[train_idx],
            sample_weights[train_idx],
        )
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predict Out-Of-Fold probabilities
        val_probs = model.predict_proba(X_val)
        oof_probs[val_idx] = val_probs

        val_preds = np.argmax(val_probs, axis=1)
        fold_f1 = f1_score(y_val, val_preds, average="macro")
        fold_scores.append(fold_f1)
        logging.info(f"Fold {fold + 1} Macro F1: {fold_f1:.4f}")

        # Accumulate test predictions (ensemble average)
        test_probs += model.predict_proba(X_test) / N_SPLITS

        # Save model
        model_path = MODELS_DIR / f"xgb_fold_{fold}.json"
        model.save_model(str(model_path))
        models.append(model)

    # 5. Out-of-Fold Evaluation
    oof_preds = np.argmax(oof_probs, axis=1)
    overall_macro_f1 = f1_score(y, oof_preds, average="macro")
    logging.info(f"==================================================")
    logging.info(f"Mean Fold Macro F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    logging.info(f"Overall OOF Macro F1 Score: {overall_macro_f1:.4f}")
    logging.info(f"==================================================")
    logging.info("\nDetailed Out-of-Fold Classification Report:\n" + classification_report(
        y, oof_preds, target_names=[IDX_TO_LABEL[i] for i in range(len(CLASSES))], digits=4
    ))

    # Save OOF predictions
    oof_df = pd.DataFrame(
        oof_probs, columns=[f"prob_{c}" for c in CLASSES]
    )
    oof_df["id"] = train_df["id"].values
    oof_df["target_true"] = [IDX_TO_LABEL[i] for i in y]
    oof_df["target_pred"] = [IDX_TO_LABEL[i] for i in oof_preds]
    oof_df.to_csv(MODELS_DIR / "oof_predictions.csv", index=False)
    logging.info(f"Saved OOF predictions to {MODELS_DIR / 'oof_predictions.csv'}")

    # 6. Generate Test Submission
    logging.info("Generating predictions for test set...")
    test_pred_indices = np.argmax(test_probs, axis=1)
    submission_df = pd.DataFrame({
        "id": test_df["id"].values,
        "Target": [IDX_TO_LABEL[idx] for idx in test_pred_indices],
    })

    OUTPUT_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    logging.info(f"Submission saved successfully to {OUTPUT_SUBMISSION_PATH}")

    # Submission validation check
    logging.info("Validating submission:")
    logging.info(f"  Shape: {submission_df.shape} (Expected: (33161, 2))")
    logging.info(f"  Columns: {list(submission_df.columns)} (Expected: ['id', 'Target'])")
    logging.info(f"  Null count: {submission_df.isnull().sum().sum()}")
    logging.info("  Predicted class distribution in test set:")
    for target_val, count in submission_df["Target"].value_counts().items():
        logging.info(f"    {target_val}: {count} ({count / len(submission_df) * 100:.2f}%)")

    return overall_macro_f1


if __name__ == "__main__":
    train_pipeline()
