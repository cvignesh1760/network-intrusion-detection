import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"

# Target Definition
TARGET_COL = "Target"
ID_COL = "id"

CLASSES = ["Benign", "Availability", "Reconnaissance", "Spoofing"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Categorical columns
CATEGORICAL_COLS = ["proto", "service"]

# Columns to drop during modeling
DROP_COLS = [ID_COL, TARGET_COL]

# Random Seed & CV
RANDOM_STATE = 42
N_SPLITS = 5
