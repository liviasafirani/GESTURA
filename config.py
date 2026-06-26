"""
config.py

Global configuration for BISINDO Detection System.
"""

from pathlib import Path
from sympy import false
import torch

# ==========================================================
# PROJECT
# ==========================================================

PROJECT_NAME = "BISINDO Detection"
PROJECT_VERSION = "2.0"

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

NUMBER_DATASET_DIR = DATASET_DIR / "numbers"
ALPHABET_DATASET_DIR = DATASET_DIR / "alphabets"
WORD_DATASET_DIR = DATASET_DIR / "words"

PROCESSED_DATASET_DIR = BASE_DIR / "processed"

MODEL_SAVE_DIR = BASE_DIR / "saved_models"

LOG_DIR = BASE_DIR / "logs"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# ==========================================================
# MODEL FILE
# ==========================================================

NUMBER_MODEL = MODEL_SAVE_DIR / "number_model.pth"
ALPHABET_MODEL = MODEL_SAVE_DIR / "alphabet_model.pth"
WORD_MODEL = MODEL_SAVE_DIR / "word_model.pth"

# ==========================================================
# CLASSES
# ==========================================================

NUMBER_CLASSES = [str(i) for i in range(11)]

ALPHABET_CLASSES = [
    chr(i)
    for i in range(ord("A"), ord("Z") + 1)
]

WORD_CLASSES = [
    "bisa",
    "contoh",
    "malu",
    "mau",
    "untuk"
]

# ==========================================================
# IMAGE
# ==========================================================

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_SIZE = (
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)

VALID_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# ==========================================================
# CAMERA
# ==========================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ==========================================================
# DATA CAPTURE
# ==========================================================

COUNTDOWN_SECONDS = 3

CAPTURE_FPS = 2
CAPTURE_INTERVAL = 1 / CAPTURE_FPS

SEQUENCE_LENGTH = 30

SEQUENCE_FPS = 15
SEQUENCE_INTERVAL = 1 / SEQUENCE_FPS

# ==========================================================
# DATASET SPLIT
# ==========================================================

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PIN_MEMORY = false

SAVE_BEST_ONLY = True

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

# ==========================================================
# INFERENCE
# ==========================================================

CONFIDENCE_THRESHOLD = 0.80

# ==========================================================
# RANDOM
# ==========================================================

SEED = 42