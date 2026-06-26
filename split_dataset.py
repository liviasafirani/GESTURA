"""
split_dataset.py

Split dataset menjadi:
70% Train
15% Validation
15% Test
"""

from pathlib import Path
import random
import shutil

# ==========================================================
# Config
# ==========================================================

DATASET_ROOT = Path("dataset")

DATASETS = [
    "numbers",
    "alphabets",
    "words"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

random.seed(RANDOM_SEED)


# ==========================================================
# Utility
# ==========================================================

def copy_files(files, destination):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in files:

        shutil.copy2(
            file,
            destination / file.name
        )


# ==========================================================
# Split
# ==========================================================

for dataset_name in DATASETS:

    dataset_dir = DATASET_ROOT / dataset_name

    classes = [

        folder

        for folder in dataset_dir.iterdir()

        if folder.is_dir()

        and folder.name not in (
            "train",
            "val",
            "test"
        )

    ]

    print(f"\n===== {dataset_name.upper()} =====")

    for class_dir in classes:

        files = [

            file

            for file in class_dir.iterdir()

            if file.suffix.lower() in IMAGE_EXTENSIONS

        ]

        random.shuffle(files)

        total = len(files)

        train_end = int(total * TRAIN_RATIO)

        val_end = train_end + int(total * VAL_RATIO)

        train_files = files[:train_end]

        val_files = files[train_end:val_end]

        test_files = files[val_end:]

        copy_files(

            train_files,

            dataset_dir /
            "train" /
            class_dir.name

        )

        copy_files(

            val_files,

            dataset_dir /
            "val" /
            class_dir.name

        )

        copy_files(

            test_files,

            dataset_dir /
            "test" /
            class_dir.name

        )

        print(

            f"{class_dir.name:<10}"

            f" Total={total:<4}"

            f" Train={len(train_files):<4}"

            f" Val={len(val_files):<4}"

            f" Test={len(test_files):<4}"

        )

print("\nDataset split selesai.")