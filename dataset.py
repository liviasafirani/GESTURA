"""
dataset.py

PyTorch Dataset & DataLoader
for BISINDO Detection System
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    IMAGE_SIZE,
    VALID_IMAGE_EXTENSIONS,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

# ==========================================================
# Transform
# ==========================================================


def build_train_transform() -> transforms.Compose:
    """
    Transform untuk training.
    """

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),

        transforms.RandomRotation(
            degrees=8
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


def build_eval_transform() -> transforms.Compose:
    """
    Transform untuk validation,
    testing,
    dan inference.
    """

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


# ==========================================================
# Utility
# ==========================================================


def is_image_file(path: Path) -> bool:
    """
    Mengecek apakah file merupakan gambar.
    """

    return (
        path.is_file()
        and
        path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def load_rgb_image(path: Path) -> Image.Image:
    """
    Membaca gambar RGB.
    """

    with Image.open(path) as image:
        return image.convert("RGB")


def create_dataloader(
    dataset: Dataset,
    *,
    shuffle: bool
) -> DataLoader:
    """
    Factory DataLoader.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )


# ==========================================================
# Static Gesture Dataset
# ==========================================================


class StaticGestureDataset(Dataset):
    """
    Dataset gesture statis.

    Struktur folder:

    train/
        A/
        B/
        ...

    atau

    train/
        0/
        1/
        ...
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None
    ) -> None:

        self.root = Path(root)

        if not self.root.exists():

            raise FileNotFoundError(
                f"Dataset tidak ditemukan:\n{self.root}"
            )

        self.transform = (
            transform
            if transform is not None
            else build_eval_transform()
        )

        self.classes: list[str] = []

        self.class_to_idx: dict[str, int] = {}

        self.idx_to_class: dict[int, str] = {}

        self.samples: list[
            tuple[Path, int]
        ] = []

        self._scan_dataset()

    # ======================================================
    # Dataset Scanner
    # ======================================================

    def _scan_dataset(self) -> None:
        """
        Membaca seluruh dataset.
        """

        self.classes = sorted(
            folder.name
            for folder in self.root.iterdir()
            if folder.is_dir()
        )

        if not self.classes:

            raise RuntimeError(
                f"Tidak ada folder kelas pada:\n{self.root}"
            )

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.classes)
        }

        self.idx_to_class = {
            idx: name
            for name, idx in self.class_to_idx.items()
        }

        for class_name in self.classes:

            class_dir = self.root / class_name

            images = sorted(

                image

                for image in class_dir.iterdir()

                if is_image_file(image)

            )

            if not images:

                raise RuntimeError(
                    f"Kelas '{class_name}' kosong."
                )

            label = self.class_to_idx[class_name]

            for image_path in images:

                self.samples.append(
                    (
                        image_path,
                        label
                    )
                )

        if not self.samples:

            raise RuntimeError(
                f"Dataset kosong:\n{self.root}"
            )

    # ======================================================
    # PyTorch Dataset
    # ======================================================

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, int]:

        image_path, label = self.samples[index]

        image = load_rgb_image(image_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label

    # ======================================================
    # Properties
    # ======================================================

    @property
    def num_classes(self) -> int:

        return len(self.classes)

    @property
    def num_samples(self) -> int:

        return len(self.samples)

    # ======================================================
    # Utility
    # ======================================================

    def get_class_name(
        self,
        index: int
    ) -> str:

        return self.idx_to_class[index]

    def get_class_index(
        self,
        class_name: str
    ) -> int:

        return self.class_to_idx[class_name]

    def class_distribution(self) -> dict[str, int]:

        distribution = {
            class_name: 0
            for class_name in self.classes
        }

        for _, label in self.samples:

            distribution[
                self.idx_to_class[label]
            ] += 1

        return distribution

    def summary(self) -> None:

        print("\n========== STATIC DATASET ==========\n")

        print(f"Root Folder : {self.root}")

        print(f"Classes     : {self.num_classes}")

        print(f"Samples     : {self.num_samples}")

        print("\nClass Distribution")

        distribution = self.class_distribution()

        for class_name in self.classes:

            print(
                f"{class_name:<12}"
                f": "
                f"{distribution[class_name]}"
            )

        print("\n====================================\n")
    
    # ==========================================================
# Dynamic Gesture Dataset
# ==========================================================


class DynamicGestureDataset(Dataset):
    """
    Dataset gesture dinamis.

    Struktur folder:

    train/
        bisa/
            seq_0001/
                frame_0001.jpg
                frame_0002.jpg
                ...
            seq_0002/
                ...
        contoh/
        ...
    """

    def __init__(
        self,
        root: str | Path,
        sequence_length: int,
        transform: Callable | None = None
    ) -> None:

        self.root = Path(root)

        if not self.root.exists():

            raise FileNotFoundError(
                f"Dataset tidak ditemukan:\n{self.root}"
            )

        self.sequence_length = sequence_length

        self.transform = (
            transform
            if transform is not None
            else build_eval_transform()
        )

        self.classes: list[str] = []

        self.class_to_idx: dict[str, int] = {}

        self.idx_to_class: dict[int, str] = {}

        self.samples: list[
            tuple[Path, int]
        ] = []

        self._scan_dataset()

    # ======================================================
    # Dataset Scanner
    # ======================================================

    def _scan_dataset(self) -> None:
        """
        Membaca seluruh sequence.
        """

        self.classes = sorted(
            folder.name
            for folder in self.root.iterdir()
            if folder.is_dir()
        )

        if not self.classes:

            raise RuntimeError(
                f"Tidak ada folder kelas pada:\n{self.root}"
            )

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.classes)
        }

        self.idx_to_class = {
            idx: name
            for name, idx in self.class_to_idx.items()
        }

        for class_name in self.classes:

            class_dir = self.root / class_name

            sequences = sorted(
                folder
                for folder in class_dir.iterdir()
                if folder.is_dir()
            )

            if not sequences:

                raise RuntimeError(
                    f"Kelas '{class_name}' kosong."
                )

            label = self.class_to_idx[class_name]

            for sequence_dir in sequences:

                frame_paths = sorted(
                    image
                    for image in sequence_dir.iterdir()
                    if is_image_file(image)
                )

                if len(frame_paths) != self.sequence_length:

                    raise RuntimeError(

                        f"{sequence_dir} "

                        f"berisi {len(frame_paths)} frame "

                        f"(harus {self.sequence_length})."

                    )

                self.samples.append(
                    (
                        sequence_dir,
                        label
                    )
                )

        if not self.samples:

            raise RuntimeError(
                f"Dataset kosong:\n{self.root}"
            )

    # ======================================================
    # PyTorch Dataset
    # ======================================================

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, int]:

        sequence_dir, label = self.samples[index]

        frame_paths = sorted(
            image
            for image in sequence_dir.iterdir()
            if is_image_file(image)
        )

        sequence: list[Tensor] = []

        for frame_path in frame_paths:

            image = load_rgb_image(frame_path)

            if self.transform is not None:

                image = self.transform(image)

            sequence.append(image)

        sequence_tensor = torch.stack(
            sequence,
            dim=0
        )

        return sequence_tensor, label

    # ======================================================
    # Properties
    # ======================================================

    @property
    def num_classes(self) -> int:

        return len(self.classes)

    @property
    def num_sequences(self) -> int:

        return len(self.samples)

    # ======================================================
    # Utility
    # ======================================================

    def get_class_name(
        self,
        index: int
    ) -> str:

        return self.idx_to_class[index]

    def get_class_index(
        self,
        class_name: str
    ) -> int:

        return self.class_to_idx[class_name]

    def class_distribution(self) -> dict[str, int]:

        distribution = {
            class_name: 0
            for class_name in self.classes
        }

        for _, label in self.samples:

            distribution[
                self.idx_to_class[label]
            ] += 1

        return distribution

    def summary(self) -> None:

        print("\n========= DYNAMIC DATASET =========\n")

        print(f"Root Folder : {self.root}")

        print(f"Classes     : {self.num_classes}")

        print(f"Sequences   : {self.num_sequences}")

        print(f"Frame/Seq   : {self.sequence_length}")

        print("\nClass Distribution")

        distribution = self.class_distribution()

        for class_name in self.classes:

            print(
                f"{class_name:<12}"
                f": "
                f"{distribution[class_name]}"
            )

        print("\n===================================\n")
    
# ==========================================================
# DataLoader Factory
# ==========================================================


def create_static_dataloader(
    root: str | Path,
    *,
    train: bool = False,
    shuffle: bool | None = None
) -> DataLoader:
    """
    Membuat DataLoader untuk dataset statis.

    Parameters
    ----------
    root : str | Path
        Folder train / val / test.

    train : bool
        True jika DataLoader digunakan
        untuk training.

    shuffle : bool | None
        Override nilai shuffle.
    """

    transform = (
        build_train_transform()
        if train
        else build_eval_transform()
    )

    dataset = StaticGestureDataset(
        root=root,
        transform=transform
    )

    if shuffle is None:
        shuffle = train

    return create_dataloader(
        dataset=dataset,
        shuffle=shuffle
    )


def create_dynamic_dataloader(
    root: str | Path,
    sequence_length: int,
    *,
    train: bool = False,
    shuffle: bool | None = None
) -> DataLoader:
    """
    Membuat DataLoader untuk dataset
    gesture dinamis.
    """

    transform = (
        build_train_transform()
        if train
        else build_eval_transform()
    )

    dataset = DynamicGestureDataset(
        root=root,
        sequence_length=sequence_length,
        transform=transform
    )

    if shuffle is None:
        shuffle = train

    return create_dataloader(
        dataset=dataset,
        shuffle=shuffle
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    print("=" * 50)
    print("      DATASET MODULE")
    print("=" * 50)
    print()
    print("Module berhasil dimuat.")
    print("Gunakan fungsi berikut:")
    print()
    print("- create_static_dataloader()")
    print("- create_dynamic_dataloader()")
    print()