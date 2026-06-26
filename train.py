"""
train.py

Training Module
for BISINDO Detection System.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
    MODEL_SAVE_DIR,
)

from dataset import (
    create_static_dataloader,
    create_dynamic_dataloader,
)

from model import (
    StaticGestureModel,
    WordGestureModel,
)


# ==========================================================
# Trainer
# ==========================================================


class Trainer:
    """
    Trainer untuk seluruh model BISINDO.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        model_path: str | Path,
    ) -> None:

        self.device = DEVICE

        self.model = model.to(
            self.device
        )

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.model_path = Path(
            model_path
        )

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(

            self.model.parameters(),

            lr=LEARNING_RATE

        )

        self.best_accuracy = 0.0

    # ======================================================
    # Utility
    # ======================================================

    def save_checkpoint(self) -> None:
        """
        Menyimpan model terbaik.
        """

        torch.save(

            self.model.state_dict(),

            self.model_path

        )

    # ======================================================
    # Helper
    # ======================================================

    @staticmethod
    def calculate_accuracy(
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Menghitung accuracy satu batch.
        """

        prediction = torch.argmax(

            outputs,

            dim=1

        )

        correct = (

            prediction == labels

        ).sum().item()

        return correct / labels.size(0)
        # ======================================================
    # Training
    # ======================================================

    def train_epoch(
        self
    ) -> tuple[float, float]:
        """
        Menjalankan satu epoch training.

        Returns
        -------
        tuple
            (loss, accuracy)
        """

        self.model.train()

        total_loss = 0.0

        total_correct = 0

        total_samples = 0

        for images, labels in self.train_loader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                images
            )

            loss = self.criterion(
                outputs,
                labels
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        average_loss = (
            total_loss
            / len(self.train_loader)
        )

        accuracy = (
            total_correct
            / total_samples
        )

        return (
            average_loss,
            accuracy
        )

    # ======================================================
    # Validation
    # ======================================================

    @torch.no_grad()
    def validate(
        self
    ) -> tuple[float, float]:
        """
        Evaluasi validation dataset.

        Returns
        -------
        tuple
            (loss, accuracy)
        """

        self.model.eval()

        total_loss = 0.0

        total_correct = 0

        total_samples = 0

        for images, labels in self.val_loader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            outputs = self.model(
                images
            )

            loss = self.criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        average_loss = (
            total_loss
            / len(self.val_loader)
        )

        accuracy = (
            total_correct
            / total_samples
        )

        return (
            average_loss,
            accuracy
        )
        # ======================================================
    # Main Training Loop
    # ======================================================

    def train(self) -> None:
        """
        Menjalankan seluruh proses training.
        """

        print("=" * 60)
        print("START TRAINING")
        print("=" * 60)

        for epoch in range(EPOCHS):

            train_loss, train_acc = self.train_epoch()

            val_loss, val_acc = self.validate()

            print()

            print(
                f"Epoch {epoch + 1}/{EPOCHS}"
            )

            print(
                f"Train Loss : {train_loss:.4f}"
            )

            print(
                f"Train Acc  : {train_acc:.4f}"
            )

            print(
                f"Val Loss   : {val_loss:.4f}"
            )

            print(
                f"Val Acc    : {val_acc:.4f}"
            )

            if val_acc > self.best_accuracy:

                self.best_accuracy = val_acc

                self.save_checkpoint()

                print(
                    "Best model updated."
                )

        print()

        print("=" * 60)

        print("TRAINING FINISHED")

        print(
            f"Best Validation Accuracy : {self.best_accuracy:.4f}"
        )

        print(
            f"Model Saved : {self.model_path}"
        )

        print("=" * 60)

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BISINDO TRAINING")
    print("=" * 60)

    print("1. Numbers")
    print("2. Alphabets")
    print("3. Words")

    choice = input("\nPilih model: ").strip()

    if choice == "1":

        train_loader = create_static_dataloader(
            root="dataset/numbers/train",
            train=True
        )

        val_loader = create_static_dataloader(
            root="dataset/numbers/val"
        )

        model = StaticGestureModel(
            num_classes=train_loader.dataset.num_classes
        )

        model_path = MODEL_SAVE_DIR / "numbers.pth"

    elif choice == "2":

        train_loader = create_static_dataloader(
            root="dataset/alphabets/train",
            train=True
        )

        val_loader = create_static_dataloader(
            root="dataset/alphabets/val"
        )

        model = StaticGestureModel(
            num_classes=train_loader.dataset.num_classes
        )

        model_path = MODEL_SAVE_DIR / "alphabets.pth"

    elif choice == "3":

        train_loader = create_dynamic_dataloader(
            root="dataset/words/train",
            sequence_length=SEQUENCE_LENGTH,
            train=True
        )

        val_loader = create_dynamic_dataloader(
            root="dataset/words/val",
            sequence_length=SEQUENCE_LENGTH
        )

        model = WordGestureModel(
            num_classes=train_loader.dataset.num_classes
        )

        model_path = MODEL_SAVE_DIR / "words.pth"

    else:

        raise ValueError("Pilihan tidak valid.")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=model_path
    )

    trainer.train()