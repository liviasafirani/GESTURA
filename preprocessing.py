"""
preprocessing.py

Dataset Preprocessing

Tahapan preprocessing sebelum training.

Static Dataset
--------------
- Resize
- Normalize

Dynamic Dataset
---------------
- Resize
- Normalize
"""

from pathlib import Path

import cv2
import numpy as np

from config import (
    DATASET_DIR,
    IMAGE_SIZE
)


class Preprocessor:

    def __init__(self):

        # ==================================================
        # Dataset
        # ==================================================

        self.dataset_root = DATASET_DIR

        self.output_root = Path("processed")

        self.extensions = (
            ".jpg",
            ".jpeg",
            ".png"
        )

        self.dataset = {}

        self.create_output_folder()

        self.scan_dataset()

    # ======================================================
    # Output Folder
    # ======================================================

    def create_output_folder(self):
        """
        Membuat folder processed.
        """

        self.output_root.mkdir(
            exist_ok=True
        )

    # ======================================================
    # Dataset
    # ======================================================

    def scan_dataset(self):
        """
        Membaca seluruh dataset.
        """

        if not self.dataset_root.exists():

            raise FileNotFoundError(
                "Folder dataset tidak ditemukan."
            )

        for mode in sorted(
            self.dataset_root.iterdir()
        ):

            if not mode.is_dir():
                continue

            self.dataset[mode.name] = {}

            output_mode = (
                self.output_root
                / mode.name
            )

            output_mode.mkdir(
                parents=True,
                exist_ok=True
            )

            for label in sorted(
                mode.iterdir()
            ):

                if not label.is_dir():
                    continue

                images = []

                output_label = (
                    output_mode
                    / label.name
                )

                output_label.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # ==========================================
                # Dynamic Dataset (Sequence)
                # ==========================================

                if mode.name == "words":

                    for sequence in sorted(
                        label.iterdir()
                    ):

                        if not sequence.is_dir():
                            continue

                        sequence_images = []

                        output_sequence = (
                            output_label
                            / sequence.name
                        )

                        output_sequence.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        for image in sorted(
                            sequence.iterdir()
                        ):

                            if image.suffix.lower() in self.extensions:

                                sequence_images.append(image)

                        images.append(
                            {
                                "sequence": sequence.name,
                                "images": sequence_images
                            }
                        )

                # ==========================================
                # Static Dataset
                # ==========================================

                else:

                    for image in sorted(
                        label.iterdir()
                    ):

                        if image.suffix.lower() in self.extensions:

                            images.append(image)

                self.dataset[
                    mode.name
                ][label.name] = images

    # ======================================================
    # Statistics
    # ======================================================

    def dataset_statistics(self):
        """
        Menampilkan ringkasan dataset.
        """

        print("\n========== DATASET ==========\n")

        total = 0

        for mode, labels in self.dataset.items():

            print(f"[{mode.upper()}]")

            mode_total = 0

            for label, data in labels.items():

                if mode == "words":

                    count = sum(
                        len(seq["images"])
                        for seq in data
                    )

                else:

                    count = len(data)

                mode_total += count
                total += count

                print(
                    f"{label:<12} : {count}"
                )

            print(
                f"Total {mode:<10}: {mode_total}\n"
            )

        print("=" * 35)
        print(f"TOTAL IMAGE : {total}")
        print("=" * 35)

        # ======================================================
    # Image Processing
    # ======================================================

    def preprocess_image(
        self,
        image_path
    ):
        """
        Resize gambar.
        """

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            return None

        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        return image

    # ======================================================
    # Save Image
    # ======================================================

    def save_image(
        self,
        image,
        output_path
    ):
        """
        Menyimpan hasil preprocessing.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            image
        )

    # ======================================================
    # Static Dataset
    # ======================================================

    def process_static(self):
        """
        Preprocessing dataset
        numbers dan alphabets.
        """

        print("\nProcessing Static Dataset...\n")

        for mode in ("numbers", "alphabets"):

            if mode not in self.dataset:
                continue

            for label, images in self.dataset[mode].items():

                for image_path in images:

                    image = self.preprocess_image(
                        image_path
                    )

                    if image is None:
                        continue

                    output = (
                        self.output_root
                        / mode
                        / label
                        / image_path.name
                    )

                    self.save_image(
                        image,
                        output
                    )

    # ======================================================
    # Dynamic Dataset
    # ======================================================

    def process_dynamic(self):
        """
        Preprocessing dataset words.
        """

        print("\nProcessing Dynamic Dataset...\n")

        if "words" not in self.dataset:
            return

        for label, sequences in self.dataset["words"].items():

            for sequence in sequences:

                sequence_name = sequence["sequence"]

                for image_path in sequence["images"]:

                    image = self.preprocess_image(
                        image_path
                    )

                    if image is None:
                        continue

                    output = (
                        self.output_root
                        / "words"
                        / label
                        / sequence_name
                        / image_path.name
                    )

                    self.save_image(
                        image,
                        output
                    )

        # ======================================================
    # Run
    # ======================================================

    def run(self):
        """
        Menjalankan seluruh proses preprocessing.
        """

        self.dataset_statistics()

        self.process_static()

        self.process_dynamic()

        print("\n===================================")
        print("Preprocessing selesai.")
        print(f"Output : {self.output_root}")
        print("===================================")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    preprocessor = Preprocessor()

    preprocessor.run()