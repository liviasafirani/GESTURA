"""
quality_check.py

Quality Checker

Memeriksa kualitas dataset sebelum
preprocessing dan training.

Pengecekan:
- Dataset Statistics
- Blur
- Brightness
- Resolution
- Duplicate (Bagian 3)
"""

from pathlib import Path

import cv2
import numpy as np

from config import (
    DATASET_DIR,
    IMAGE_SIZE
)


class QualityChecker:

    def __init__(self):

        self.dataset_root = DATASET_DIR

        self.extensions = (
            ".jpg",
            ".jpeg",
            ".png"
        )

        self.dataset = {}

        self.report = {
            "total_images": 0,
            "classes": {},
            "blur": [],
            "dark": [],
            "bright": [],
            "resolution": [],
            "duplicate": []
        }

        self.scan_dataset()

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

            for label in sorted(
                mode.iterdir()
            ):

                if not label.is_dir():
                    continue

                images = []

                # Dataset sequence (words)
                if mode.name == "words":

                    for sequence in sorted(
                        label.iterdir()
                    ):

                        if not sequence.is_dir():
                            continue

                        for image in sorted(
                            sequence.iterdir()
                        ):

                            if image.suffix.lower() in self.extensions:

                                images.append(image)

                # Dataset static
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
        Menghitung jumlah dataset.
        """

        total = 0

        print("\n========== DATASET ==========\n")

        for mode, labels in self.dataset.items():

            print(f"[{mode.upper()}]")

            self.report["classes"][mode] = {}

            mode_total = 0

            for label, images in labels.items():

                count = len(images)

                mode_total += count

                total += count

                self.report["classes"][mode][label] = count

                print(
                    f"{label:<12} : {count}"
                )

            print(
                f"Total {mode:<10}: {mode_total}\n"
            )

        self.report["total_images"] = total

        print("=" * 30)
        print(
            f"TOTAL IMAGE : {total}"
        )
        print("=" * 30)

        # ======================================================
    # Blur Detection
    # ======================================================

    def check_blur(
        self,
        threshold=100
    ):
        """
        Mendeteksi gambar blur menggunakan
        Variance of Laplacian.
        """

        print("\nChecking Blur...")

        for mode, labels in self.dataset.items():

            for label, images in labels.items():

                for image_path in images:

                    image = cv2.imread(
                        str(image_path)
                    )

                    if image is None:
                        continue

                    gray = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2GRAY
                    )

                    score = cv2.Laplacian(
                        gray,
                        cv2.CV_64F
                    ).var()

                    if score < threshold:

                        self.report["blur"].append({

                            "file": image_path,

                            "score": round(score, 2)

                        })

    # ======================================================
    # Brightness Detection
    # ======================================================

    def check_brightness(
        self,
        dark_threshold=50,
        bright_threshold=205
    ):
        """
        Mendeteksi gambar terlalu gelap
        atau terlalu terang.
        """

        print("Checking Brightness...")

        for mode, labels in self.dataset.items():

            for label, images in labels.items():

                for image_path in images:

                    image = cv2.imread(
                        str(image_path)
                    )

                    if image is None:
                        continue

                    gray = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2GRAY
                    )

                    brightness = np.mean(gray)

                    if brightness < dark_threshold:

                        self.report["dark"].append({

                            "file": image_path,

                            "value": round(
                                brightness,
                                2
                            )

                        })

                    elif brightness > bright_threshold:

                        self.report["bright"].append({

                            "file": image_path,

                            "value": round(
                                brightness,
                                2
                            )

                        })

    # ======================================================
    # Resolution Check
    # ======================================================

    def check_resolution(self):
        """
        Memastikan seluruh gambar memiliki
        ukuran yang konsisten.
        """

        print("Checking Resolution...")

        expected_width, expected_height = (
            IMAGE_SIZE
        )

        for mode, labels in self.dataset.items():

            for label, images in labels.items():

                for image_path in images:

                    image = cv2.imread(
                        str(image_path)
                    )

                    if image is None:
                        continue

                    height, width = image.shape[:2]

                    if (
                        width != expected_width
                        or
                        height != expected_height
                    ):

                        self.report[
                            "resolution"
                        ].append({

                            "file": image_path,

                            "size": (
                                width,
                                height
                            )

                        })
        # ======================================================
    # Duplicate Detection
    # ======================================================

    def check_duplicate(self):
        """
        Mendeteksi gambar duplikat berdasarkan
        Average Hash (aHash).
        """

        print("Checking Duplicate...")

        hashes = {}

        for mode, labels in self.dataset.items():

            for label, images in labels.items():

                for image_path in images:

                    image = cv2.imread(str(image_path))

                    if image is None:
                        continue

                    image = cv2.resize(
                        image,
                        (8, 8)
                    )

                    gray = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2GRAY
                    )

                    avg = gray.mean()

                    hash_value = "".join(

                        "1" if pixel > avg else "0"

                        for pixel in gray.flatten()

                    )

                    if hash_value in hashes:

                        self.report["duplicate"].append({

                            "original": hashes[hash_value],

                            "duplicate": image_path

                        })

                    else:

                        hashes[hash_value] = image_path

    # ======================================================
    # Report
    # ======================================================

    def print_report(self):
        """
        Menampilkan hasil quality check.
        """

        print("\n========== QUALITY REPORT ==========\n")

        print(
            f"Total Images      : {self.report['total_images']}"
        )

        print(
            f"Blur Images       : {len(self.report['blur'])}"
        )

        print(
            f"Dark Images       : {len(self.report['dark'])}"
        )

        print(
            f"Bright Images     : {len(self.report['bright'])}"
        )

        print(
            f"Wrong Resolution  : {len(self.report['resolution'])}"
        )

        print(
            f"Duplicate Images  : {len(self.report['duplicate'])}"
        )

        print("\n====================================")

    # ======================================================
    # Run
    # ======================================================

    def run(self):

        self.dataset_statistics()

        self.check_blur()

        self.check_brightness()

        self.check_resolution()

        self.check_duplicate()

        self.print_report()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    checker = QualityChecker()

    checker.run()