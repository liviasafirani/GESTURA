"""
predict.py

Inference Module
for BISINDO Detection System.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from config import (
    DEVICE,
    CONFIDENCE_THRESHOLD,
)

from dataset import (
    build_eval_transform,
)

from model import (
    StaticGestureModel,
    WordGestureModel,
)


# ==========================================================
# Static Predictor
# ==========================================================


class StaticPredictor:
    """
    Predictor untuk model
    angka dan abjad.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str]
    ) -> None:

        self.device = DEVICE

        self.class_names = class_names

        self.transform = build_eval_transform()

        self.model = StaticGestureModel(

            num_classes=len(class_names)

        )

        self.model.load_state_dict(

            torch.load(

                model_path,

                map_location=self.device

            )

        )

        self.model.to(

            self.device

        )

        self.model.eval()

    # ======================================================
    # Prediction
    # ======================================================

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image
    ) -> dict:

        image = image.convert("RGB")

        image = self.transform(image)

        image = image.unsqueeze(0)

        image = image.to(self.device)

        logits = self.model(image)

        probabilities = F.softmax(

            logits,

            dim=1

        )

        confidence, index = torch.max(

            probabilities,

            dim=1

        )

        confidence = confidence.item()

        index = index.item()

        if confidence < CONFIDENCE_THRESHOLD:

            return {

                "label": "Unknown",

                "confidence": confidence,

                "index": -1

            }

        return {

            "label": self.class_names[index],

            "confidence": confidence,

            "index": index

        }
    # ==========================================================
# Word Predictor
# ==========================================================


class WordPredictor:
    """
    Predictor untuk model
    gesture dinamis (kata).
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str]
    ) -> None:

        self.device = DEVICE

        self.class_names = class_names

        self.transform = build_eval_transform()

        self.model = WordGestureModel(

            num_classes=len(class_names)

        )

        self.model.load_state_dict(

            torch.load(

                model_path,

                map_location=self.device

            )

        )

        self.model.to(

            self.device

        )

        self.model.eval()

    # ======================================================
    # Prediction
    # ======================================================

    @torch.no_grad()
    def predict(
        self,
        sequence: list[Image.Image]
    ) -> dict:
        """
        Parameters
        ----------
        sequence :
            List berisi frame ROI.

        Returns
        -------
        dict
        """

        if len(sequence) == 0:

            raise ValueError(
                "Sequence kosong."
            )

        frames = []

        for image in sequence:

            image = image.convert("RGB")

            image = self.transform(image)

            frames.append(image)

        sequence_tensor = torch.stack(

            frames,

            dim=0

        )

        sequence_tensor = sequence_tensor.unsqueeze(0)

        sequence_tensor = sequence_tensor.to(

            self.device

        )

        logits = self.model(

            sequence_tensor

        )

        probabilities = F.softmax(

            logits,

            dim=1

        )

        confidence, index = torch.max(

            probabilities,

            dim=1

        )

        confidence = confidence.item()

        index = index.item()

        if confidence < CONFIDENCE_THRESHOLD:

            return {

                "label": "Unknown",

                "confidence": confidence,

                "index": -1

            }

        return {

            "label": self.class_names[index],

            "confidence": confidence,

            "index": index

        }

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BISINDO PREDICTION MODULE")
    print("=" * 60)
    print()

    print("Gunakan predictor sesuai model.")
    print()

    print("# Number Model")
    print(
        "number_predictor = StaticPredictor("
    )
    print("    model_path=NUMBER_MODEL,")
    print("    class_names=NUMBER_CLASSES")
    print(")")
    print()

    print("# Alphabet Model")
    print(
        "alphabet_predictor = StaticPredictor("
    )
    print("    model_path=ALPHABET_MODEL,")
    print("    class_names=ALPHABET_CLASSES")
    print(")")
    print()

    print("# Word Model")
    print(
        "word_predictor = WordPredictor("
    )
    print("    model_path=WORD_MODEL,")
    print("    class_names=WORD_CLASSES")
    print(")")
    print()

    print("Prediction API")
    print()

    print("result = predictor.predict(...)")
    print()

    print("Return Value")

    print("{")
    print('    "label": "...",')
    print('    "confidence": 0.98,')
    print('    "index": 0')
    print("}")