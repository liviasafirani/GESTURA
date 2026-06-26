"""
model.py

Model Definition
for BISINDO Detection System.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

# ==========================================================
# Static Gesture Model
# ==========================================================


class StaticGestureModel(nn.Module):
    """
    MobileNetV3 Small
    untuk angka dan abjad.
    """

    def __init__(
        self,
        num_classes: int
    ) -> None:

        super().__init__()

        self.backbone = mobilenet_v3_small(

            weights=MobileNet_V3_Small_Weights.DEFAULT

        )

        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier[-1] = nn.Linear(

            in_features,

            num_classes

        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.backbone(x)
    # ==========================================================
# CNN Feature Extractor
# ==========================================================


class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor
    untuk setiap frame pada gesture dinamis.

    Input
    -----
    (B, 3, 224, 224)

    Output
    ------
    (B, 256)
    """

    def __init__(self) -> None:

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                output_size=1
            )

        )

        self.flatten = nn.Flatten()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.flatten(x)

        return x
    # ==========================================================
# Word Gesture Model
# ==========================================================


class WordGestureModel(nn.Module):
    """
    CNN + GRU
    untuk klasifikasi gesture dinamis.

    Input
    -----
    (B, T, 3, H, W)

    Output
    ------
    (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ) -> None:

        super().__init__()

        self.feature_extractor = CNNFeatureExtractor()

        self.gru = nn.GRU(

            input_size=256,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout

        )

        self.classifier = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(

                hidden_size,

                num_classes

            )

        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x :
            Shape = (B, T, C, H, W)

        Returns
        -------
        logits :
            Shape = (B, num_classes)
        """

        batch_size, sequence_length, channels, height, width = x.shape

        # ==================================================
        # CNN Feature Extraction
        # ==================================================

        x = x.view(

            batch_size * sequence_length,

            channels,

            height,

            width

        )

        x = self.feature_extractor(x)

        x = x.view(

            batch_size,

            sequence_length,

            -1

        )

        # ==================================================
        # GRU
        # ==================================================

        output, hidden = self.gru(x)

        last_hidden = hidden[-1]

        # ==================================================
        # Classification
        # ==================================================

        logits = self.classifier(

            last_hidden

        )

        return logits