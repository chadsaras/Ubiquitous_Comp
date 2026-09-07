"""
Model 1 (PLAN.md Step 10): a small 1D convolutional network over the raw,
6-channel, 25 Hz burst -- the neural counterpart to the Random Forest / HGB
tree models in scripts/train_classifier.py, trained and evaluated on the exact
same 5-fold official user split for a fair, apples-to-apples comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """
    ~31k parameters. Three conv blocks (widen 6->32->64->96 channels, each
    followed by a 2x max-pool) then global average pooling and a small
    classifier head. Kept deliberately small: the challenge brief asks for an
    edge-deployable recognizer, and the tree-model ablation (results/ablation.csv)
    already shows the current accuracy ceiling is mostly label/data noise, not
    model capacity -- so a bigger net is unlikely to buy much, and a small one
    keeps this a fair efficiency comparison against the Random Forest.
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 7, dropout: float = 0.3):
        super().__init__()

        def block(cin: int, cout: int, k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.blocks = nn.Sequential(
            block(n_channels, 32, 7),
            block(32, 64, 5),
            block(64, 96, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 6, RAW_T) -> logits (batch, n_classes)."""
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
