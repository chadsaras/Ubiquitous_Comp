"""
PLAN.md Step 10: small neural nets over the raw, 6-channel, 25 Hz burst -- the
neural counterparts to the Random Forest / HGB tree models in
scripts/train_classifier.py, trained and evaluated on the exact same 5-fold
official user split for a fair, apples-to-apples comparison.

Two architectures, deliberately kept to a similar (~30-40k) parameter budget so
the comparison between them isolates the ARCHITECTURE choice, not just "which
one has more capacity":

  Model 1 - TinyCNN:  convolution only, no notion of sequence order beyond what
                       the conv filters see locally.
  Model 2 - CNNGRU:   a short conv front-end (same job: pull out local motion
                       shapes like a footstep or pedal stroke) feeding a small
                       GRU, which can track how those shapes evolve across the
                       whole ~20 s burst -- e.g. a genuinely periodic gait
                       cadence -- something TinyCNN's global-average-pool head
                       cannot represent directly. This directly tests whether
                       *learned* temporal structure beats the hand-computed
                       dominant-frequency features the tree models rely on.

Both Model 1 and Model 2 consistently trailed Random Forest by ~8 accuracy
points, on every fold -- the ceiling turned out to be the harder job of
learning useful features from raw signal at all, not which raw-signal
architecture does it. FeatureMLP below removes that handicap: it trains on
the SAME 302 engineered features Random Forest uses (see
scripts/train_classifier.py's aggregate_minutes/feature_mask), so it is not
capacity-matched to Models 1/2 -- it is meant to be a genuine competitor to
Random Forest, not another point in a fair small-model ablation.
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


class CNNGRU(nn.Module):
    """
    ~33k parameters (comparable to TinyCNN's ~31k, by design). A 2-block 1D-CNN
    front-end downsamples the raw 500-sample burst to a 125-step sequence of
    48-dim local motion features, which a small unidirectional GRU consumes.
    The classifier head mean-pools the GRU's per-step outputs (the recurrent
    analogue of TinyCNN's global average pool) rather than using only the
    final hidden state, so it isn't overly sensitive to where in the burst the
    informative motion happened to occur.
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 7, hidden: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(cin: int, cout: int, k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(block(n_channels, 32, 7), block(32, 48, 5))
        self.gru = nn.GRU(input_size=48, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 6, RAW_T) -> logits (batch, n_classes)."""
        x = self.conv(x)                    # (batch, 48, RAW_T//4)
        x = x.transpose(1, 2)                # (batch, seq_len, 48) -- GRU wants channels-last
        out, _ = self.gru(x)                 # (batch, seq_len, hidden)
        pooled = out.mean(dim=1)             # mean-pool over time, not just the last step
        return self.head(pooled)


class FeatureMLP(nn.Module):
    """
    ~112k parameters -- larger than Models 1/2 on purpose (see module docstring).
    Plain feedforward net: 302 -> 256 -> 128 -> 7, BatchNorm+ReLU+Dropout between
    layers. Input is a (batch, n_in) vector of the same mean+std-aggregated,
    noori-subset features Random Forest trains on, not a raw signal -- no
    convolution or recurrence needed, the feature engineering already did the
    representation-learning job.
    """

    def __init__(self, n_in: int = 302, n_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_in) -> logits (batch, n_classes)."""
        return self.net(x)


MODELS = {"cnn": TinyCNN, "cnngru": CNNGRU, "mlp": FeatureMLP}


def build_model(arch: str, **kwargs) -> nn.Module:
    if arch not in MODELS:
        raise ValueError(f"unknown arch {arch!r}; choose from {list(MODELS)}")
    return MODELS[arch](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
