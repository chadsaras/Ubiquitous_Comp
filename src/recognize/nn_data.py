"""
Raw-signal data prep for the neural recognizer (PLAN.md Step 10, Model 1: 1D-CNN).

Unlike src/recognize/features.py (which hand-computes 175 descriptive numbers per
window for the tree models), this module keeps the ALIGNED, RESAMPLED, G-NORMALISED
RAW SIGNAL itself -- one fixed-length (RAW_T, 6) array per labelled minute -- and
lets the network learn its own representation directly from it, instead of from
hand-picked statistics.

RAW_T=500 at 25 Hz = 20.0 s, covering the large majority of ExtraSensory's ~20 s
bursts; longer bursts are truncated, shorter ones zero-padded at the end. The real
(unpadded) length of every burst is kept alongside it (`n_valid`), so anything that
needs to know what part of the array is real signal -- e.g. a "did the phone ever
move" check for the signal-consistency evaluation -- doesn't get fooled by the
padding.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

HZ = 25.0
RAW_T = 500                  # 20 s at 25 Hz
CHANNELS = 6                 # acc_x acc_y acc_z gyro_x gyro_y gyro_z
STILL_G = 0.03               # matches scripts/train_classifier.py's STILL_G


def fixed_burst(acc: np.ndarray, gyro: np.ndarray, t: int = RAW_T) -> tuple[np.ndarray, int]:
    """
    acc, gyro: (n, 3) resampled, g-normalised arrays (from src/preprocess/load.py).
    Returns ((t, 6) float32, n_valid): trimmed to a common length, padded/truncated
    to `t`; n_valid is how many of the first rows are real (unpadded) signal.
    """
    n = min(len(acc), len(gyro), t)
    out = np.zeros((t, CHANNELS), dtype=np.float32)
    if n > 0:
        out[:n, :3] = acc[:n]
        out[:n, 3:] = gyro[:n]
    return out, n


def still_mask(X: np.ndarray, n_valid: np.ndarray, still_g: float = STILL_G) -> np.ndarray:
    """
    True where the accelerometer's magnitude barely varies over the REAL (unpadded)
    portion of the burst -- i.e. the phone was not on a moving body, whatever the
    label says. Mirrors the STILL_G check scripts/train_classifier.py does from the
    engineered `acc_mag__std` feature, computed here directly from the raw signal.
    """
    mag = np.linalg.norm(X[:, :, :3], axis=2)          # (n, T)
    out = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        v = max(int(n_valid[i]), 2)
        out[i] = mag[i, :v].std() < still_g
    return out


class MinuteRawDataset(Dataset):
    """Wraps in-memory (N, RAW_T, 6) arrays -> (channels-first tensor, label)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3 and X.shape[1:] == (RAW_T, CHANNELS), X.shape
        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        # torch.tensor(), not torch.from_numpy(): the latter is a zero-copy path tied
        # to the exact NumPy C-API version torch was built against, and breaks with a
        # bare "RuntimeError: Numpy is not available" on any environment where that
        # doesn't line up (hit this on Kaggle's numpy 2.x base image).
        x = torch.tensor(self.X[i], dtype=torch.float32).transpose(0, 1)   # (RAW_T, 6) -> (6, RAW_T)
        return x, int(self.y[i])
