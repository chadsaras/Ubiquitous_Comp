"""
Windowing and hand-engineered features for the 7-class recognition backbone.

Design notes
------------
* Every burst is ~20 s at 25 Hz. We cut it into WIN_SEC windows with HOP_SEC hop and
  compute one feature vector per window. At training time each window inherits the
  minute's label; at inference, window probabilities are averaged per minute.
* Features are computed per signal for 8 signals: Acc X/Y/Z, Gyro X/Y/Z, |Acc|, |Gyro|.
  Every feature has a human-readable name (see feature_names()) so the explanation layer
  can later say "dominant frequency of |Acc| was 1.9 Hz" from real computed values.
* Frequency features come from an rFFT of the mean-removed window. Band edges are chosen
  around gait physics: ~0.5-3 Hz covers walking/running/pedalling cadence.
"""
from __future__ import annotations

import numpy as np

HZ = 25.0
WIN_SEC = 5.0
HOP_SEC = 2.5
WIN = int(WIN_SEC * HZ)   # 125 samples
HOP = int(HOP_SEC * HZ)   # 62 samples

SIGNALS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "acc_mag", "gyro_mag"]
BANDS = [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.5)]
TIME_FEATS = ["mean", "std", "min", "max", "range", "median", "iqr", "rms",
              "skew", "kurt", "zcr", "mad"]
FREQ_FEATS = ["dom_freq", "dom_power", "spec_entropy", "spec_centroid"] + \
             [f"band_{lo}_{hi}" for lo, hi in BANDS]
CROSS_FEATS = ["acc_corr_xy", "acc_corr_xz", "acc_corr_yz",
               "gyro_corr_xy", "gyro_corr_xz", "gyro_corr_yz",
               "acc_gyro_mag_corr"]


def feature_names() -> list[str]:
    names = []
    for s in SIGNALS:
        names += [f"{s}__{f}" for f in TIME_FEATS + FREQ_FEATS]
    names += CROSS_FEATS
    return names


N_FEATURES = len(feature_names())


# --------------------------------------------------------------------------- helpers
def align(acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """Trim both modalities to a common length and stack -> (n, 6) [ax ay az gx gy gz]."""
    n = min(len(acc), len(gyro))
    return np.hstack([acc[:n], gyro[:n]])


def windows(x: np.ndarray, win: int = WIN, hop: int = HOP) -> list[tuple[int, np.ndarray]]:
    """Yield (start_sample, window) pairs. A short final segment is dropped."""
    out = []
    for s in range(0, len(x) - win + 1, hop):
        out.append((s, x[s:s + win]))
    if not out and len(x) >= win // 2:      # very short burst: use it whole, zero-pad
        pad = np.zeros((win - len(x), x.shape[1]))
        out.append((0, np.vstack([x, pad])))
    return out


def _time_feats_mat(V: np.ndarray) -> np.ndarray:
    """V: (n, k) signals as columns -> (k, 12) time-domain features, order = TIME_FEATS."""
    mean = V.mean(0)
    D = V - mean
    std = D.std(0)
    safe = np.where(std > 1e-8, std, 1.0)
    Z = D / safe
    q25, med, q75 = np.percentile(V, [25, 50, 75], axis=0)
    zcr = (np.abs(np.diff(np.sign(D), axis=0)) > 0).mean(0)
    skew = np.where(std > 1e-8, (Z ** 3).mean(0), 0.0)
    kurt = np.where(std > 1e-8, (Z ** 4).mean(0) - 3.0, 0.0)
    mad = np.median(np.abs(V - med), axis=0)
    return np.column_stack([
        mean, std, V.min(0), V.max(0), V.max(0) - V.min(0), med, q75 - q25,
        np.sqrt((V ** 2).mean(0)), skew, kurt, zcr, mad,
    ])


def _freq_feats_mat(V: np.ndarray, hz: float = HZ) -> np.ndarray:
    """V: (n, k) -> (k, 9) frequency-domain features, order = FREQ_FEATS."""
    D = V - V.mean(0)
    spec = np.abs(np.fft.rfft(D * np.hanning(len(D))[:, None], axis=0)) ** 2   # (nf, k)
    freqs = np.fft.rfftfreq(len(D), d=1.0 / hz)
    p = spec / (spec.sum(0) + 1e-12)
    k = 1 + spec[1:].argmax(0)                                                  # skip DC bin
    dom_freq = freqs[k]
    dom_power = p[k, np.arange(p.shape[1])]
    entropy = -(p * np.log(p + 1e-12)).sum(0) / np.log(len(p))
    centroid = (freqs[:, None] * p).sum(0)
    bands = [p[(freqs >= lo) & (freqs < hi)].sum(0) for lo, hi in BANDS]
    return np.column_stack([dom_freq, dom_power, entropy, centroid] + bands)


def _freq_feats(v: np.ndarray, hz: float = HZ) -> list[float]:
    """Single-signal convenience wrapper (used by signal_summary)."""
    return _freq_feats_mat(v[:, None], hz)[0].tolist()


def _corr_cols(V: np.ndarray, pairs: list[tuple[int, int]]) -> list[float]:
    std = V.std(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(V, rowvar=False)
    out = []
    for i, j in pairs:
        out.append(0.0 if std[i] < 1e-8 or std[j] < 1e-8 else float(C[i, j]))
    return out


# --------------------------------------------------------------------------- main API
def window_features(w: np.ndarray) -> np.ndarray:
    """
    w: (WIN, 6) array [ax ay az gx gy gz] at 25 Hz -> feature vector of length N_FEATURES.
    All 8 signals are processed as one matrix; output order matches feature_names().
    """
    acc, gyr = w[:, :3], w[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    V = np.column_stack([acc, gyr, acc_mag, gyr_mag])          # (WIN, 8), order = SIGNALS
    per_signal = np.hstack([_time_feats_mat(V), _freq_feats_mat(V)])   # (8, 21)
    cross = _corr_cols(V, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (6, 7)])
    out = np.concatenate([per_signal.reshape(-1), cross]).astype(np.float32)
    assert out.shape[0] == N_FEATURES, (out.shape, N_FEATURES)
    return out


def burst_features(acc: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Full burst -> (starts_sec, X) where X is (n_windows, N_FEATURES) and starts_sec gives
    each window's start offset in seconds from the beginning of the burst.
    """
    x = align(acc, gyro)
    ws = windows(x)
    if not ws:
        return np.zeros(0), np.zeros((0, N_FEATURES), dtype=np.float32)
    starts = np.array([s / HZ for s, _ in ws])
    X = np.stack([window_features(w) for _, w in ws])
    return starts, X


def signal_summary(w: np.ndarray) -> dict[str, float]:
    """
    A handful of physically meaningful numbers for one window, used by the explanation
    layer. Keys are stable and human-readable.
    """
    acc, gyr = w[:, :3], w[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    ff = _freq_feats(acc_mag)
    return {
        "acc_mag_mean": float(acc_mag.mean()),
        "acc_mag_std": float(acc_mag.std()),
        "acc_mag_range": float(acc_mag.max() - acc_mag.min()),
        "acc_dom_freq_hz": ff[0],
        "acc_band_0.5_3hz": ff[5] + ff[6],
        "gyro_mag_mean": float(gyr_mag.mean()),
        "gyro_mag_std": float(gyr_mag.std()),
        "gyro_dom_freq_hz": _freq_feats(gyr_mag)[0],
    }
