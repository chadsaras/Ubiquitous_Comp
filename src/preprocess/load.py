"""
ExtraSensory loading utilities for the "Ask the Sensors" challenge.

Layout assumed (produced by scripts/prepare_data.py):
    data/raw/features_labels/<uuid>.features_labels.csv.gz
    data/raw/original_labels/<uuid>.original_labels.csv.gz
    data/raw/raw_acc/<uuid>/<timestamp>.m_raw_acc.dat        (verify names with inspect_data.py)
    data/raw/raw_gyro/<uuid>/<timestamp>.m_proc_gyro.dat
    data/raw/cv5Folds/...

Each raw .dat file is one ~20 s burst at ~40 Hz, whitespace-separated rows.
We normalise every burst to a (n, 3) float array sampled at TARGET_HZ.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

TARGET_HZ = 25.0
NATIVE_HZ = 40.0  # phone acc/gyro in ExtraSensory
MAX_BURST_SEC = 60.0  # bursts are ~20 s; anything longer is a clock glitch

# The seven challenge classes, in a fixed order used everywhere downstream.
CLASSES = [
    "lying_down",
    "sitting",
    "standing_in_place",
    "standing_and_moving",
    "walking",
    "running",
    "bicycling",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# If a minute carries more than one of our labels (rare, labels are self-reported),
# resolve with this priority: the most dynamic activity wins.
PRIORITY = ["bicycling", "running", "walking", "standing_and_moving",
            "standing_in_place", "sitting", "lying_down"]

# Cleaned-label columns (features_labels csv) -> our class. Standing is collapsed there,
# so the two standing classes come from the ORIGINAL labels file instead.
CLEANED_MAP = {
    "label:LYING_DOWN": "lying_down",
    "label:SITTING": "sitting",
    "label:FIX_walking": "walking",
    "label:FIX_running": "running",
    "label:BICYCLING": "bicycling",
}
ORIGINAL_SUBSTR = {  # matched case-insensitively against original-label column names
    "STANDING_IN_PLACE": "standing_in_place",
    "STANDING_AND_MOVING": "standing_and_moving",
}


# --------------------------------------------------------------------------- paths
def find_users(data_dir: str | Path = "data") -> list[str]:
    d = Path(data_dir) / "raw" / "features_labels"
    return sorted(p.name.split(".")[0] for p in d.rglob("*.features_labels.csv.gz"))


def _find_one(d: Path, pattern: str) -> Path:
    hits = list(d.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern} under {d}")
    return hits[0]


# --------------------------------------------------------------------------- labels
def load_labels(uuid: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Per-minute label table for one user.

    Returns a DataFrame indexed by unix `timestamp` (int seconds) with:
        one 0/1 column per class in CLASSES, and
        `label` : the resolved single class name, or None if none of the seven applies.
    """
    root = Path(data_dir) / "raw"
    cleaned = pd.read_csv(_find_one(root / "features_labels", f"{uuid}*.csv.gz"))
    orig = pd.read_csv(_find_one(root / "original_labels", f"{uuid}*.csv.gz"))

    out = pd.DataFrame({"timestamp": cleaned["timestamp"].astype(int)})
    for col, cls in CLEANED_MAP.items():
        out[cls] = cleaned[col].fillna(0).astype(int).values if col in cleaned else 0

    # standing split from original labels, joined on timestamp
    orig = orig.rename(columns={c: c for c in orig.columns})
    orig_ts = orig["timestamp"].astype(int)
    for sub, cls in ORIGINAL_SUBSTR.items():
        cols = [c for c in orig.columns if sub.lower() in c.lower()]
        vals = orig[cols[0]].fillna(0).astype(int) if cols else pd.Series(0, index=orig.index)
        out[cls] = out["timestamp"].map(dict(zip(orig_ts, vals))).fillna(0).astype(int)

    def resolve(row) -> str | None:
        for cls in PRIORITY:
            if row[cls] == 1:
                return cls
        return None

    out["label"] = out.apply(resolve, axis=1)
    return out.set_index("timestamp").sort_index()


# --------------------------------------------------------------------------- raw bursts
def read_burst(path: Path) -> np.ndarray | None:
    """
    Read one raw .dat burst -> (n, 4) array [t_sec, x, y, z] at native rate,
    or None if the file is a 'nan' dummy / unreadable.
    """
    try:
        # pandas' C parser is ~10x faster than np.loadtxt on these 80 KB text files
        arr = pd.read_csv(path, sep=r"\s+", header=None, engine="c",
                          dtype=np.float64, na_filter=False).to_numpy()
    except Exception:
        try:
            arr = np.loadtxt(path)
        except Exception:
            return None
    if arr.ndim != 2 or arr.shape[0] < 10:
        return None
    if arr.shape[1] == 4:
        t = arr[:, 0] - arr[0, 0]
        xyz = arr[:, 1:4]
        # In-the-wild timestamp glitches: non-monotonic steps or an absurd span.
        # Fall back to the nominal sampling grid rather than trusting a broken clock,
        # otherwise resample() would try to build a grid with billions of points.
        dt = np.diff(t)
        if (not np.all(np.isfinite(t)) or t[-1] <= 0 or t[-1] > MAX_BURST_SEC
                or np.any(dt <= 0)):
            t = np.arange(arr.shape[0]) / NATIVE_HZ
    elif arr.shape[1] == 3:  # no timestamp column: assume native rate
        t = np.arange(arr.shape[0]) / NATIVE_HZ
        xyz = arr
    else:
        return None
    if not np.all(np.isfinite(xyz)):
        return None
    return np.column_stack([t, xyz])


def resample(burst: np.ndarray, hz: float = TARGET_HZ) -> np.ndarray:
    """Linear-interpolate an [t,x,y,z] burst onto a uniform grid at `hz`. Returns (m, 3)."""
    t = burst[:, 0]
    end = min(float(t[-1]), MAX_BURST_SEC)          # hard cap: never build a giant grid
    grid = np.arange(0.0, end, 1.0 / hz)
    return np.column_stack([np.interp(grid, t, burst[:, k]) for k in (1, 2, 3)])


@dataclass
class Minute:
    """One labelled minute with both modalities at TARGET_HZ, or None where missing."""
    uuid: str
    timestamp: int          # unix seconds, start of the minute's 20 s burst
    label: str | None
    acc: np.ndarray | None  # (m, 3) at TARGET_HZ
    gyro: np.ndarray | None # (m, 3) at TARGET_HZ


def iter_minutes(uuid: str, data_dir: str | Path = "data",
                 require_both: bool = True) -> Iterator[Minute]:
    """
    Yield each labelled minute for a user with its resampled acc and gyro bursts.
    File naming inside the raw zips is matched by timestamp prefix so it survives
    small naming differences; run scripts/inspect_data.py once to confirm.
    """
    root = Path(data_dir) / "raw"
    labels = load_labels(uuid, data_dir)
    acc_dir = next(iter((root / "raw_acc").rglob(uuid)), None)
    gyr_dir = next(iter((root / "raw_gyro").rglob(uuid)), None)

    def index(d: Path | None) -> dict[int, Path]:
        """One directory scan -> {minute_timestamp: file}. Avoids a glob per minute."""
        if d is None:
            return {}
        out: dict[int, Path] = {}
        for p in d.iterdir():
            head = p.name.split(".")[0]
            if head.isdigit():
                out[int(head)] = p
        return out

    acc_idx, gyr_idx = index(acc_dir), index(gyr_dir)

    def burst_for(idx: dict[int, Path], ts: int) -> np.ndarray | None:
        p = idx.get(ts)
        if p is None:
            return None
        b = read_burst(p)
        return resample(b) if b is not None else None

    for ts, row in labels.iterrows():
        acc = burst_for(acc_idx, ts)
        gyr = burst_for(gyr_idx, ts)
        if require_both and (acc is None or gyr is None):
            continue
        yield Minute(uuid, int(ts), row["label"], acc, gyr)


def load_folds(data_dir: str | Path = "data") -> list[dict[str, list[str]]]:
    """
    The official 5-fold user partition. Returns [{'train': [...uuids], 'test': [...]}, ...].
    Adjust the filename patterns if the unpacked folder differs.
    """
    d = Path(data_dir) / "raw" / "cv5Folds"
    folds = []
    for i in range(5):
        tr = list(d.rglob(f"*fold_{i}*train*uuids*"))
        te = list(d.rglob(f"*fold_{i}*test*uuids*"))
        if not tr or not te:
            raise FileNotFoundError(f"fold {i}: pattern miss under {d}; inspect the folder")
        folds.append({
            "train": tr[0].read_text().split(),
            "test": te[0].read_text().split(),
        })
    return folds
