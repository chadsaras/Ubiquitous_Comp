"""
Aggregation layer: a raw recording -> Timeline of activity intervals.

    from src.aggregate.timeline import build_timeline
    tl = build_timeline("recording.csv", "results/model_rf.joblib")
    print(tl); tl.to_json("timeline.json")

Input recording: CSV with columns
    timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
at any sampling rate; timestamps may be unix seconds or seconds from start (we subtract the first).
Accelerometer may be in g or m/s^2 (normalised per contiguous chunk). Gaps are allowed.

Pipeline
    1. split into contiguous chunks wherever the timestamp gap exceeds GAP_SEC
    2. resample each chunk to 25 Hz, normalise acc to g
    3. 5 s windows / 2.5 s hop -> 175 features -> classifier probabilities
    4. temporal smoothing of window probabilities (moving average over neighbours)
    5. merge same-label windows into intervals; bridge gaps up to MERGE_GAP_SEC;
       absorb episodes shorter than MIN_EPISODE_SEC into their neighbours
    6. bursty recordings (ExtraSensory-style 20 s per minute) are detected and each burst is
       padded to its nominal period so a labelled minute counts as a minute

Every interval carries confidence, runner-up class, and mean signal statistics for grounding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.aggregate.schema import CLASSES, Interval, Timeline          # noqa: E402
from src.preprocess.load import to_g                                  # noqa: E402
from src.recognize.features import (HOP, HZ, WIN, align, signal_summary,  # noqa: E402
                                    window_features, windows)

GAP_SEC = 1.0          # a hole longer than this splits the recording into chunks
MERGE_GAP_SEC = 90.0   # same-label intervals separated by less than this are joined
MIN_EPISODE_SEC = 10.0 # shorter episodes are absorbed into the surrounding activity
SMOOTH_K = 1           # window probabilities averaged with +-K neighbours before argmax

COLS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ------------------------------------------------------------------ loading
def load_recording(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for want in COLS:
        for cand in (want, want.replace("_", ""), want.replace("acc_", "acc").replace("gyro_", "gyr_"),
                     want.replace("gyro_", "gyr")):
            if cand in lower:
                ren[lower[cand]] = want
                break
    df = df.rename(columns=ren)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"recording is missing columns {missing}; have {list(df.columns)}")
    df = df.dropna(subset=COLS).sort_values("timestamp").reset_index(drop=True)
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def split_chunks(df: pd.DataFrame, gap_sec: float = GAP_SEC) -> list[pd.DataFrame]:
    gaps = np.diff(df["t"].to_numpy())
    cut = [0] + list(np.where(gaps > gap_sec)[0] + 1) + [len(df)]
    return [df.iloc[a:b] for a, b in zip(cut, cut[1:]) if b - a >= 10]


def resample_chunk(chunk: pd.DataFrame, hz: float = HZ) -> tuple[np.ndarray, np.ndarray]:
    """-> (t_grid seconds from recording start, (n, 6) [acc g, gyro] at hz)."""
    t = chunk["t"].to_numpy()
    t, idx = np.unique(t, return_index=True)          # drop duplicate timestamps
    vals = chunk[COLS[1:]].to_numpy()[idx]
    grid = np.arange(t[0], t[-1], 1.0 / hz)
    if len(grid) < 2:
        return grid, np.zeros((0, 6))
    X = np.column_stack([np.interp(grid, t, vals[:, k]) for k in range(6)])
    X[:, :3] = to_g(X[:, :3])
    return grid, X


# ------------------------------------------------------------------ recognition over windows
def _context_features(F: np.ndarray, context: int) -> np.ndarray:
    """
    Match the minute-level training distribution: for each window, [mean, std] of the window
    feature vectors in a centred context of `context` windows (clipped at chunk edges).
    """
    if context <= 1:
        return F
    h = context // 2
    out = np.empty((len(F), 2 * F.shape[1]), dtype=np.float32)
    for i in range(len(F)):
        ctx = F[max(0, i - h): i + h + 1]
        out[i, :F.shape[1]] = ctx.mean(0)
        out[i, F.shape[1]:] = ctx.std(0)
    return out


def window_predictions(df: pd.DataFrame, bundle: dict, whole_chunk: bool = False) -> pd.DataFrame:
    """
    One row per analysis window: start, end, p_<class>..., plus signal summary keys.

    `whole_chunk` (used for bursty recordings) aggregates over every window of the burst, so the
    feature distribution matches training exactly: a minute-level model was fitted on the mean and
    std of all 7 windows of a 20 s burst. Every window of the burst then shares that prediction.
    """
    model, mask = bundle["model"], bundle.get("mask")
    context = int(bundle.get("context", 1))
    out_rows, feats = [], []
    for chunk in split_chunks(df):
        grid, X = resample_chunk(chunk)
        if len(X) < WIN // 2:
            continue
        rows, F = [], []
        for s, w in windows(X):
            F.append(window_features(w))
            rows.append({"start": float(grid[s]), "end": float(grid[min(s + WIN, len(grid)) - 1]) + 1 / HZ,
                         **{f"sig_{k}": v for k, v in signal_summary(w).items()}})
        F = np.nan_to_num(np.stack(F), nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            F = F[:, mask]
        if context > 1 and whole_chunk:
            agg = np.concatenate([F.mean(0), F.std(0)])[None, :]
            feats.append(np.repeat(agg, len(F), axis=0))
        else:
            feats.append(_context_features(F, context))      # context never crosses a gap
        out_rows += rows
    if not out_rows:
        return pd.DataFrame()
    Fa = np.vstack(feats)
    P = np.zeros((len(Fa), len(CLASSES)))
    P[:, model.classes_] = model.predict_proba(Fa)
    out = pd.DataFrame(out_rows)
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = P[:, i]
    return out


def smooth_probs(P: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    if k <= 0 or len(P) < 3:
        return P
    pad = np.pad(P, ((k, k), (0, 0)), mode="edge")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.column_stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(P.shape[1])])


# ------------------------------------------------------------------ intervals
def detect_bursty(df: pd.DataFrame) -> tuple[bool, float]:
    """ExtraSensory-style recordings: short bursts every ~60 s. Returns (bursty, period_sec)."""
    chunks = split_chunks(df)
    if len(chunks) < 3:
        return False, 0.0
    starts = np.array([c["t"].iloc[0] for c in chunks])
    lengths = np.array([c["t"].iloc[-1] - c["t"].iloc[0] for c in chunks])
    period = float(np.median(np.diff(starts)))
    return bool(np.median(lengths) < 0.6 * period), period


def merge_windows(wdf: pd.DataFrame, pad_to: float | None, recording_sec: float,
                  min_episode: float = MIN_EPISODE_SEC) -> list[Interval]:
    P = smooth_probs(wdf[[f"p_{c}" for c in CLASSES]].to_numpy())
    lab = P.argmax(1)
    starts, ends = wdf["start"].to_numpy(), wdf["end"].to_numpy()
    sig_cols = [c for c in wdf.columns if c.startswith("sig_")]

    # pass 1: raw runs of identical label, bridging small gaps
    runs: list[dict] = []
    for i in range(len(lab)):
        if runs and runs[-1]["lab"] == lab[i] and starts[i] - runs[-1]["end"] <= MERGE_GAP_SEC:
            runs[-1]["idx"].append(i); runs[-1]["end"] = ends[i]
        else:
            runs.append({"lab": lab[i], "idx": [i], "start": starts[i], "end": ends[i]})

    # pass 2: absorb short episodes into the longer TEMPORALLY ADJACENT neighbour, until stable.
    # A neighbour separated by more than MERGE_GAP_SEC is across a data gap: absorbing into it
    # would stretch an interval over time where nothing was recorded, so it is not a candidate.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, r in enumerate(runs):
            if r["end"] - r["start"] >= min_episode:
                continue
            nb = []
            if j > 0 and r["start"] - runs[j - 1]["end"] <= MERGE_GAP_SEC:
                nb.append(runs[j - 1])
            if j + 1 < len(runs) and runs[j + 1]["start"] - r["end"] <= MERGE_GAP_SEC:
                nb.append(runs[j + 1])
            if not nb:                       # isolated short episode: keep it, it is all we saw
                continue
            tgt = max(nb, key=lambda q: q["end"] - q["start"])
            tgt["idx"] = sorted(tgt["idx"] + r["idx"])
            tgt["start"], tgt["end"] = min(tgt["start"], r["start"]), max(tgt["end"], r["end"])
            runs.pop(j); changed = True
            break
        # re-merge neighbours that became the same label
        k = 0
        while k < len(runs) - 1:
            if runs[k]["lab"] == runs[k + 1]["lab"] and runs[k + 1]["start"] - runs[k]["end"] <= MERGE_GAP_SEC:
                runs[k]["idx"] += runs[k + 1]["idx"]; runs[k]["end"] = runs[k + 1]["end"]; runs.pop(k + 1)
            else:
                k += 1

    out = []
    for r in runs:
        idx = np.array(r["idx"])
        pm = P[idx].mean(0)
        order = np.argsort(pm)[::-1]
        observed = float(sum(min(ends[i], r["end"]) - starts[i] for i in idx))  # overlapping windows ~double count
        observed = min(observed / 2 + (ends[idx[-1]] - starts[idx[-1]]) / 2, r["end"] - r["start"])
        end = r["end"]
        if pad_to:                                   # bursty: last burst counts as a full period
            end = max(end, float(starts[idx[-1]]) + pad_to)
        end = min(end, recording_sec)
        out.append(Interval(
            activity=CLASSES[int(order[0])], start=float(r["start"]), end=float(end),
            confidence=float(pm[order[0]]), n_windows=len(idx), observed_sec=round(observed, 1),
            runner_up=CLASSES[int(order[1])], runner_up_prob=float(pm[order[1]]),
            summary={c[4:]: round(float(wdf[c].iloc[idx].mean()), 4) for c in sig_cols},
        ))
    # overlapping windows make adjacent intervals overlap by one hop; snap to the midpoint
    for a, b in zip(out, out[1:]):
        if b.start < a.end:
            mid = round((a.end + b.start) / 2, 2)
            a.end, b.start = mid, mid
    return out


def split_on_gaps(ivs: list[Interval], wdf: pd.DataFrame, pad_to: float | None,
                  max_gap: float = MERGE_GAP_SEC) -> list[Interval]:
    """
    An interval must never span time where nothing was recorded, or its cited timestamps would
    claim evidence that does not exist. Split each interval wherever its supporting windows leave
    a hole longer than max_gap.
    """
    starts, ends = wdf["start"].to_numpy(), wdf["end"].to_numpy()
    out: list[Interval] = []
    for iv in ivs:
        sel = np.where((starts >= iv.start - 1e-6) & (starts < iv.end))[0]
        if len(sel) == 0:
            out.append(iv); continue
        pieces, cur = [], [sel[0]]
        for a, b in zip(sel, sel[1:]):
            (cur.append(b) if starts[b] - ends[a] <= max_gap else (pieces.append(cur), cur := [b]))
        pieces.append(cur)
        if len(pieces) == 1:
            out.append(iv); continue
        for p in pieces:
            e = float(ends[p[-1]])
            if pad_to:
                e = max(e, float(starts[p[-1]]) + pad_to)
            piece = Interval(activity=iv.activity, start=float(starts[p[0]]), end=min(e, iv.end),
                             confidence=iv.confidence, n_windows=len(p),
                             observed_sec=round(min(e, iv.end) - float(starts[p[0]]), 1),
                             runner_up=iv.runner_up, runner_up_prob=iv.runner_up_prob,
                             summary=dict(iv.summary))
            if piece.end > piece.start:
                out.append(piece)
    out.sort(key=lambda x: x.start)
    # splitting can leave adjacent same-label pieces; re-join those that touch
    joined: list[Interval] = []
    for iv in out:
        p = joined[-1] if joined else None
        if p and p.activity == iv.activity and iv.start - p.end <= max_gap:
            p.end = max(p.end, iv.end)
            p.n_windows += iv.n_windows
            p.observed_sec = round(p.observed_sec + iv.observed_sec, 1)
            p.confidence = max(p.confidence, iv.confidence)
        else:
            joined.append(iv)
    return joined


# ------------------------------------------------------------------ public API
def load_model(path: str | Path) -> dict:
    """Returns the training bundle {model, mask, context, ...}; wraps a bare estimator if needed."""
    bundle = joblib.load(path)
    return bundle if isinstance(bundle, dict) else {"model": bundle, "mask": None, "context": 1}


def build_timeline(recording: str | Path | pd.DataFrame, model_path: str | Path | None = None,
                   model=None) -> Timeline:
    if model is None:
        if model_path is None:
            raise ValueError("need model_path or model")
        bundle = load_model(model_path)
    else:
        bundle = model if isinstance(model, dict) else {"model": model, "mask": None, "context": 1}
    df = recording if isinstance(recording, pd.DataFrame) else load_recording(recording)
    if "t" not in df:
        df = df.copy(); df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    recording_sec = float(df["t"].iloc[-1])
    bursty, period = detect_bursty(df)
    pad_to = period if bursty else None
    if bursty:
        recording_sec = max(recording_sec, float(split_chunks(df)[-1]["t"].iloc[0]) + period)

    wdf = window_predictions(df, bundle, whole_chunk=bursty)
    # in a bursty recording the natural unit is one burst period, so an "episode" shorter than
    # that cannot be resolved; for a continuous stream keep the 10 s default.
    min_ep = max(MIN_EPISODE_SEC, 1.5 * period) if bursty else MIN_EPISODE_SEC
    ivs = merge_windows(wdf, pad_to, recording_sec, min_ep) if len(wdf) else []
    ivs = split_on_gaps(ivs, wdf, pad_to) if ivs else ivs
    return Timeline(intervals=ivs, recording_sec=recording_sec, hz=HZ,
                    meta={"bursty": bursty, "burst_period_sec": period if bursty else None,
                          "n_windows": int(len(wdf)), "n_samples": int(len(df)),
                          "model": str(model_path) if model_path else type(bundle["model"]).__name__,
                          "feature_subset": bundle.get("subset", "all"), "context_windows": int(bundle.get("context", 1)),
                          "min_episode_sec": round(min_ep, 1), "whole_burst_aggregation": bursty,
                          "window_sec": WIN / HZ, "hop_sec": HOP / HZ})