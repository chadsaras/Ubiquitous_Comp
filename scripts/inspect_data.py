#!/usr/bin/env python3
"""
Day-1 sanity checks. Run after prepare_data.py has fetched at least labels + folds
(and ideally raw acc/gyro for one user).

    python scripts/inspect_data.py               # first user found
    python scripts/inspect_data.py --uuid <UUID>

Prints: raw file naming, per-class minute counts, burst length / sampling rate,
and saves results/fig_walk_vs_sit_magnitude.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import (CLASSES, find_users, load_labels,  # noqa: E402
                                 read_burst, resample)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--uuid", default=None)
    args = ap.parse_args()
    root = Path(args.data_dir) / "raw"

    users = find_users(args.data_dir)
    print(f"users with label files: {len(users)}")
    uuid = args.uuid or users[0]
    print(f"inspecting user {uuid}\n")

    # 1. labels
    lab = load_labels(uuid, args.data_dir)
    counts = lab["label"].value_counts(dropna=False).reindex(CLASSES + [None])
    print("labelled minutes per class:")
    print(counts.to_string(), "\n")
    print(f"minutes with >1 of our labels: {(lab[CLASSES].sum(axis=1) > 1).sum()}")
    print(f"minutes with none of our labels: {lab['label'].isna().sum()}\n")

    # 2. raw file naming + one burst
    for mod in ("raw_acc", "raw_gyro"):
        d = next(iter((root / mod).rglob(uuid)), None)
        if d is None:
            print(f"[{mod}] no folder for this user (not downloaded yet?)")
            continue
        files = sorted(d.iterdir())
        print(f"[{mod}] {len(files)} files, e.g. {files[0].name}")
        b = next((read_burst(f) for f in files[:20] if read_burst(f) is not None), None)
        if b is None:
            print("   could not read any of the first 20 bursts")
            continue
        dur = b[-1, 0] - b[0, 0]
        hz = (len(b) - 1) / dur if dur > 0 else float("nan")
        print(f"   burst: {len(b)} samples over {dur:.2f} s  -> ~{hz:.1f} Hz native; "
              f"{len(resample(b))} samples after resampling to 25 Hz\n")

    # 3. walking vs sitting magnitude plot (needs raw acc)
    acc_dir = next(iter((root / "raw_acc").rglob(uuid)), None)
    if acc_dir is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    def first_burst(cls: str):
        for ts in lab.index[lab["label"] == cls]:
            hits = list(acc_dir.glob(f"{ts}*"))
            if hits and (b := read_burst(hits[0])) is not None:
                return resample(b)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, cls in zip(axes, ("sitting", "walking")):
        xyz = first_burst(cls)
        if xyz is None:
            ax.set_title(f"{cls}: no burst found")
            continue
        mag = np.linalg.norm(xyz, axis=1)
        t = np.arange(len(mag)) / 25.0
        ax.plot(t, mag, lw=0.8)
        ax.set_title(f"{cls}  (std={mag.std():.3f})")
        ax.set_xlabel("seconds")
    axes[0].set_ylabel("|acc| (g)")
    out = Path("results") / "fig_walk_vs_sit_magnitude.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
