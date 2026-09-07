#!/usr/bin/env python3
"""
Cache fixed-length raw (aligned, 25 Hz, g-normalised) bursts for every labelled
minute of every user -- the input the CNN recognizer (src/recognize/nn_models.py)
trains on, as opposed to scripts/build_features.py's hand-engineered features.

Mirrors build_features.py's structure and options (same resumable per-user
caching, same --cap down-sampling of the majority classes with the same seed)
so the CNN trains on the IDENTICAL set of minutes as the tree models -- required
for the comparison in scripts/train_cnn.py to be fair.

    python scripts/build_raw_cache.py                # all users, 8 workers
    python scripts/build_raw_cache.py --users 2       # smoke test
    python scripts/build_raw_cache.py --workers 4     # gentler on a slow NFS mount
    python scripts/build_raw_cache.py --merge-only    # just rebuild raw_minutes.npz from cache

Output: data/processed/raw_minutes.npz with
    X (n_minutes, RAW_T, 6) float32 | y int class idx | n_valid int (real, unpadded
    length of each burst) | ts unix sec | uuid str
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import CLASS_TO_IDX, find_users, load_labels, iter_minutes  # noqa: E402
from src.recognize.nn_data import RAW_T, CHANNELS, fixed_burst                        # noqa: E402

CAPPED = {"lying_down", "sitting", "standing_and_moving"}
PER_USER = Path("data/processed/per_user_raw")


def one_user(args):
    uuid, data_dir, cap, seed = args
    out = PER_USER / f"{uuid}.npz"
    t0 = time.time()

    # decide which minutes to keep BEFORE touching any raw file (same policy as
    # build_features.py, same seed -> same minutes selected for the capped classes)
    keep = None
    if cap:
        lab = load_labels(uuid, data_dir)
        rng = np.random.default_rng(seed)
        keep = set()
        for cls, grp in lab.groupby("label"):
            ts = grp.index.to_numpy()
            if cls in CAPPED and len(ts) > cap:
                ts = rng.choice(ts, cap, replace=False)
            keep.update(int(t) for t in ts)

    X, y, nv, ts_ = [], [], [], []
    for m in iter_minutes(uuid, data_dir, require_both=True):
        if m.label is None or (keep is not None and m.timestamp not in keep):
            continue
        burst, n = fixed_burst(m.acc, m.gyro)
        X.append(burst); y.append(CLASS_TO_IDX[m.label]); nv.append(n); ts_.append(m.timestamp)

    if not X:
        np.savez_compressed(out, X=np.zeros((0, RAW_T, CHANNELS), np.float32),
                            y=np.zeros(0, int), n_valid=np.zeros(0, int), ts=np.zeros(0, int))
        return uuid, 0, time.time() - t0
    np.savez_compressed(out, X=np.stack(X).astype(np.float32), y=np.array(y, int),
                        n_valid=np.array(nv, int), ts=np.array(ts_, int))
    return uuid, len(y), time.time() - t0


def merge(out_path: Path) -> None:
    parts = sorted(PER_USER.glob("*.npz"))
    Xs, ys, nvs, tss, uus = [], [], [], [], []
    for p in parts:
        d = np.load(p)
        if len(d["y"]) == 0:
            continue
        Xs.append(d["X"]); ys.append(d["y"]); nvs.append(d["n_valid"]); tss.append(d["ts"])
        uus.append(np.full(len(d["y"]), p.stem))
    y = np.concatenate(ys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=np.vstack(Xs), y=y, n_valid=np.concatenate(nvs),
                        ts=np.concatenate(tss), uuid=np.concatenate(uus))
    print(f"\nmerged {len(parts)} users -> {out_path}: {len(y)} minutes")
    print("minutes per class:", np.bincount(y, minlength=7).tolist())
    size_mb = out_path.stat().st_size / 1e6
    print(f"file size: {size_mb:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--users", type=int, default=None, help="limit to first N users")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=int, default=400,
                    help="max minutes per user for each of %s (0 = keep all)" % sorted(CAPPED))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/raw_minutes.npz")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    PER_USER.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(Path(args.out)); return

    users = find_users(args.data_dir)[: args.users]
    todo = [u for u in users if not (PER_USER / f"{u}.npz").exists()]
    print(f"{len(users)} users, {len(users)-len(todo)} cached, {len(todo)} to do, "
          f"cap={args.cap or 'none'}, workers={args.workers}", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        jobs = [(u, args.data_dir, args.cap, args.seed) for u in todo]
        for i, (uuid, n_min, dt) in enumerate(pool.imap_unordered(one_user, jobs), 1):
            print(f"  [{i}/{len(todo)}] {uuid[:8]}  {n_min:5d} min  {dt:5.0f}s  "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    merge(Path(args.out))


if __name__ == "__main__":
    main()
