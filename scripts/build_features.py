#!/usr/bin/env python3
"""
Extract windowed features for every labelled minute of every user, one cache file per
user, then merge. Resumable: users already in data/processed/per_user/ are skipped.

    python scripts/build_features.py                       # all users, 8 workers
    python scripts/build_features.py --users 2             # smoke test
    python scripts/build_features.py --workers 4           # gentler on a slow NFS mount
    python scripts/build_features.py --cap 400             # <=400 minutes per sedentary class per user
    python scripts/build_features.py --merge-only          # just rebuild features.npz from the cache

Output: data/processed/features.npz with
    X (n_windows, N_FEATURES) float32 | y int class idx | uuid str | ts unix sec | start offset sec | names
Unlabelled minutes are skipped. --cap applies only to CAPPED classes (all rare-class minutes are kept);
it is a deliberate down-sampling of the majority classes and must be stated in the report.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import CLASS_TO_IDX, find_users, load_labels, iter_minutes  # noqa: E402
from src.recognize.features import burst_features, feature_names                    # noqa: E402

CAPPED = {"lying_down", "sitting", "standing_and_moving"}
PER_USER = Path("data/processed/per_user")


def one_user(args):
    uuid, data_dir, cap, seed = args
    out = PER_USER / f"{uuid}.npz"
    t0 = time.time()

    # decide which minutes to keep BEFORE touching any raw file
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

    X, y, ts_, st = [], [], [], []
    for m in iter_minutes(uuid, data_dir, require_both=True):
        if m.label is None or (keep is not None and m.timestamp not in keep):
            continue
        starts, F = burst_features(m.acc, m.gyro)
        if len(F) == 0:
            continue
        X.append(F); y.append(np.full(len(F), CLASS_TO_IDX[m.label]))
        ts_.append(np.full(len(F), m.timestamp)); st.append(starts)

    n_min = len(X)
    if not X:
        np.savez_compressed(out, X=np.zeros((0, 1), np.float32), y=np.zeros(0, int),
                            ts=np.zeros(0, int), start=np.zeros(0), n_min=0)
        return uuid, 0, 0, time.time() - t0
    y = np.concatenate(y)
    np.savez_compressed(out, X=np.vstack(X), y=y, ts=np.concatenate(ts_),
                        start=np.concatenate(st), n_min=n_min)
    return uuid, n_min, len(y), time.time() - t0


def merge(out_path: Path) -> None:
    parts = sorted(PER_USER.glob("*.npz"))
    Xs, ys, tss, sts, uus = [], [], [], [], []
    for p in parts:
        d = np.load(p)
        if len(d["y"]) == 0:
            continue
        Xs.append(d["X"]); ys.append(d["y"]); tss.append(d["ts"]); sts.append(d["start"])
        uus.append(np.full(len(d["y"]), p.stem))
    y = np.concatenate(ys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=np.vstack(Xs), y=y, ts=np.concatenate(tss),
                        start=np.concatenate(sts), uuid=np.concatenate(uus),
                        names=np.array(feature_names()))
    print(f"\nmerged {len(parts)} users -> {out_path}: {len(y)} windows")
    print("windows per class:", np.bincount(y, minlength=7).tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--users", type=int, default=None, help="limit to first N users")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=int, default=400,
                    help="max minutes per user for each of %s (0 = keep all)" % sorted(CAPPED))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/features.npz")
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
        for i, (uuid, n_min, n_win, dt) in enumerate(pool.imap_unordered(one_user, jobs), 1):
            print(f"  [{i}/{len(todo)}] {uuid[:8]}  {n_min:5d} min  {n_win:6d} win  "
                  f"{dt:5.0f}s  (elapsed {time.time()-t0:.0f}s)", flush=True)

    merge(Path(args.out))


if __name__ == "__main__":
    main()
