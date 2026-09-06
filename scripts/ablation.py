#!/usr/bin/env python3
"""
Ablation over the recognition design, on the cached features (no re-extraction).

    python scripts/ablation.py                  # folds 0 and 1, RF 100 trees
    python scripts/ablation.py --folds 0 1 2 3 4 --trees 200

Variants
    all_window        baseline: all 175 features, one sample per 5 s window (current)
    inv_window        orientation-invariant features only (|acc|, |gyro|, their correlation)
    noori_window      all features minus pure orientation ones (per-axis mean/median/min/max)
    all_minute        per-minute aggregation: mean + std of the 175 window features
    noori_minute      same aggregation on the no-orientation subset      <- expected best
    hgb_noori_minute  gradient boosting on noori_minute

Every variant is scored at the MINUTE level with the official user-level folds, so numbers are
directly comparable to train_classifier.py. Writes results/ablation.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import CLASSES, load_folds  # noqa: E402

ORIENTATION = {"mean", "median", "min", "max"}
AXES = {"acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}


def subsets(names: np.ndarray) -> dict[str, np.ndarray]:
    names = [str(n) for n in names]
    inv = np.array([n.startswith(("acc_mag", "gyro_mag")) or n == "acc_gyro_mag_corr" for n in names])
    noori = np.array([not (n.split("__")[0] in AXES and n.split("__")[-1] in ORIENTATION) for n in names])
    return {"all": np.ones(len(names), bool), "inv": inv, "noori": noori}


def to_minutes(X, y, uuid, ts):
    """Aggregate windows -> one row per (uuid, ts): [mean(F), std(F)]."""
    key = pd.factorize(pd.Series(uuid).astype(str) + "_" + pd.Series(ts).astype(str))[0]
    n = key.max() + 1
    sums = np.zeros((n, X.shape[1])); sq = np.zeros_like(sums); cnt = np.zeros(n)
    np.add.at(sums, key, X); np.add.at(sq, key, X ** 2); np.add.at(cnt, key, 1)
    mean = sums / cnt[:, None]
    std = np.sqrt(np.maximum(sq / cnt[:, None] - mean ** 2, 0))
    ym = np.zeros(n, int); ym[key] = y
    um = np.empty(n, object); um[key] = uuid
    return np.hstack([mean, std]).astype(np.float32), ym, um


def minute_from_windows(prob, y, uuid, ts):
    key = pd.factorize(pd.Series(uuid).astype(str) + "_" + pd.Series(ts).astype(str))[0]
    n = key.max() + 1
    P = np.zeros((n, prob.shape[1])); np.add.at(P, key, prob)
    yt = np.zeros(n, int); yt[key] = y
    return yt, P.argmax(1)


def score(yt, yp):
    return {"acc": (yt == yp).mean(),
            "macro_f1": f1_score(yt, yp, average="macro", labels=range(7), zero_division=0),
            "bal_acc": balanced_accuracy_score(yt, yp),
            "walk_f1": f1_score(yt == 4, yp == 4), "run_f1": f1_score(yt == 5, yp == 5),
            "bike_f1": f1_score(yt == 6, yp == 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.npz")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y, uuid, ts, names = d["X"], d["y"], d["uuid"].astype(str), d["ts"], d["names"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    subs = subsets(names)
    print(f"{len(y)} windows; feature subsets: " + ", ".join(f"{k}={m.sum()}" for k, m in subs.items()))
    Xm_all, ym, um = to_minutes(X, y, uuid, ts)
    print(f"{len(ym)} minutes after aggregation")
    folds = load_folds()

    def rf():
        return RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=3, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=0)

    def hgb():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced",
                                              early_stopping=True, random_state=0)

    variants = {
        "all_window":       ("window", subs["all"], rf),
        "inv_window":       ("window", subs["inv"], rf),
        "noori_window":     ("window", subs["noori"], rf),
        "all_minute":       ("minute", subs["all"], rf),
        "noori_minute":     ("minute", subs["noori"], rf),
        "hgb_noori_minute": ("minute", subs["noori"], hgb),
    }
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}

    rows = []
    for name, (level, mask, make) in variants.items():
        yts, yps = [], []
        t0 = time.time()
        for k in args.folds:
            tr_u, te_u = set(folds[k]["train"]), set(folds[k]["test"])
            if level == "window":
                tr, te = np.isin(uuid, list(tr_u)), np.isin(uuid, list(te_u))
                m = make().fit(X[tr][:, mask], y[tr])
                prob = np.zeros((te.sum(), 7)); prob[:, m.classes_] = m.predict_proba(X[te][:, mask])
                yt, yp = minute_from_windows(prob, y[te], uuid[te], ts[te])
            else:
                mm = np.concatenate([mask, mask])                     # mean block + std block
                tr, te = np.isin(um, list(tr_u)), np.isin(um, list(te_u))
                m = make().fit(Xm_all[tr][:, mm], ym[tr])
                yt, yp = ym[te], m.predict(Xm_all[te][:, mm])
            yts.append(yt); yps.append(yp)
        s = score(np.concatenate(yts), np.concatenate(yps))
        s.update(variant=name, n_feat=int(mask.sum()) * (2 if level == "minute" else 1), sec=round(time.time() - t0))
        rows.append(s)
        print(f"{name:<18} acc={s['acc']:.3f} macroF1={s['macro_f1']:.3f} balAcc={s['bal_acc']:.3f} "
              f"| walk={s['walk_f1']:.2f} run={s['run_f1']:.2f} bike={s['bike_f1']:.2f} "
              f"| {s['n_feat']} feats, {s['sec']}s", flush=True)

    df = pd.DataFrame(rows)[["variant", "n_feat", "acc", "macro_f1", "bal_acc", "walk_f1", "run_f1", "bike_f1", "sec"]]
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/ablation.csv", index=False)
    print("\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
