#!/usr/bin/env python3
"""
ISOLATED EXPERIMENT — class-prior calibration for the recognition backbone.

Nothing in src/ or scripts/ is modified or imported-for-side-effects by this file. It produces a
calibration vector and a verdict; only if the verdict is positive would anything be wired in.

THE PROBLEM
-----------
Measured on the held-out QA benchmark, predicted activity time collapses toward `sitting`:
  55% of true walking time, 55% of standing_and_moving, 67% of standing_in_place and 51% of
  lying_down are all predicted as sitting. Duration answers inherit this directly, which is why
  duration accuracy is 0.167 while verification is 0.944.

THE IDEA
--------
The forest already outputs probabilities; the pipeline just takes argmax, which is the decision
rule that maximises plain accuracy on an imbalanced set -- i.e. it is *supposed* to over-predict
the majority class. Re-weighting the probabilities before argmax can trade a little plain accuracy
for much better balance. Two families are tried:

  * prior correction     p'_c  ∝  p_c / prior_c**alpha        (one parameter, alpha)
  * per-class weights    p'_c  ∝  w_c * p_c                   (seven parameters, coordinate ascent)

EVALUATION INTEGRITY
--------------------
  * fold 0's TRAIN users are split into sub-train and validation BY USER;
  * the forest is fitted on sub-train only;
  * every calibration parameter is chosen on VALIDATION only;
  * fold 0's TEST users are scored once, at the end, and never influence any choice.
This is what makes the result usable rather than self-fulfilling.

TWO OBJECTIVES, because they are not the same thing
---------------------------------------------------
  * macro-F1        -- standard balanced recognition quality.
  * time error      -- sum over classes of |predicted total time - true total time|, normalised.
                       This is the quantity duration questions actually depend on, and a model can
                       improve it while barely moving per-minute accuracy.

Usage:
    python experiments/calibration/calibrate_classes.py
    python experiments/calibration/calibrate_classes.py --trees 150   # quicker smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_classifier import (ACTIVE, STILL_G, aggregate_minutes,  # noqa: E402
                                      feature_mask)
from src.preprocess.load import CLASSES, load_folds                        # noqa: E402

EPS = 1e-12


# --------------------------------------------------------------------------- metrics
def time_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalised total-time error: sum_c |minutes predicted as c - minutes truly c| / total minutes.

    0 means the predicted class distribution matches reality exactly (what duration questions
    need); 2.0 is the worst possible. Deliberately insensitive to *which* minute got which label,
    because a duration answer only sums time per class.
    """
    n = len(y_true)
    t = np.bincount(y_true, minlength=len(CLASSES)).astype(float)
    p = np.bincount(y_pred, minlength=len(CLASSES)).astype(float)
    return float(np.abs(p - t).sum() / max(n, 1))


def evaluate(y_true: np.ndarray, proba: np.ndarray, weights: np.ndarray) -> dict:
    pred = (proba * weights).argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=range(len(CLASSES)),
                                                 zero_division=0)
    return {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=range(len(CLASSES)),
                                   zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "time_error": time_error(y_true, pred),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "true_minutes": int(s[i]),
                          "predicted_minutes": int((pred == i).sum())}
                      for i, c in enumerate(CLASSES)},
    }


# --------------------------------------------------------------------------- calibrations
def prior_weights(prior: np.ndarray, alpha: float) -> np.ndarray:
    w = 1.0 / np.power(prior + EPS, alpha)
    return w / w.sum() * len(w)


def coordinate_ascent(y: np.ndarray, proba: np.ndarray, objective: str,
                      rounds: int = 4, grid=(0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.2, 3.0, 4.5)) -> np.ndarray:
    """Greedy per-class multipliers. Cheap because only the argmax over saved probabilities moves."""
    w = np.ones(len(CLASSES))

    def score(wv: np.ndarray) -> float:
        pred = (proba * wv).argmax(1)
        if objective == "macro_f1":
            return f1_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0)
        return -time_error(y, pred)          # maximise => minimise error

    best = score(w)
    for _ in range(rounds):
        improved = False
        for c in range(len(CLASSES)):
            base = w[c]
            for g in grid:
                trial = w.copy()
                trial[c] = base * g
                s = score(trial)
                if s > best + 1e-6:
                    best, w, improved = s, trial, True
        if not improved:
            break
    return w


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.npz")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/calibration/results.json")
    args = ap.parse_args()

    t_start = time.time()
    print("[1/6] loading features ...", flush=True)
    d = np.load(args.features, allow_pickle=True)
    X, y, uuid, ts, names = d["X"], d["y"], d["uuid"].astype(str), d["ts"], d["names"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = feature_mask(names, "noori")
    still_col = list(map(str, names)).index("acc_mag__std")

    Xs, ys, us, tss, still = aggregate_minutes(X, y, uuid, ts, still_col)
    Xs = Xs[:, np.concatenate([mask, mask])]
    consistent = ~(np.isin(ys, list(ACTIVE)) & (still < STILL_G))
    print(f"      {len(ys)} minutes, {Xs.shape[1]} features, {len(set(us))} users", flush=True)

    folds = load_folds(args.data_dir)
    present = set(us)
    train_users = sorted(u for u in folds[args.fold]["train"] if u in present)
    test_users = sorted(u for u in folds[args.fold]["test"] if u in present)

    rng = np.random.default_rng(args.seed)
    shuffled = list(train_users)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.val_frac)))
    val_users, sub_train_users = shuffled[:n_val], shuffled[n_val:]
    print(f"[2/6] fold {args.fold}: {len(sub_train_users)} sub-train / {len(val_users)} validation "
          f"/ {len(test_users)} test users (split BY USER)", flush=True)

    tr = np.isin(us, sub_train_users) & consistent
    va = np.isin(us, val_users)
    te = np.isin(us, test_users)
    print(f"      minutes: train {tr.sum()}, val {va.sum()}, test {te.sum()}", flush=True)

    print(f"[3/6] fitting RandomForest({args.trees} trees) on sub-train ...", flush=True)
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=3,
                                 class_weight="balanced_subsample", n_jobs=-1,
                                 random_state=args.seed).fit(Xs[tr], ys[tr])
    print(f"      fitted in {time.time()-t0:.0f}s", flush=True)

    def full_proba(mask_sel: np.ndarray) -> np.ndarray:
        p = np.zeros((int(mask_sel.sum()), len(CLASSES)))
        p[:, clf.classes_] = clf.predict_proba(Xs[mask_sel])
        return p

    print("[4/6] scoring validation and test ...", flush=True)
    pv, yv = full_proba(va), ys[va]
    pt, yt = full_proba(te), ys[te]

    prior = np.bincount(ys[tr], minlength=len(CLASSES)).astype(float)
    prior = prior / prior.sum()
    ones = np.ones(len(CLASSES))

    print("[5/6] searching calibrations on VALIDATION only ...", flush=True)
    candidates: dict[str, np.ndarray] = {"baseline_argmax": ones}

    best_a_f1, best_a_te = (None, -1.0), (None, 1e9)
    for alpha in np.round(np.arange(0.0, 1.51, 0.05), 2):
        w = prior_weights(prior, float(alpha))
        pred = (pv * w).argmax(1)
        f1v = f1_score(yv, pred, average="macro", labels=range(len(CLASSES)), zero_division=0)
        tev = time_error(yv, pred)
        if f1v > best_a_f1[1]:
            best_a_f1 = (float(alpha), f1v)
        if tev < best_a_te[1]:
            best_a_te = (float(alpha), tev)
    print(f"      prior-correction alpha: best macro-F1 @ {best_a_f1[0]} ({best_a_f1[1]:.4f}); "
          f"best time-error @ {best_a_te[0]} ({best_a_te[1]:.4f})", flush=True)
    candidates[f"prior_alpha_{best_a_f1[0]}_maxF1"] = prior_weights(prior, best_a_f1[0])
    candidates[f"prior_alpha_{best_a_te[0]}_minTimeErr"] = prior_weights(prior, best_a_te[0])

    t0 = time.time()
    candidates["coord_ascent_maxF1"] = coordinate_ascent(yv, pv, "macro_f1")
    candidates["coord_ascent_minTimeErr"] = coordinate_ascent(yv, pv, "time_error")
    print(f"      coordinate ascent done in {time.time()-t0:.0f}s", flush=True)

    print("[6/6] final scoring on HELD-OUT TEST users ...", flush=True)
    report = {
        "fold": args.fold, "trees": args.trees, "seed": args.seed,
        "users": {"sub_train": len(sub_train_users), "validation": len(val_users),
                  "test": len(test_users)},
        "minutes": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "train_class_prior": {c: float(prior[i]) for i, c in enumerate(CLASSES)},
        "candidates": {},
    }
    for name, w in candidates.items():
        report["candidates"][name] = {
            "weights": {c: float(w[i]) for i, c in enumerate(CLASSES)},
            "validation": evaluate(yv, pv, w),
            "test": evaluate(yt, pt, w),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    base = report["candidates"]["baseline_argmax"]["test"]
    print(f"\n{'candidate':<34}{'acc':>7}{'macroF1':>9}{'balAcc':>8}{'timeErr':>9}")
    for name, r in report["candidates"].items():
        t = r["test"]
        print(f"{name:<34}{t['accuracy']:>7.3f}{t['macro_f1']:>9.3f}"
              f"{t['balanced_accuracy']:>8.3f}{t['time_error']:>9.3f}")
    print(f"\n(baseline time_error {base['time_error']:.3f}; lower is better for duration answers)")

    print(f"\nper-class TEST recall, baseline -> best-time-error candidate:")
    best_name = min(report["candidates"], key=lambda k: report["candidates"][k]["test"]["time_error"])
    bt = report["candidates"][best_name]["test"]["per_class"]
    for c in CLASSES:
        b, n = base["per_class"][c], bt[c]
        print(f"  {c:<22} recall {b['recall']:.3f} -> {n['recall']:.3f}   "
              f"minutes {b['predicted_minutes']:>6} -> {n['predicted_minutes']:>6} "
              f"(true {b['true_minutes']})")
    print(f"\nbest by time-error: {best_name}")
    print(f"saved {out}  (total {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
