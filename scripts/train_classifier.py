#!/usr/bin/env python3
"""
Train the 7-class recognition backbone with the official 5-fold user-level split.

    python scripts/train_classifier.py                       # minute-level, no-orientation subset, cleaned
    python scripts/train_classifier.py --level window        # old per-window baseline
    python scripts/train_classifier.py --no-clean            # keep physically-still 'active' minutes in training
    python scripts/train_classifier.py --subset all
    python scripts/train_classifier.py --model hgb --folds 0 1

Design (see results/ablation.csv):
  * level=minute: each labelled minute is one sample = [mean, std] of its 5 s window feature
    vectors (7 windows per 20 s burst). +5 points over per-window classification.
  * subset=noori: drop per-axis mean/median/min/max (pure phone-orientation features).
  * clean: drop training minutes labelled walking/running/bicycling whose |acc| std never exceeded
    STILL_G in any window - the phone was not on the body, so the label cannot be learned.
    Test data is NEVER cleaned for the headline numbers; a second table shows the
    'signal-consistent' subset so the label-noise cost is visible.

Outputs (results/):
    confusion_matrix.png      Figure 2, all test minutes
    classifier_metrics.json   per-fold + pooled metrics (all and signal-consistent), per-class P/R/F1
    feature_importance.csv    (RF)
    model_<name>.joblib       bundle: model, feature mask, level, context, classes -> used by timeline.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import CLASSES, load_folds  # noqa: E402

PRETTY = ["Lying", "Sitting", "Stand (place)", "Stand (moving)", "Walking", "Running", "Bicycling"]
ACTIVE = {4, 5, 6}            # walking, running, bicycling
STILL_G = 0.03                # |acc| std (g) below which the phone is considered motionless
ORIENTATION = {"mean", "median", "min", "max"}
AXES = {"acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
CONTEXT = 7                   # windows aggregated per minute (matches 20 s burst / 2.5 s hop)


def feature_mask(names, subset: str) -> np.ndarray:
    names = [str(n) for n in names]
    if subset == "all":
        return np.ones(len(names), bool)
    if subset == "inv":
        return np.array([n.startswith(("acc_mag", "gyro_mag")) or n == "acc_gyro_mag_corr" for n in names])
    if subset == "noori":
        return np.array([not (n.split("__")[0] in AXES and n.split("__")[-1] in ORIENTATION) for n in names])
    raise ValueError(subset)


def aggregate_minutes(X, y, uuid, ts, still_col):
    """windows -> one row per (uuid, ts): [mean(F), std(F)], label, uuid, ts, max |acc| std."""
    key, uniq = pd.factorize(pd.Series(uuid).astype(str) + "\t" + pd.Series(ts).astype(str))
    n = len(uniq)
    sums = np.zeros((n, X.shape[1])); sq = np.zeros_like(sums); cnt = np.zeros(n)
    np.add.at(sums, key, X); np.add.at(sq, key, X.astype(np.float64) ** 2); np.add.at(cnt, key, 1)
    mean = sums / cnt[:, None]
    std = np.sqrt(np.maximum(sq / cnt[:, None] - mean ** 2, 0))
    ym = np.zeros(n, int); ym[key] = y
    um = np.array([k.split("\t")[0] for k in uniq]); tm = np.array([int(k.split("\t")[1]) for k in uniq])
    still_max = np.zeros(n); np.maximum.at(still_max, key, X[:, still_col])
    return np.hstack([mean, std]).astype(np.float32), ym, um, tm, still_max


def make_model(name, args):
    if name == "rf":
        return RandomForestClassifier(n_estimators=args.trees, max_depth=args.max_depth, min_samples_leaf=3,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              class_weight="balanced", early_stopping=True, random_state=0)
    raise ValueError(name)


def report(yt, yp, label) -> dict:
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=range(7), zero_division=0)
    out = {"n": int(len(yt)), "accuracy": float((yt == yp).mean()),
           "macro_f1": float(f1_score(yt, yp, average="macro", labels=range(7), zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
           "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                         for i, c in enumerate(CLASSES)}}
    print(f"\n{label}: n={len(yt)}  acc={out['accuracy']:.3f}  macroF1={out['macro_f1']:.3f}  "
          f"balAcc={out['balanced_accuracy']:.3f}")
    print(f"  {'class':<20}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>8}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<20}{p[i]:7.3f}{r[i]:7.3f}{f[i]:7.3f}{s[i]:8d}")
    return out


def plot_cm(cm, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), PRETTY, rotation=40, ha="right"); ax.set_yticks(range(7), PRETTY)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.npz")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", choices=["rf", "hgb"], default="rf")
    ap.add_argument("--level", choices=["minute", "window"], default="minute")
    ap.add_argument("--subset", choices=["all", "noori", "inv"], default="noori")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None, help="suffix for output files")
    args = ap.parse_args()
    clean = not args.no_clean
    tag = args.tag or f"{args.model}_{args.level}_{args.subset}{'_clean' if clean else ''}"

    d = np.load(args.features, allow_pickle=True)
    X, y, uuid, ts, names = d["X"], d["y"], d["uuid"].astype(str), d["ts"], d["names"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = feature_mask(names, args.subset)
    still_col = list(map(str, names)).index("acc_mag__std")
    print(f"{len(y)} windows, {len(np.unique(uuid))} users; subset={args.subset} ({mask.sum()} feats), "
          f"level={args.level}, clean={clean}")

    if args.level == "minute":
        Xs, ys, us, tss, still = aggregate_minutes(X, y, uuid, ts, still_col)
        fmask = np.concatenate([mask, mask])
        print(f"{len(ys)} minutes; per class: {np.bincount(ys, minlength=7).tolist()}")
    else:
        Xs, ys, us, tss, still, fmask = X, y, uuid, ts, X[:, still_col], mask
    Xs = Xs[:, fmask]
    consistent = ~(np.isin(ys, list(ACTIVE)) & (still < STILL_G))     # label agrees with the signal
    print(f"label/signal-inconsistent active samples: {(~consistent).sum()} of {np.isin(ys, list(ACTIVE)).sum()} active "
          f"({(~consistent).sum()/max(1,np.isin(ys, list(ACTIVE)).sum()):.1%})")

    folds = load_folds(args.data_dir)
    present = set(us)
    out = Path(args.out); out.mkdir(exist_ok=True)

    def predict_minutes(model, Xte, yte, ute, tste):
        prob = np.zeros((len(Xte), 7)); prob[:, model.classes_] = model.predict_proba(Xte)
        if args.level == "minute":
            return yte, prob.argmax(1), np.arange(len(yte))
        key = pd.factorize(pd.Series(ute) + "\t" + pd.Series(tste).astype(str))[0]
        P = np.zeros((key.max() + 1, 7)); np.add.at(P, key, prob)
        yt = np.zeros(key.max() + 1, int); yt[key] = yte
        return yt, P.argmax(1), key

    all_yt, all_yp, all_cons, per_fold = [], [], [], {}
    for k in args.folds:
        tr = np.isin(us, [u for u in folds[k]["train"] if u in present])
        te = np.isin(us, [u for u in folds[k]["test"] if u in present])
        if clean:
            tr = tr & consistent
        if tr.sum() == 0 or te.sum() == 0:
            print(f"fold {k}: skipped"); continue
        t0 = time.time()
        model = make_model(args.model, args).fit(Xs[tr], ys[tr])
        yt, yp, key = predict_minutes(model, Xs[te], ys[te], us[te], tss[te])
        cons_te = consistent[te]
        if args.level == "window":       # minute is consistent if all its windows are
            c = np.ones(key.max() + 1, bool); np.logical_and.at(c, key, cons_te); cons_te = c
        print(f"\n=== fold {k}: train {tr.sum()} / test {te.sum()} samples, "
              f"{len(np.unique(us[te]))} test users, {time.time()-t0:.0f}s ===")
        per_fold[k] = {"all": report(yt, yp, "all test minutes"),
                       "signal_consistent": report(yt[cons_te], yp[cons_te], "signal-consistent test minutes")}
        all_yt.append(yt); all_yp.append(yp); all_cons.append(cons_te)

    if not all_yt:
        sys.exit("no folds evaluated")
    yt, yp, cons = map(np.concatenate, (all_yt, all_yp, all_cons))
    pooled_all = report(yt, yp, "POOLED, all test minutes (headline)")
    pooled_cons = report(yt[cons], yp[cons], "POOLED, signal-consistent test minutes")
    cm = confusion_matrix(yt, yp, labels=range(7))
    plot_cm(cm, out / f"confusion_matrix.png",
            f"Activity confusion matrix (row-normalised), minute level,\n5-fold user-level CV, {tag}")
    plot_cm(confusion_matrix(yt[cons], yp[cons], labels=range(7)), out / f"confusion_matrix_consistent.png",
            f"Confusion matrix on signal-consistent test minutes\n(active labels with a moving phone), {tag}")

    (out / "classifier_metrics.json").write_text(json.dumps({
        "config": vars(args) | {"clean": clean, "n_features": int(fmask.sum()), "still_g": STILL_G, "context": CONTEXT},
        "folds": {str(k): v for k, v in per_fold.items()},
        "pooled_all": pooled_all, "pooled_signal_consistent": pooled_cons,
        "confusion_matrix": cm.tolist(), "classes": CLASSES}, indent=2))

    print("\ntraining final model on all users ...")
    tr = consistent if clean else np.ones(len(ys), bool)
    final = make_model(args.model, args).fit(Xs[tr], ys[tr])
    bundle = {"model": final, "feature_names": [str(n) for n in names], "mask": mask, "level": args.level,
              "context": CONTEXT if args.level == "minute" else 1, "classes": CLASSES, "subset": args.subset,
              "cleaned": clean, "still_g": STILL_G}
    mpath = out / f"model_{args.model}.joblib"
    joblib.dump(bundle, mpath, compress=3)
    size_mb = mpath.stat().st_size / 1e6
    extra = f", {sum(t.tree_.node_count for t in final.estimators_)} tree nodes" if args.model == "rf" else ""
    print(f"saved {mpath}  ({size_mb:.1f} MB on disk{extra})")
    if args.model == "rf":
        fn = [str(n) for n in names]
        used = [n for n, m in zip(fn, mask) if m]
        used = [f"mean:{n}" for n in used] + [f"std:{n}" for n in used] if args.level == "minute" else used
        imp = sorted(zip(final.feature_importances_, used), reverse=True)
        with open(out / "feature_importance.csv", "w") as f:
            f.write("importance,feature\n"); f.writelines(f"{v:.5f},{n}\n" for v, n in imp)
        print("top features:", [n for _, n in imp[:10]])


if __name__ == "__main__":
    main()
