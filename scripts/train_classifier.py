#!/usr/bin/env python3
"""
Train the 7-class recognition backbone with the official 5-fold user-level split.

    python scripts/train_classifier.py                       # RF, all folds
    python scripts/train_classifier.py --model hgb           # HistGradientBoosting
    python scripts/train_classifier.py --folds 0 1           # quick check
    python scripts/train_classifier.py --trees 100 --max-depth 16

Evaluation is at the MINUTE level (window probabilities averaged per minute), because
the minute is the labelled unit and the timeline unit. Window-level numbers are also
printed for reference.

Outputs (results/):
    confusion_matrix.png      Figure 2 (row-normalised heatmap over 7 classes)
    classifier_metrics.json   per-fold + pooled accuracy, macro-F1, balanced acc, per-class P/R/F1
    feature_importance.csv    top features (RF only)
    model_<name>.joblib       model trained on ALL users, plus feature names, for inference
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import CLASSES, load_folds  # noqa: E402

PRETTY = ["Lying", "Sitting", "Stand (place)", "Stand (moving)", "Walking", "Running", "Bicycling"]


def make_model(name: str, args) -> object:
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=args.trees, max_depth=args.max_depth, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, class_weight="balanced",
            early_stopping=True, random_state=0)
    raise ValueError(name)


def minute_level(prob: np.ndarray, y: np.ndarray, uuid: np.ndarray, ts: np.ndarray):
    """Average window probabilities per (uuid, ts) minute; return (y_true, y_pred) per minute."""
    key = np.char.add(uuid.astype(str), ts.astype(str))
    _, inv = np.unique(key, return_inverse=True)
    n = inv.max() + 1
    P = np.zeros((n, prob.shape[1]))
    np.add.at(P, inv, prob)
    yt = np.zeros(n, dtype=int)
    yt[inv] = y
    return yt, P.argmax(1)


def report(yt, yp, label: str) -> dict:
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=range(7), zero_division=0)
    out = {
        "accuracy": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=range(7), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(CLASSES)},
    }
    print(f"\n{label}: acc={out['accuracy']:.3f}  macroF1={out['macro_f1']:.3f}  "
          f"balAcc={out['balanced_accuracy']:.3f}")
    print(f"  {'class':<20}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>8}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<20}{p[i]:7.3f}{r[i]:7.3f}{f[i]:7.3f}{s[i]:8d}")
    return out


def plot_cm(cm: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), PRETTY, rotation=40, ha="right")
    ax.set_yticks(range(7), PRETTY)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Activity confusion matrix (row-normalised), minute level,\n5-fold user-level CV")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.npz")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", choices=["rf", "hgb"], default="rf")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y, uuid, ts, names = d["X"], d["y"], d["uuid"].astype(str), d["ts"], d["names"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"{len(y)} windows, {len(np.unique(uuid))} users, {X.shape[1]} features")
    print("windows per class:", np.bincount(y, minlength=7).tolist())

    folds = load_folds(args.data_dir)
    present = set(uuid)
    out = Path(args.out); out.mkdir(exist_ok=True)

    all_yt, all_yp, all_ytw, all_ypw, per_fold = [], [], [], [], {}
    for k in args.folds:
        tr = np.isin(uuid, [u for u in folds[k]["train"] if u in present])
        te = np.isin(uuid, [u for u in folds[k]["test"] if u in present])
        if tr.sum() == 0 or te.sum() == 0:
            print(f"fold {k}: skipped (train={tr.sum()}, test={te.sum()} windows in this feature file)")
            continue
        t0 = time.time()
        model = make_model(args.model, args).fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])
        # models may not have seen every class in a small subset; pad columns
        full = np.zeros((len(prob), 7)); full[:, model.classes_] = prob
        ytw, ypw = y[te], full.argmax(1)
        yt, yp = minute_level(full, y[te], uuid[te], ts[te])
        print(f"\n=== fold {k}: train {tr.sum()} / test {te.sum()} windows, "
              f"{len(np.unique(uuid[te]))} test users, {time.time()-t0:.0f}s ===")
        per_fold[k] = report(yt, yp, "minute-level")
        all_yt.append(yt); all_yp.append(yp); all_ytw.append(ytw); all_ypw.append(ypw)

    if not all_yt:
        sys.exit("no folds evaluated")
    yt, yp = np.concatenate(all_yt), np.concatenate(all_yp)
    pooled = report(yt, yp, "POOLED minute-level over evaluated folds")
    pooled_w = report(np.concatenate(all_ytw), np.concatenate(all_ypw), "POOLED window-level")
    cm = confusion_matrix(yt, yp, labels=range(7))
    plot_cm(cm, out / "confusion_matrix.png")

    metrics = {"model": args.model, "folds": {str(k): v for k, v in per_fold.items()},
               "pooled_minute": pooled, "pooled_window": pooled_w,
               "confusion_matrix": cm.tolist(), "classes": CLASSES}
    (out / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))

    # final model on everything, for inference
    print("\ntraining final model on all users ...")
    final = make_model(args.model, args).fit(X, y)
    mpath = out / f"model_{args.model}.joblib"
    joblib.dump({"model": final, "feature_names": list(names), "classes": CLASSES}, mpath)
    size_mb = mpath.stat().st_size / 1e6
    n_nodes = sum(t.tree_.node_count for t in final.estimators_) if args.model == "rf" else None
    print(f"saved {mpath}  ({size_mb:.1f} MB on disk"
          + (f", {n_nodes} tree nodes)" if n_nodes else ")"))
    if args.model == "rf":
        imp = sorted(zip(final.feature_importances_, names), reverse=True)
        with open(out / "feature_importance.csv", "w") as f:
            f.write("importance,feature\n")
            f.writelines(f"{v:.5f},{n}\n" for v, n in imp)
        print("top features:", [n for _, n in imp[:10]])


if __name__ == "__main__":
    main()
