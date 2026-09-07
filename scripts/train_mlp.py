#!/usr/bin/env python3
"""
Train the final-pick model: a small feedforward net (FeatureMLP) over Random
Forest's own 302 engineered features, evaluated on the exact same official
5-fold user-level split as scripts/train_classifier.py and scripts/train_cnn.py,
so all four models -- RF, HGB, TinyCNN, CNN+GRU, and this one -- are directly
comparable.

    python scripts/train_mlp.py                 # CPU, all 5 folds

Prerequisite: scripts/build_features.py must have already produced
data/processed/features.npz (it has -- this is the same cache RF/HGB train on).

Design: unlike Model 1/2 (raw signal, capacity-matched to each other), this
model trains on the SAME mean+std-aggregated, noori-subset (302) features RF
uses -- see src/recognize/nn_models.py's module docstring for why. Same
class-weighted loss, same "clean training / honest test" policy, same
validation-slice-from-train-users-only early stopping as train_cnn.py.

Outputs (results/):
    confusion_matrix_mlp.png
    classifier_metrics_mlp.json   same schema as the other classifier_metrics_*.json
    model_mlp.pt                  gitignored, like the other model files
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import CLASSES, load_folds       # noqa: E402
from src.recognize.nn_models import FeatureMLP, count_params  # noqa: E402

PRETTY = ["Lying", "Sitting", "Stand (place)", "Stand (moving)", "Walking", "Running", "Bicycling"]
ACTIVE = {4, 5, 6}
STILL_G = 0.03
ORIENTATION = {"mean", "median", "min", "max"}
AXES = {"acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}


# --------------------------------------------------------------------------- feature prep (mirrors train_classifier.py)
def feature_mask(names, subset: str) -> np.ndarray:
    names = [str(n) for n in names]
    if subset == "all":
        return np.ones(len(names), bool)
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


# --------------------------------------------------------------------------- training (mirrors train_cnn.py)
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, n_classes: int = 7) -> torch.Tensor:
    count = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = len(y) / (n_classes * np.maximum(count, 1))
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, device, optimizer=None, criterion=None) -> tuple[float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    tot_loss, n, yt_all, yp_all = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * len(yb); n += len(yb)
            yt_all.append(yb.cpu().numpy()); yp_all.append(logits.argmax(1).cpu().numpy())
    return tot_loss / max(n, 1), np.concatenate(yt_all), np.concatenate(yp_all)


def train_one_model(Xtr, ytr, Xval, yval, device, n_in, epochs, batch_size, lr, wd, patience, seed):
    set_seed(seed)
    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
                           batch_size=batch_size, shuffle=True, drop_last=len(ytr) > batch_size)
    val_loader = DataLoader(TensorDataset(torch.tensor(Xval, dtype=torch.float32), torch.tensor(yval, dtype=torch.long)),
                            batch_size=256, shuffle=False)

    model = FeatureMLP(n_in=n_in).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(weight=class_weights(ytr).to(device))

    best_f1, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    history = []
    for ep in range(1, epochs + 1):
        tr_loss, _, _ = run_epoch(model, tr_loader, device, optimizer, criterion)
        val_loss, yv, pv = run_epoch(model, val_loader, device, criterion=criterion)
        val_f1 = f1_score(yv, pv, average="macro", labels=range(7), zero_division=0)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"    epoch {ep:3d}  train_loss={tr_loss:.3f}  val_loss={val_loss:.3f}  val_macroF1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop at epoch {ep} (best val macroF1={best_f1:.3f})")
                break
    model.load_state_dict(best_state)
    return model, {"best_val_macro_f1": best_f1, "epochs_run": len(history)}


@torch.no_grad()
def predict(model, X, device, batch_size=256) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    out = []
    for (xb,) in loader:
        out.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


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


def plot_cm(cm, path, title) -> None:
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


def user_val_split(train_users: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    users = np.array(sorted(train_users))
    n_val = max(1, round(len(users) * val_frac))
    val = set(rng.choice(users, n_val, replace=False).tolist())
    return [u for u in train_users if u not in val], list(val)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.npz")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    clean = not args.no_clean

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available; falling back to cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"device: {device}  arch: mlp (FeatureMLP over RF's engineered features)")

    d = np.load(args.features, allow_pickle=True)
    X, y, uuid, ts, names = d["X"], d["y"], d["uuid"].astype(str), d["ts"], d["names"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = feature_mask(names, "noori")
    still_col = list(map(str, names)).index("acc_mag__std")
    print(f"{len(y)} windows, {len(np.unique(uuid))} users; noori subset ({mask.sum()} feats)")

    Xs, ys, us, tss, still = aggregate_minutes(X, y, uuid, ts, still_col)
    fmask = np.concatenate([mask, mask])
    Xs = Xs[:, fmask]
    n_in = Xs.shape[1]
    print(f"{len(ys)} minutes; {n_in} features per minute; per class: {np.bincount(ys, minlength=7).tolist()}")

    consistent = ~(np.isin(ys, list(ACTIVE)) & (still < STILL_G))
    n_active = np.isin(ys, list(ACTIVE)).sum()
    print(f"label/signal-inconsistent active minutes: {(~consistent).sum()} of {n_active} active "
          f"({(~consistent).sum()/max(1,n_active):.1%})")

    folds = load_folds(args.data_dir)
    present = set(us)
    out = Path(args.out); out.mkdir(exist_ok=True)

    all_yt, all_yp, all_cons, per_fold = [], [], [], {}
    for k in args.folds:
        tr_users_all = [u for u in folds[k]["train"] if u in present]
        te_users = [u for u in folds[k]["test"] if u in present]
        tr_users, val_users = user_val_split(tr_users_all, args.val_frac, args.seed + k)

        tr_mask = np.isin(us, tr_users)
        val_mask = np.isin(us, val_users)
        te_mask = np.isin(us, te_users)
        if clean:
            tr_mask = tr_mask & consistent
            val_mask = val_mask & consistent
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            print(f"fold {k}: skipped"); continue

        t0 = time.time()
        print(f"\n=== fold {k}: train {tr_mask.sum()} ({len(tr_users)} users) / "
              f"val {val_mask.sum()} ({len(val_users)} users) / test {te_mask.sum()} ({len(te_users)} users) ===")
        model, hist = train_one_model(Xs[tr_mask], ys[tr_mask], Xs[val_mask], ys[val_mask], device, n_in,
                                      args.epochs, args.batch_size, args.lr, args.weight_decay,
                                      args.patience, args.seed + k)
        yp = predict(model, Xs[te_mask], device)
        yt = ys[te_mask]
        cons_te = consistent[te_mask]
        print(f"  fold {k} done in {time.time()-t0:.0f}s")
        per_fold[k] = {"all": report(yt, yp, "all test minutes"),
                       "signal_consistent": report(yt[cons_te], yp[cons_te], "signal-consistent test minutes"),
                       "training": {"best_val_macro_f1": hist["best_val_macro_f1"], "epochs_run": hist["epochs_run"]}}
        all_yt.append(yt); all_yp.append(yp); all_cons.append(cons_te)

    if not all_yt:
        sys.exit("no folds evaluated")
    yt, yp, cons = map(np.concatenate, (all_yt, all_yp, all_cons))
    pooled_all = report(yt, yp, "POOLED, all test minutes (headline)")
    pooled_cons = report(yt[cons], yp[cons], "POOLED, signal-consistent test minutes")
    cm = confusion_matrix(yt, yp, labels=range(7))
    plot_cm(cm, out / "confusion_matrix_mlp.png",
            "Activity confusion matrix (row-normalised), FeatureMLP,\n5-fold user-level CV")

    print("\ntraining final model on all users (small internal val slice for early stopping)...")
    all_users = sorted(present)
    fit_users, val_users = user_val_split(all_users, args.val_frac, args.seed + 100)
    fit_mask = np.isin(us, fit_users); val_mask = np.isin(us, val_users)
    if clean:
        fit_mask = fit_mask & consistent; val_mask = val_mask & consistent
    final_model, final_hist = train_one_model(Xs[fit_mask], ys[fit_mask], Xs[val_mask], ys[val_mask], device, n_in,
                                              args.epochs, args.batch_size, args.lr, args.weight_decay,
                                              args.patience, args.seed)

    mpath = out / "model_mlp.pt"
    torch.save({"state_dict": final_model.state_dict(), "arch": "FeatureMLP", "n_in": n_in,
               "classes": CLASSES, "feature_mask": fmask.tolist(), "feature_names": [str(n) for n in names]}, mpath)
    size_mb = mpath.stat().st_size / 1e6
    n_params = count_params(final_model)

    final_model.eval()
    dummy = torch.zeros(1, n_in, device=device)
    with torch.no_grad():
        for _ in range(10):
            final_model(dummy)
        t0 = time.time()
        n_reps = 200
        for _ in range(n_reps):
            final_model(dummy)
        latency_ms = (time.time() - t0) / n_reps * 1000

    print(f"saved {mpath}  ({size_mb:.2f} MB on disk, {n_params:,} params, "
          f"{latency_ms:.2f} ms/prediction on {device})")

    (out / "classifier_metrics_mlp.json").write_text(json.dumps({
        "config": vars(args) | {"clean": clean, "arch_name": "FeatureMLP", "n_in": n_in,
                                "still_g": STILL_G, "device": str(device)},
        "folds": {str(k): v for k, v in per_fold.items()},
        "pooled_all": pooled_all, "pooled_signal_consistent": pooled_cons,
        "confusion_matrix": cm.tolist(), "classes": CLASSES,
        "cost": {"n_params": n_params, "model_size_mb": round(size_mb, 3),
                 "latency_ms_per_prediction": round(latency_ms, 3), "device": str(device)},
        "final_model_training": {"best_val_macro_f1": final_hist["best_val_macro_f1"],
                                 "epochs_run": final_hist["epochs_run"]},
    }, indent=2))
    print(f"\nsaved {out / 'classifier_metrics_mlp.json'}")


if __name__ == "__main__":
    main()
