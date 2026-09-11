#!/usr/bin/env python3
"""
Figure 2b -- activity confusion matrix measured on the SAME held-out benchmark as Figures 1, 3
and 5, so the results section is internally consistent.

Why this exists alongside the existing Figure 2:
  * `results/confusion_matrix.png` characterises the recognition backbone the standard way --
    minute-level, 5-fold user cross-validation over all 60 users. That is the right figure for
    "how good is the recognizer", and it is what the brief's Figure 2 asks for.
  * Every QA figure, however, is produced by the deployed *pipeline* (recognizer + smoothing +
    interval merging) on 12 held-out recordings. Errors introduced by the aggregation layer do
    not appear in the minute-level matrix at all.

This figure closes that gap. It is measured in SECONDS of overlap rather than minutes classified,
because that is what duration and grounding answers are actually built from: an activity can be
recognised on most of its minutes yet still lose most of its *time* to a neighbour when short
episodes are absorbed during merging.

Usage:
    python scripts/make_benchmark_confusion.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregate.schema import CLASSES, Timeline   # noqa: E402

PRETTY = ["Lying", "Sitting", "Stand (place)", "Stand (moving)", "Walking", "Running", "Bicycling"]


def build_matrix(benchmark: str, timelines: str) -> tuple[np.ndarray, int]:
    """M[i, j] = seconds whose reference activity is i and predicted activity is j."""
    M = np.zeros((len(CLASSES), len(CLASSES)))
    idx = {c: i for i, c in enumerate(CLASSES)}
    n_rec = 0
    for truth_path in sorted(glob.glob(f"{benchmark}/*/truth.json")):
        rec_id = os.path.basename(os.path.dirname(truth_path))
        hits = sorted(glob.glob(f"{timelines}/{rec_id}__*.json"))
        if not hits:
            continue
        n_rec += 1
        tl = Timeline.from_json(hits[0])
        for t in json.loads(Path(truth_path).read_text())["intervals"]:
            ti = idx.get(t["activity"])
            if ti is None:
                continue
            for p in tl.intervals:
                pj = idx.get(p.activity)
                if pj is None:
                    continue
                overlap = max(0.0, min(p.end, t["end"]) - max(p.start, t["start"]))
                if overlap > 0:
                    M[ti, pj] += overlap
    return M, n_rec


def _largest_confusion(M, i: int) -> str | None:
    """Class that absorbs most of class i's time, or None if nothing is misattributed."""
    off = [(CLASSES[j], float(M[i, j])) for j in range(len(CLASSES)) if j != i]
    best = max(off, key=lambda kv: kv[1], default=None)
    return best[0] if best and best[1] > 0 else None


def plot(M: np.ndarray, out: Path, n_rec: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), PRETTY, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), PRETTY)
    ax.set_xlabel("Predicted activity (pipeline output)")
    ax.set_ylabel("True activity (reference labels)")
    ax.set_title("Figure 2b — Activity confusion by TIME, full pipeline\n"
                 f"{n_rec} held-out real recordings (fold 0 test users); "
                 "row-normalised share of each activity's seconds")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if M[i, j] > 0 or i == j:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the true activity's time")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", nargs="*", default=["data/eval_benchmark"],
                    help="one or more benchmark roots; pass all folds to pool them")
    ap.add_argument("--timelines", nargs="*", default=None,
                    help="matching timeline cache dirs; defaults to <benchmark>/_timelines")
    ap.add_argument("--out", default="results/fig2b_benchmark_confusion.png")
    ap.add_argument("--json-out", default="results/qa_eval/benchmark_confusion.json")
    args = ap.parse_args()

    benches = args.benchmark
    tls = args.timelines or [f"{b}/_timelines" for b in benches]
    M = np.zeros((len(CLASSES), len(CLASSES)))
    n_rec = 0
    for b, t in zip(benches, tls):
        m, n = build_matrix(b, t)
        M += m; n_rec += n
    if n_rec == 0:
        sys.exit("no cached timelines found -- run scripts/evaluate_qa.py first")

    per_class = {}
    for i, c in enumerate(CLASSES):
        true_t, pred_t, hit = M[i].sum(), M[:, i].sum(), M[i, i]
        per_class[c] = {
            "true_seconds": round(float(true_t), 1),
            "predicted_seconds": round(float(pred_t), 1),
            "correct_seconds": round(float(hit), 1),
            "time_recall": round(float(hit / true_t), 4) if true_t else None,
            "time_precision": round(float(hit / pred_t), 4) if pred_t else None,
            # None when the class has no off-diagonal mass at all, rather than naming whichever
            # class happens to sort first among a row of zeros.
            "largest_confusion": _largest_confusion(M, i),
        }

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(
        {"n_recordings": n_rec, "matrix_seconds": M.tolist(), "classes": CLASSES,
         "per_class": per_class}, indent=2))
    plot(M, Path(args.out), n_rec)

    print(f"\n{'class':<22}{'true s':>9}{'pred s':>9}{'recall':>8}{'prec':>7}  largest confusion")
    for c, v in per_class.items():
        r = f"{v['time_recall']:.2f}" if v["time_recall"] is not None else "-"
        p = f"{v['time_precision']:.2f}" if v["time_precision"] is not None else "-"
        print(f"{c:<22}{v['true_seconds']:>9.0f}{v['predicted_seconds']:>9.0f}{r:>8}{p:>7}  "
              f"{v['largest_confusion'] or '-'}")
    print(f"\nsaved {args.json_out}")


if __name__ == "__main__":
    main()
