#!/usr/bin/env python3
"""
Pool the per-fold QA evaluations into one result over the full official 5-fold split.

Each fold is scored with its OWN held-out model against its OWN held-out users, exactly as
scripts/train_classifier.py already does for the recognition backbone. Pooling the per-fold
answers then gives a single figure computed over all 60 users rather than the 12 of one fold,
and -- because the per-recording spread is wide (sd ~0.16) -- a materially tighter confidence
interval on the headline number.

Run the per-fold evaluations first, e.g.:
    for F in 0 1 2 3 4; do
      python scripts/evaluate_qa.py --fold-tag $F
    done
then:
    python scripts/pool_folds.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_qa import QTYPES, aggregate, strictness_sweep   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/qa_eval")
    ap.add_argument("--tags", nargs="*", default=["rf_fold0", "rf_fold1", "rf_fold2",
                                                  "rf_fold3", "rf_fold4"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="results/qa_eval/summary_all_folds.json")
    args = ap.parse_args()

    rows, per_fold, used = [], {}, []
    for tag in args.tags:
        p = Path(args.dir) / f"answers_{tag}.json"
        if not p.exists():
            print(f"  (skipping {tag}: {p} not found)")
            continue
        r = json.loads(p.read_text())
        rows += r
        s = aggregate(r, args.iou)
        per_fold[tag] = {"n_questions": len(r),
                         "macro": s["overall_qa_accuracy_macro"],
                         "micro": s["overall_qa_accuracy_micro"],
                         "grounded_and_correct": s["grounded_and_correct"],
                         "duration": s["per_type"].get("duration", {}).get("accuracy"),
                         "n_recordings": (s.get("per_recording_spread") or {}).get("n_recordings")}
        used.append(tag)

    if not rows:
        sys.exit("no per-fold answer files found")

    pooled = aggregate(rows, args.iou)
    pooled["strictness"] = strictness_sweep(rows)
    pooled["folds_pooled"] = used
    pooled["per_fold"] = per_fold
    pooled["n_recordings"] = sum(v["n_recordings"] or 0 for v in per_fold.values())

    Path(args.out).write_text(json.dumps(pooled, indent=2))

    print(f"\n{'fold':<12}{'recs':>6}{'questions':>11}{'macro':>8}{'duration':>10}{'grounded':>10}")
    for tag, v in per_fold.items():
        print(f"{tag:<12}{v['n_recordings'] or 0:>6}{v['n_questions']:>11}{v['macro']:>8.3f}"
              f"{(v['duration'] or 0):>10.3f}{(v['grounded_and_correct'] or 0):>10.3f}")

    macros = [v["macro"] for v in per_fold.values()]
    print(f"{'-'*57}")
    print(f"{'POOLED':<12}{pooled['n_recordings']:>6}{pooled['n_questions']:>11}"
          f"{pooled['overall_qa_accuracy_macro']:>8.3f}"
          f"{pooled['per_type']['duration']['accuracy']:>10.3f}"
          f"{pooled['grounded_and_correct']:>10.3f}")
    if len(macros) > 1:
        mean = sum(macros) / len(macros)
        sd = (sum((m - mean) ** 2 for m in macros) / (len(macros) - 1)) ** 0.5
        print(f"\nbetween-fold spread of macro accuracy: mean {mean:.3f}, sd {sd:.3f}, "
              f"range {min(macros):.3f}-{max(macros):.3f}")
    sp = pooled.get("per_recording_spread")
    if sp:
        print(f"across all {sp['n_recordings']} held-out recordings: "
              f"{sp['mean']:.3f} +/- {sp['ci95_halfwidth']:.3f} (95% CI), sd {sp['sd']:.3f}")

    print(f"\n{'type':<16}{'n':>6}{'accuracy':>10}{'mean IoU':>10}")
    for t in QTYPES:
        e = pooled["per_type"].get(t)
        if e:
            miou = f"{e['mean_iou']:.3f}" if e.get("mean_iou") is not None else "-"
            print(f"{t:<16}{e['n']:>6}{e['accuracy']:>10.3f}{miou:>10}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
