#!/usr/bin/env python3
"""
ISOLATED EXPERIMENT, stage 2 — does class calibration actually improve QA answers?

Stage 1 (calibrate_classes.py) showed calibration cuts minute-level "time error" by up to 31%.
Time error is only a *proxy* for what duration questions need, so this stage runs the real
end-to-end QA harness with the calibration applied and compares the metrics that are actually
reported.

Nothing in src/ is edited. The calibration is applied by wrapping `window_predictions` at runtime
and re-weighting the per-class probability columns before the aggregation layer takes its argmax,
which is exactly where a decision-rule change belongs. If the verdict is positive, the same
re-weighting would be wired into the model bundle properly.

Usage:
    python experiments/calibration/apply_and_evaluate.py                      # all candidates
    python experiments/calibration/apply_and_evaluate.py --only baseline prior_alpha_0.4_maxF1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.aggregate.timeline as tl_mod                       # noqa: E402
import src.query.explain as explain_mod                       # noqa: E402
import src.query.operations as ops                            # noqa: E402
from src.aggregate.schema import CLASSES                      # noqa: E402
from src.eval.questions import build_questions                # noqa: E402
from src.query.intent import parse_intent                     # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_qa import aggregate, score_one                   # noqa: E402

_ORIGINAL_WP = tl_mod.window_predictions


def install_weights(weights: np.ndarray | None) -> None:
    """Re-weight class probabilities before the aggregation layer's argmax."""
    if weights is None:
        tl_mod.window_predictions = _ORIGINAL_WP
        return

    def weighted(df, bundle, whole_chunk=False):
        out = _ORIGINAL_WP(df, bundle, whole_chunk=whole_chunk)
        if len(out) == 0:
            return out
        cols = [f"p_{c}" for c in CLASSES]
        P = out[cols].to_numpy() * weights
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        for i, c in enumerate(cols):
            out[c] = P[:, i]
        return out

    tl_mod.window_predictions = weighted


def run_once(rec_dirs, model: str, weights, iou: float) -> dict:
    install_weights(weights)
    rows = []
    for rec in rec_dirs:
        truth = json.loads((rec / "truth.json").read_text())["intervals"]
        qs = build_questions(rec.name, truth)
        tl = tl_mod.build_timeline(rec / "recording.csv", model)
        for q in qs:
            rows.append(score_one(q, ops.execute_query(parse_intent(q["question"]), tl), truth))
    install_weights(None)
    return aggregate(rows, iou)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/eval_benchmark")
    ap.add_argument("--model", default="results/model_rf_fold0.joblib")
    ap.add_argument("--calibration", default="experiments/calibration/results.json")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/calibration/qa_comparison.json")
    args = ap.parse_args()

    # explanations are deterministic here: none of the compared metrics depend on the SLM
    orig = explain_mod.generate_explanation
    patched = lambda a, d, i, **k: orig(a, d, i, use_llm=False)   # noqa: E731
    explain_mod.generate_explanation = patched
    ops.generate_explanation = patched

    calib = json.loads(Path(args.calibration).read_text())
    candidates: dict[str, np.ndarray | None] = {"baseline": None}
    for name, c in calib["candidates"].items():
        if name == "baseline_argmax":
            continue
        candidates[name] = np.array([c["weights"][cl] for cl in CLASSES], dtype=float)
    if args.only:
        candidates = {k: v for k, v in candidates.items() if k in args.only}

    rec_dirs = sorted(p for p in Path(args.benchmark).iterdir()
                      if p.is_dir() and (p / "truth.json").exists())
    if args.limit:
        rec_dirs = rec_dirs[:args.limit]
    print(f"{len(rec_dirs)} held-out recordings | {len(candidates)} candidates | model {args.model}\n")

    results = {}
    for name, w in candidates.items():
        t0 = time.time()
        results[name] = run_once(rec_dirs, args.model, w, args.iou)
        s = results[name]
        print(f"{name:<32} macro={s['overall_qa_accuracy_macro']:.3f} "
              f"duration={s['per_type']['duration']['accuracy']:.3f} "
              f"count={s['per_type']['count']['accuracy']:.3f} "
              f"grounded={s['grounded_and_correct']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).write_text(json.dumps(
        {"model": args.model, "n_recordings": len(rec_dirs), "results": results}, indent=2))

    print(f"\n{'candidate':<32}{'macro':>8}{'ident':>7}{'dur':>7}{'count':>7}{'cmp':>7}"
          f"{'ground':>8}{'open':>7}{'G&C':>7}{'durMAE':>9}")
    for name, s in results.items():
        pt = s["per_type"]
        print(f"{name:<32}{s['overall_qa_accuracy_macro']:>8.3f}"
              f"{pt['identification']['accuracy']:>7.3f}{pt['duration']['accuracy']:>7.3f}"
              f"{pt['count']['accuracy']:>7.3f}{pt['comparison']['accuracy']:>7.3f}"
              f"{pt['grounding']['accuracy']:>8.3f}{pt['open_world']['accuracy']:>7.3f}"
              f"{s['grounded_and_correct']:>7.3f}{pt['duration']['mae']:>9.1f}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
