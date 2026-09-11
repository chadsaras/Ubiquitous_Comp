#!/usr/bin/env python3
"""
Required Figure 5 -- robustness curve.

"A line plot with a controlled degradation of the input on the x-axis (added sensor noise, the
 percentage of dropped samples, or a sampling rate below 25 Hz) and accuracy on a fixed question
 set on the y-axis."

All three degradations are swept. The question set is held fixed across every level -- only the
input signal is degraded -- so the curve isolates input quality from question difficulty. The
whole pipeline is re-run per level (corrupt signal -> features -> timeline -> intents -> answers),
not just the classifier, so this measures end-to-end robustness as the brief intends.

Explanations run without the SLM here (--no-llm equivalent): the scored quantities are all
deterministic, and skipping it keeps a ~200-run sweep tractable.

Usage:
    python scripts/robustness_test.py --n-recordings 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregate.timeline import build_timeline              # noqa: E402
from src.eval.questions import build_questions                 # noqa: E402
from src.query.intent import parse_intent                      # noqa: E402

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

NOISE_G = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]      # added Gaussian sigma, in g / (rad/s)
DROP_PCT = [0, 10, 20, 30, 50]                      # percentage of samples removed
RATE_HZ = [25, 20, 15, 10, 5]                       # effective input sampling rate


def corrupt(df: pd.DataFrame, kind: str, level: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    if kind == "noise" and level > 0:
        for c in CHANNELS:
            out[c] = out[c].to_numpy() + rng.normal(0.0, level, len(out))
    elif kind == "drop" and level > 0:
        keep = rng.random(len(out)) >= (level / 100.0)
        if keep.sum() < 100:
            keep[:100] = True
        out = out.loc[keep].reset_index(drop=True)
    elif kind == "rate" and level < 25:
        step = max(1, int(round(25.0 / level)))
        out = out.iloc[::step].reset_index(drop=True)
    return out


def macro_accuracy(rows: list[dict]) -> float:
    """Macro-average of per-type accuracy -- the same headline rule the QA harness reports."""
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(r["type"], []).append(bool(r["correct"]))
    return sum(sum(v) / len(v) for v in by.values()) / len(by) if by else 0.0


def run_level(rec_dirs: list[Path], model: str, kind: str, level: float, seed: int) -> dict:
    from src.eval.scoring import (parse_number, parse_yes_no, score_categorical,  # local: keeps
                                  score_numeric)                                   # import cost off
    from src.query.operations import execute_query

    rows = []
    for rec in rec_dirs:
        truth = json.loads((rec / "truth.json").read_text())["intervals"]
        qs = build_questions(rec.name, truth)
        df = pd.read_csv(rec / "recording.csv")
        tl = build_timeline(corrupt(df, kind, level, seed), model_path=model)
        for q in qs:
            block = execute_query(parse_intent(q["question"]), tl)
            t = q["type"]
            if t in ("identification", "comparison"):
                ok = score_categorical(block.answer, q["truth_answer"])
            elif t in ("verification", "open_world", "grounding"):
                pred = parse_yes_no(block.answer)
                ok = pred is not None and pred == (q["truth_answer"] == "yes")
            else:
                ok = score_numeric(parse_number(block.answer), q["truth_value"], t)["correct"]
            rows.append({"type": t, "correct": ok})
    return {"kind": kind, "level": level, "n_questions": len(rows),
            "macro_accuracy": macro_accuracy(rows)}


def figure5(results: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [("noise", "Added Gaussian sensor noise (sigma, g and rad/s)", "#4C72B0"),
             ("drop", "Dropped samples (%)", "#C44E52"),
             ("rate", "Input sampling rate (Hz)", "#55A868")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (kind, xlabel, color) in zip(axes, specs):
        pts = sorted([r for r in results if r["kind"] == kind], key=lambda r: r["level"])
        xs = [p["level"] for p in pts]
        ys = [p["macro_accuracy"] for p in pts]
        ax.plot(xs, ys, marker="o", color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if kind == "rate":
            ax.invert_xaxis()          # degradation increases to the right
    axes[0].set_ylabel("Overall QA accuracy (macro-averaged)")
    fig.suptitle("Figure 5 — Robustness to controlled input degradation\n"
                 "held-out real ExtraSensory recordings, fixed question set, full pipeline re-run "
                 "at each level", y=1.06)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/eval_benchmark")
    ap.add_argument("--model", default="results/model_rf_fold0.joblib")
    ap.add_argument("--n-recordings", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # explanations are deterministic-only for this sweep (see module docstring)
    import src.query.explain as ex
    import src.query.operations as ops
    _orig = ex.generate_explanation
    patched = lambda a, d, i, **kw: _orig(a, d, i, use_llm=False)   # noqa: E731
    ex.generate_explanation = patched
    ops.generate_explanation = patched

    rec_dirs = sorted(p for p in Path(args.benchmark).iterdir()
                      if p.is_dir() and (p / "truth.json").exists())[:args.n_recordings]
    print(f"robustness sweep over {len(rec_dirs)} recordings, model {args.model}")

    results, t0 = [], time.time()
    for kind, levels in (("noise", NOISE_G), ("drop", DROP_PCT), ("rate", RATE_HZ)):
        for lvl in levels:
            t1 = time.time()
            r = run_level(rec_dirs, args.model, kind, lvl, args.seed)
            results.append(r)
            print(f"  {kind:6s} level={lvl:<6} macro_acc={r['macro_accuracy']:.3f} "
                  f"({r['n_questions']} questions, {time.time()-t1:.0f}s)", flush=True)

    out = Path(args.out); out.mkdir(exist_ok=True)
    (out / "robustness_results.json").write_text(json.dumps(
        {"model": args.model, "n_recordings": len(rec_dirs), "seed": args.seed,
         "wall_time_sec": round(time.time() - t0, 1), "results": results}, indent=2))
    figure5(results, out / "fig5_robustness.png")
    print(f"saved {out}/robustness_results.json  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
