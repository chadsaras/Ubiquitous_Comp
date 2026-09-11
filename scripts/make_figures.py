#!/usr/bin/env python3
"""
Required Figures 1 and 3 from the QA evaluation summary.

  Figure 1 -- Accuracy by question type: grouped bar chart over identification, verification,
              duration, count, comparison, grounding and open-world reasoning, with a final bar
              for the overall (macro-averaged) score. The caption states the correctness rule per
              group, because the brief requires it and the rule genuinely differs per group.
  Figure 3 -- Accuracy versus strictness: IoU threshold swept 0.1-0.9 for temporal/cited
              intervals, and the tolerance swept for numeric answers.

(Figure 2, the activity confusion matrix, is produced by scripts/train_classifier.py.
 Figure 4 is the extra-credit accuracy-vs-overhead Pareto plot and is out of scope.
 Figure 5 is produced by scripts/robustness_test.py.)

Usage:
    python scripts/make_figures.py --summary results/qa_eval/summary_rf_fold0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

LABELS = {"identification": "Identification", "verification": "Verification",
          "duration": "Duration", "count": "Count", "comparison": "Comparison",
          "grounding": "Grounding", "open_world": "Open-world"}

RULES = ("Correctness rule per group: identification & comparison - exact match on the activity "
         "label; verification, grounding & open-world - exact match on the yes/no verdict; "
         "duration - within max(5 s, 10%) of the reference; count - within +-1 episode. "
         "Overall is the macro-average across question types.")


def figure1(summary: dict, out: Path) -> None:
    types = [t for t in LABELS if t in summary["per_type"]]
    accs = [summary["per_type"][t]["accuracy"] for t in types]
    names = [LABELS[t] for t in types]
    ns = [summary["per_type"][t]["n"] for t in types]

    names.append("OVERALL\n(macro)")
    accs.append(summary["overall_qa_accuracy_macro"])
    ns.append(summary["n_questions"])

    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = ["#4C72B0"] * (len(accs) - 1) + ["#C44E52"]
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.6)
    for b, a, n in zip(bars, accs, ns):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.2f}\n(n={n})",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy under the rule for that question type")
    ax.set_title("Figure 1 — QA accuracy by question type\n"
                 "held-out real ExtraSensory recordings (fold 0 test users), "
                 f"{summary['n_recordings']} recordings, {summary['n_questions']} questions")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.02, RULES, ha="center", va="top", fontsize=7.5, wrap=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


def figure3(summary: dict, out: Path) -> None:
    s = summary["strictness"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    iou = s["iou_threshold"]
    xs = sorted(float(k) for k in iou)
    axes[0].plot(xs, [iou[str(x) if str(x) in iou else f"{x:.1f}"] for x in xs],
                 marker="o", color="#4C72B0")
    axes[0].axvline(summary["iou_threshold"], ls="--", color="grey", lw=1)
    axes[0].text(summary["iou_threshold"] + 0.02, 0.94, f"reported tau={summary['iou_threshold']}",
                 fontsize=8, color="grey")
    axes[0].set_xlabel("IoU threshold")
    axes[0].set_ylabel("Fraction of cited intervals accepted")
    axes[0].set_title("Temporal / cited-interval grounding")
    axes[0].set_ylim(0, 1.02); axes[0].grid(alpha=0.3)

    dur = s["duration_rel_tolerance"]
    dxs = sorted(float(k) for k in dur)
    axes[1].plot([x * 100 for x in dxs], [dur[str(x)] for x in dxs],
                 marker="o", label="Duration (relative tolerance)", color="#55A868")
    cnt = s["count_abs_tolerance"]
    cxs = sorted(float(k) for k in cnt)
    ax2 = axes[1].twiny()
    ax2.plot(cxs, [cnt[str(int(x)) if str(int(x)) in cnt else str(x)] for x in cxs],
             marker="s", ls="--", label="Count (+-N episodes)", color="#C44E52")
    ax2.set_xlabel("Count tolerance (+- episodes)", color="#C44E52")
    axes[1].set_xlabel("Duration tolerance (%)", color="#55A868")
    axes[1].set_ylabel("Fraction of numeric answers accepted")
    axes[1].set_title("Numeric answers")
    axes[1].set_ylim(0, 1.02); axes[1].grid(alpha=0.3)
    lines = axes[1].get_lines() + ax2.get_lines()
    axes[1].legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)

    fig.suptitle("Figure 3 — Accuracy versus strictness", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/qa_eval/summary_rf_fold0.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    figure1(summary, out / "fig1_accuracy_by_question_type.png")
    figure3(summary, out / "fig3_accuracy_vs_strictness.png")


if __name__ == "__main__":
    main()
