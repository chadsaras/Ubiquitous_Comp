#!/usr/bin/env python3
"""
Export every evaluated question to one spreadsheet-readable CSV.

The per-fold JSON files are the machine-readable record; this is the same content flattened so it
can be opened in Excel, sorted, and filtered -- which is what error analysis for the report
actually needs ("show me every duration question we got wrong, worst first").

Usage:
    python scripts/export_answers_csv.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

FIELDS = ["fold", "recording", "type", "question", "predicted_answer", "true_answer",
          "correct", "abs_error", "pct_error", "iou", "cited_timestamps",
          "cited_modality", "cited_channels", "explanation"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/qa_eval")
    ap.add_argument("--out", default="results/qa_eval/all_answers.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(f"{args.dir}/answers_*.json")):
        fold = Path(path).stem.replace("answers_rf_fold", "")
        for r in json.loads(Path(path).read_text()):
            rows.append({
                "fold": fold,
                "recording": r["id"].split("::")[0],
                "type": r["type"],
                "question": r["question"],
                "predicted_answer": r.get("pred_answer"),
                "true_answer": r.get("truth"),
                "correct": r.get("correct"),
                "abs_error": r.get("abs_error"),
                "pct_error": (round(r["pct_error"], 1) if r.get("pct_error") is not None else None),
                "iou": (round(r["iou"], 3) if r.get("iou") is not None else None),
                "cited_timestamps": r.get("pred_timestamps"),
                "cited_modality": r.get("pred_modality"),
                "cited_channels": r.get("pred_channels"),
                "explanation": (r.get("explanation") or "").replace("\n", " "),
            })

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_wrong = sum(1 for r in rows if r["correct"] is False)
    print(f"exported {len(rows)} questions ({n_wrong} incorrect) -> {out}")
    print(f"open it in Excel; filter on 'correct = FALSE' to review failures")


if __name__ == "__main__":
    main()
