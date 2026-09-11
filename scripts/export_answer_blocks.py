#!/usr/bin/env python3
"""
Write every evaluated question and the system's answer, in the challenge's required output format.

The per-fold JSON is the machine-readable record and all_answers.csv is the spreadsheet view;
this produces the human-readable transcript -- 1,241 answers exactly as the system emits them,
each followed by the reference answer and the grade. It is the artefact to point at when asked
"show me the system actually answering in the required format", and the source for quoting real
examples (including real failures) in the report.

Usage:
    python scripts/export_answer_blocks.py
    python scripts/export_answer_blocks.py --only-wrong --type duration
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/qa_eval")
    ap.add_argument("--out", default="results/qa_eval/all_answer_blocks.txt")
    ap.add_argument("--only-wrong", action="store_true")
    ap.add_argument("--type", default=None)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(f"{args.dir}/answers_*.json")):
        rows += json.loads(Path(path).read_text())
    if args.type:
        rows = [r for r in rows if r["type"] == args.type]
    if args.only_wrong:
        rows = [r for r in rows if r.get("correct") is False]

    lines, n_block = [], 0
    for r in rows:
        block = r.get("formatted_block")
        if not block:
            continue
        n_block += 1
        verdict = {True: "CORRECT", False: "INCORRECT"}.get(r.get("correct"), "n/a")
        lines.append("=" * 78)
        lines.append(f'Recording: {r["id"].split("::")[0]}   [{r["type"]}]')
        lines.append(f'Query: "{r["question"]}"')
        lines.append("-" * 78)
        lines.append(block)
        lines.append("-" * 78)
        detail = f"Reference answer: {r.get('truth')}    ->  {verdict}"
        if r.get("abs_error") is not None:
            detail += f"   (off by {r['abs_error']:.0f})"
        if r.get("iou") is not None:
            detail += f"   evidence IoU {r['iou']:.2f}"
        lines.append(detail)
        lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {n_block} answer blocks -> {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
