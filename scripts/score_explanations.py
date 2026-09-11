#!/usr/bin/env python3
"""
Explanation faithfulness scoring (the brief's open-world reasoning requirement).

"The explanation calls for more than a correctness check, so judge its faithfulness and
 plausibility with a short rubric on a one to five scale, applied either by human graders or by a
 language model given a fixed rubric, rating whether the reasoning cites real signal features,
 whether those features support the conclusion, and whether the conclusion is plausible. Report the
 mean rubric score..."

Two complementary measures are produced, because they fail in different ways:

  1. NUMERIC FIDELITY (deterministic, objective). Every number the explanation quotes is checked
     against the cited interval's own measured signal summary. This directly tests the central
     design claim of the system -- that the SLM phrases evidence but never invents it -- and needs
     no judge at all, so it cannot be gamed by a weak evaluator.

  2. RUBRIC SCORE 1-5 (LLM judge with a fixed rubric), as the brief specifies.

Known limitation, stated rather than hidden: the judge is the same Qwen 2.5 1.5B that wrote the
explanations. A model grading its own output is a weak evaluator and tends to be generous. The
rubric score should be read alongside the deterministic numeric-fidelity figure, which is the more
trustworthy of the two. A stronger setup would use a larger independent judge or human graders.

Usage:
    python scripts/score_explanations.py --answers results/qa_eval/answers_rf_fold0.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUBRIC = """You are grading a single explanation produced by a wearable-sensor question answering
system. Score it 1-5 on faithfulness and plausibility using EXACTLY this rubric:

5 - Cites specific measured signal features; those features genuinely support the conclusion; the
    conclusion is plausible.
4 - Cites specific measured features and the conclusion is plausible, but the link between them is
    partly loose or incomplete.
3 - Refers to signal evidence only in general terms, or the reasoning is thin but not wrong.
2 - Reasoning is largely generic, or partly contradicts the cited evidence.
1 - Cites no real evidence, or the reasoning contradicts the evidence or is implausible.

Reply with ONLY a single digit 1-5. No words, no punctuation, no explanation."""


def numbers_in(text: str) -> list[float]:
    """
    Numbers a reader would take as measurements of THIS interval.

    The lookbehind matters: in "usual range 0.3-0.5 g" a naive pattern reads "-0.5 g" as a
    negative measurement, when it is really the upper end of a reference band quoted as context.
    Measurements here are never negative, so a value preceded by "<digit>-" is a range endpoint
    and is skipped.
    """
    out = []
    for m in re.finditer(r"(?<![\d.]-)(?<![\d.])(\d+(?:\.\d+)?)\s*(g\b|Hz\b|rad/s)", text or ""):
        out.append(float(m.group(1)))
    return out


def numeric_fidelity(row: dict, tol: float = 0.02) -> dict | None:
    """
    Every measurement quoted in the explanation must match a value the system actually computed
    for the cited interval (within rounding). Returns None when the explanation quotes no
    measurements at all, so those rows do not silently count as passes.
    """
    quoted = numbers_in(row.get("explanation", ""))
    if not quoted:
        return None
    available = []
    for values in (row.get("interval_summary") or {}).values():
        for v in (values if isinstance(values, list) else [values]):
            if isinstance(v, (int, float)):
                v = float(v)
                available.extend([v, round(v, 3), round(v, 2), round(v, 1)])
    # ranges quoted from the rubric bands in the prompt are legitimate context, not measurements
    band_values = {0.002, 0.006, 0.10, 0.15, 0.3, 0.5, 1.8, 2.2, 2.6, 3.0, 0.12, 0.16}
    unmatched = [q for q in quoted
                 if q not in band_values
                 and not any(abs(q - a) <= max(tol * max(abs(a), 1e-6), 5e-4) for a in available)]
    return {"quoted": quoted, "unmatched": unmatched,
            "faithful": len(unmatched) == 0}


def judge(explanation: str, question: str, answer: str, model: str) -> int | None:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=model, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RUBRIC),
        ("human", "Question: {q}\nAnswer given: {a}\nExplanation to grade: {e}\n\nScore (1-5):"),
    ])
    try:
        resp = (prompt | llm).invoke({"q": question, "a": answer, "e": explanation})
        m = re.search(r"[1-5]", getattr(resp, "content", str(resp)))
        return int(m.group()) if m else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="results/qa_eval/answers_rf_fold0.json")
    ap.add_argument("--timelines", default="data/eval_benchmark/_timelines")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--types", nargs="*", default=["open_world", "verification", "grounding"])
    ap.add_argument("--no-judge", action="store_true", help="deterministic fidelity check only")
    ap.add_argument("--out", default="results/qa_eval/explanation_scores.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.answers).read_text())
    rows = [r for r in rows if r["type"] in args.types and r.get("explanation")]

    # attach the measured signal summary of each cited interval, for the fidelity check
    from src.aggregate.schema import Timeline
    cache: dict[str, Timeline] = {}
    for r in rows:
        rec_id = r["id"].split("::")[0]
        if rec_id not in cache:
            hits = list(Path(args.timelines).glob(f"{rec_id}__*.json"))
            cache[rec_id] = Timeline.from_json(str(hits[0])) if hits else None
        tl = cache[rec_id]
        # Collect the summaries of EVERY cited interval, not just the first: generate_explanation()
        # summarises the LONGEST interval it was given, so checking only the first would flag
        # perfectly faithful explanations as inventing numbers.
        summary: dict[str, list[float]] = {}
        if tl and r.get("pred_intervals"):
            for s, e in r["pred_intervals"]:
                for iv in tl.intervals:
                    if max(0.0, min(iv.end, e) - max(iv.start, s)) > 0:
                        for k, v in (iv.summary or {}).items():
                            if isinstance(v, (int, float)):
                                summary.setdefault(k, []).append(float(v))
        r["interval_summary"] = summary

    scored = []
    for i, r in enumerate(rows, 1):
        fid = numeric_fidelity(r)
        rating = None if args.no_judge else judge(r["explanation"], r["question"],
                                                  r["pred_answer"], args.model)
        scored.append({"id": r["id"], "type": r["type"], "question": r["question"],
                       "answer": r["pred_answer"], "explanation": r["explanation"],
                       "numeric_fidelity": fid, "rubric_score": rating})
        if i % 25 == 0:
            print(f"  scored {i}/{len(rows)}", flush=True)

    with_nums = [s for s in scored if s["numeric_fidelity"] is not None]
    faithful = [s for s in with_nums if s["numeric_fidelity"]["faithful"]]
    ratings = [s["rubric_score"] for s in scored if s["rubric_score"] is not None]

    summary = {
        "answers_file": args.answers,
        "judge_model": None if args.no_judge else args.model,
        "n_explanations": len(scored),
        "numeric_fidelity": {
            "n_quoting_measurements": len(with_nums),
            "n_faithful": len(faithful),
            "rate": round(len(faithful) / len(with_nums), 4) if with_nums else None,
            "definition": "every g / Hz / rad-s value quoted in the explanation matches a value "
                          "actually computed for the cited interval (2% tolerance)",
        },
        "rubric": {
            "n_scored": len(ratings),
            "mean": round(statistics.mean(ratings), 3) if ratings else None,
            "median": statistics.median(ratings) if ratings else None,
            "distribution": {k: ratings.count(k) for k in range(1, 6)} if ratings else None,
            "caveat": "judged by the same model that generated the explanations; treat as "
                      "indicative and read alongside numeric fidelity",
        },
        "per_type_mean_rubric": {
            t: round(statistics.mean([s["rubric_score"] for s in scored
                                      if s["type"] == t and s["rubric_score"] is not None]), 3)
            for t in args.types
            if any(s["type"] == t and s["rubric_score"] is not None for s in scored)
        },
    }

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "scored": scored}, indent=2))

    nf = summary["numeric_fidelity"]
    print(f"\nexplanations scored          : {summary['n_explanations']}")
    print(f"quoting measurements         : {nf['n_quoting_measurements']}")
    print(f"NUMERIC FIDELITY (no invented values): {nf['rate']}  ({nf['n_faithful']}/{nf['n_quoting_measurements']})")
    if ratings:
        print(f"MEAN RUBRIC SCORE (1-5)      : {summary['rubric']['mean']}  "
              f"distribution={summary['rubric']['distribution']}")
        print(f"per type                     : {summary['per_type_mean_rubric']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
