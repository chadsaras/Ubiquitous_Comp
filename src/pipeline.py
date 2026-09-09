#!/usr/bin/env python3
"""
End-to-End Prediction & Question-Answering Pipeline for 'Ask the Sensors'.

Usage:
    # 1. Ask a single question directly:
    python src/pipeline.py --recording data/test_recording.csv \
                           --question "How long was the user walking?"

    # 2. Ask a batch of questions from a file using Random Forest:
    python src/pipeline.py --recording data/test_recording.csv \
                           --questions data/sample_questions.txt \
                           --model models/model_rf.joblib

    # 3. Use Histogram Gradient Boosting for faster inference:
    python src/pipeline.py --recording data/test_recording.csv \
                           --questions data/sample_questions.txt \
                           --model models/model_hgb.joblib \
                           --out results/answers.txt

    # 4. Skip sensor processing by passing an already cached timeline:
    python src/pipeline.py --timeline tests/fixtures/00EABED2_590_160/timeline.json \
                           --question "Did the user spend more time walking or sitting?"
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List, Union, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.aggregate.schema import Timeline
from src.aggregate.timeline import build_timeline
from src.query.engine import answer


def load_questions(questions_input: Union[str, Path, List[str]]) -> List[str]:
    """Load questions from a list, file path, or newline-delimited string."""
    if isinstance(questions_input, list):
        return [q.strip() for q in questions_input if q.strip()]

    path = Path(questions_input)
    if path.exists() and path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # If treated as a single query string
    return [str(questions_input).strip()]


def run_pipeline(
    recording: Optional[Union[str, Path]] = None,
    timeline: Optional[Timeline] = None,
    questions: Union[str, Path, List[str]] = "What activity is the user performing?",
    model_path: Union[str, Path] = "models/model_rf.joblib",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> List[str]:
    """
    Run the end-to-end question answering pipeline.

    Args:
        recording: Path to raw sensor recording.csv (optional if timeline provided).
        timeline: Existing Timeline instance (optional if recording provided).
        questions: Single question, list of questions, or path to questions.txt.
        model_path: Path to model_rf.joblib or model_hgb.joblib.
        output_path: Optional path to save formatted answers.
        verbose: Whether to print progress to stdout.

    Returns:
        List of formatted answer blocks as strings.
    """
    # 1. Obtain Timeline
    if timeline is None:
        if recording is None:
            raise ValueError("Must provide either a 'recording' CSV path or an existing 'timeline'.")
        rec_p = Path(recording)
        if not rec_p.exists():
            raise FileNotFoundError(f"Sensor recording not found: {rec_p}")

        if verbose:
            print(f"[Pipeline] Processing recording: {rec_p.name} with model: {Path(model_path).name}...")
        t0 = time.time()
        timeline = build_timeline(rec_p, model_path)
        dt = time.time() - t0
        if verbose:
            print(f"[Pipeline] Generated {len(timeline.intervals)} intervals in {dt:.2f}s.")
    else:
        if verbose:
            print(f"[Pipeline] Using pre-loaded timeline with {len(timeline.intervals)} intervals.")

    # 2. Process Questions
    q_list = load_questions(questions)
    if verbose:
        print(f"[Pipeline] Answering {len(q_list)} question(s)...\n" + "=" * 60)

    results = []
    for i, q in enumerate(q_list, 1):
        if verbose:
            print(f"\n[Q{i}]: \"{q}\"")
            print("-" * 40)

        # Query Engine (LangChain + Ollama Qwen 2.5 + Deterministic Timeline)
        ans_block = answer(q, timeline)
        results.append(ans_block)

        if verbose:
            print(ans_block)
            print("-" * 40)

    # 3. Save to file if requested
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for q, res in zip(q_list, results):
                f.write(f"Query: \"{q}\"\n{res}\n\n" + "=" * 60 + "\n\n")
        if verbose:
            print(f"\n[Pipeline] All answers saved to: {out_p}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ask the Sensors: Grounded Activity Question-Answering Pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recording", help="Path to input sensor recording CSV")
    group.add_argument("--timeline", help="Path to cached timeline.json (skips model inference)")

    parser.add_argument("--model", default="models/model_rf.joblib",
                        help="Path to trained classifier (model_rf.joblib or model_hgb.joblib)")
    parser.add_argument("--question", help="Single natural language question")
    parser.add_argument("--questions", help="Path to text file with questions (one per line)")
    parser.add_argument("--out", default=None, help="Path to output answers file")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")

    args = parser.parse_args()

    # Determine question source
    if args.question:
        q_input = args.question
    elif args.questions:
        q_input = args.questions
    else:
        # Default test questions if none supplied
        q_input = [
            "What activity is the user performing?",
            "Is the user walking?",
            "How long was the user walking?",
            "How many times did she walk?",
            "Did the user begin walking at any point, and if so, when?",
            "Did the user spend more time walking or sitting?"
        ]

    tl_instance = Timeline.from_json(args.timeline) if args.timeline else None

    run_pipeline(
        recording=args.recording,
        timeline=tl_instance,
        questions=q_input,
        model_path=args.model,
        output_path=args.out,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
