#!/usr/bin/env python3
"""
Verification test for Stage 2 & 3:
Connecting QueryIntent with real Timeline fixtures and generating formatted answers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregate.schema import Timeline
from src.query.engine import answer

FIXTURE_PATH = Path("tests/fixtures/00EABED2_590_160/timeline.json")

test_cases = [
    "What activity is the user performing?",
    "Is the user walking?",
    "How long was the user walking?",
    "How many times did she walk?",
    "Did the user begin walking at any point, and if so, when?",
    "Did the user spend more time walking or sitting?",
    "Did the user lie down for a prolonged period?",
]

def main():
    if not FIXTURE_PATH.exists():
        print(f"Error: Fixture {FIXTURE_PATH} not found!")
        sys.exit(1)

    print(f"Loading Timeline fixture from {FIXTURE_PATH}...")
    timeline = Timeline.from_json(str(FIXTURE_PATH))
    print(f"Loaded {len(timeline.intervals)} intervals.\n")
    print("=" * 60)

    for q in test_cases:
        print(f"\n[QUERY]: \"{q}\"")
        result = answer(q, timeline)
        print("-" * 40)
        print(result)
        print("=" * 60)

if __name__ == "__main__":
    main()
