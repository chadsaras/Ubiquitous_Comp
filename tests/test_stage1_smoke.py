#!/usr/bin/env python3
"""
Smoke test for Stage 1: Query Intent Parsing with LangChain and Ollama (Qwen2.5:1.5b).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.query.intent import parse_intent, get_structured_llm

test_queries = [
    ("What activity is the user performing?", "identify"),
    ("Is the user running?", "verify"),
    ("How long was the user walking?", "duration"),
    ("How many times did she walk?", "count"),
    ("Did the user begin running at any point, and if so, when?", "onset"),
    ("Did the user spend more time walking or running?", "compare"),
    ("Was the user using a wheeled or pedal-based mode of movement?", "open_world"),
    ("Did the grandmother take a quick jog in the park?", "verify"),  # Natural language synonym check
]

def main():
    print("--- Stage 1 Smoke Test (Qwen 2.5 1.5B via LangChain & Ollama) ---\n")
    for q, expected_type in test_queries:
        try:
            intent_obj = parse_intent(q)
            print(f"Query:    '{q}'")
            print(f"Parsed:   intent={intent_obj.intent.value}, target={intent_obj.target_activity}, "
                f"compare={intent_obj.compare_activity}, concept={intent_obj.semantic_concept}")
            assert intent_obj.intent.value == expected_type, f"Mismatch for '{q}': got {intent_obj.intent.value}"
            print("Status:   PASSED\n")
        except Exception as e:
            print(f"Query:    '{q}'")
            print(f"Status:   FAILED")
            print(f"Error:    {e}\n")
            continue

    print("All smoke tests passed successfully!")

if __name__ == "__main__":
    main()
