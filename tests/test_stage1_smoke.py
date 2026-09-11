#!/usr/bin/env python3
"""
Stage 1: question -> QueryIntent routing.

These run entirely on the deterministic rule fast-path, so they need no Ollama server and are
safe in CI. That is deliberate: the fast-path is what answers the overwhelming majority of
questions, and a regression in it silently pushes load onto the SLM, which is both slower and
measurably less accurate at this job.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.query.intent import parse_intent_fast_rules  # noqa: E402

# (question, expected intent, expected target_activity or None to skip the check)
CASES = [
    ("What activity is the user performing?",                       "identify",   None),
    ("What is the user doing?",                                     "identify",   None),
    ("Is the user running?",                                        "verify",     "running"),
    ("Is the user walking?",                                        "verify",     "walking"),
    ("How long was the user walking?",                              "duration",   "walking"),
    ("How many times did she walk?",                                "count",      "walking"),
    ("Did the user begin running at any point, and if so, when?",   "onset",      "running"),
    ("When did the user begin walking?",                            "onset",      "walking"),
    ("Did the user spend more time walking or running?",            "compare",    "walking"),
    ("Was the user using a wheeled or pedal-based mode of movement?", "open_world", None),
    ("How long was the user standing in place?",                    "duration",   "standing_in_place"),
    ("How long was the user standing and moving?",                  "duration",   "standing_and_moving"),
]


@pytest.mark.parametrize("question,expected_intent,expected_target", CASES)
def test_fast_path_routing(question, expected_intent, expected_target):
    intent = parse_intent_fast_rules(question)
    assert intent is not None, f"fast path failed to classify {question!r} (would fall back to the SLM)"
    assert intent.intent.value == expected_intent, f"{question!r} -> {intent.intent.value}"
    if expected_target is not None:
        assert intent.target_activity == expected_target


def test_verify_accepts_subjects_other_than_the_user():
    """
    Regression: the rule was anchored on the literal phrase "the user", so any other subject
    ("she", a name, "the grandmother") fell through to the SLM, which then misclassified it.
    """
    intent = parse_intent_fast_rules("Did the grandmother take a quick jog in the park?")
    assert intent is not None, "fell through to the SLM"
    assert intent.intent.value == "verify"
    assert intent.target_activity == "running"      # "jog" -> running


def test_how_much_spend_is_a_duration_question():
    """
    Regression: "How much of the afternoon did she spend resting?" -- phrasing taken verbatim from
    the brief's opening scenario -- was routed to open-world instead of duration.
    """
    intent = parse_intent_fast_rules("How much of the afternoon did she spend resting?")
    assert intent is not None
    assert intent.intent.value == "duration"


def test_did_user_spend_a_prolonged_period_is_not_a_duration_question():
    """The counterpart guard: a yes/no question must not be captured by the word "spend"."""
    intent = parse_intent_fast_rules("Did the user spend a prolonged period resting?")
    assert intent is not None
    assert intent.intent.value == "open_world"


def test_named_posture_narrows_prolonged_rest_but_generic_rest_does_not():
    """
    "lie down" asks about lying down specifically; bare "rest" means sedentary behaviour in
    general and must not be silently narrowed (the synonym table maps "resting" -> lying_down).
    """
    assert parse_intent_fast_rules("Did the user lie down for a prolonged period?").target_activity == "lying_down"
    assert parse_intent_fast_rules("Did the user spend a prolonged period resting?").target_activity is None
