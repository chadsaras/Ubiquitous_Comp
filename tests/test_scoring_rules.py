#!/usr/bin/env python3
"""
Unit tests for the evaluation scoring rules (src/eval/scoring.py).

These guard the numbers the report will quote. A silent bug here would not crash anything -- it
would just produce a wrong accuracy figure, which is the worst kind of failure for a submission
that is graded on reported results.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import scoring as S  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("Standing in place", "standing_in_place"),
    ("standing_in_place", "standing_in_place"),
    ("Lying Down", "lying_down"),
    ("Walking", "walking"),
])
def test_activity_normalisation(raw, expected):
    assert S.norm_activity(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("2251 seconds", 2251.0),
    ("2 episodes", 2.0),
    ("0 seconds", 0.0),
    ("Yes, walking began at 239 seconds", 239.0),
    ("N/A", None),
])
def test_number_parsing(raw, expected):
    assert S.parse_number(raw) == expected


def test_zero_is_parsed_not_treated_as_missing():
    """A "0 seconds" answer is a real prediction (and usually a wrong one), not a missing value."""
    assert S.parse_number("0 seconds") == 0.0
    assert S.parse_number("0 seconds") is not None


@pytest.mark.parametrize("raw,expected", [
    ("905 to 1420, 2110 to 2295 (seconds from start)", [(905.0, 1420.0), (2110.0, 2295.0)]),
    ("239 to 1348 (seconds from start)", [(239.0, 1348.0)]),
    ("N/A", []),
    (None, []),
])
def test_interval_parsing(raw, expected):
    assert S.parse_intervals(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Yes", True), ("No", False), ("Likely yes", True), ("Likely no", False),
    ("Yes, walking began at 239 seconds", True), ("Sitting", None),
])
def test_yes_no_parsing(raw, expected):
    assert S.parse_yes_no(raw) is expected


def test_iou_basics():
    assert S.iou((0, 100), (0, 100)) == pytest.approx(1.0)
    assert S.iou((0, 100), (100, 200)) == pytest.approx(0.0)
    assert S.iou((0, 100), (50, 150)) == pytest.approx(50 / 150)


def test_interval_set_iou_perfect_and_disjoint():
    assert S.interval_sets_iou([(0, 100)], [(0, 100)]) == pytest.approx(1.0)
    assert S.interval_sets_iou([(0, 100)], [(500, 600)]) == pytest.approx(0.0)
    assert S.interval_sets_iou([], [(0, 100)]) == 0.0


def test_interval_set_iou_handles_differing_episode_counts():
    """Predicted 1 long episode vs 2 true episodes: must still score, not crash or return 0."""
    score = S.interval_sets_iou([(0, 200)], [(0, 90), (110, 200)])
    assert 0.5 < score < 1.0


def test_duration_tolerance_is_the_larger_of_absolute_and_relative():
    # 10% of 1000 = 100 s, so 1080 is within tolerance but 1200 is not
    assert S.score_numeric(1080, 1000, "duration")["correct"] is True
    assert S.score_numeric(1200, 1000, "duration")["correct"] is False
    # for tiny references the 5 s absolute floor applies instead of 10%
    assert S.score_numeric(13, 10, "duration")["correct"] is True


def test_count_tolerance_is_plus_or_minus_one():
    assert S.score_numeric(3, 4, "count")["correct"] is True
    assert S.score_numeric(2, 4, "count")["correct"] is False


def test_unparseable_answer_never_counts_as_correct():
    assert S.score_numeric(None, 100, "duration")["correct"] is False
    assert S.binary_counts(None, True) == "fn"
    assert S.binary_counts(None, False) == "tn"


def test_binary_counts():
    assert S.binary_counts(True, True) == "tp"
    assert S.binary_counts(False, True) == "fn"
    assert S.binary_counts(True, False) == "fp"
    assert S.binary_counts(False, False) == "tn"


def test_grounding_precision_requires_the_cited_span_to_be_mostly_that_activity():
    truth = [{"activity": "walking", "start": 0, "end": 100},
             {"activity": "sitting", "start": 100, "end": 1000}]
    assert S.covers_activity([(0, 90)], truth, "walking") is True
    assert S.covers_activity([(200, 400)], truth, "walking") is False
    assert S.covers_activity([], truth, "walking") is False


def test_macro_f1_ignores_classes_that_never_occur():
    pairs = [("walking", "walking"), ("sitting", "sitting"), ("walking", "sitting")]
    f1, bal = S.macro_f1(pairs)
    assert 0.0 < f1 <= 1.0
    assert 0.0 < bal <= 1.0
