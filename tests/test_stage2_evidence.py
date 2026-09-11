#!/usr/bin/env python3
"""
Stage 2/3: deterministic query execution and evidence grounding over a Timeline.

Runs against a hand-built Timeline rather than a trained model, so these tests need no model
file, no dataset and no Ollama server -- they check the query layer's own logic in isolation.
Explanations are generated on the deterministic template path for the same reason.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.query.explain as explain_mod            # noqa: E402
import src.query.operations as ops                 # noqa: E402
from src.aggregate.schema import Interval, Timeline  # noqa: E402
from src.query.intent import parse_intent_fast_rules  # noqa: E402


@pytest.fixture(autouse=True)
def no_slm(monkeypatch):
    """Force the deterministic explanation path; operations.py binds the symbol at import."""
    original = explain_mod.generate_explanation
    patched = lambda a, d, i, **kw: original(a, d, i, use_llm=False)   # noqa: E731
    monkeypatch.setattr(explain_mod, "generate_explanation", patched)
    monkeypatch.setattr(ops, "generate_explanation", patched)


def _iv(activity, start, end, **summary):
    return Interval(activity=activity, start=start, end=end, confidence=0.8, n_windows=10,
                    observed_sec=end - start, runner_up="sitting", runner_up_prob=0.1,
                    summary=summary or {"acc_mag_std": 0.11, "acc_dom_freq_hz": 2.0,
                                        "gyro_mag_std": 0.07})


@pytest.fixture
def timeline():
    """Walking twice (300 s total), a long sitting stretch (2000 s), and no running at all."""
    return Timeline(intervals=[
        _iv("walking", 100, 250),
        _iv("sitting", 250, 2250, acc_mag_std=0.003, acc_dom_freq_hz=0.2, gyro_mag_std=0.01),
        _iv("walking", 2250, 2400),
    ], recording_sec=2400.0)


def answer_for(question, timeline):
    return ops.execute_query(parse_intent_fast_rules(question), timeline)


def test_duration_sums_all_episodes(timeline):
    block = answer_for("How long was the user walking?", timeline)
    assert "300" in block.answer                    # 150 + 150
    assert block.timestamps != "N/A"


def test_count_counts_episodes(timeline):
    block = answer_for("How many times did the user walk?", timeline)
    assert block.answer.startswith("2")


def test_comparison_picks_the_longer_activity(timeline):
    block = answer_for("Did the user spend more time walking or sitting?", timeline)
    assert "sitting" in block.answer.lower()


def test_onset_reports_the_first_episode_start(timeline):
    block = answer_for("Did the user begin walking at any point, and if so, when?", timeline)
    assert block.answer.lower().startswith("yes")
    assert "100" in block.answer


def test_verification_of_an_absent_activity_says_no(timeline):
    block = answer_for("Is the user running?", timeline)
    assert block.answer.strip().lower().startswith("no")


@pytest.mark.parametrize("question", [
    "Did the user spend a prolonged period resting?",
    "Did the user lie down for a prolonged period?",
    "Was the user using a wheeled or pedal-based mode of movement?",
    "Was the user doing anything strenuous?",
])
def test_open_world_always_cites_evidence(question, timeline):
    """
    The brief makes evidence and explanation MANDATORY at Tier 4, including for a "no" verdict.
    Regression: every negative answer used to emit N/A for all three evidence fields, which
    scores zero under the combined grounded-and-correct rule even when the verdict is right.
    """
    block = answer_for(question, timeline)
    assert block.timestamps.strip().upper() != "N/A", f"{question}: no timestamps cited"
    assert block.sensor_modality.strip().upper() != "N/A", f"{question}: no modality cited"
    assert block.sensor_channels.strip().upper() != "N/A", f"{question}: no channels cited"
    assert block.explanation.strip().upper() != "N/A"


def test_prolonged_rest_counts_a_long_sitting_stretch(timeline):
    """Generic "resting" must consider sitting, not only lying down."""
    block = answer_for("Did the user spend a prolonged period resting?", timeline)
    assert block.answer.lower().startswith(("yes", "likely yes"))
    assert "sitting" in block.activity_event.lower()


def test_lie_down_question_is_not_satisfied_by_sitting(timeline):
    """The counterpart: a long sit must NOT be reported as prolonged lying down."""
    block = answer_for("Did the user lie down for a prolonged period?", timeline)
    assert block.answer.lower().startswith(("no", "likely no"))


def test_output_block_has_all_six_fields_in_order(timeline):
    text = answer_for("How long was the user walking?", timeline).to_output_string()
    order = ["Answer:", "Activity/Event:", "Evidence:", "Timestamp(s):",
             "Sensor Modality:", "Sensor Channel(s):", "Explanation:"]
    positions = [text.find(f) for f in order]
    assert all(p >= 0 for p in positions), f"missing field in output block:\n{text}"
    assert positions == sorted(positions), f"fields out of order:\n{text}"
