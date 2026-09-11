#!/usr/bin/env python3
"""
End-to-end integration: raw recording CSV -> timeline -> answered question.

Skipped automatically when the trained model or a recording is unavailable, because both are
git-ignored (models are Git LFS, recordings are dataset-derived). A fresh clone therefore runs
the rest of the suite green rather than reporting spurious failures, and this test starts working
as soon as `scripts/prepare_data.py` / `scripts/build_eval_benchmark.py` have been run.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.query.explain as explain_mod            # noqa: E402
import src.query.operations as ops                 # noqa: E402
from src.aggregate.schema import Timeline          # noqa: E402
from src.query.intent import parse_intent_fast_rules  # noqa: E402

MODEL_CANDIDATES = [ROOT / "models" / "model_rf.joblib",
                    ROOT / "results" / "model_rf.joblib",
                    ROOT / "results" / "model_rf_fold0.joblib"]


def _first_existing(paths):
    return next((p for p in paths if p.exists() and p.stat().st_size > 1_000_000), None)


def _first_recording():
    for base in (ROOT / "data" / "eval_benchmark", ROOT / "tests" / "fixtures"):
        if base.is_dir():
            for d in sorted(base.iterdir()):
                rec = d / "recording.csv"
                if rec.exists():
                    return rec
    return None


@pytest.fixture(autouse=True)
def no_slm(monkeypatch):
    original = explain_mod.generate_explanation
    patched = lambda a, d, i, **kw: original(a, d, i, use_llm=False)   # noqa: E731
    monkeypatch.setattr(explain_mod, "generate_explanation", patched)
    monkeypatch.setattr(ops, "generate_explanation", patched)


def test_recording_to_answer_end_to_end():
    model = _first_existing(MODEL_CANDIDATES)
    rec = _first_recording()
    if model is None:
        pytest.skip("no trained model available (Git LFS not pulled, or not yet trained)")
    if rec is None:
        pytest.skip("no recording.csv available (run scripts/build_eval_benchmark.py)")

    from src.aggregate.timeline import build_timeline
    tl = build_timeline(rec, model)

    assert len(tl.intervals) > 0, "no activity intervals produced"
    assert tl.recording_sec > 0
    for a, b in zip(tl.intervals, tl.intervals[1:]):
        assert b.start >= a.end - 1e-6, "timeline intervals must not overlap"

    block = ops.execute_query(parse_intent_fast_rules("How long was the user walking?"), tl)
    text = block.to_output_string()
    for field in ("Answer:", "Activity/Event:", "Evidence:", "Explanation:"):
        assert field in text


def test_committed_fixture_timeline_is_well_formed():
    """Runs on a fresh clone: the fixture timelines are committed, unlike the recordings."""
    fixtures = sorted((ROOT / "tests" / "fixtures").glob("*/timeline.json"))
    if not fixtures:
        pytest.skip("no committed fixture timelines")
    for f in fixtures:
        tl = Timeline.from_json(str(f))
        assert tl.intervals, f"{f} has no intervals"
        assert tl.recording_sec > 0
        for iv in tl.intervals:
            assert iv.end > iv.start, f"{f}: non-positive interval {iv}"
