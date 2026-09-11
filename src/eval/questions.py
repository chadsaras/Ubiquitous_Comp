"""
Build the benchmark question set with ground truth, from a recording's reference intervals.

The brief keeps the real evaluation questions hidden, so this generates our own across every
question type it names -- identification, verification, duration, count, comparison, grounding
and open-world reasoning -- with the ground-truth answer AND the ground-truth intervals needed
for IoU scoring.

Phrasings are deliberately varied (including phrasings taken from the brief's own scenario, e.g.
"How much of the ... did she spend resting?") so the harness measures generalisation rather than
one canned template.
"""
from __future__ import annotations

PROLONGED_SEC = 1200      # same threshold the system documents for "prolonged" rest
SEDENTARY = ("lying_down", "sitting")
STRENUOUS = ("running", "bicycling")
MIN_PRESENT_SEC = 60      # an activity counts as "occurred" only with at least this much time


def _by_activity(intervals: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for iv in intervals:
        out.setdefault(iv["activity"], []).append(iv)
    return out


def _total(ivs: list[dict]) -> float:
    return sum(iv["end"] - iv["start"] for iv in ivs)


def _pairs(ivs: list[dict]) -> list[tuple[float, float]]:
    return [(iv["start"], iv["end"]) for iv in ivs]


def pretty(act: str) -> str:
    return act.replace("_", " ")


def build_questions(recording_id: str, intervals: list[dict]) -> list[dict]:
    """
    -> [{id, recording, type, question, truth_answer, truth_intervals, truth_value, ...}]

    `truth_answer` is the reference for categorical scoring, `truth_value` for numeric scoring,
    `truth_intervals` for IoU / grounding.
    """
    by_act = _by_activity(intervals)
    present = {a: ivs for a, ivs in by_act.items() if _total(ivs) >= MIN_PRESENT_SEC}
    absent = [a for a in ("walking", "running", "bicycling", "lying_down", "sitting")
              if a not in present]
    qs: list[dict] = []

    def add(qtype: str, question: str, **kw) -> None:
        qs.append({"id": f"{recording_id}::{qtype}::{len(qs)}", "recording": recording_id,
                   "type": qtype, "question": question, **kw})

    if not present:
        return qs

    # ---- identification: the activity occupying the most time in the reference labels
    dominant = max(present, key=lambda a: _total(present[a]))
    add("identification", "What activity is the user performing?",
        truth_answer=dominant, truth_intervals=_pairs(present[dominant]))

    # ---- verification: one activity that DID occur and one that did NOT (the negative case is
    #      what makes specificity meaningful -- the brief warns plain accuracy hides a "no" bias)
    add("verification", f"Is the user {pretty(dominant)}?",
        truth_answer="yes", truth_activity=dominant, truth_intervals=_pairs(present[dominant]))
    for a in sorted(present):
        if a != dominant:
            add("verification", f"Did the user spend any time {pretty(a)}?",
                truth_answer="yes", truth_activity=a, truth_intervals=_pairs(present[a]))
            break
    if absent:
        add("verification", f"Was the user {pretty(absent[0])}?",
            truth_answer="no", truth_activity=absent[0], truth_intervals=[])

    # ---- duration + count, for up to three activities that actually occur
    for a in sorted(present, key=lambda x: -_total(present[x]))[:3]:
        add("duration", f"How long was the user {pretty(a)}?",
            truth_value=_total(present[a]), truth_activity=a, truth_intervals=_pairs(present[a]))
        add("count", f"How many times did the user {pretty(a)}?",
            truth_value=float(len(present[a])), truth_activity=a, truth_intervals=_pairs(present[a]))

    # ---- comparison between the two most common activities
    ranked = sorted(present, key=lambda x: -_total(present[x]))
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        winner = a if _total(present[a]) >= _total(present[b]) else b
        add("comparison", f"Did the user spend more time {pretty(a)} or {pretty(b)}?",
            truth_answer=winner, truth_intervals=_pairs(present[winner]))

    # ---- grounding: onset of an activity that occurs, and one that does not
    for a in sorted(present, key=lambda x: -_total(present[x]))[:2]:
        first = min(present[a], key=lambda iv: iv["start"])
        add("grounding", f"Did the user begin {pretty(a)} at any point, and if so, when?",
            truth_answer="yes", truth_value=first["start"], truth_activity=a,
            truth_intervals=[(first["start"], first["end"])])
    if absent:
        add("grounding", f"Did the user begin {pretty(absent[0])} at any point, and if so, when?",
            truth_answer="no", truth_activity=absent[0], truth_intervals=[])

    # ---- open-world reasoning
    sed = [iv for a in SEDENTARY for iv in by_act.get(a, [])]
    longest_sed = max(sed, key=lambda iv: iv["end"] - iv["start"], default=None)
    rest_yes = longest_sed is not None and (longest_sed["end"] - longest_sed["start"]) >= PROLONGED_SEC
    add("open_world", "Did the user spend a prolonged period resting?",
        truth_answer="yes" if rest_yes else "no",
        truth_intervals=[(longest_sed["start"], longest_sed["end"])] if rest_yes else [])

    lie = by_act.get("lying_down", [])
    longest_lie = max(lie, key=lambda iv: iv["end"] - iv["start"], default=None)
    lie_yes = longest_lie is not None and (longest_lie["end"] - longest_lie["start"]) >= PROLONGED_SEC
    add("open_world", "Did the user lie down for a prolonged period?",
        truth_answer="yes" if lie_yes else "no",
        truth_intervals=[(longest_lie["start"], longest_lie["end"])] if lie_yes else [])

    bike = [iv for iv in by_act.get("bicycling", [])]
    bike_yes = _total(bike) >= MIN_PRESENT_SEC
    add("open_world", "Was the user using a wheeled or pedal-based mode of movement?",
        truth_answer="yes" if bike_yes else "no",
        truth_intervals=_pairs(bike) if bike_yes else [])

    stren = [iv for a in STRENUOUS for iv in by_act.get(a, [])]
    stren_yes = _total(stren) >= MIN_PRESENT_SEC
    add("open_world", "Was the user doing anything strenuous?",
        truth_answer="yes" if stren_yes else "no",
        truth_intervals=_pairs(stren) if stren_yes else [])

    return qs
