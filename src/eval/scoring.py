"""
Scoring rules for the QA evaluation harness, implemented to the letter of the challenge brief
("Accuracy Reporting and Required Figures").

Each rule below quotes the requirement it implements, because the brief is explicit that the
correctness rule differs per answer kind and must be stated alongside the numbers.
"""
from __future__ import annotations

import re

CATEGORICAL_TYPES = {"identification", "verification", "comparison", "open_world"}
NUMERIC_TYPES = {"duration", "count"}
TEMPORAL_TYPES = {"grounding"}

# "an answer is correct when its absolute error is at most a chosen threshold, whether absolute,
#  for instance a few seconds, or plus or minus one for a count, or relative, for instance within
#  about ten percent for a duration"
DURATION_ABS_TOL = 5.0      # seconds
DURATION_REL_TOL = 0.10     # 10%
COUNT_ABS_TOL = 1           # +-1 episode
DEFAULT_IOU = 0.5           # "default tau = 0.5"


# --------------------------------------------------------------------------- parsing helpers
def norm_activity(s: str | None) -> str:
    """'Standing in place' / 'standing_in_place' / 'Lying Down' -> 'standing_in_place'."""
    if not s:
        return ""
    return re.sub(r"[^a-z]+", "_", str(s).strip().lower()).strip("_")


def parse_number(s: str | None) -> float | None:
    """First number in a string: '2251 seconds' -> 2251.0, '2 episodes' -> 2.0."""
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def parse_intervals(s: str | None) -> list[tuple[float, float]]:
    """'905 to 1420, 2110 to 2295 (seconds from start)' -> [(905,1420),(2110,2295)]."""
    if not s or str(s).strip().upper() == "N/A":
        return []
    return [(float(a), float(b)) for a, b in
            re.findall(r"(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", str(s))]


def parse_yes_no(s: str | None) -> bool | None:
    """'Yes' / 'Likely yes' / 'Yes, walking began at 239 seconds' -> True; 'No'/'Likely no' -> False."""
    if s is None:
        return None
    t = str(s).strip().lower()
    if t.startswith(("yes", "likely yes", "probably yes")) or ", yes" in t:
        return True
    if t.startswith(("no", "likely no", "probably no")):
        return False
    return None


# --------------------------------------------------------------------------- interval metrics
def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over Union of two intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def interval_sets_iou(pred: list[tuple[float, float]], true: list[tuple[float, float]]) -> float:
    """
    "When several intervals are involved, match the predicted to the true by overlap and average,
     or compute a temporal precision as overlap over predicted length and a recall as overlap over
     true length and take their F1."

    Uses the temporal precision/recall F1 form, which is well defined for any interval counts
    (including a differing number of predicted vs. true episodes) and does not depend on a
    matching heuristic.
    """
    if not pred or not true:
        return 0.0
    overlap = 0.0
    for p in pred:
        for t in true:
            overlap += max(0.0, min(p[1], t[1]) - max(p[0], t[0]))
    len_p = sum(b - a for a, b in pred)
    len_t = sum(b - a for a, b in true)
    if len_p <= 0 or len_t <= 0:
        return 0.0
    precision, recall = min(overlap / len_p, 1.0), min(overlap / len_t, 1.0)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def covers_activity(pred: list[tuple[float, float]], truth_intervals: list[dict], activity: str) -> bool:
    """
    "grounding precision, the fraction of answers whose cited interval, according to the
     ground-truth activity labels, in fact contains the activity that the answer names"

    True when a majority of the cited time is genuinely that activity in the reference labels.
    """
    act = norm_activity(activity)
    if not pred or not act:
        return False
    cited = sum(b - a for a, b in pred)
    if cited <= 0:
        return False
    match = 0.0
    for p in pred:
        for t in truth_intervals:
            if norm_activity(t["activity"]) == act:
                match += max(0.0, min(p[1], t["end"]) - max(p[0], t["start"]))
    return match / cited >= 0.5


# --------------------------------------------------------------------------- per-answer scoring
def score_numeric(pred: float | None, true: float, kind: str) -> dict:
    """Accuracy-within-tolerance plus the error itself (MAE/MAPE are aggregated by the caller)."""
    if pred is None:
        return {"correct": False, "abs_error": None, "pct_error": None}
    err = abs(pred - true)
    if kind == "count":
        ok = err <= COUNT_ABS_TOL
    else:
        ok = err <= max(DURATION_ABS_TOL, DURATION_REL_TOL * true)
    return {"correct": bool(ok), "abs_error": err,
            "pct_error": (err / true * 100.0) if true > 0 else None}


def score_categorical(pred: str | None, true: str) -> bool:
    return norm_activity(pred) == norm_activity(true)


def binary_counts(pred: bool | None, true: bool) -> str:
    """-> 'tp' | 'fp' | 'tn' | 'fn' for verification-style answers."""
    if pred is None:
        return "fn" if true else "tn"          # unparseable answer counts as a miss, never a hit
    if true and pred:
        return "tp"
    if true and not pred:
        return "fn"
    if not true and pred:
        return "fp"
    return "tn"


# --------------------------------------------------------------------------- aggregation
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def macro_f1(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """
    -> (macro_F1, balanced_accuracy) over (true, pred) label pairs.
    "for each class compute precision as TP/(TP+FP) and recall as TP/(TP+FN), combine them as
     F1 = 2 x precision x recall / (precision + recall), and average the per-class values;
     balanced accuracy, the mean of the per-class recall, serves the same end."
    """
    if not pairs:
        return 0.0, 0.0
    labels = sorted({norm_activity(t) for t, _ in pairs} | {norm_activity(p) for _, p in pairs})
    f1s, recalls = [], []
    for lab in labels:
        tp = sum(1 for t, p in pairs if norm_activity(t) == lab and norm_activity(p) == lab)
        fp = sum(1 for t, p in pairs if norm_activity(t) != lab and norm_activity(p) == lab)
        fn = sum(1 for t, p in pairs if norm_activity(t) == lab and norm_activity(p) != lab)
        if tp + fn == 0:                      # class never actually occurs: not a real class here
            continue
        _, r, f = prf(tp, fp, fn)
        f1s.append(f); recalls.append(r)
    return (sum(f1s) / len(f1s) if f1s else 0.0,
            sum(recalls) / len(recalls) if recalls else 0.0)
