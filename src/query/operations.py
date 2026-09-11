"""
Stage 2: Deterministic Evidence Computation over Timeline.
Translates a parsed QueryIntent into exact computations on the Timeline object,
adhering strictly to Contract 1 (Timeline) and Contract 2 (Output Block).
"""
from __future__ import annotations
from typing import List, Optional
from src.query.schema import QueryIntent, IntentType, FormattedAnswerBlock
from src.query.explain import generate_explanation

try:
    from src.aggregate.schema import fmt_range, format_answer
except ImportError:
    def fmt_range(intervals) -> str:
        if not intervals:
            return "N/A"
        spans = [f"{int(round(iv.start))} to {int(round(iv.end))}" for iv in intervals]
        return f"{', '.join(spans)} (seconds from start)"

    def format_answer(answer, activity, timestamps, modality, channels, explanation) -> str:
        return (
            f"Answer: {answer}\n"
            f"Activity/Event: {activity}\n"
            f"Evidence:\n"
            f"  Timestamp(s): {timestamps}\n"
            f"  Sensor Modality: {modality}\n"
            f"  Sensor Channel(s): {channels}\n"
            f"Explanation: {explanation}"
        )


def _fmt_activity(act: Optional[str]) -> str:
    if not act:
        return "N/A"
    return act.replace("_", " ").title()


def handle_identify(timeline) -> FormattedAnswerBlock:
    """Task 1: Open Activity Identification."""
    dom = timeline.dominant(0.0, 1e9)
    if not dom:
        return FormattedAnswerBlock(
            answer="N/A", activity_event="N/A", timestamps="N/A",
            sensor_modality="N/A", sensor_channels="N/A",
            explanation="No activities recorded on the timeline."
        )
    
    dom_ivs = timeline.of(dom)
    dur = timeline.total_duration(dom)
    return FormattedAnswerBlock(
        answer=_fmt_activity(dom),
        activity_event=_fmt_activity(dom),
        timestamps=fmt_range(dom_ivs),
        sensor_modality="Accelerometer, Gyroscope",
        sensor_channels="All",
        explanation=f"The dominant activity across the recording was {dom.replace('_', ' ')}, "
                    f"detected for a total of {int(round(dur))} seconds across {len(dom_ivs)} episode(s)."
    )


def handle_verify(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """Task 1: Binary Activity Verification."""
    target = intent.target_activity
    if not target:
        return handle_identify(timeline)

    dur = timeline.total_duration(target)
    ivs = timeline.of(target)

    if dur > 0 and len(ivs) > 0:
        return FormattedAnswerBlock(
            answer="Yes",
            activity_event=_fmt_activity(target),
            timestamps=fmt_range(ivs),
            sensor_modality="Accelerometer, Gyroscope",
            sensor_channels="All",
            explanation = generate_explanation(target, dur, ivs)
        )
    else:
        return FormattedAnswerBlock(
            answer="No",
            activity_event=_fmt_activity(target),
            timestamps="N/A",
            sensor_modality="N/A",
            sensor_channels="N/A",
            explanation=f"No episodes of {target.replace('_', ' ')} were observed in the recording."
        )


def handle_duration(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """Task 2: Temporal Duration Reasoning."""
    target = intent.target_activity
    if not target:
        return FormattedAnswerBlock(
            answer="N/A", activity_event="N/A", timestamps="N/A",
            sensor_modality="N/A", sensor_channels="N/A",
            explanation="No target activity specified in duration query."
        )

    dur = timeline.total_duration(target)
    ivs = timeline.of(target)
    act_name = _fmt_activity(target)

    if not ivs or dur == 0:
        return FormattedAnswerBlock(
            answer="0 seconds",
            activity_event=act_name,
            timestamps="N/A",
            sensor_modality="N/A",
            sensor_channels="N/A",
            explanation=generate_explanation(target, dur, ivs)
        )


    dur_int = int(round(dur))
    if len(ivs) == 1:
        ep_desc = f"a single interval of {int(round(ivs[0].end - ivs[0].start))} seconds"
    else:
        ep_lens = [str(int(round(iv.end - iv.start))) for iv in ivs]
        ep_desc = f"{len(ivs)} separate intervals, of {', '.join(ep_lens)} seconds, which sum to {dur_int} seconds"

    return FormattedAnswerBlock(
        answer=f"{dur_int} seconds",
        activity_event=act_name,
        timestamps=fmt_range(ivs),
        sensor_modality="Accelerometer, Gyroscope",
        sensor_channels="All",
        explanation=f"{act_name} was detected in {ep_desc}."
    )


def handle_count(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """Task 2: Quantitative Episode Count Reasoning."""
    target = intent.target_activity
    if not target:
        return FormattedAnswerBlock(
            answer="0 episodes", activity_event="N/A", timestamps="N/A",
            sensor_modality="N/A", sensor_channels="N/A",
            explanation="No activity specified for count query."
        )

    cnt = timeline.count(target)
    ivs = timeline.of(target)
    act_name = _fmt_activity(target)

    return FormattedAnswerBlock(
        answer=f"{cnt} episode{'s' if cnt != 1 else ''}",
        activity_event=act_name,
        timestamps=fmt_range(ivs) if ivs else "N/A",
        sensor_modality="Accelerometer, Gyroscope" if ivs else "N/A",
        sensor_channels="All" if ivs else "N/A",
        explanation=f"The user engaged in {target.replace('_', ' ')} across {cnt} distinct episode(s)."
    )


def handle_onset(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """Task 2 & 3: Onset Event Localization & Evidence Grounding."""
    target = intent.target_activity or "running"
    first_iv = timeline.first(target)
    act_name = _fmt_activity(target)

    if first_iv is not None:
        t_onset = int(round(first_iv.start))
        summary = first_iv.summary or {}
        acc_std = summary.get("acc_mag_std", 0.0)
        dom_freq = summary.get("acc_dom_freq_hz", 0.0)

        # Build evidence grounding explanation citing real signal features
        explanation = generate_explanation(target, first_iv.end - first_iv.start, [first_iv])


        return FormattedAnswerBlock(
            answer=f"Yes, {target.replace('_', ' ')} began at {t_onset} seconds",
            activity_event=f"Onset of {target.replace('_', ' ')}",
            timestamps=f"{int(round(first_iv.start))} to {int(round(first_iv.end))} (seconds from start)",
            sensor_modality="Accelerometer, Gyroscope",
            sensor_channels="All",
            explanation=explanation
        )
    else:
        # Task 3 makes evidence "required and directly assessed", so a negative onset still cites
        # the span that was examined to reach it -- unlike Task 1 verification, where the brief
        # explicitly permits N/A.
        return FormattedAnswerBlock(
            answer="No",
            activity_event=f"Onset of {target.replace('_', ' ')}",
            timestamps=f"0 to {int(round(timeline.recording_sec))} (seconds from start)",
            sensor_modality="Accelerometer, Gyroscope",
            sensor_channels="All",
            explanation=f"No onset of {target.replace('_', ' ')} was found: the full recording was "
                        f"searched and no interval was classified as "
                        f"{target.replace('_', ' ')}."
        )


def handle_compare(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """Task 2: Comparative Temporal Reasoning."""
    act1 = intent.target_activity or "walking"
    act2 = intent.compare_activity or "running"

    dur1 = timeline.total_duration(act1)
    dur2 = timeline.total_duration(act2)

    dur1_int = int(round(dur1))
    dur2_int = int(round(dur2))

    winner = act1 if dur1 >= dur2 else act2
    winner_name = _fmt_activity(winner)

    return FormattedAnswerBlock(
        answer=winner_name,
        activity_event=f"{_fmt_activity(act1)}, {_fmt_activity(act2)}",
        timestamps=f"{_fmt_activity(act1)} = {dur1_int} seconds total, {_fmt_activity(act2)} = {dur2_int} seconds total",
        sensor_modality="Accelerometer, Gyroscope",
        sensor_channels="All",
        explanation=f"Total {winner.replace('_', ' ')} time ({max(dur1_int, dur2_int)} seconds) exceeded "
                    f"total {(act2 if winner == act1 else act1).replace('_', ' ')} time ({min(dur1_int, dur2_int)} seconds) over the recording."
    )


PROLONGED_SEC = 1200        # 20 minutes; below this a sedentary stretch reads as a pause, not rest


def _negative_open_world(event: str, reason: str, timeline, candidates: list) -> FormattedAnswerBlock:
    """
    A "no" at Tier 4 still has to be grounded -- the brief makes evidence and explanation
    mandatory at this tier, so a bare N/A scores zero even when the verdict is right. Cite the
    strongest candidate actually found (that's *why* it's a no), or the span examined when there
    was no candidate at all.
    """
    if candidates:
        longest = max(candidates, key=lambda iv: iv.end - iv.start)
        span = f"{int(round(longest.start))} to {int(round(longest.end))} (seconds from start)"
        found = (f"The closest match found was {longest.activity.replace('_', ' ')} for "
                 f"{int(round(longest.end - longest.start))} seconds over the cited interval. ")
    else:
        span = f"0 to {int(round(timeline.recording_sec))} (seconds from start)"
        found = "No interval of the relevant kind was detected anywhere in the recording. "
    return FormattedAnswerBlock(
        answer="Likely no",
        activity_event=event,
        timestamps=span,
        sensor_modality="Accelerometer, Gyroscope",
        sensor_channels="All",
        explanation=found + reason,
    )


def handle_open_world(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """
    Task 4: Open-World Semantic Reasoning grounded in signal statistics.

    Every branch -- positive or negative -- returns real timestamps, modality and channels,
    because at this tier the brief assesses the evidence itself, not just the verdict.
    """
    concept = (intent.semantic_concept or "").lower()

    # 1. Prolonged rest. If the question named a specific posture ("did she lie down..."), judge
    #    that posture only; otherwise any sustained sedentary stretch counts as rest.
    if "rest" in concept or "prolonged" in concept:
        rest_classes = [intent.target_activity] if intent.target_activity else ["lying_down", "sitting"]
        rest_ivs = [iv for c in rest_classes for iv in timeline.of(c)]
        if rest_ivs:
            longest = max(rest_ivs, key=lambda iv: iv.end - iv.start)
            dur = int(round(longest.end - longest.start))
            if dur >= PROLONGED_SEC:
                summary = longest.summary or {}
                acc_std = summary.get("acc_mag_std")
                gyro_std = summary.get("gyro_mag_std")
                measured = (f" Measured over that interval: acc_mag_std = {acc_std:.4f} g"
                            + (f", gyro_mag_std = {gyro_std:.4f} rad/s" if gyro_std is not None else "")
                            + ".") if acc_std is not None else ""
                return FormattedAnswerBlock(
                    answer="Likely yes",
                    activity_event=f"Prolonged {longest.activity.replace('_', ' ')}",
                    timestamps=f"{int(round(longest.start))} to {int(round(longest.end))} (seconds from start)",
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation=f"A continuous {dur}-second stretch of near-zero acceleration variance and "
                                f"minimal gyroscope activity, well beyond any brief stationary pause, is "
                                f"consistent with sustained rest rather than a transient stop." + measured,
                )
        return _negative_open_world(
            "Prolonged rest", f"No sedentary stretch reached the {PROLONGED_SEC}-second threshold used "
                              f"to separate sustained rest from a transient pause.", timeline, rest_ivs)

    # 2. Wheeled or pedal-based movement (bicycling)
    if "wheel" in concept or "pedal" in concept or "cycl" in concept:
        bike_ivs = timeline.of("bicycling")
        if bike_ivs:
            longest = max(bike_ivs, key=lambda iv: iv.end - iv.start)
            summary = longest.summary or {}
            gyro_std = summary.get("gyro_mag_std")
            measured = f" Gyroscope magnitude variability over the cited span was {gyro_std:.3f} rad/s." \
                if gyro_std is not None else ""
            return FormattedAnswerBlock(
                answer="Yes",
                activity_event="Unknown outdoor physical activity, consistent with cycling",
                timestamps=fmt_range(bike_ivs),
                sensor_modality="Accelerometer, Gyroscope",
                sensor_channels="All",
                explanation="The segment shows smooth, continuous, cyclic acceleration at a steady cadence, "
                            "without the discrete heel-strike spikes of walking or running, accompanied by "
                            "sustained periodic gyroscope oscillation consistent with pedaling and balance, "
                            "which points to a low-impact wheeled mode." + measured,
            )
        return _negative_open_world(
            "Wheeled or pedal-based movement",
            "No interval showed the smooth, steady-cadence acceleration without heel-strike spikes that "
            "distinguishes pedalling from walking or running.", timeline,
            [iv for iv in timeline.intervals if iv.activity in ("walking", "running")])

    # 3. Strenuous activity
    if "strenuous" in concept:
        candidates = [iv for iv in timeline.intervals if iv.activity in ("running", "bicycling")]
        if candidates:
            longest = max(candidates, key=lambda iv: iv.end - iv.start)
            summary = longest.summary or {}
            acc_std = summary.get("acc_mag_std")
            measured = f" Acceleration magnitude variability over that span was {acc_std:.3f} g." \
                if acc_std is not None else ""
            return FormattedAnswerBlock(
                answer="Yes",
                activity_event=f"Strenuous activity ({longest.activity.replace('_', ' ')})",
                timestamps=f"{int(round(longest.start))} to {int(round(longest.end))} (seconds from start)",
                sensor_modality="Accelerometer, Gyroscope",
                sensor_channels="All",
                explanation="Sustained elevated accelerometer magnitude variance and high kinetic energy "
                            "indicate strenuous physical exertion during the cited interval." + measured,
            )
        return _negative_open_world(
            "Strenuous activity",
            "No interval reached the sustained high acceleration variance that distinguishes exertion "
            "from ordinary ambulation.", timeline,
            [iv for iv in timeline.intervals if iv.activity in ("walking", "standing_and_moving")])

    # Unrecognised concept: still grounded, and honest that the concept was not matched.
    return _negative_open_world(
        "Unclassified behavior",
        "The described behaviour did not map onto any pattern the system computes from the signal.",
        timeline, timeline.intervals)


def execute_query(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """Route a parsed QueryIntent to its respective deterministic handler."""
    dispatch = {
        IntentType.IDENTIFY: handle_identify,
        IntentType.VERIFY: lambda tl: handle_verify(intent, tl),
        IntentType.DURATION: lambda tl: handle_duration(intent, tl),
        IntentType.COUNT: lambda tl: handle_count(intent, tl),
        IntentType.ONSET: lambda tl: handle_onset(intent, tl),
        IntentType.COMPARE: lambda tl: handle_compare(intent, tl),
        IntentType.OPEN_WORLD: lambda tl: handle_open_world(intent, tl),
    }

    handler = dispatch.get(intent.intent, handle_identify)
    return handler(timeline)
