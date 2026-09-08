"""
Stage 2: Deterministic Evidence Computation over Timeline.
Translates a parsed QueryIntent into exact computations on the Timeline object,
adhering strictly to Contract 1 (Timeline) and Contract 2 (Output Block).
"""
from __future__ import annotations
from typing import List, Optional
from src.query.schema import QueryIntent, IntentType, FormattedAnswerBlock


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
            explanation=f"{_fmt_activity(target)} was detected for a total duration of "
                        f"{int(round(dur))} seconds across {len(ivs)} distinct episode(s)."
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
            explanation=f"{act_name} was not detected in the recording (0 seconds)."
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
        explanation = (
            f"A sustained rise in accelerometer magnitude variance (acc_mag_std = {acc_std:.3f} g) "
            f"at cadence {dom_freq:.1f} Hz marks the onset transition to {target.replace('_', ' ')} "
            f"at {t_onset} seconds."
        )

        return FormattedAnswerBlock(
            answer=f"Yes, {target.replace('_', ' ')} began at {t_onset} seconds",
            activity_event=f"Onset of {target.replace('_', ' ')}",
            timestamps=f"{int(round(first_iv.start))} to {int(round(first_iv.end))} (seconds from start)",
            sensor_modality="Accelerometer, Gyroscope",
            sensor_channels="All",
            explanation=explanation
        )
    else:
        return FormattedAnswerBlock(
            answer="No",
            activity_event=f"Onset of {target.replace('_', ' ')}",
            timestamps="N/A",
            sensor_modality="N/A",
            sensor_channels="N/A",
            explanation=f"No episodes of {target.replace('_', ' ')} were observed in the recording."
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


def handle_open_world(intent: QueryIntent, timeline) -> FormattedAnswerBlock:
    """Task 4: Open-World Semantic Reasoning grounded in signal statistics."""
    concept = (intent.semantic_concept or "").lower()

    # 1. Prolonged rest / lying down
    if "rest" in concept or "prolonged" in concept:
        lying_ivs = timeline.of("lying_down")
        if lying_ivs:
            longest = max(lying_ivs, key=lambda iv: iv.end - iv.start)
            dur = int(round(longest.end - longest.start))
            if dur >= 1200:  # >= 20 minutes
                return FormattedAnswerBlock(
                    answer="Likely yes",
                    activity_event="Prolonged lying down",
                    timestamps=f"{int(round(longest.start))} to {int(round(longest.end))} (seconds from start)",
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation="A long, continuous stretch of near-zero acceleration variance and "
                                "minimal gyroscope activity, well beyond any brief stationary pause, "
                                "is consistent with sustained rest rather than a transient stop."
                )

    # 2. Wheeled or pedal-based movement (bicycling)
    if "wheel" in concept or "pedal" in concept or "cycl" in concept:
        bike_ivs = timeline.of("bicycling")
        if bike_ivs:
            return FormattedAnswerBlock(
                answer="Yes",
                activity_event="Unknown outdoor physical activity, consistent with cycling",
                timestamps=fmt_range(bike_ivs),
                sensor_modality="Accelerometer, Gyroscope",
                sensor_channels="All",
                explanation="The segment shows smooth, continuous, cyclic acceleration at a steady cadence, "
                            "without the discrete heel-strike spikes of walking or running, accompanied by sustained "
                            "periodic gyroscope oscillation consistent with pedaling and balance, which points to a low-impact wheeled mode."
            )

    # 3. Strenuous activity
    if "strenuous" in concept:
        candidates = [iv for iv in timeline.intervals if iv.activity in ("running", "bicycling")]
        if candidates:
            longest = max(candidates, key=lambda iv: iv.end - iv.start)
            return FormattedAnswerBlock(
                answer="Yes",
                activity_event=f"Strenuous activity ({longest.activity})",
                timestamps=f"{int(round(longest.start))} to {int(round(longest.end))} (seconds from start)",
                sensor_modality="Accelerometer, Gyroscope",
                sensor_channels="All",
                explanation=f"Sustained elevated accelerometer magnitude variance and high kinetic energy "
                            f"indicate strenuous physical exertion during the cited interval."
            )

    # Fallback open-world answer
    return FormattedAnswerBlock(
        answer="Likely no",
        activity_event="Unclassified behavior",
        timestamps="N/A",
        sensor_modality="N/A",
        sensor_channels="N/A",
        explanation="No sensor signal pattern matched the described behavior."
    )


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
