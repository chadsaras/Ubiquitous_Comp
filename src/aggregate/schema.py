"""
Shared contracts between the recognition/aggregation layers and the query/eval layers.

Contract 1: the Timeline — a sorted, non-overlapping list of Interval objects. It is the single
            source of truth for every answer. Nothing downstream may invent a number that is not
            derivable from it.
Contract 2: format_answer() — the only place the challenge's output block is written.

Time convention: all times are SECONDS FROM THE START OF THE RECORDING (floats).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable

CLASSES = ["lying_down", "sitting", "standing_in_place", "standing_and_moving",
           "walking", "running", "bicycling"]

PRETTY = {
    "lying_down": "Lying down", "sitting": "Sitting", "standing_in_place": "Standing in place",
    "standing_and_moving": "Standing and moving", "walking": "Walking", "running": "Running",
    "bicycling": "Bicycling",
}


@dataclass
class Interval:
    activity: str                 # one of CLASSES
    start: float                  # seconds from recording start
    end: float                    # seconds from recording start (end > start)
    confidence: float             # mean of the winning class probability over its windows
    n_windows: int                # how many analysis windows support it
    observed_sec: float           # seconds actually covered by sensor data (<= end - start if gaps)
    runner_up: str | None = None  # second most likely class over the interval
    runner_up_prob: float = 0.0
    summary: dict = field(default_factory=dict)   # mean signal_summary() over its windows

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, t0: float, t1: float) -> float:
        """Seconds of overlap with [t0, t1]."""
        return max(0.0, min(self.end, t1) - max(self.start, t0))


@dataclass
class Timeline:
    intervals: list[Interval]
    recording_sec: float                  # total span of the recording
    hz: float = 25.0
    time_base: str = "seconds from start"
    meta: dict = field(default_factory=dict)   # bursty?, sampling info, model name, ...

    # ---------------------------------------------------------------- basic queries
    def of(self, activity: str) -> list[Interval]:
        return [iv for iv in self.intervals if iv.activity == activity]

    def total_duration(self, activity: str) -> float:
        return sum(iv.duration for iv in self.of(activity))

    def count(self, activity: str) -> int:
        return len(self.of(activity))

    def first(self, activity: str) -> Interval | None:
        ivs = self.of(activity)
        return ivs[0] if ivs else None

    def last(self, activity: str) -> Interval | None:
        ivs = self.of(activity)
        return ivs[-1] if ivs else None

    def longest(self, activity: str | None = None) -> Interval | None:
        ivs = self.of(activity) if activity else self.intervals
        return max(ivs, key=lambda iv: iv.duration) if ivs else None

    def at(self, t: float) -> Interval | None:
        for iv in self.intervals:
            if iv.start <= t < iv.end:
                return iv
        return None

    def between(self, t0: float, t1: float) -> list[Interval]:
        return [iv for iv in self.intervals if iv.overlaps(t0, t1) > 0]

    def dominant(self, t0: float | None = None, t1: float | None = None) -> str | None:
        """Activity with the most time in [t0, t1] (whole recording by default)."""
        t0 = 0.0 if t0 is None else t0
        t1 = self.recording_sec if t1 is None else t1
        tot: dict[str, float] = {}
        for iv in self.between(t0, t1):
            tot[iv.activity] = tot.get(iv.activity, 0.0) + iv.overlaps(t0, t1)
        return max(tot, key=tot.get) if tot else None

    def transitions(self) -> list[tuple[float, str, str]]:
        """(time, from_activity, to_activity) for each change between adjacent intervals."""
        out = []
        for a, b in zip(self.intervals, self.intervals[1:]):
            if a.activity != b.activity:
                out.append((b.start, a.activity, b.activity))
        return out

    def durations(self) -> dict[str, float]:
        return {c: self.total_duration(c) for c in CLASSES}

    # ---------------------------------------------------------------- (de)serialisation
    def to_json(self, path: str | None = None) -> str:
        d = {"intervals": [asdict(iv) for iv in self.intervals], "recording_sec": self.recording_sec,
             "hz": self.hz, "time_base": self.time_base, "meta": self.meta}
        s = json.dumps(d, indent=1)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, path_or_str: str) -> "Timeline":
        try:
            with open(path_or_str) as f:
                d = json.load(f)
        except (FileNotFoundError, OSError):
            d = json.loads(path_or_str)
        return cls(intervals=[Interval(**iv) for iv in d["intervals"]], recording_sec=d["recording_sec"],
                   hz=d.get("hz", 25.0), time_base=d.get("time_base", "seconds from start"),
                   meta=d.get("meta", {}))

    def __str__(self) -> str:
        lines = [f"Timeline: {len(self.intervals)} intervals over {self.recording_sec:.0f} s"]
        for iv in self.intervals:
            lines.append(f"  {iv.start:8.1f} - {iv.end:8.1f}  {iv.activity:<20} "
                         f"conf={iv.confidence:.2f}  n={iv.n_windows}")
        return "\n".join(lines)


# ==================================================================== Contract 2: output block
def fmt_time(t: float) -> str:
    return f"{t:.0f}"


def fmt_range(ivs: Iterable[Interval] | Iterable[tuple[float, float]]) -> str:
    """'905 to 1420, 2110 to 2295 (seconds from start)' from intervals or (start, end) pairs."""
    parts = []
    for iv in ivs:
        s, e = (iv.start, iv.end) if isinstance(iv, Interval) else iv
        parts.append(f"{fmt_time(s)} to {fmt_time(e)}")
    return (", ".join(parts) + " (seconds from start)") if parts else "N/A"


def format_answer(answer: str, activity: str = "N/A", timestamps: str = "N/A",
                  modality: str = "N/A", channels: str = "N/A", explanation: str = "N/A") -> str:
    """
    The one and only writer of the challenge's output format. Fields in the mandated order.
    Multi-line explanations are indented to match the brief's examples.
    """
    expl_lines = str(explanation).strip().splitlines() or ["N/A"]
    expl = expl_lines[0] + "".join("\n             " + ln.strip() for ln in expl_lines[1:])
    return (
        f"Answer: {answer}\n"
        f"Activity/Event: {activity}\n"
        f"Evidence:\n"
        f" Timestamp(s): {timestamps}\n"
        f" Sensor Modality: {modality}\n"
        f" Sensor Channel(s): {channels}\n"
        f"Explanation: {expl}"
    )
