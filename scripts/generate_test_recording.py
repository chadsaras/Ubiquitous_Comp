#!/usr/bin/env python3
"""
Generate realistic synthetic recording.csv for testing build_timeline and model inference.
Generates 25 Hz 6-channel sensor streams (Acc X/Y/Z in g, Gyro X/Y/Z in rad/s).

Usage:
    python scripts/generate_test_recording.py
    python scripts/generate_test_recording.py --out tests/fixtures/00EABED2_590_160/recording.csv
    python scripts/generate_test_recording.py --from-fixture tests/fixtures/00EABED2_590_160/truth.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

FS = 25.0  # 25 Hz sampling rate
DT = 1.0 / FS


def synthesize_signal(activity: str, duration_sec: float, rng: np.random.Generator):
    """Generate realistic 6-channel sensor streams for an activity."""
    n_samples = int(round(duration_sec * FS))
    t = np.arange(n_samples) * DT

    # Base gravity vector (device in pocket or on wrist)
    gx_base, gy_base, gz_base = 0.0, 0.1, 0.98

    act = activity.lower()
    if "sit" in act or "lie" in act:
        # Sedentary: pure static gravity + sensor noise
        acc_x = gx_base + rng.normal(0, 0.003, n_samples)
        acc_y = gy_base + rng.normal(0, 0.003, n_samples)
        acc_z = gz_base + rng.normal(0, 0.004, n_samples)
        gyro_x = rng.normal(0, 0.005, n_samples)
        gyro_y = rng.normal(0, 0.005, n_samples)
        gyro_z = rng.normal(0, 0.005, n_samples)

    elif "walk" in act:
        # Walking: 2.0 Hz cadence (~120 steps/min), acc_mag_std ~ 0.12 g
        f = 2.0
        acc_x = gx_base + 0.06 * np.sin(2 * np.pi * (f / 2) * t) + rng.normal(0, 0.02, n_samples)
        acc_y = gy_base + 0.08 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.02, n_samples)
        acc_z = gz_base + 0.15 * np.cos(2 * np.pi * f * t) + rng.normal(0, 0.03, n_samples)
        gyro_x = 0.40 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.05, n_samples)
        gyro_y = 0.25 * np.cos(2 * np.pi * f * t) + rng.normal(0, 0.05, n_samples)
        gyro_z = 0.15 * np.sin(2 * np.pi * (f / 2) * t) + rng.normal(0, 0.04, n_samples)

    elif "run" in act:
        # Running: 2.8 Hz cadence, acc_mag_std ~ 0.42 g, high-impact peaks
        f = 2.8
        impact = 0.20 * np.maximum(0, np.sin(2 * np.pi * f * t)) ** 3
        acc_x = gx_base + 0.18 * np.sin(2 * np.pi * (f / 2) * t) + rng.normal(0, 0.05, n_samples)
        acc_y = gy_base + 0.22 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.05, n_samples)
        acc_z = gz_base + 0.45 * np.cos(2 * np.pi * f * t) + impact + rng.normal(0, 0.06, n_samples)
        gyro_x = 1.10 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.10, n_samples)
        gyro_y = 0.80 * np.cos(2 * np.pi * f * t) + rng.normal(0, 0.10, n_samples)
        gyro_z = 0.40 * np.sin(2 * np.pi * (f / 2) * t) + rng.normal(0, 0.08, n_samples)

    elif "bicycl" in act:
        # Bicycling: smooth pedaling at 1.3 Hz (~80 RPM), sustained gyro oscillations
        f = 1.3
        acc_x = gx_base + 0.05 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.02, n_samples)
        acc_y = gy_base + 0.09 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.02, n_samples)
        acc_z = gz_base + 0.10 * np.cos(2 * np.pi * f * t) + rng.normal(0, 0.03, n_samples)
        gyro_x = 0.60 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.04, n_samples)
        gyro_y = 0.45 * np.cos(2 * np.pi * f * t) + rng.normal(0, 0.04, n_samples)
        gyro_z = 0.20 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.03, n_samples)

    else:
        # Standing & moving / fidgeting
        acc_x = gx_base + 0.03 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.01, n_samples)
        acc_y = gy_base + 0.03 * np.cos(2 * np.pi * 0.4 * t) + rng.normal(0, 0.01, n_samples)
        acc_z = gz_base + rng.normal(0, 0.01, n_samples)
        gyro_x = rng.normal(0, 0.02, n_samples)
        gyro_y = rng.normal(0, 0.02, n_samples)
        gyro_z = rng.normal(0, 0.02, n_samples)

    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z


def generate_recording_from_intervals(intervals, out_path: Path):
    """Synthesize recording dataframe from a list of interval definitions."""
    rng = np.random.default_rng(42)
    dfs = []
    
    for iv in intervals:
        start = float(iv.get("start", 0.0))
        end = float(iv.get("end", 0.0))
        dur = end - start
        if dur <= 0:
            continue
        
        act = iv.get("activity", "sitting")
        ax, ay, az, gx, gy, gz = synthesize_signal(act, dur, rng)
        n_samples = len(ax)
        t = start + np.arange(n_samples) * DT

        df_iv = pd.DataFrame({
            "timestamp": t,
            "acc_x": ax,
            "acc_y": ay,
            "acc_z": az,
            "gyro_x": gx,
            "gyro_y": gy,
            "gyro_z": gz,
        })
        dfs.append(df_iv)

    full_df = pd.concat(dfs, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"Generated recording with {len(full_df)} samples ({len(full_df)*DT:.1f}s) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/test_recording.csv", help="Output path for CSV")
    parser.add_argument("--from-fixture", default=None, help="Path to truth.json or timeline.json")
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.from_fixture and Path(args.from_fixture).exists():
        print(f"Synthesizing recording from fixture: {args.from_fixture}")
        with open(args.from_fixture) as f:
            data = json.load(f)
            # Support both {"intervals": [...]} and [...]
            intervals = data.get("intervals", data) if isinstance(data, dict) else data
    else:
        # Default scenario: 10-minute session: Sitting -> Walking -> Running -> Sitting -> Bicycling
        print("Synthesizing default multi-activity scenario (10 minutes)...")
        intervals = [
            {"start": 0.0, "end": 120.0, "activity": "sitting"},
            {"start": 120.0, "end": 320.0, "activity": "walking"},
            {"start": 320.0, "end": 440.0, "activity": "running"},
            {"start": 440.0, "end": 520.0, "activity": "sitting"},
            {"start": 520.0, "end": 640.0, "activity": "bicycling"},
        ]

    generate_recording_from_intervals(intervals, out_path)


if __name__ == "__main__":
    main()
