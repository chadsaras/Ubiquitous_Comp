#!/usr/bin/env python3
"""
Make a realistic test recording from ExtraSensory and check the timeline builder against truth.

    python scripts/make_test_recording.py --uuid <UUID> --minutes 90
    python scripts/make_test_recording.py --uuid <UUID> --start-minute 400 --minutes 120 --model results/model_rf.joblib

Writes to data/fixtures/<uuid8>_<start>_<n>/:
    recording.csv     timestamp, acc_x..gyro_z  (unix seconds, raw units, bursty like the source)
    truth.json        ground-truth intervals from the minute labels, seconds from start
    timeline.json     the built timeline (if --model given)
and prints the timeline next to the truth with a minute-level agreement score.

Pick a stretch with variety: the script lists the label sequence so you can choose --start-minute.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.load import find_users, load_labels, read_burst          # noqa: E402


def truth_intervals(lab: pd.DataFrame, t0: int, period: int = 60) -> list[dict]:
    """Merge consecutive same-label minutes into intervals (seconds from t0)."""
    out = []
    for ts, row in lab.iterrows():
        if row["label"] is None or (isinstance(row["label"], float) and np.isnan(row["label"])):
            continue
        s, e = float(ts - t0), float(ts - t0 + period)
        if out and out[-1]["activity"] == row["label"] and s - out[-1]["end"] <= 90:
            out[-1]["end"] = e
        else:
            if out:                                   # never let a padded end overrun the next start
                out[-1]["end"] = min(out[-1]["end"], s)
            out.append({"activity": row["label"], "start": s, "end": e})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", default=None)
    ap.add_argument("--start-minute", type=int, default=0, help="index into the user's labelled minutes")
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--model", default=None, help="results/model_rf.joblib to also build + score a timeline")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--list", action="store_true", help="just print the label sequence and exit")
    args = ap.parse_args()

    uuid = args.uuid or find_users(args.data_dir)[0]
    lab = load_labels(uuid, args.data_dir)
    lab = lab[lab["label"].notna()]
    if args.list:
        seq = lab["label"].to_numpy()
        i = 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            print(f"  minutes {i:5d}-{j-1:5d}  {seq[i]}")
            i = j
        return

    sel = lab.iloc[args.start_minute: args.start_minute + args.minutes]
    root = Path(args.data_dir) / "raw"
    acc_dir = next((root / "raw_acc").rglob(uuid)); gyr_dir = next((root / "raw_gyro").rglob(uuid))
    acc_idx = {int(p.name.split(".")[0]): p for p in acc_dir.iterdir()}
    gyr_idx = {int(p.name.split(".")[0]): p for p in gyr_dir.iterdir()}

    frames, kept = [], []
    for ts in sel.index:
        pa, pg = acc_idx.get(ts), gyr_idx.get(ts)
        if pa is None or pg is None:
            continue
        a, g = read_burst(pa), read_burst(pg)
        if a is None or g is None:
            continue
        n = min(len(a), len(g))
        t = ts + a[:n, 0]
        frames.append(pd.DataFrame({"timestamp": t, "acc_x": a[:n, 1], "acc_y": a[:n, 2], "acc_z": a[:n, 3],
                                    "gyro_x": g[:n, 1], "gyro_y": g[:n, 2], "gyro_z": g[:n, 3]}))
        kept.append(ts)
    if not frames:
        sys.exit("no usable minutes in that range")
    rec = pd.concat(frames, ignore_index=True)
    t0 = int(kept[0])
    truth = truth_intervals(sel.loc[kept], t0)

    out = Path(args.data_dir) / "fixtures" / f"{uuid[:8]}_{args.start_minute}_{args.minutes}"
    out.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out / "recording.csv", index=False)
    (out / "truth.json").write_text(json.dumps({"t0_unix": t0, "intervals": truth}, indent=1))
    print(f"wrote {out}/recording.csv  ({len(rec)} rows, {len(kept)} minutes, {rec['timestamp'].iloc[-1]-t0:.0f} s)")
    print("truth:")
    for iv in truth:
        print(f"  {iv['start']:8.0f} - {iv['end']:8.0f}  {iv['activity']}")

    if not args.model:
        return
    from src.aggregate.timeline import build_timeline
    tl = build_timeline(out / "recording.csv", args.model)
    tl.to_json(str(out / "timeline.json"))
    print("\npredicted:"); print(tl)

    # minute-level agreement: label the truth minute vs the timeline's dominant activity in it
    hits = tot = 0
    for ts in kept:
        s = float(ts - t0)
        pred = tl.dominant(s, s + 60)
        tot += 1; hits += pred == sel.loc[ts, "label"]
    print(f"\nminute-level agreement: {hits}/{tot} = {hits/tot:.2f}")
    print("total duration per activity (truth vs predicted):")
    for c in sorted({iv['activity'] for iv in truth} | {iv.activity for iv in tl.intervals}):
        tt = sum(iv["end"] - iv["start"] for iv in truth if iv["activity"] == c)
        print(f"  {c:<20} {tt:7.0f} s   {tl.total_duration(c):7.0f} s")


if __name__ == "__main__":
    main()