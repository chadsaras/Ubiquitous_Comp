#!/usr/bin/env python3
"""
Fetch and unpack the ExtraSensory dataset pieces needed for the challenge.

Usage:
    python scripts/prepare_data.py                 # labels + folds only (small, ~220 MB)
    python scripts/prepare_data.py --raw           # also phone accelerometer + gyroscope (~15 GB)
    python scripts/prepare_data.py --raw --only acc

Downloads are resumable (re-run if interrupted). Nothing is committed to git;
data/ is in .gitignore.

Dataset: Vaizman, Ellis, Lanckriet. "Recognizing Detailed Human Context In-the-Wild
from Smartphones and Smartwatches." IEEE Pervasive Computing, 2017.
http://extrasensory.ucsd.edu/
"""
import argparse
import sys
import zipfile
from pathlib import Path

import requests

BASE = "http://extrasensory.ucsd.edu/data"
FILES = {
    # name: (url, unpack subdir)
    "labels":  (f"{BASE}/primary_data_files/ExtraSensory.per_uuid_features_labels.zip", "features_labels"),
    "orig":    (f"{BASE}/additional_data_files/ExtraSensory.per_uuid_original_labels.zip", "original_labels"),
    "folds":   (f"{BASE}/cv5Folds.zip", "cv5Folds"),
    "acc":     (f"{BASE}/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip", "raw_acc"),
    "gyro":    (f"{BASE}/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip", "raw_gyro"),
}
SMALL = ["labels", "orig", "folds"]
RAW = ["acc", "gyro"]


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream a file to disk, resuming from a partial download if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:            # range not satisfiable -> already complete
            print(f"  {dest.name}: already complete")
            return
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have and r.status_code == 206 else "wb"
        done = have if mode == "ab" else 0
        with open(dest, mode) as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    sys.stdout.write(f"\r  {dest.name}: {done/1e9:6.2f} / {total/1e9:.2f} GB ({pct:5.1f}%)")
                    sys.stdout.flush()
    print()


def unpack(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  {out_dir.name}: already unpacked")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  unpacking {zip_path.name} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="root data directory")
    ap.add_argument("--raw", action="store_true", help="also fetch raw acc + gyro (~15 GB)")
    ap.add_argument("--only", nargs="*", choices=list(FILES), help="fetch only these keys")
    ap.add_argument("--no-unpack", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    zips = root / "zips"
    keys = args.only or (SMALL + RAW if args.raw else SMALL)

    for key in keys:
        url, sub = FILES[key]
        print(f"[{key}] {url}")
        zp = zips / Path(url).name
        download(url, zp)
        if not args.no_unpack:
            unpack(zp, root / "raw" / sub)

    print("\nDone. Layout:")
    for p in sorted((root / "raw").glob("*")):
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        print(f"  {p}  ({n} files)")


if __name__ == "__main__":
    main()
