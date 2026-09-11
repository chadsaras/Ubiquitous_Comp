#!/usr/bin/env python3
"""
Verify that a model pickled under one scikit-learn version predicts identically under another.

Why this matters: the models were fitted on the training server (scikit-learn 1.9.0) and are
loaded for evaluation on the laptop (scikit-learn 1.6.0). scikit-learn emits
InconsistentVersionWarning for exactly this and warns it "might lead to breaking code or invalid
results". Every accuracy figure in the report is produced by the cross-version load, so this must
be checked rather than assumed.

Run this on BOTH machines and compare the printed fingerprint. Identical fingerprints mean the
cross-version load is faithful and the reported numbers stand.

    python scripts/verify_model_portability.py --model results/model_rf_fold0.joblib
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model_rf_fold0.joblib")
    ap.add_argument("--recording", default=None)
    ap.add_argument("--max-windows", type=int, default=400)
    args = ap.parse_args()

    import sklearn
    from src.aggregate.timeline import (load_model, load_recording, resample_chunk,  # noqa: E402
                                        split_chunks, _context_features)
    from src.recognize.features import WIN, window_features, windows

    rec = Path(args.recording) if args.recording else sorted(
        p for p in Path("data/eval_benchmark").iterdir()
        if p.is_dir() and (p / "recording.csv").exists())[0] / "recording.csv"

    bundle = load_model(args.model)
    model, mask = bundle["model"], bundle.get("mask")
    context = int(bundle.get("context", 1))

    df = load_recording(rec)
    F = []
    for chunk in split_chunks(df):
        grid, X = resample_chunk(chunk)
        if len(X) < WIN // 2:
            continue
        for s, w in windows(X):
            F.append(window_features(w))
            if len(F) >= args.max_windows:
                break
        if len(F) >= args.max_windows:
            break

    F = np.nan_to_num(np.stack(F), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        F = F[:, np.asarray(mask, dtype=bool)]
    Fa = _context_features(F, context)

    proba = model.predict_proba(Fa)
    pred = proba.argmax(1)

    # Round before hashing: identical trees can differ in the last float bits across BLAS builds,
    # which would be a false alarm. 6 decimals is far tighter than anything that affects argmax.
    digest = hashlib.sha256(np.round(proba, 6).tobytes()).hexdigest()[:16]

    print(f"scikit-learn      : {sklearn.__version__}")
    print(f"numpy             : {np.__version__}")
    print(f"model             : {args.model}")
    print(f"recording         : {rec}")
    print(f"windows scored    : {len(Fa)}")
    print(f"predicted classes : {np.bincount(pred, minlength=7).tolist()}")
    print(f"proba column means: {np.round(proba.mean(0), 6).tolist()}")
    print(f"FINGERPRINT       : {digest}")


if __name__ == "__main__":
    main()
