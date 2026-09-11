#!/usr/bin/env python3
"""
Mandatory resource-cost measurement (the brief's "Accuracy and the Cost of Running It").

"At a minimum, alongside your accuracy figures, report the resource cost of your system: the model
 size, both in parameters and on disk, the peak memory used during inference, and the time taken to
 answer a single query. Measure these on a defined target and state clearly what that target is."

This is required for full marks even without attempting the efficiency extra credit -- only the
accuracy-versus-overhead *curve* (Figure 4, two operating points) is extra credit.

TARGET DEVICE: the laptop this repository is developed on, CPU only, no GPU used at any stage.
The exact CPU/RAM/OS are captured at runtime and written into the results file, so the numbers are
attributable rather than asserted.

Both halves of the deployed system are measured, because both run per query:
  * recognizer  -- Random Forest over 302 engineered features
  * SLM         -- Qwen 2.5 1.5B (Q4_K_M) served locally by Ollama

Costs are separated into one-time-per-recording (building the activity timeline) and
per-query (intent parsing -> deterministic timeline query -> grounded explanation), because
conflating them would misrepresent the cost of answering a question about an already-ingested
recording.

Usage:
    python scripts/measure_cost.py
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = [
    "What activity is the user performing?",           # identify   (no SLM)
    "How long was the user walking?",                  # duration   (no SLM)
    "How many times did the user walk?",               # count      (no SLM)
    "Did the user spend more time walking or sitting?", # compare   (no SLM)
    "Is the user walking?",                            # verify     (SLM)
    "Did the user begin walking at any point, and if so, when?",   # onset (SLM)
]


def device_info() -> dict:
    info = {"os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "gpu_used": False}
    try:
        import psutil
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return info


def rf_model_size(path: Path) -> dict:
    """Disk footprint and a parameter count meaningful for a forest (total decision-tree nodes)."""
    import joblib
    bundle = joblib.load(path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    out = {"file": str(path), "disk_bytes": path.stat().st_size,
           "disk_mb": round(path.stat().st_size / 1e6, 2)}
    if hasattr(model, "estimators_"):
        nodes = sum(t.tree_.node_count for t in model.estimators_)
        out.update({"n_estimators": len(model.estimators_), "total_tree_nodes": nodes,
                    "parameter_proxy": nodes,
                    "parameter_note": "decision-tree nodes; a forest has no weights, so node count "
                                      "is the comparable capacity measure"})
    out["n_features_in"] = int(getattr(model, "n_features_in_", 0))
    return out


def ollama_model_size(model_name: str) -> dict:
    """Disk size and parameter count of the SLM, read from Ollama rather than assumed."""
    out = {"model": model_name}
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") == model_name:
                    d = m.get("details", {})
                    out.update({"disk_bytes": m.get("size"),
                                "disk_mb": round(m.get("size", 0) / 1e6, 1),
                                "parameters": d.get("parameter_size"),
                                "quantization": d.get("quantization_level"),
                                "context_length": d.get("context_length")})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def ollama_rss_mb() -> dict:
    """
    Resident memory of the whole Ollama process tree, in MB.

    Must include children: `ollama.exe` itself is only ~36 MB because the weights are actually
    held by a `llama-server` child process. Note also that the GGUF file is memory-mapped, so
    resident size sits below the on-disk size -- both figures are reported rather than conflated.
    """
    try:
        import psutil
    except ImportError:
        return {}
    seen: dict[int, tuple[str, int, int]] = {}
    for p in psutil.process_iter(["name", "pid"]):
        try:
            name = (p.info["name"] or "").lower()
            if "ollama" not in name and "llama" not in name:
                continue
            proc = psutil.Process(p.info["pid"])
            for q in [proc] + proc.children(recursive=True):
                n = q.name()
                if "conhost" in n.lower():
                    continue
                mi = q.memory_full_info()
                private = getattr(mi, "private", None) or getattr(mi, "pagefile", 0)
                seen[q.pid] = (n, mi.rss, private)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not seen:
        return {}
    return {"total_rss_mb": round(sum(r for _, r, _ in seen.values()) / 1e6, 1),
            "total_private_commit_mb": round(sum(pv for _, _, pv in seen.values()) / 1e6, 1),
            "processes": {n: {"rss_mb": round(r / 1e6, 1), "private_commit_mb": round(pv / 1e6, 1)}
                          for n, r, pv in seen.values()},
            "note": "Resident size (RSS) badly under-reports this workload on Windows: the GGUF "
                    "weights are memory-mapped, so pages are evicted and re-faulted and RSS reads "
                    "~39 MB while the process actually reserves ~2 GB. Private/commit is the "
                    "meaningful peak-memory figure; on-disk size is 986 MB."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model_rf.joblib")
    ap.add_argument("--recording", default=None,
                    help="defaults to the first held-out benchmark recording")
    ap.add_argument("--slm", default="qwen2.5:1.5b")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/cost_report.json")
    args = ap.parse_args()

    from src.aggregate.timeline import build_timeline, load_model
    from src.query.intent import parse_intent
    from src.query.operations import execute_query

    rec = Path(args.recording) if args.recording else sorted(
        p for p in Path("data/eval_benchmark").iterdir()
        if p.is_dir() and (p / "recording.csv").exists())[0] / "recording.csv"

    report = {"target_device": device_info(),
              "note": "CPU only; no GPU used at any stage of the deployed pipeline.",
              "recognizer": rf_model_size(Path(args.model)),
              "slm": ollama_model_size(args.slm),
              "recording_used": str(rec)}
    print(f"target: {report['target_device']}")
    print(f"recording: {rec}")

    # ---------------- one-time per recording: building the activity timeline
    tracemalloc.start()
    t0 = time.perf_counter()
    tl = build_timeline(rec, args.model)
    t_build = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_samples = sum(1 for _ in open(rec)) - 1
    report["timeline_build"] = {
        "seconds": round(t_build, 2),
        "peak_python_memory_mb": round(peak / 1e6, 1),
        "recording_samples": n_samples,
        "recording_span_sec": round(tl.recording_sec, 1),
        "intervals_produced": len(tl.intervals),
        "realtime_factor": round(tl.recording_sec / t_build, 1),
        "note": "one-time ingestion cost per recording, not paid per question",
    }
    print(f"timeline: {t_build:.2f}s, peak {peak/1e6:.1f} MB, "
          f"{tl.recording_sec/t_build:.0f}x realtime")

    # ---------------- per query, split by whether the SLM is invoked
    import src.query.explain as ex
    import src.query.operations as ops
    orig = ex.generate_explanation

    def timed(use_llm: bool) -> dict:
        patched = (orig if use_llm else
                   (lambda a, d, i, **k: orig(a, d, i, use_llm=False)))
        ex.generate_explanation, ops.generate_explanation = patched, patched
        per_q = {}
        for q in QUESTIONS:
            samples = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                intent = parse_intent(q)
                t1 = time.perf_counter()
                execute_query(intent, tl)
                t2 = time.perf_counter()
                samples.append((t1 - t0, t2 - t1, t2 - t0))
            per_q[q] = {"intent_ms": round(statistics.median(s[0] for s in samples) * 1e3, 2),
                        "query_ms": round(statistics.median(s[1] for s in samples) * 1e3, 2),
                        "total_ms": round(statistics.median(s[2] for s in samples) * 1e3, 2)}
        return per_q

    tracemalloc.start()
    no_llm = timed(use_llm=False)
    _, peak_q = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Sample the Ollama process tree WHILE generating: it unloads the model after an idle
    # keep-alive window, so measuring afterwards reports an empty runner (~38 MB) instead of the
    # loaded one (~400 MB).
    import threading
    peak_slm = {"total_rss_mb": 0.0, "total_private_commit_mb": 0.0, "processes": {}}
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            snap = ollama_rss_mb()
            if snap and snap.get("total_private_commit_mb", 0) > peak_slm.get("total_private_commit_mb", 0):
                peak_slm.update(snap)
            time.sleep(0.1)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    with_llm = timed(use_llm=True)
    stop.set()
    watcher.join(timeout=2)
    ex.generate_explanation, ops.generate_explanation = orig, orig

    slm_rss = peak_slm if peak_slm.get("total_private_commit_mb") else ollama_rss_mb()
    det_totals = [v["total_ms"] for v in no_llm.values()]
    llm_totals = [v["total_ms"] for k, v in with_llm.items()
                  if with_llm[k]["total_ms"] > no_llm[k]["total_ms"] * 2]   # SLM-invoking questions

    report["per_query"] = {
        "repeats": args.repeats,
        "deterministic_only": no_llm,
        "with_slm_explanation": with_llm,
        "median_ms_deterministic": round(statistics.median(det_totals), 2),
        "median_ms_with_slm": round(statistics.median(llm_totals), 1) if llm_totals else None,
        "peak_python_memory_mb_deterministic": round(peak_q / 1e6, 1),
        "slm_process_memory": slm_rss,
        "note": "measured against an already-built timeline; the SLM is invoked only for "
                "verification and onset explanations, so most question types cost the "
                "deterministic figure",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\nper-query (median of {args.repeats}):")
    for q in QUESTIONS:
        a, b = no_llm[q]["total_ms"], with_llm[q]["total_ms"]
        print(f"  {a:8.2f} ms | {b:9.1f} ms  {q[:52]}")
    print(f"\nrecognizer: {report['recognizer']['disk_mb']} MB on disk, "
          f"{report['recognizer'].get('total_tree_nodes', 0):,} tree nodes")
    print(f"SLM: {report['slm'].get('disk_mb')} MB on disk, "
          f"{report['slm'].get('parameters')} params, {report['slm'].get('quantization')}")
    print(f"SLM peak memory: {slm_rss.get('total_private_commit_mb')} MB private/commit "
          f"({slm_rss.get('total_rss_mb')} MB RSS -- see note)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
