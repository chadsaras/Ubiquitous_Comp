#!/usr/bin/env python3
"""
Turn the saved metrics into report-ready tables, with counts next to every percentage.

    python scripts/report_tables.py                    # print markdown to stdout
    python scripts/report_tables.py --latex            # LaTeX tabulars as well
    python scripts/report_tables.py --out report/tables.md

Reads:
    results/classifier_metrics.json   written by scripts/train_classifier.py
    results/variants/*.log            the design-progression variants (optional)
    data/processed/features.npz       for the label-noise measurement (optional)
"""
import argparse
import json
import re
import sys
from pathlib import Path

PRETTY = {"lying_down": "Lying down", "sitting": "Sitting", "standing_in_place": "Standing in place",
          "standing_and_moving": "Standing and moving", "walking": "Walking", "running": "Running",
          "bicycling": "Bicycling"}
ACTIVE = ["walking", "running", "bicycling"]
STILL_G = 0.03


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(header, rows, caption, label):
    cols = "l" + "r" * (len(header) - 1)
    out = ["\\begin{table}[t]", "\\centering", f"\\begin{{tabular}}{{{cols}}}", "\\hline",
           " & ".join(header) + " \\\\", "\\hline"]
    out += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out += ["\\hline", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            "\\end{table}"]
    return "\n".join(out)


def overall_rows(m):
    rows = []
    for key, name in [("pooled_all", "All test minutes (headline)"),
                      ("pooled_signal_consistent", "Signal-consistent test minutes")]:
        if key not in m:
            continue
        p = m[key]
        n = p["n"]
        correct = round(p["accuracy"] * n)
        rows.append([name, f"{n:,}", f"{p['accuracy']:.3f}", f"{correct:,}",
                     f"{p['macro_f1']:.3f}", f"{p['balanced_accuracy']:.3f}"])
    return rows


def per_class_rows(p):
    rows = []
    for cls, s in p["per_class"].items():
        rows.append([PRETTY.get(cls, cls), f"{s['precision']:.3f}", f"{s['recall']:.3f}",
                     f"{s['f1']:.3f}", f"{s['support']:,}", f"{round(s['recall']*s['support']):,}"])
    return rows


def fold_rows(m):
    rows = []
    for k, f in sorted(m.get("folds", {}).items()):
        a = f["all"]
        rows.append([f"Fold {k}", f"{a['n']:,}", f"{a['accuracy']:.3f}",
                     f"{round(a['accuracy']*a['n']):,}", f"{a['macro_f1']:.3f}",
                     f"{a['balanced_accuracy']:.3f}"])
    return rows


def variant_rows(vdir: Path):
    """Parse the pooled lines out of results/variants/*.log."""
    names = {"rf_window_all_baseline": "Per-window, all 175 features (baseline)",
             "rf_minute_noori_noclean": "Minute aggregation, no-orientation features",
             "rf_minute_noori_clean": "+ cleaned training set (chosen)"}
    order = ["rf_window_all_baseline", "rf_minute_noori_noclean", "rf_minute_noori_clean"]
    rows = []
    for stem in order:
        p = vdir / f"{stem}.log"
        if not p.exists():
            continue
        txt = p.read_text()
        m = re.search(r"POOLED, all test minutes.*?acc=([\d.]+)\s+macroF1=([\d.]+)\s+balAcc=([\d.]+)", txt)
        if not m:
            continue
        run = re.search(r"^  running\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt, re.M)
        walk = re.search(r"^  walking\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt, re.M)
        rows.append([names[stem], m.group(1), m.group(2), m.group(3),
                     run.group(1) if run else "-", walk.group(3) if walk else "-"])
    return rows


def noise_rows(features: Path):
    try:
        import numpy as np, pandas as pd
    except ImportError:
        return []
    if not features.exists():
        return []
    d = np.load(features, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    std = d["X"][:, names.index("acc_mag__std")]
    key = pd.Series(d["uuid"].astype(str)) + "\t" + pd.Series(d["ts"]).astype(str)
    df = pd.DataFrame({"k": key, "y": d["y"], "std": std, "dom": d["X"][:, names.index("acc_mag__dom_freq")]})
    g = df.groupby("k").agg(y=("y", "first"), std_max=("std", "max"), dom=("dom", "median"))
    classes = list(PRETTY)
    rows, tot_n, tot_still = [], 0, 0
    for i, cls in enumerate(classes):
        sub = g[g.y == i]
        if not len(sub):
            continue
        still = int((sub.std_max < STILL_G).sum())
        rows.append([PRETTY[cls], f"{len(sub):,}", f"{still:,}", f"{still/len(sub):.1%}",
                     f"{sub.dom.median():.1f}"])
        if cls in ACTIVE:
            tot_n += len(sub); tot_still += still
    if tot_n:
        rows.append(["**All active classes**", f"**{tot_n:,}**", f"**{tot_still:,}**",
                     f"**{tot_still/tot_n:.1%}**", ""])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/classifier_metrics.json")
    ap.add_argument("--variants", default="results/variants")
    ap.add_argument("--features", default="data/processed/features.npz")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mp = Path(args.metrics)
    if not mp.exists():
        sys.exit(f"{mp} not found - run scripts/train_classifier.py first")
    m = json.loads(mp.read_text())
    cfg = m.get("config", {})

    sec = []
    sec.append("# Recognition backbone: results\n")
    sec.append(f"Configuration: {cfg.get('model','rf')}, level={cfg.get('level','minute')}, "
               f"feature subset={cfg.get('subset','noori')} ({cfg.get('n_features','?')} features), "
               f"cleaned training={cfg.get('clean', True)}, "
               f"{len(m.get('folds', {}))} folds, leave-users-out.\n")

    sec.append("## Table 1: Overall accuracy\n")
    sec.append(md_table(["Evaluation set", "Minutes", "Accuracy", "Correct", "Macro-F1", "Balanced acc."],
                        overall_rows(m)))
    sec.append("\nThe second row excludes minutes labelled walking, running or bicycling whose "
               f"accelerometer never exceeded {STILL_G} g of variation in any window; the gap between "
               "the two rows is the cost of self-reported labels.\n")

    sec.append("## Table 2: Per-class performance (all test minutes)\n")
    sec.append(md_table(["Class", "Precision", "Recall", "F1", "Test minutes", "Correctly predicted"],
                        per_class_rows(m["pooled_all"])))

    if "pooled_signal_consistent" in m:
        sec.append("\n## Table 3: Per-class performance (signal-consistent minutes)\n")
        sec.append(md_table(["Class", "Precision", "Recall", "F1", "Test minutes", "Correctly predicted"],
                            per_class_rows(m["pooled_signal_consistent"])))

    fr = fold_rows(m)
    if fr:
        sec.append("\n## Table 4: Per-fold results\n")
        sec.append(md_table(["Fold", "Minutes", "Accuracy", "Correct", "Macro-F1", "Balanced acc."], fr))

    vr = variant_rows(Path(args.variants))
    if vr:
        sec.append("\n## Table 5: Design progression\n")
        sec.append(md_table(["Variant", "Accuracy", "Macro-F1", "Balanced acc.", "Running prec.", "Walking F1"], vr))

    nr = noise_rows(Path(args.features))
    if nr:
        sec.append("\n## Table 6: Label noise measured against the signal\n")
        sec.append(md_table(["Labelled as", "Minutes", "Phone never moved", "Share", "Median cadence (Hz)"], nr))
        sec.append(f"\n'Phone never moved' means the |Acc| standard deviation stayed below {STILL_G} g "
                   "in every 5 s window of the minute's 20 s burst.\n")

    text = "\n".join(sec)

    if args.latex:
        text += "\n\n# LaTeX versions\n\n"
        text += tex_table(["Evaluation set", "Minutes", "Accuracy", "Correct", "Macro-F1", "Bal. acc."],
                          overall_rows(m), "Overall recognition accuracy, leave-users-out.", "tab:overall") + "\n\n"
        text += tex_table(["Class", "Precision", "Recall", "F1", "Minutes", "Correct"],
                          per_class_rows(m["pooled_all"]),
                          "Per-class performance over all test minutes.", "tab:perclass")
        if vr:
            text += "\n\n" + tex_table(["Variant", "Acc.", "Macro-F1", "Bal. acc.", "Run prec.", "Walk F1"],
                                       vr, "Effect of each design decision.", "tab:ablation")
        if nr:
            text += "\n\n" + tex_table(["Labelled as", "Minutes", "Still", "Share", "Cadence (Hz)"],
                                       nr, "Label noise measured against the recorded signal.", "tab:noise")

    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\n[written to {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()