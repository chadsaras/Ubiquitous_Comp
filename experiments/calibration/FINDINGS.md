# Experiment: class-prior calibration for the recognition backbone

**Verdict: does not work end to end. Not adopted. Kept as a documented negative result.**

Run 2026-09-11 on the institute server (stage 1) and locally (stage 2). Nothing in `src/` or
`scripts/` was modified — the calibration was applied by wrapping `window_predictions` at runtime.

---

## The problem it was meant to solve

On the held-out QA benchmark, predicted activity time collapses toward `sitting`: 55% of true
walking time, 55% of standing-and-moving, 67% of standing-in-place and 51% of lying-down are all
predicted as sitting. Duration answers inherit this, which is why duration accuracy is 0.167 while
verification is 0.944.

The forest outputs probabilities and the pipeline takes a plain argmax — the decision rule that
maximises raw accuracy on an imbalanced dataset, i.e. one that is *supposed* to over-predict the
majority class. Re-weighting before the argmax should trade a little accuracy for much better
balance.

## Method, and why the result can be trusted

- fold 0's TRAIN users split **by user** into 34 sub-train / 11 validation;
- Random Forest (300 trees) fitted on sub-train only;
- every calibration parameter chosen on **validation only**;
- fold 0's 12 TEST users scored once at the end and never used to choose anything.

Two families: single-parameter prior correction `p_c / prior_c**alpha`, and seven-parameter
per-class weights by coordinate ascent. Each tuned against two different objectives — macro-F1,
and a "time error" measure (`sum_c |predicted minutes of c - true minutes of c| / total`) chosen
because it is what duration questions actually depend on.

## Stage 1 — minute level: the proxy improved a lot

| Candidate | Accuracy | Macro-F1 | Balanced acc. | **Time error** |
|---|---|---|---|---|
| Baseline (plain argmax) | 0.492 | 0.429 | 0.413 | 0.435 |
| Prior correction, alpha = 0.4 | 0.481 | **0.432** | **0.450** | 0.349 |
| Coordinate ascent (min time error) | 0.471 | 0.397 | 0.443 | **0.300** |

Time error fell by up to **31%**, and prior-correction alpha=0.4 was *better than baseline on
macro-F1 and balanced accuracy simultaneously*. Per-class recalls moved in exactly the intended
direction — running 0.195 -> 0.439, bicycling 0.594 -> 0.646, standing_in_place 0.039 -> 0.092,
and over-predicted sitting minutes fell from 7,935 to 6,483 (true 4,792).

On this evidence alone, adopting calibration looks obviously correct.

## Stage 2 — end to end: the improvement does not survive

Running the actual QA harness over all 12 held-out recordings (216 questions):

| Candidate | Macro QA | Ident. | **Duration** | Count | Comp. | Ground | Open | Grounded&correct | Duration MAE |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | **0.633** | 0.500 | 0.167 | **0.361** | 0.667 | **0.917** | **0.875** | **0.229** | 1334.5 |
| prior alpha=0.4 (max F1) | 0.629 | 0.583 | **0.194** | 0.306 | 0.750 | 0.861 | 0.792 | 0.229 | 1316.7 |
| prior alpha=0.45 (min time err) | 0.621 | 0.583 | 0.111 | 0.333 | 0.750 | 0.861 | 0.792 | 0.223 | 1302.6 |
| coord ascent (max F1) | **0.642** | 0.583 | 0.139 | 0.361 | **0.833** | 0.889 | 0.771 | 0.210 | **1213.7** |
| coord ascent (min time err) | 0.614 | 0.583 | 0.111 | 0.306 | 0.750 | 0.861 | 0.771 | 0.223 | 1281.8 |

**No candidate dominates the baseline.** Every one trades something away:
- the best duration *accuracy* (0.194, prior alpha=0.4) costs count (-0.055), open-world (-0.083)
  and grounding (-0.056), and leaves overall macro slightly worse;
- the best overall macro (0.642, coordinate ascent) has *worse* duration accuracy than baseline
  and the worst grounded-and-correct of any candidate;
- the candidate that won stage 1 outright (min time error, 0.300) produced the **worst** duration
  accuracy end to end (0.111).

## Why the proxy misled

1. **Duration is scored with a tolerance**, `max(5 s, 10%)`. Calibration shifts time between
   classes globally, which pulls aggregate totals closer without necessarily pulling any individual
   recording inside its 10% band. Aggregate time error and per-question duration accuracy are
   simply different objectives.
2. **Predicting more rare-class minutes creates more episodes**, so count accuracy falls. Duration
   and count pull in opposite directions.
3. **More fragmented timelines cite worse intervals**, so grounding and grounded-and-correct fall
   even when the answer is right.

## Caveat on significance — the differences are small

With 36 duration questions, **one question is worth 0.028**. The headline duration gain of +0.027
is therefore literally a single question changing. Macro differences of 0.01 are ~2 questions out
of 216. None of these gaps is large enough to call a real improvement on this sample size, which
reinforces rather than weakens the verdict: there is no reliable win here.

## Conclusion

Not adopted. The baseline plain argmax stays.

The useful finding for the report is the *mechanism*, not the score: the sitting-collapse is not a
decision-threshold artefact that re-weighting can fix. Re-weighting successfully moves predictions
toward the rare classes at the minute level — the stage-1 numbers prove the knob works — but the
minutes it moves are not reliably the *right* minutes, so aggregated answers do not improve. That
points at genuine class confusability in the features (sedentary postures being near-indistinguishable
from a phone accelerometer), not at a miscalibrated classifier, and it is consistent with the
~47-49% ceiling that six different models independently hit.

## Reproducing

```bash
# stage 1 (needs data/processed/features.npz; ~70 s on the server)
python experiments/calibration/calibrate_classes.py

# stage 2 (needs data/eval_benchmark + results/model_rf_fold0.joblib; ~4 min)
python experiments/calibration/apply_and_evaluate.py
```
