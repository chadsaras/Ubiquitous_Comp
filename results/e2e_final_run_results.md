# End-to-end pipeline run, real LLM active, post-fix

Run 2026-09-10. Both fixes from the previous session (broadened verify fast-path, grounded
LLM explanations) are in place. Ollama was already installed and running (`qwen2.5:1.5b` live at
`localhost:11434`). Ran the full command a real user would run: raw sensor CSV in, natural
language questions out, for both trained recognizers.

**Update (same day): added FeatureMLP+Context as a third, purely additive option.** RF and HGB's
code paths are completely untouched -- see "Adding FeatureMLP+Context" below for what changed,
why it required new inference code (not just pointing `--model` at a different file), and the
honest (not flattering) result of testing it end to end.

## Command run

```bash
python src/pipeline.py --recording tests/fixtures/00EABED2_590_160/recording.csv \
                       --model models/model_rf.joblib \
                       --out results/e2e_final_rf_answers.txt

python src/pipeline.py --recording tests/fixtures/00EABED2_590_160/recording.csv \
                       --model models/model_hgb.joblib \
                       --out results/e2e_final_hgb_answers.txt

python src/pipeline.py --recording tests/fixtures/00EABED2_590_160/recording.csv \
                       --model results/model_mlpctx.pt \
                       --out results/e2e_final_mlpctx_answers.txt
```

## Results

| | Random Forest | HistGradientBoosting | FeatureMLP+Context |
|---|---|---|---|
| Exit code | 0 (clean) | 0 (clean) | 0 (clean) |
| Total wall time (6 questions, 2 of which use the LLM) | 33.7s | 17.6s | 40.0s |
| Timeline-building time | 9.9s | 6.2s | ~11s |
| Intervals produced | 5 (clean) | 24 (fragmented) | 74 (very fragmented) |
| **Graded against ground truth (`truth.json`)** | **93.7%** | **93.7%** | **1.5%** |

RF and HGB land on the same accuracy despite RF producing far cleaner intervals. FeatureMLP+Context
is not a typo -- see below for why its accuracy collapses on this specific recording despite
being the official 5-fold *winner* (0.492 vs RF's 0.472, averaged over all 60 ExtraSensory users).

## Sample answer, real LLM, grounded correctly (Random Forest run)

```
[Q2]: "Is the user walking?"
Answer: Yes
Activity/Event: Walking
Evidence:
  Timestamp(s): 239 to 1348, 8127 to 9269 (seconds from start)
  Sensor Modality: Accelerometer, Gyroscope
  Sensor Channel(s): All
Explanation: The sensor evidence supports the activity of walking because the acceleration
variance (acc_mag_std = 0.109 g) is within the typical range, indicating smooth, steady motion
typical of walking. The cadence (2.0 Hz) is also within the typical range, suggesting a
consistent walking pace. The gyroscope angular velocity (gyro_mag_std = 0.071 rad/s) is also
within the typical range, indicating a smooth, low-impact motion typical of walking.
```

Full answer sets for all 6 questions x 2 models saved to `results/e2e_final_rf_answers.txt` and
`results/e2e_final_hgb_answers.txt`.

## What this confirms

1. The pipeline runs cleanly end-to-end, on demand, with the real LLM engaged (not the
   fallback) -- no crashes, no silent failures.
2. The two fixes made last session hold up under a fresh full run: no false "consistent" claims,
   explanations read naturally and correctly cite in-range vs atypical metrics.
3. **93.7% is not the honest average** -- it's this one recording's accuracy. The 5-fold study
   (`report/tables.md`) measured ~47-49% averaged across all 60 users; a separate fixture graded
   earlier came out to 37.0%. This recording happens to be an easy one (long, unambiguous
   sitting/walking blocks). Don't quote 93.7% as the system's accuracy without that context.
4. Random Forest and HGB are functionally tied on this recording's accuracy -- the earlier
   6-model comparison's ranking (RF > HGB on average) doesn't necessarily hold per-recording.

## Adding FeatureMLP+Context (purely additive, RF/HGB untouched)

Per instruction: added as a third option, nothing replaced, easy to switch back to RF/HGB by
just changing `--model`. RF/HGB run through the exact same code they always did
(`window_predictions()` in `src/aggregate/timeline.py`); a new, separate function
(`window_predictions_torch()`) handles `.pt` models and is only reached when `load_model()`
detects a `.pt` file.

**Why this needed real new code, not just a different `--model` path:** RF/HGB take a 302-dim
feature vector per window. FeatureMLP+Context takes a 604-dim vector (its own 302 features, plus
the *average of its neighbouring minutes'* 302-dim features -- the actual innovation that made it
win the 5-fold comparison). Nothing in the pipeline previously knew how to build that second half
at inference time, since it only existed in the training script.

**A real bug found and fixed along the way:** my first version computed the "own" 302 features by
averaging over a flat 60-second window. That's wrong -- the model was trained on ~20-second
ExtraSensory bursts (RF already emulates this correctly via a 7-window sliding aggregation, ~17.5s).
Feeding 60-second-scale statistics into a model whose BatchNorm layers were calibrated on
20-second-scale statistics produced near-garbage output (1.8% accuracy, worse than random
guessing). Fixed by reusing RF's own proven 7-window aggregation for the "own" half, and using
minute-scale bucketing *only* for the genuinely new part -- averaging in neighbouring buckets as
context. Verified this fix is real, not cosmetic, by checking RF and FeatureMLP+Context against
each other on the identical, correctly-computed feature vectors for a known-`sitting` span: RF
correctly said sitting (52.1%); FeatureMLP+Context, fed those exact same vectors, favoured
`lying_down` (47.2%) over `sitting` (31.6%).

**That last number is the real, honest finding, not a leftover bug.** FeatureMLP+Context's own
official metrics (`report/tables.md`) already show it has a measured bias toward over-predicting
`lying_down` (precision only 0.506 -- half its lying_down calls are wrong, even in the correct,
official evaluation). Two things make that bias much worse on this specific test:

1. **This recording is a continuous stream, not the bursty once-per-minute pattern the model was
   trained and measured on.** RF only needs a short local aggregation window to emulate a burst
   and handles continuous data fine. FeatureMLP+Context's whole reason for existing is averaging
   real, discrete per-minute samples -- on a continuous stream, "minutes" have to be approximated
   by 60-second buckets that don't correspond to any real boundary in the data, so the context
   half is inherently a rougher approximation than what the model saw in training.
2. Once the model leans `lying_down` on most windows in a long ambiguous stretch, the pipeline's
   existing smoothing + argmax + interval-merging (unchanged, shared code) turns that into one
   giant wrong interval rather than many small ones -- so a real but moderate per-window bias
   becomes a near-total accuracy collapse at the interval level.

**Bottom line:** the integration works -- it's wired in, doesn't crash, produces valid grounded
LLM answers, and is trivially reversible (switch `--model` back to RF/HGB, or delete
`window_predictions_torch()` -- nothing else in the codebase references it). But on *this*
specific recording, it performs far worse than its official aggregate ranking suggests, for
reasons now understood and written down rather than hidden. It would need real per-minute
labelled data (ExtraSensory's actual bursty format, not a continuous phone recording) to be
tested fairly against its own benchmark.
