# Recognition backbone: results

Six models were trained and evaluated on the identical official 5-fold user-level split, over the identical set of minutes: two tree ensembles (Random Forest, HistGradientBoosting) over 302 hand-engineered features, two raw-signal neural nets capacity-matched to each other (TinyCNN, CNN+GRU), a feedforward net over the same 302 engineered features as the tree models (FeatureMLP), and a final variant of that net given multi-minute temporal context (FeatureMLP+Context). All numbers below are measured, not estimated.

## Final comparison across all six models

| Model | Accuracy (all) | Macro-F1 | Balanced acc. | Accuracy (signal-consistent) | Params | Size | Latency |
|---|---|---|---|---|---|---|---|
| **FeatureMLP + Context (final pick)** | **0.492** | **0.431** | **0.439** | **0.526** | 189,447 | 0.77 MB | 2.62 ms (CPU) |
| Random Forest | 0.472 | 0.402 | 0.387 | 0.507 | — | 223 MB | not measured |
| FeatureMLP | 0.448 | 0.376 | 0.398 | 0.481 | 112,135 | 0.46 MB | 1.08 ms (CPU) |
| HistGradientBoosting | 0.438* | 0.411* | 0.438* | — | — | 7.96 MB | not measured |
| CNN+GRU | 0.393 | 0.308 | 0.330 | 0.425 | 31,607 | 0.134 MB | 0.655 ms (CUDA, T4) |
| TinyCNN | 0.389 | 0.308 | 0.331 | 0.422 | 31,271 | 0.135 MB | 63.4 ms (CPU) |

\* HistGradientBoosting was only evaluated on 2 of the 5 folds (a quick comparison run), not the full 5-fold split every other model uses -- not directly comparable to the other rows, shown for reference only.

**Final pick: FeatureMLP + Context.** It beats Random Forest on every pooled metric and on 4 of 5 individual folds (essentially tied on the 5th), while being roughly 290x smaller on disk. The two raw-signal nets (TinyCNN, CNN+GRU) consistently trailed Random Forest by ~8 accuracy points on every fold -- the representation-learning job from raw signal alone, at a small/edge-appropriate parameter budget, proved harder than starting from good hand-engineered features. Handing a neural net those same 302 features (FeatureMLP) closed most of that gap; adding multi-minute temporal context on top of that (FeatureMLP + Context) closed the rest and overtook Random Forest. The honest ceiling on this task, even with the best approach found, remains in the high-40s/low-50s percent, not higher -- traced in Table 6 below to measured label noise and physically ambiguous posture classes, not a model-capacity limitation five different architectures all hit the same wall on.

## Model progression (the neural-network side of the design story)

| Step | Approach | Accuracy | Macro-F1 | What changed |
|---|---|---|---|---|
| 1 | TinyCNN | 0.389 | 0.308 | Raw-signal 1D-CNN, learns its own features from scratch |
| 2 | CNN+GRU | 0.393 | 0.308 | Adds recurrence for temporal structure -- capacity-matched to TinyCNN; net accuracy unchanged, but per-class it helped periodicity-driven classes (bicycling, walking) while losing ground on ambiguous postures (sitting, standing) |
| 3 | FeatureMLP | 0.448 | 0.376 | Switches input from raw signal to Random Forest's own 302 engineered features -- removes the raw-signal representation-learning handicap entirely |
| 4 | **FeatureMLP + Context** | **0.492** | **0.431** | Concatenates each minute's own features with the mean of its temporal neighbours (up to 2 minutes before/after, only within real time-continuity) -- resolves single-minute ambiguity using surrounding context |

## Winning model (FeatureMLP + Context): per-fold results

| Fold | Minutes | Accuracy | Macro-F1 | Balanced acc. |
|---|---|---|---|---|
| Fold 0 | 20,394 | 0.491 | 0.398 | 0.467 |
| Fold 1 | 20,424 | 0.464 | 0.415 | 0.518 |
| Fold 2 | 20,008 | 0.543 | 0.480 | 0.475 |
| Fold 3 | 20,242 | 0.489 | 0.387 | 0.392 |
| Fold 4 | 14,541 | 0.463 | 0.428 | 0.460 |

## Winning model (FeatureMLP + Context): per-class performance (all test minutes)

| Class | Precision | Recall | F1 | Test minutes | Correctly predicted |
|---|---|---|---|---|---|
| Lying down | 0.506 | 0.726 | 0.596 | 20,855 | 15,141 |
| Sitting | 0.533 | 0.476 | 0.503 | 22,536 | 10,727 |
| Standing in place | 0.221 | 0.284 | 0.248 | 7,909 | 2,246 |
| Standing and moving | 0.318 | 0.215 | 0.257 | 17,073 | 3,671 |
| Walking | 0.670 | 0.557 | 0.608 | 21,417 | 11,929 |
| Running | 0.124 | 0.159 | 0.139 | 1,078 | 171 |
| Bicycling | 0.673 | 0.653 | 0.663 | 4,741 | 3,096 |

## Winning model (FeatureMLP + Context): per-class performance (signal-consistent minutes)

| Class | Precision | Recall | F1 | Test minutes | Correctly predicted |
|---|---|---|---|---|---|
| Lying down | 0.537 | 0.726 | 0.617 | 20,855 | 15,141 |
| Sitting | 0.568 | 0.476 | 0.518 | 22,536 | 10,727 |
| Standing in place | 0.271 | 0.284 | 0.277 | 7,909 | 2,246 |
| Standing and moving | 0.364 | 0.215 | 0.270 | 17,073 | 3,671 |
| Walking | 0.667 | 0.740 | 0.702 | 15,905 | 11,770 |
| Running | 0.127 | 0.259 | 0.171 | 653 | 169 |
| Bicycling | 0.676 | 0.758 | 0.715 | 4,019 | 3,046 |

---

## Appendix: Random Forest baseline deep-dive

Configuration: rf, level=minute, feature subset=noori (302 features), cleaned training=True, 5 folds, leave-users-out.

## Table 1: Overall accuracy

| Evaluation set | Minutes | Accuracy | Correct | Macro-F1 | Balanced acc. |
|---|---|---|---|---|---|
| All test minutes (headline) | 95,609 | 0.472 | 45,141 | 0.402 | 0.387 |
| Signal-consistent test minutes | 88,950 | 0.507 | 45,127 | 0.450 | 0.442 |

The second row excludes minutes labelled walking, running or bicycling whose accelerometer never exceeded 0.03 g of variation in any window; the gap between the two rows is the cost of self-reported labels.

## Table 2: Per-class performance (all test minutes)

| Class | Precision | Recall | F1 | Test minutes | Correctly predicted |
|---|---|---|---|---|---|
| Lying down | 0.536 | 0.550 | 0.543 | 20,855 | 11,470 |
| Sitting | 0.396 | 0.666 | 0.497 | 22,536 | 15,008 |
| Standing in place | 0.225 | 0.062 | 0.098 | 7,909 | 494 |
| Standing and moving | 0.265 | 0.143 | 0.186 | 17,073 | 2,446 |
| Walking | 0.597 | 0.603 | 0.600 | 21,417 | 12,916 |
| Running | 0.890 | 0.120 | 0.211 | 1,078 | 129 |
| Bicycling | 0.859 | 0.565 | 0.681 | 4,741 | 2,678 |

## Table 3: Per-class performance (signal-consistent minutes)

| Class | Precision | Recall | F1 | Test minutes | Correctly predicted |
|---|---|---|---|---|---|
| Lying down | 0.576 | 0.550 | 0.563 | 20,855 | 11,470 |
| Sitting | 0.440 | 0.666 | 0.530 | 22,536 | 15,008 |
| Standing in place | 0.304 | 0.062 | 0.104 | 7,909 | 494 |
| Standing and moving | 0.291 | 0.143 | 0.192 | 17,073 | 2,446 |
| Walking | 0.597 | 0.811 | 0.688 | 15,905 | 12,902 |
| Running | 0.890 | 0.198 | 0.323 | 653 | 129 |
| Bicycling | 0.859 | 0.666 | 0.751 | 4,019 | 2,678 |

## Table 4: Per-fold results

| Fold | Minutes | Accuracy | Correct | Macro-F1 | Balanced acc. |
|---|---|---|---|---|---|
| Fold 0 | 20,394 | 0.492 | 10,036 | 0.408 | 0.415 |
| Fold 1 | 20,424 | 0.458 | 9,352 | 0.454 | 0.435 |
| Fold 2 | 20,008 | 0.515 | 10,311 | 0.447 | 0.424 |
| Fold 3 | 20,242 | 0.442 | 8,940 | 0.332 | 0.331 |
| Fold 4 | 14,541 | 0.447 | 6,502 | 0.390 | 0.415 |

## Table 5: Design progression

| Variant | Accuracy | Macro-F1 | Balanced acc. | Running prec. | Walking F1 |
|---|---|---|---|---|---|
| Per-window, all 175 features (baseline) | 0.435 | 0.363 | 0.358 | 0.938 | 0.662 |
| Minute aggregation, no-orientation features | 0.477 | 0.400 | 0.390 | 0.962 | 0.642 |
| + cleaned training set (chosen) | 0.472 | 0.402 | 0.387 | 0.923 | 0.642 |

## Table 6: Label noise measured against the signal

| Labelled as | Minutes | Phone never moved | Share | Median cadence (Hz) |
|---|---|---|---|---|
| Lying down | 20,855 | 19,782 | 94.9% | 4.4 |
| Sitting | 22,536 | 17,199 | 76.3% | 4.4 |
| Standing in place | 7,909 | 5,075 | 64.2% | 4.0 |
| Standing and moving | 17,073 | 10,572 | 61.9% | 4.0 |
| Walking | 21,417 | 5,512 | 25.7% | 2.2 |
| Running | 1,078 | 425 | 39.4% | 2.8 |
| Bicycling | 4,741 | 722 | 15.2% | 4.0 |
| **All active classes** | **27,236** | **6,659** | **24.4%** |  |

'Phone never moved' means the |Acc| standard deviation stayed below 0.03 g in every 5 s window of the minute's 20 s burst.


# LaTeX versions

\begin{table}[t]
\centering
\begin{tabular}{lrrrrr}
\hline
Evaluation set & Minutes & Accuracy & Correct & Macro-F1 & Bal. acc. \\
\hline
All test minutes (headline) & 95,609 & 0.472 & 45,141 & 0.402 & 0.387 \\
Signal-consistent test minutes & 88,950 & 0.507 & 45,127 & 0.450 & 0.442 \\
\hline
\end{tabular}
\caption{Overall recognition accuracy, leave-users-out.}
\label{tab:overall}
\end{table}

\begin{table}[t]
\centering
\begin{tabular}{lrrrrr}
\hline
Class & Precision & Recall & F1 & Minutes & Correct \\
\hline
Lying down & 0.536 & 0.550 & 0.543 & 20,855 & 11,470 \\
Sitting & 0.396 & 0.666 & 0.497 & 22,536 & 15,008 \\
Standing in place & 0.225 & 0.062 & 0.098 & 7,909 & 494 \\
Standing and moving & 0.265 & 0.143 & 0.186 & 17,073 & 2,446 \\
Walking & 0.597 & 0.603 & 0.600 & 21,417 & 12,916 \\
Running & 0.890 & 0.120 & 0.211 & 1,078 & 129 \\
Bicycling & 0.859 & 0.565 & 0.681 & 4,741 & 2,678 \\
\hline
\end{tabular}
\caption{Per-class performance over all test minutes.}
\label{tab:perclass}
\end{table}

\begin{table}[t]
\centering
\begin{tabular}{lrrrrr}
\hline
Variant & Acc. & Macro-F1 & Bal. acc. & Run prec. & Walk F1 \\
\hline
Per-window, all 175 features (baseline) & 0.435 & 0.363 & 0.358 & 0.938 & 0.662 \\
Minute aggregation, no-orientation features & 0.477 & 0.400 & 0.390 & 0.962 & 0.642 \\
+ cleaned training set (chosen) & 0.472 & 0.402 & 0.387 & 0.923 & 0.642 \\
\hline
\end{tabular}
\caption{Effect of each design decision.}
\label{tab:ablation}
\end{table}

\begin{table}[t]
\centering
\begin{tabular}{lrrrr}
\hline
Labelled as & Minutes & Still & Share & Cadence (Hz) \\
\hline
Lying down & 20,855 & 19,782 & 94.9% & 4.4 \\
Sitting & 22,536 & 17,199 & 76.3% & 4.4 \\
Standing in place & 7,909 & 5,075 & 64.2% & 4.0 \\
Standing and moving & 17,073 & 10,572 & 61.9% & 4.0 \\
Walking & 21,417 & 5,512 & 25.7% & 2.2 \\
Running & 1,078 & 425 & 39.4% & 2.8 \\
Bicycling & 4,741 & 722 & 15.2% & 4.0 \\
**All active classes** & **27,236** & **6,659** & **24.4%** &  \\
\hline
\end{tabular}
\caption{Label noise measured against the recorded signal.}
\label{tab:noise}
\end{table}

\begin{table}[t]
\centering
\begin{tabular}{lrrrrrr}
\hline
Model & Acc. (all) & Macro-F1 & Bal. acc. & Acc. (consistent) & Params & Size \\
\hline
FeatureMLP + Context (final) & 0.492 & 0.431 & 0.439 & 0.526 & 189,447 & 0.77 MB \\
Random Forest & 0.472 & 0.402 & 0.387 & 0.507 & -- & 223 MB \\
FeatureMLP & 0.448 & 0.376 & 0.398 & 0.481 & 112,135 & 0.46 MB \\
HistGradientBoosting* & 0.438 & 0.411 & 0.438 & -- & -- & 7.96 MB \\
CNN+GRU & 0.393 & 0.308 & 0.330 & 0.425 & 31,607 & 0.134 MB \\
TinyCNN & 0.389 & 0.308 & 0.331 & 0.422 & 31,271 & 0.135 MB \\
\hline
\end{tabular}
\caption{Final comparison across all six recognition models, identical 5-fold split. *HistGradientBoosting evaluated on 2 of 5 folds only, not directly comparable.}
\label{tab:final-comparison}
\end{table}

\begin{table}[t]
\centering
\begin{tabular}{lrrrr}
\hline
Fold & Minutes & Accuracy & Macro-F1 & Bal. acc. \\
\hline
Fold 0 & 20,394 & 0.491 & 0.398 & 0.467 \\
Fold 1 & 20,424 & 0.464 & 0.415 & 0.518 \\
Fold 2 & 20,008 & 0.543 & 0.480 & 0.475 \\
Fold 3 & 20,242 & 0.489 & 0.387 & 0.392 \\
Fold 4 & 14,541 & 0.463 & 0.428 & 0.460 \\
\hline
\end{tabular}
\caption{FeatureMLP + Context: per-fold results.}
\label{tab:mlpctx-folds}
\end{table}

\begin{table}[t]
\centering
\begin{tabular}{lrrrrr}
\hline
Class & Precision & Recall & F1 & Minutes & Correct \\
\hline
Lying down & 0.506 & 0.726 & 0.596 & 20,855 & 15,141 \\
Sitting & 0.533 & 0.476 & 0.503 & 22,536 & 10,727 \\
Standing in place & 0.221 & 0.284 & 0.248 & 7,909 & 2,246 \\
Standing and moving & 0.318 & 0.215 & 0.257 & 17,073 & 3,671 \\
Walking & 0.670 & 0.557 & 0.608 & 21,417 & 11,929 \\
Running & 0.124 & 0.159 & 0.139 & 1,078 & 171 \\
Bicycling & 0.673 & 0.653 & 0.663 & 4,741 & 3,096 \\
\hline
\end{tabular}
\caption{FeatureMLP + Context: per-class performance, all test minutes.}
\label{tab:mlpctx-perclass}
\end{table}
