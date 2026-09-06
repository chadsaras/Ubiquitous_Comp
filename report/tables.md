# Recognition backbone: results

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
