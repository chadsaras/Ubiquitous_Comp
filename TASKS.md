# TASKS.md — remaining work to complete CS60055 Challenge 1

Working checklist, in dependency order. Derived from the assignment brief
(`CS60055_Challenge1.pdf`) rather than from memory, so the requirement column below quotes what
the brief actually asks for. Extra credit (Figure 4 / compressed operating point) is deliberately
out of scope. Report *writing* is deferred, but every task records its outcome here so the report
can be written from this file later.

Legend: **[ ]** not started · **[~]** partly done · **[x]** done

---

## What the brief actually requires (checked against the PDF, not assumed)

| Requirement | Where in brief | Status |
|---|---|---|
| 7 classes, acc+gyro only, resampled to 25 Hz | "The Data" | **[x]** done |
| Exact 6-field output format, consistent time base, stated | "The Output Format" | **[x]** done (seconds from start, stated in README) |
| Task 1 — identification + binary verification | "Task 1" | **[x]** implemented |
| Task 2 — duration, count, onset, comparison; evidence expected | "Task 2" | **[x]** implemented |
| Task 3 — evidence grounding, all fields required and scored | "Task 3" | **[x]** implemented + measured (grounded-and-correct 0.210) |
| Task 4 — open-world; evidence + explanation **mandatory** | "Task 4" | **[x]** N/A-evidence gap fixed; accuracy 0.833 |
| Resource cost: params, disk size, **peak memory**, per-query latency, on a **stated target device** | "Accuracy and the Cost of Running It" — *mandatory, not extra credit* | **[x]** measured on stated laptop target |
| Figure 1 — accuracy by question type | "Required Figures" | **[x]** `results/fig1_accuracy_by_question_type.png` |
| Figure 2 — confusion matrix + per-class P/R/F1 | "Required Figures" | **[x]** `results/confusion_matrix*.png` |
| Figure 3 — accuracy vs strictness (tolerance / IoU 0.1–0.9) | "Required Figures" | **[x]** `results/fig3_accuracy_vs_strictness.png` |
| Figure 4 — accuracy vs overhead (Pareto) | "Required Figures" | **out of scope** (extra credit) |
| Figure 5 — robustness curve | "Required Figures" | **[x]** `results/fig5_robustness.png` |
| Overall QA accuracy, **macro-averaged across question types** | "Accuracy Reporting" | **[x]** **0.623** |
| Verification scored with precision/recall/F1 **+ specificity** | "Accuracy Reporting" | **[x]** P 1.00 / R 0.92 / spec 1.00 |
| Numeric scored by tolerance + **MAE** (optionally MAPE) | "Accuracy Reporting" | **[x]** duration MAE 1551.8 s, count MAE 4.7 |
| Temporal scored by **IoU**; multi-interval via overlap-matched average or temporal P/R → F1 | "Accuracy Reporting" | **[x]** temporal P/R→F1 form |
| Combined "grounded **and** correct" = answer correct **and** IoU ≥ τ **and** modality+channels match | "Accuracy Reporting" | **[x]** **0.210** |
| Grounding precision (cited interval actually contains the named activity) | "Accuracy Reporting" | **[x]** 0.379 |
| Open-world explanation faithfulness, 1–5 rubric, mean score reported | "Accuracy Reporting" | **[x]** mean 3.37; numeric fidelity **1.000** |
| Script to run the system on a **fresh recording** in the required format | "Deliverables" / "Groups and the GitHub Repository" | **[x]** verified + `sample_questions.txt` |
| README explaining setup + how to reproduce results | "Groups and the GitHub Repository" | **[~]** exists, has TODOs |
| Raw dataset **not** committed | same | **[x]** gitignored |
| TA access: `sandipc-iitkgp`, `sayantan-kuila`, `debjit2001` | same | **[ ]** user action |
| Report ~10–12 pages, contribution statement, AI-use disclosure | "Deliverables" | **deferred** (record kept here) |

---

## Ordered task list

### Task 1 — Lock in the final recognizer choice and document it
**Why first:** everything downstream (eval harness, efficiency numbers, robustness curve) must
measure *one* fixed system. Changing the model later invalidates every number.

Decision reached: **Random Forest is the default recognizer** for the live pipeline.
FeatureMLP+Context won the offline 5-fold benchmark (0.492 vs 0.472) but underperforms badly on
continuous (non-bursty) recordings — its multi-minute context mechanism assumes ExtraSensory's
discrete one-burst-per-minute structure. Both remain selectable via `--model`.

**Outcome: [x] done.** `PLAN.md` Step 12 rewritten to record the decision, the evidence behind it
(RF vs FeatureMLP+Context on identical feature vectors for a known-`sitting` span: 0.521 vs 0.472
favouring lying_down), and the BatchNorm feature-scale integration bug found and fixed during the
work. `PLAN.md`'s "Current status" section also corrected — it claimed the pipeline still needed
re-pointing, and did not mention that the main test fixture is synthetic.

**Also decided this session (user):** benchmark = real ExtraSensory only; cost-reporting target
device = this Windows laptop, CPU-only (it is where the complete RF + Qwen 2.5 system actually
runs end to end).

---

### Task 2 — Close the Task-4 evidence gap (correctness bug, blocks grounding score)
The brief states evidence and explanation are **mandatory** at Tier 4. `handle_open_world()` in
`src/query/operations.py` currently falls through to a generic answer with
`Timestamp(s): N/A, Sensor Modality: N/A, Sensor Channel(s): N/A` whenever no pattern matches.
Under the brief's combined grounding rule, an N/A evidence field scores zero even if the verdict
("Likely no") is right. Needs to cite the interval it actually inspected to reach that verdict.

**Outcome: [x] done.** Three real defects fixed, all verified against the fixture timeline:

1. **Tier-4 evidence was N/A on every negative verdict.** `handle_open_world()` fell through to a
   generic block with `Timestamp(s): N/A`. Under the brief's combined grounding rule that scores
   zero even when the verdict is right. Rewritten so *every* branch — positive and negative —
   cites real evidence: the strongest candidate interval actually found (which is *why* the answer
   is "no"), or the full span examined when no candidate exists. Positive verdicts now also quote
   the measured `acc_mag_std` / `gyro_mag_std` from the cited interval rather than generic prose.
2. **"Prolonged rest" was silently narrowed to lying down.** The synonym table maps
   `"resting" -> lying_down`, so *"Did the user spend a prolonged period resting?"* answered **no**
   on a recording containing a 6,331-second sitting stretch. Fixed in `intent.py`: a named posture
   ("lie", "lying", "sit", …) narrows the question; the generic word "rest" does not, and now
   resolves against sedentary behaviour as a whole. That question now correctly answers
   "Likely yes", citing 1693–8024 s with `acc_mag_std = 0.0027 g, gyro_mag_std = 0.0166 rad/s`.
3. **"How much … did she spend resting?" was routed to open-world instead of duration.** The
   duration rule only matched "how long" / "duration" / "time spent". This phrasing is quoted
   verbatim from the brief's own opening scenario, so it is very likely to appear in the hidden
   evaluation set. Duration now also matches the "how much" question form and "total time" —
   deliberately *not* the bare word "spend", which would swallow the yes/no
   *"Did the user spend a prolonged period resting?"*. Verified: 12 question forms across all
   seven intent types now route correctly.

**Considered and deliberately not changed:**
- *`Sensor Channel(s)` hardcoded to `All`.* The brief's own examples use `All` for Task 2, 3 and 4
  answers, and the recognizer genuinely uses all 302 features across all 8 signals — so per-answer
  channel attribution would be a claim we cannot substantiate without per-interval feature
  importance. Inventing a narrower citation risks being scored as *wrong* rather than vague.
- *Two output formatters* (`format_answer()` in `src/aggregate/schema.py`, 1-space indent, vs
  `FormattedAnswerBlock.to_output_string()`, 2-space). Only the latter is ever executed; the
  former is imported in `operations.py` but never called. Harmless, and both match the brief's
  field names and order. Left alone to avoid churn in a shared contract file.

**Open question deferred to the harness (do not guess, measure):** a *duration* question about
"resting" still resolves to `lying_down` only, so it would miss sitting time. Whether "resting"
should mean lying-down-only or sedentary-in-general is genuinely ambiguous; decide it with
measured evidence once the benchmark exists rather than by assumption.

---

### Task 3 — Build the benchmark question set with ground truth
**Blocker to resolve first:** the only labelled ground truth available locally is two fixtures
(`tests/fixtures/*/truth.json`). A 2-recording benchmark is too thin to macro-average over seven
question types. Options: generate more labelled recordings from the ExtraSensory data on the
institute server, or construct synthetic-but-faithful recordings from known label sequences.

**Outcome: [x] done.** `scripts/build_eval_benchmark.py` builds the benchmark from **real
ExtraSensory bursts**; `src/eval/questions.py` generates the question set with ground truth.

**Benchmark:** 12 recordings, ~2 hours of labelled minutes each, cut from genuine raw
accelerometer/gyroscope bursts with their real ~40 s inter-burst gaps preserved. 216 questions
(18 per recording) spanning all seven question types.

**Two integrity rules enforced, both of which materially change the numbers:**
1. *Real signal only.* The pre-existing fixture `tests/fixtures/00EABED2_590_160/recording.csv` is
   synthetic (generated by `scripts/generate_test_recording.py`: two distinct timestamp deltas in
   the whole file, `acc_z` mean exactly the generator's hardcoded 0.98). Any figure measured on it
   — including the 93.7% quoted earlier in this project — describes performance on idealised sine
   waves, not on ExtraSensory.
2. *Held-out users AND a held-out model.* Benchmark recordings come only from fold 0's **test**
   users, and evaluation uses `results/model_rf_fold0.joblib`, refitted on fold 0's **train** users
   only (70,172 minutes, 45 users, 100 s to fit). The shipped `results/model_rf.joblib` was fitted
   on all 60 users, so scoring it against any ExtraSensory user would be scoring on training data.

**Two defects found and fixed in the generator itself:**
- *Overlapping ground truth.* Real minute timestamps drift (61 s apart, not 60), so a fixed
  `start + 60` produced intervals like `walking 1261→1321` immediately followed by
  `lying_down 1320→1620` — an overlap that makes IoU ill-defined. Minutes are now clamped to end
  at the next minute's start. Verified: **0 overlapping or non-positive intervals** across all 12.
- *Under-merging.* Consecutive same-label minutes only merged on exact float adjacency, which
  drifting timestamps never satisfy — one recording had 110 intervals where it should have had
  ~54. Now merges on genuine adjacency.
- *Performance.* The first version read every raw burst for every candidate user before selecting
  anything (thousands of NFS reads, effectively hung). Selection now uses labels only, and signal
  is read for the ~120 chosen minutes; probe time dropped from many minutes to **1 s per user**.

---

### Task 4 — Implement the scoring rules exactly as the brief defines them
One module implementing, precisely:
- categorical → exact match, macro-F1, balanced accuracy
- verification → precision / recall / F1 on the positive class **plus specificity**
- numeric → accuracy-within-tolerance (absolute and relative variants), MAE, MAPE
- temporal → IoU; multi-interval via overlap matching, or temporal precision/recall → F1
- combined grounded-and-correct → answer correct **and** IoU ≥ τ **and** modality/channels match
- grounding precision → cited interval genuinely contains the named activity
- open-world → categorical where a reference exists, plus a fixed 1–5 faithfulness rubric
- headline → overall QA accuracy, **macro-averaged across question types**

**Outcome: [x] done.** `src/eval/scoring.py` (rules) + `scripts/evaluate_qa.py` (harness).
Every rule carries the brief sentence it implements as a comment. Multi-interval IoU uses the
temporal-precision/recall-F1 form the brief offers, which is well defined when the predicted and
reference episode counts differ (they usually do).

### HEADLINE RESULTS — 12 held-out real recordings, 216 questions, held-out model

| Question type | n | Accuracy | Mean IoU | Other metrics |
|---|---|---|---|---|
| Identification | 12 | 0.500 | 0.627 | macro-F1 0.29, balanced acc 0.33 |
| Verification | 36 | **0.944** | 0.343 | P 1.00, R 0.92, specificity 1.00 |
| Duration | 36 | **0.139** | 0.298 | MAE 1551.8 s, MAPE 125% |
| Count | 36 | 0.361 | 0.298 | MAE 4.7 episodes, MAPE 95% |
| Comparison | 12 | 0.667 | — | macro-F1 0.52, balanced acc 0.62 |
| Grounding (onset) | 36 | **0.917** | 0.160 | P 1.00, R 0.88, specificity 1.00 |
| Open-world | 48 | 0.833 | 0.567 | P 0.62, R 1.00, specificity 0.77 |

- **Overall QA accuracy (macro-averaged across types): 0.623** ← the brief's headline definition
- Overall QA accuracy (micro, all 216 questions): 0.644
- **Grounded AND correct (answer right AND IoU ≥ 0.5 AND modality/channels cited): 0.210** (n=157)
- Grounding precision (cited interval genuinely contains the named activity): 0.379
- Wall time: 132 s for the full run with SLM explanations active

**The gap between 0.623 answer accuracy and 0.210 grounded-and-correct is the price of demanding
evidence** — the brief explicitly asks for both to be reported side by side, and this is it.

### Where the system is strong and where it fails, with the cause identified

*Strong:* yes/no reasoning — verification 0.944 and onset grounding 0.917, both with specificity
1.00, meaning it does **not** have the "always answer no" bias the brief warns about.

*Weak:* **duration (0.139) and count (0.361)**, and the cause is not the query layer. Inspection of
every duration answer shows the errors are inherited wholesale from per-class recognition:

| Activity asked about | Truth (s) | Predicted (s) |
|---|---|---|
| walking | 1107 | 1096 ← accurate |
| standing and moving | 3780 | 616 |
| standing and moving | 3421 | 204 |
| standing in place | 540 | 0 |
| sitting | 947 | 5155 |

`standing_and_moving` and `standing_in_place` are drastically under-predicted (sometimes zero) and
that time is absorbed into `sitting`. This is exactly the per-class weakness already measured in
`report/tables.md` (standing_and_moving F1 0.257) surfacing at the QA level. Walking durations,
where recognition is reliable, are accurate to ~1%. So Task 2 numeric accuracy is bounded by the
recognition backbone, not by the temporal reasoning built on top of it.

---

### Task 5 — Generate Figure 1 and Figure 3
- **Figure 1:** grouped bar chart, accuracy by question type (identification, verification,
  duration, count, comparison, grounding, open-world) + a final overall bar. Caption must state
  the correctness rule per group.
- **Figure 3:** accuracy vs strictness — error tolerance on the x-axis for numeric answers, IoU
  threshold 0.1→0.9 for temporal/cited intervals; y-axis is fraction accepted.

**Outcome: [x] done.** `scripts/make_figures.py` -> `results/fig1_accuracy_by_question_type.png`
and `results/fig3_accuracy_vs_strictness.png`. Figure 1 carries the per-group correctness rule in
its caption as the brief requires.

*What Figure 3 shows:* the IoU curve falls smoothly from 0.55 at tau=0.1 to 0.02 at tau=0.9, so
cited intervals are approximately but not precisely right. The numeric curve is the opposite and
more damning: duration acceptance only climbs from 0.11 to 0.33 even when the tolerance is relaxed
all the way to +-50%. Duration errors are therefore **systematic, not near-misses** - consistent
with whole activities being assigned to the wrong class rather than boundaries being slightly off.
This is precisely the diagnostic the brief says this figure exists to provide.

---

### Task 6 — Build the robustness test and Figure 5

**Outcome: [x] done.** `scripts/robustness_test.py` -> `results/fig5_robustness.png` and
`results/robustness_results.json`. All three degradations the brief names are swept, over 5
held-out recordings x 90 questions x 16 levels, re-running the **entire** pipeline at each level
(not just the classifier). 281 s total.

| Degradation | Result |
|---|---|
| Added Gaussian noise | 0.533 -> 0.210 by sigma = 0.1 g. Graceful to ~0.05 g, then a cliff. |
| Dropped samples | **Flat**: 0.533 at 0% vs 0.555 at 50% dropped. |
| Input sampling rate | Accuracy *rises*: 0.533 at 25 Hz -> 0.636 at 5 Hz. |

*Why dropping half the samples barely matters:* the pipeline resamples to a 25 Hz grid and every
feature is a statistic aggregated over a 5 s window, so removing half the points leaves means,
standard deviations and dominant frequencies almost unchanged. Genuine robustness, and it directly
answers the brief's concern about uneven and missing data.

*Why lower sampling rate looks BETTER - checked rather than assumed.* The obvious guess is that
smoothing pushes predictions toward the sedentary majority class. That guess was **wrong**;
measuring the predicted class distribution shows the opposite:

| Class | 25 Hz | 5 Hz |
|---|---|---|
| sitting | 44.3% | 37.1% |
| lying_down | 39.8% | 32.6% |
| walking | 10.0% | 18.1% |
| standing_and_moving | 2.6% | 8.5% |
| running | 0.7% | **0%** |

Downsampling moves predictions *away* from the over-predicted sedentary classes and *toward* the
under-predicted ones, partially cancelling the recognizer's known class bias - so the gain is real
but is compensating for a defect rather than evidence of quality. The cost is hidden by this
benchmark's class balance: `running` detection collapses to zero at 5 Hz, as expected once the
~2.8 Hz running cadence exceeds the 2.5 Hz Nyquist limit. **Report this with the caveat, not as a
claim that 5 Hz is fine.**

---

### Task 7 — Measure and record the resource cost (mandatory, not extra credit)
The brief requires, at minimum: model size in **parameters and on disk**, **peak memory during
inference**, and **time to answer a single query** — measured on a **defined, stated target**.
Covers the recognizer *and* the SLM together, since both run per query.

**Outcome: [x] done.** `scripts/measure_cost.py` -> `results/cost_report.json`.

**TARGET DEVICE (stated, as the brief requires):** Windows 11 laptop, AMD Ryzen (Family 25 Model
80), 6 physical / 12 logical cores, **6.3 GB RAM**, Python 3.12.2, **CPU only - no GPU at any
stage**. Captured programmatically at runtime, not asserted.

| Component | Parameters | On disk | Peak memory | Latency |
|---|---|---|---|---|
| Recognizer (Random Forest) | 5,959,242 tree nodes over 300 trees, 302 input features | 234.1 MB | 719 MB (timeline build) | 17.1 s per 2-hour recording = **421x realtime** |
| SLM (Qwen 2.5 1.5B, Q4_K_M) | 1.5 B | 986.1 MB | **2,147 MB** private/commit | ~1.9 s per explanation |
| **Per query** (timeline already built) | — | — | — | **0.5-0.7 ms** deterministic; **~1.9 s** when the SLM is invoked |

**Key structural point for the report:** timeline construction is a *one-time ingestion cost per
recording*, not paid per question. Once ingested, four of the seven question types
(identification, duration, count, comparison) are answered in **under a millisecond** with no SLM
involvement at all. Only verification and onset explanations invoke the SLM, and they dominate
per-query cost by a factor of roughly 3,000x.

**A measurement trap worth documenting.** Peak SLM memory first read as 45 MB, which is
implausible for a 1.5 B model. Two distinct causes, both found by checking rather than assuming:
1. The weights are not held by `ollama.exe` (~20 MB) but by a **`llama-server.exe` child
   process**, which the initial process scan missed entirely.
2. Even then, **RSS under-reports this workload badly on Windows**: the GGUF file is
   memory-mapped, so pages are evicted and re-faulted and RSS reads ~39-75 MB while the process
   actually reserves **~2.0-2.1 GB of private/commit memory**. Ollama also unloads the model after
   an idle keep-alive window, so sampling after generation finishes measures an empty runner.
   The final script samples the whole process tree *during* generation and reports private/commit.

**Edge-deployment implication:** the deployed system needs roughly **3 GB** of memory
(2.1 GB SLM + 0.7 GB ingestion peak) and 1.2 GB of storage. That fits a 4 GB single-board computer
but rules out a 2 GB class device - a concrete, honest statement about where this can actually run,
which is what the brief's cost section is asking for.

---

### Task 8 — Convert the placeholder tests into real automated tests

**Outcome: [x] done. `pytest tests/` now collects and passes 59 tests (previously 0).**

| File | Covers |
|---|---|
| `tests/test_stage1_smoke.py` | Intent routing across all 7 question types, plus regression tests for the three parser bugs fixed in Task 2. Rule fast-path only, so no Ollama needed. |
| `tests/test_stage2_evidence.py` | Query handlers over a hand-built `Timeline` — no model, no dataset, no SLM. Includes the Tier-4 "evidence is never N/A" guard and the 6-field output-order check. |
| `tests/test_scoring_rules.py` | Unit tests for every scoring rule the report's numbers depend on: tolerance boundaries, IoU with differing episode counts, "0 seconds" being a real prediction rather than a missing value, unparseable answers never scoring as correct. |
| `tests/test_timeline_models.py` | Full recording -> timeline -> answer integration, skipping gracefully when the LFS model or dataset-derived recording is absent so a fresh clone stays green. |

Most tests deliberately need **no model file, no dataset and no Ollama server**, so the suite is
meaningful on a fresh clone — which is precisely the situation the teaching team will be in.

### Cross-version model portability — verified, not assumed

The models were fitted on the training server and are loaded for evaluation on the laptop, under a
**different scikit-learn version**. scikit-learn warns that this "might lead to breaking code or
invalid results", and every accuracy figure in this document comes from that cross-version load,
so it had to be checked. `scripts/verify_model_portability.py` fingerprints `predict_proba` over a
fixed set of 400 windows:

| | scikit-learn | numpy | Predicted class counts | Fingerprint |
|---|---|---|---|---|
| Laptop | 1.6.0 | 2.2.3 | `[51, 264, 2, 8, 75, 0, 0]` | `0cb9a9dc1de93c45` |
| Server | 1.9.0 | 2.4.6 | `[51, 264, 2, 8, 75, 0, 0]` | `0cb9a9dc1de93c45` |

**Identical**, with probability means matching to 6 decimal places. The warning is benign across
this version range and the reported results stand. `requirements.txt` records this finding at the
point of use rather than pinning an exact version and making setup brittle.

`requirements.txt` was also missing `psutil` (needed for the mandatory memory measurement) and
`pytest`; both added, and every entry now carries a version floor.

---

### Task 9 — Make the fresh-recording path a first-class deliverable

**Outcome: [x] functionally done** (README prose deferred with the rest of the writing).

- Added **`sample_questions.txt`** at the repo root, covering all four tiers. It lives outside
  `data/` deliberately: `data/` is git-ignored, so the path referenced in `pipeline.py`'s docstring
  would not exist on a fresh clone.
- Verified the loader against realistic input variation a grader's recording might have — header
  styles `acc_x` / `accx` / `ACC_X` / `gyr_x`, and unsorted rows. **All five variants load
  correctly and are sorted into time order.**
- Full fresh-recording run: 15 questions across all four tiers on a recording the pipeline had
  never seen, **32 s wall clock**, correct 6-field output for every question.
- The grounding fix is visibly working on unseen real data — the SLM wrote *"The cadence (3.3 Hz)
  is atypical for walking, falling outside the usual range of 1.8-2.2 Hz"* rather than asserting
  false consistency.

**One more evidence gap found and fixed during this run:** `handle_onset()`'s negative branch still
emitted `N/A` for all evidence fields. Task 1 verification may do that (the brief's own example
does), but **Task 3 states evidence is "required and directly assessed"**, so a negative onset now
cites the span searched. Re-verified afterwards: all 59 tests pass and the headline accuracy is
unchanged at 0.623.

---

### Task 10 — Duration accuracy: root-caused a real aggregation bug (not just a model limit)

**Outcome: [x] done — duration 0.139 -> 0.167, overall macro QA 0.623 -> 0.633, with no regression
on any question type.**

The earlier conclusion that duration error was purely inherited from the recognizer was
**incomplete**. Measuring predicted-vs-reference total time per recording exposed a genuine bug in
the aggregation layer:

| Recording | Reference time | Predicted time | Ratio |
|---|---|---|---|
| 1155FF54-63D3 | 2,220 s | **11,073 s** | **4.99x** |
| (other 11) | ~7,200 s | ~7,200 s | ~1.00 |

**Cause.** `build_timeline()` padded every burst out to the *median inter-burst period*, assuming
ExtraSensory's usual one-burst-per-minute cadence. That user's labelled minutes are ~5 minutes
apart, so `detect_bursty()` returned `period = 300 s` and each **16 s burst was stretched to
300 s — a 19x inflation**, claiming activity across time where nothing was recorded. Every
duration answer for that user was inflated ~5x, dragging the whole metric down.

**Fix.** An ExtraSensory label describes **one minute**, however far apart the labelled minutes
are, so burst padding is now capped at 60 s (`LABEL_MINUTE_SEC`). This is a correctness fix, not a
tuned parameter, and it also removes a grounding defect: intervals no longer claim evidence over
unrecorded gaps.

| Metric | Before | After |
|---|---|---|
| Overall QA accuracy (macro) | 0.623 | **0.633** |
| Duration accuracy | 0.139 | **0.167** |
| Duration MAE | 1551.8 s | **1334.5 s** |
| Duration MAPE | 125% | **92%** |
| Open-world accuracy | 0.833 | **0.875** |
| Grounded AND correct | 0.210 | **0.229** |
| Grounding precision | 0.379 | **0.437** |

**A second change was tested and deliberately rejected.** Capping the minimum resolvable episode
at 60 s (rather than 1.5x the period) was also tried, on the argument that a one-minute episode is
genuine. Measured: it improved open-world (0.875 -> 0.917) but **degraded count accuracy
0.361 -> 0.278** and lowered overall macro to 0.627. Since that threshold exists precisely to
suppress recognizer flicker, and the count regression is direct evidence that the flicker is real,
the original value was kept. Only the unambiguous bug fix was retained. *(Noted for transparency:
two configurations were compared on the benchmark; the minimal, independently-justified change was
kept rather than the highest-scoring one.)*

**What the remaining duration error actually is.** With the inflation removed, the residual error
is genuinely the recognizer. In time terms across all 12 recordings:

| True activity | Where its time is predicted to go |
|---|---|
| sitting (31,014 s) | sitting 67%, lying_down 29% |
| lying_down (20,051 s) | **sitting 51%**, lying_down 48% |
| standing_and_moving (16,162 s) | **sitting 55%**, lying_down 33%, walking 10% |
| walking (8,128 s) | **sitting 55%**, walking 38% |
| standing_in_place (2,278 s) | **sitting 67%**, walking 21% |
| bicycling (683 s) | bicycling 100% |
| running (240 s) | running 100% |

Everything collapses toward `sitting`. Note bicycling and running are perfect but are only 1.2% of
the time — the imbalance the brief warns about, visible directly.

---

### Task 11 — Class-prior calibration: tested in isolation, REJECTED (documented negative result)

**Outcome: [x] done — does not work end to end. Baseline kept. Full write-up in
`experiments/calibration/FINDINGS.md`.**

Tested whether the sitting-collapse is a decision-threshold artefact that probability
re-weighting could fix. Run in isolation (nothing in `src/` changed; calibration applied by
wrapping `window_predictions` at runtime). Stage 1 on the institute server, stage 2 locally.

**Integrity:** fold 0 TRAIN users split by user into 34 sub-train / 11 validation; forest fitted on
sub-train only; every parameter chosen on validation only; the 12 test users scored once at the end.

**Stage 1 (minute level) looked like a clear win.** Time error (`sum_c |predicted minutes -
true minutes| / total`) fell 0.435 -> 0.300, a 31% reduction. Prior correction at alpha=0.4 beat
baseline on macro-F1 *and* balanced accuracy at once. Rare-class recall moved exactly as intended
(running 0.195 -> 0.439, bicycling 0.594 -> 0.646), and over-predicted sitting minutes fell from
7,935 to 6,483 against a true 4,792.

**Stage 2 (real QA harness) did not confirm it.** No candidate dominates the baseline:

| Candidate | Macro QA | Duration | Count | Grounded&correct |
|---|---|---|---|---|
| **Baseline** | **0.633** | 0.167 | **0.361** | **0.229** |
| prior alpha=0.4 | 0.629 | **0.194** | 0.306 | 0.229 |
| coord ascent (max F1) | **0.642** | 0.139 | 0.361 | 0.210 |
| coord ascent (min time error) | 0.614 | 0.111 | 0.306 | 0.223 |

The candidate that won stage 1 outright produced the **worst** end-to-end duration accuracy. Best
duration accuracy (+0.027) costs count (-0.055), open-world (-0.083) and grounding (-0.056).

**Why the proxy misled:** duration is scored within `max(5 s, 10%)`, so pulling aggregate class
totals closer does not necessarily pull any individual recording inside its band; and predicting
more rare-class minutes creates more episodes, which hurts count and fragments the timeline, which
hurts grounding. Duration and count pull in opposite directions.

**Significance caveat:** with 36 duration questions, one question is worth 0.028 — the headline
+0.027 is a single question changing. No gap here is large enough to call a real improvement.

**What this is worth in the report:** the mechanism, not the score. Re-weighting demonstrably works
as a knob (stage 1 proves it) but moves the *wrong minutes*, so the sitting-collapse is genuine
class confusability in the features — sedentary postures being near-indistinguishable from a phone
accelerometer — not a miscalibrated classifier. That independently corroborates the ~47-49% ceiling
six different models already hit, and it is a stronger claim than "we tried and accuracy is low".

---

### Task 7b — Explanation faithfulness (brief-mandated, was outstanding)

**Outcome: [x] done.** `scripts/score_explanations.py` -> `results/qa_eval/explanation_scores.json`.
Two complementary measures, because they fail differently:

| Measure | Result |
|---|---|
| **Numeric fidelity** (deterministic): every g / Hz / rad-s value quoted in an explanation matches a value actually computed for the cited interval | **1.000 — 62/62** |
| **Rubric score 1-5** (LLM judge, fixed rubric, as the brief specifies) | mean **3.37**, median 3 (open-world 3.63, grounding 3.22, verification 3.17) |

**Numeric fidelity of 1.000 is the important number.** It is objective, needs no judge, and
directly validates the system's central design claim — the SLM phrases evidence but never invents
it. Not one of the 62 explanations that quoted a measurement quoted a value the pipeline had not
computed.

**Two bugs in my own checker had to be fixed before this number could be trusted** — it first read
0.719, which would have been a false and damaging claim:
1. The number regex read `"usual range 0.3-0.5 g"` as a *negative* measurement `-0.5 g`. Those are
   reference bands quoted as context, not measurements. Fixed with a lookbehind.
2. The checker compared against the *first* cited interval, but `generate_explanation()` summarises
   the *longest* one. Perfectly faithful explanations were being flagged. Now checks every cited
   interval.

**Stated limitation:** the rubric judge is the same Qwen 2.5 1.5B that wrote the explanations, so
it is a weak and probably generous evaluator — and its score distribution is degenerate
(36x score 1, 26x score 3, 58x score 5; no 2s or 4s at all), which is characteristic of a small
model snapping to round values. The rubric figure should be read as indicative only; numeric
fidelity is the trustworthy measure. A larger independent judge or human graders would be the
proper fix.

---

### Task 10 — Repo hygiene and TA access (user action)
- Add `sandipc-iitkgp`, `sayantan-kuila`, `debjit2001` as collaborators.
- Fresh-clone test: clone into an empty folder, follow only the README, confirm it runs.
- Confirm no raw dataset committed; `.gitignore` sane; commit history reads as incremental work.

**Outcome:** _(to fill in)_

---

## Part B follow-ups (quality items) — completed

### B4(a) — Phrasing robustness: the biggest hidden weakness found and fixed

**Outcome: [x] done — fast-path routing on realistic paraphrases went 47% -> 100%.**

The auto-generated benchmark asks every question in one of 18 templates the rules were written
against, so it **structurally could not detect a phrasing weakness**. The brief says the hidden
evaluation "will include difficult and edge-case questions chosen to probe robustness", so this
was the most likely way to lose marks on the day.

Wrote 32 deliberately different paraphrases (`tests/test_intent_robustness.py`), several lifted
from the brief's own scenario. **Only 15/32 routed correctly** — 17 fell through to the SLM, which
is ~2000x slower per query and measurably worse at routing.

The cause was systematic: rules anchored on rigid prefixes. `startswith("how long")` missed *"For
how long did she walk?"* and *"Roughly how long was she on a bike?"*; `startswith("when did")`
missed *"At what point did walking start?"*; identification only matched two literal phrases.

Fixed by matching cues anywhere in the question instead of at the start, separating count from
duration on the **noun** ("how many *episodes*" vs "how many *minutes*") rather than word order,
accepting indirect requests ("Can you confirm...", "Is there any evidence of..."), and adding
natural synonyms for the open-world concepts (vigorous/exertion -> strenuous;
inactivity/motionless/extended period -> rest).

**Result: 32/32.** All 93 tests pass and the QA evaluation was byte-identical afterwards
(macro 0.633, every per-type number unchanged) — pure coverage gain, no regression. The test
asserts a 90% floor so a future regression fails the build rather than silently shifting load onto
the SLM.

### B5 — Sample sizes on the two weakest question types

**Outcome: [x] done.** Identification had n=12 (one per recording), so a single answer moved that
bar by 8 points. Now asks three phrasings per recording (**n=36**), which doubles as a check that
the answer does not depend on wording. Comparison now also asks the dominant-vs-rarest and
second-vs-third pairs (**n=34**), so the type is no longer measured only on the easy
dominant-vs-runner-up case.

Effect: **overall macro QA 0.633 -> 0.639**, comparison 0.667 -> 0.706, grounded-and-correct
0.229 -> 0.236, on 236 questions instead of 216.

### B7 — A confusion matrix consistent with the QA figures

**Outcome: [x] done.** `scripts/make_benchmark_confusion.py` ->
`results/fig2b_benchmark_confusion.png`. The existing Figure 2 is minute-level 5-fold CV over all
60 users, which is the right way to characterise the *recognizer*; every QA figure instead comes
from the full *pipeline* on 12 held-out recordings, so aggregation-layer errors never appeared in
it. Figure 2b closes that gap and is measured in **seconds**, because that is what duration and
grounding answers are built from.

| True activity | True time | Predicted time | Time recall | Time precision | Absorbed mostly by |
|---|---|---|---|---|---|
| lying_down | 20,051 s | 24,421 s | 0.48 | 0.39 | sitting |
| sitting | 31,014 s | 46,006 s | 0.67 | 0.45 | lying_down |
| standing_in_place | 2,278 s | **108 s** | **0.00** | 0.00 | sitting |
| standing_and_moving | 16,164 s | **1,327 s** | **0.01** | 0.13 | sitting |
| walking | 8,128 s | 5,649 s | 0.38 | 0.55 | sitting |
| running | 240 s | 256 s | **1.00** | 0.94 | — |
| bicycling | 683 s | 791 s | **1.00** | 0.86 | — |

> **Superseded by the 5-fold numbers below — do not quote this table in the report.** On fold 0
> alone, running and bicycling showed recall 1.00, which looked like "the system nails periodic
> motion". That was an artefact of tiny samples: fold 0 contains just 240 s of running and 683 s of
> bicycling. Pooled over all five folds (3,086 s and 29,560 s) the true figures are 0.27 and 0.63.
> The standing-posture finding, by contrast, held up.

See Task 12 for the corrected, pooled confusion matrix.

---

### Task 12 — B4(b): full 5-fold evaluation over all 60 users

**Outcome: [x] done. Headline is now `0.641` macro QA accuracy over 1,241 questions on 57 held-out
recordings spanning all 60 users, using the dataset's official 5-fold user-level split.**

Each fold was scored with its **own** held-out Random Forest (fitted only on that fold's train
users) against its **own** held-out recordings — the same protocol
`scripts/train_classifier.py` already uses for the recognition backbone, so the QA layer and the
recognizer are now reported consistently. Models and benchmarks were built on the institute server
(~7 min), and all five evaluations were run in one local environment so no cross-machine difference
can affect comparability.

| Fold | Recordings | Questions | Macro QA | Duration | Grounded & correct |
|---|---|---|---|---|---|
| 0 | 12 | 262 | 0.639 | 0.167 | 0.236 |
| 1 | 12 | 263 | 0.668 | 0.167 | 0.336 |
| 2 | 11 | 238 | 0.621 | 0.062 | 0.288 |
| 3 | 12 | 259 | 0.609 | 0.057 | 0.195 |
| 4 | 10 | 219 | 0.670 | 0.200 | 0.293 |
| **POOLED** | **57** | **1,241** | **0.641** | **0.130** | **0.269** |

**Between-fold spread is small: sd 0.027, range 0.609-0.670.** The headline is stable across
completely disjoint user groups, which is much stronger evidence than a single fold. The confidence
interval on the headline tightened from **±0.091 to ±0.041**.

**The headline barely moved (0.639 -> 0.641) but several component numbers moved a lot**, which is
the real justification for having run this:

| Question type | Fold 0 only | All 5 folds | Shift |
|---|---|---|---|
| Identification | 0.500 | **0.649** | +0.149 (fold 0 was unlucky) |
| Count | 0.361 | **0.426** | +0.065 |
| Comparison | 0.706 | **0.735** | +0.029 |
| Verification | 0.944 | **0.883** | -0.061 |
| Open-world | 0.875 | **0.763** | -0.112 |
| Duration | 0.167 | **0.130** | -0.037 |
| Grounded & correct | 0.236 | **0.269** | +0.033 |

**It also caught a false claim before it reached the report.** On fold 0, running and bicycling both
showed time-recall 1.00, supporting a tidy story that the system handles periodic motion perfectly.
Fold 0 simply contains very little of either (240 s and 683 s). Pooled over 3,086 s of running and
29,560 s of bicycling, the real figures are **0.27 and 0.63**. Corrected pooled confusion by time:

| True activity | True time | Predicted time | Time recall | Time precision | Absorbed mostly by |
|---|---|---|---|---|---|
| sitting | 151,381 s | 201,833 s | 0.70 | 0.53 | lying_down |
| standing_and_moving | 64,095 s | 24,186 s | **0.08** | 0.20 | sitting |
| lying_down | 62,786 s | 64,930 s | 0.38 | 0.37 | sitting |
| walking | 49,882 s | 61,117 s | 0.54 | 0.44 | sitting |
| bicycling | 29,560 s | 19,174 s | 0.63 | **0.97** | walking |
| standing_in_place | 14,735 s | 3,203 s | **0.04** | 0.19 | sitting |
| running | 3,086 s | 1,081 s | 0.27 | 0.78 | sitting |

The durable finding: **the two standing postures remain effectively invisible** (0.04 and 0.08 time
recall) and their time is absorbed into `sitting`, which is over-predicted by a third
(201,833 s predicted against 151,381 s true). Bicycling has the highest precision of any class
(0.97) — when the system says bicycling it is almost always right, it just misses a third of it.
This is the root cause of the duration weakness and of why class-prior calibration could not fix it
(Task 11).

All figures (1, 2b, 3) regenerated from the pooled data; Figure 5 remains fold-0-based, which is
noted on it.

---

## Deferred by explicit instruction
- **Extra credit** — Figure 4, compressed/quantized operating point, Pareto curve.
- **Report writing** — the 10–12 page report, contribution statement, AI-use disclosure.
  (The AI-use disclosure will need to state that Claude was used; keep that in mind.)
- **HistGradientBoosting full 5-fold run** — only 2 of 5 folds were ever run. Not required by the
  brief (it asks for *a* recognition backbone, not a six-model comparison); relevant only to the
  completeness of the internal comparison table. Low priority.
- **Same-device latency re-measurement of TinyCNN vs CNN+GRU** — same reasoning; the mandatory
  cost reporting (Task 7) covers the *deployed* system, which is RF + Qwen.
