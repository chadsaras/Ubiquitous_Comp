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

## Deferred by explicit instruction
- **Extra credit** — Figure 4, compressed/quantized operating point, Pareto curve.
- **Report writing** — the 10–12 page report, contribution statement, AI-use disclosure.
  (The AI-use disclosure will need to state that Claude was used; keep that in mind.)
- **HistGradientBoosting full 5-fold run** — only 2 of 5 folds were ever run. Not required by the
  brief (it asks for *a* recognition backbone, not a six-model comparison); relevant only to the
  completeness of the internal comparison table. Low priority.
- **Same-device latency re-measurement of TinyCNN vs CNN+GRU** — same reasoning; the mandatory
  cost reporting (Task 7) covers the *deployed* system, which is RF + Qwen.
