# PLAN.md — CS60055 Hackathon Challenge 1: "Ask the Sensors"

**Grounded, Explainable Activity Question Answering from Wearable Signals**

## Context / assumptions this plan was built for
- Team of 3 (you + chadsaras + one more member joining).
- Compute available: institute GPU access + free cloud GPU (e.g. Colab/Kaggle) as backup.
- Timeline: tight, ~1–2 weeks.
- Extra-credit efficiency bonus: decide later, once the core system works — treated as an optional final stage, not a blocker.
- Priority is accuracy and working end-to-end functionality over strict repo hygiene. Keep everything important (code, plan, report) in this repo, but don't burn time polishing things that don't move accuracy or correctness.

## Team parallel tracks

Once Phase 1 (data pipeline) is solid, the remaining work splits into three tracks that don't block each other, as long as everyone agrees on two contracts up front: the **interval format** produced by Step 14 (`(activity, start_time, end_time)`) and the **output template** from Step 16. Anyone can build against those contracts with fake/sample data before the real recognizer is ready.

- **Track A — Recognition:** Phase 2 (features + classifier + confusion matrix) → Phase 7 (compression, if attempted).
- **Track B — Reasoning/QA:** Phase 3 (aggregation/timeline) → Phase 4 (question-answering interface, all 4 tiers).
- **Track C — Evaluation & reporting:** Phase 5 (test questions + scoring + figures) → Phase 6 (efficiency measurement) → the report, written incrementally as each track lands results.

Assign people to tracks once the third member is confirmed; all three can start as soon as Phase 1 output (labeled, resampled per-minute data) exists.

## Current status (updated as of this commit)

Phase 0 and most of Phase 1 are done, contributed by chadsaras:
- Repo scaffold, `requirements.txt`, `.gitignore` — **done**.
- `scripts/prepare_data.py` — downloads/unpacks ExtraSensory (labels, original labels, official 5-fold splits, raw acc/gyro) — **done** (Step 4).
- `src/preprocess/load.py` — resamples every burst to 25 Hz, resolves the 7 challenge classes (including pulling the two standing classes from the original-labels file, since the cleaned file collapses them), resolves overlapping labels by a most-dynamic-activity-wins priority order, and loads the **official 5-fold user split** — **done** (Steps 5–6, using the official folds instead of a hand-rolled split).
- `scripts/inspect_data.py` — sanity-check script (label counts, burst/sampling check, walk-vs-sit magnitude plot) — **done** (Step 4's exploration pass, kept as reusable code).

**Design decision already locked in by the data layer:** ExtraSensory doesn't give a continuous all-day stream — it gives one labeled minute at a time, each backed by a ~20s raw burst. So "windowing" (Step 7) means working at this native per-minute granularity rather than inventing arbitrary fixed windows across a continuous recording. Duration-type answers (Phase 3) will be computed by counting labeled minutes, treating each one as representative of its full minute. State this explicitly in the report as a deliberate choice driven by the dataset's real structure.

Folder structure below is updated to match what's actually in the repo (library code under `src/`, one-off entry points under `scripts/`) rather than the original flat proposal.

## Decisions already made (so you don't re-litigate them mid-build)

| Decision | Choice | Why |
|---|---|---|
| Language | Python | Standard for ML/signal processing; every needed library (numpy, pandas, scikit-learn, PyTorch, HuggingFace) is Python-first. |
| Activity classifier | Start with engineered features (mean/std/energy/FFT peaks per window) + a light model (Random Forest), then compare against a small 1D-CNN/LSTM trained on GPU | 7-class HAR from accel+gyro is well-solved at small model sizes; matches the brief's push toward edge-friendly design; also gives a real comparison to justify in the report. |
| How NL questions reach the data | Two-stage: (1) a rule-based / small-SLM intent parser turns the question into a structured request like `{type: duration, activity: walking}`; (2) a separate deterministic function computes the real answer from the timeline. The SLM only phrases language, never invents numbers/timestamps. | This is the only reliable way to satisfy the "grounding" requirement — a language model computing the answer itself risks hallucination; a plain function reading from real intervals cannot. |
| Dataset scope while building | Start with a handful of users end-to-end, expand to all 60 once the pipeline works | Matches the brief's own advice: "begin with a small, correct pipeline... only then push on accuracy." |
| Report writing | Written incrementally, section by section, right after the corresponding phase is built — not left to the end | Tight timeline; also produces a better report since details are fresh. |

## The core rule to hold onto through every phase

Every answer the system produces must be traceable to the **interval list** built in Phase 3 (Step 14). If the SLM ever states a number, timestamp, or channel name that didn't come from that list, that single mistake costs points across accuracy, grounding, and design simultaneously. Be paranoid about this boundary.

---

## Phase 0 — Setup (Day 1, morning)

**1. Create the GitHub repo now, with a real commit today.** — **Done.** Repo is `chadsaras/Ubiquitous_Comp`. Still to do: add the three TA GitHub accounts as collaborators (`sandipc-iitkgp`, `sayantan-kuila`, `debjit2001`) and add the third team member.

**2. Folder structure** (as actually used in the repo — library code under `src/`, one-off runnable scripts under `scripts/`):
```
data/                  (fetched by scripts/prepare_data.py — gitignored, never committed)
scripts/               (CLI entry points: prepare_data.py, inspect_data.py, ...)
src/
  preprocess/          (load.py — parsing, resampling, label resolution, official folds)
  recognizer/          (feature extraction + classifier training/eval)     [Track A]
  aggregation/         (timeline builder, duration/count/comparison queries) [Track B]
  qa/                  (intent parsing, answer generation, output formatting) [Track B]
  eval/                (test questions, scoring, required figures)        [Track C]
report/
README.md
PLAN.md
```

**3. Set up your Python environment** (conda/venv), pin versions in a `requirements.txt` from day one — this directly serves the "reproducibility" grading criterion, so don't skip it even under time pressure.

---

## Phase 1 — Data pipeline (Day 1 afternoon – Day 2)

**4. Download ExtraSensory and do a 30-minute exploration pass.** — **Done** (`scripts/prepare_data.py`, `scripts/inspect_data.py`).

**5. Write the parser + resampler.** — **Done** (`src/preprocess/load.py`, resamples to 25 Hz, resolves the 7 classes, handles overlapping labels). Still worth double-checking: run `inspect_data.py` on a few more users (not just the first one found) to confirm the raw-file naming assumptions hold across the dataset, not just for one user.

**6. Split users into train / validation / test.** — **Done, via the official 5-fold split** (`load_folds`) rather than a hand-rolled one — better than originally planned, since it's the standard, citable split for this dataset. Pick one fold's test users as your held-out set and don't touch them until final evaluation; use the remaining folds for train/validation.

**7. Window the streams.** — **Adapted to the dataset's real structure**: instead of arbitrary fixed windows over a continuous stream, `iter_minutes()` yields one labeled minute at a time (its ~20s burst at 25 Hz), which is what ExtraSensory actually provides. This is the unit Phase 2 trains on. (See "Current status" above for why, and note it in the report as a deliberate call.)

---

## Phase 2 — Activity recognizer (Day 2 – Day 4)

**8. Engineer simple features per window** (mean, std, min/max, energy, dominant frequency via FFT, correlation between axes) for both accel and gyro. This is the classic HAR (Human Activity Recognition) recipe — cheap, well-understood, and edge-friendly.

**9. Train a baseline classifier** (Random Forest or gradient boosting) on these features. This gives you a working, if imperfect, recognizer fast — matching the brief's advice to "begin with a small, correct pipeline" before optimizing.

**10. Train a small neural net (1D-CNN or small LSTM) directly on raw windows on your GPU**, as a second, likely more accurate option. Since you have GPU access, this is cheap to try and strengthens your "design reasoning" section (you can justify picking whichever wins, with numbers).

**11. Evaluate both on the validation set: accuracy, per-class precision/recall/F1, and the confusion matrix.** This *is* Required Figure #2 — build it now while you're already in this code, don't defer it.

**12. Pick the better model as your final recognizer**, and note *why* in a paragraph for the report (this feeds the 25% design-quality score directly).

---

## Phase 3 — Aggregation layer / the "timeline" (Day 4)

**13. Run the recognizer over a full recording to get a per-window label sequence.** Smooth it lightly (e.g., merge single flickering windows) so "walking" doesn't randomly blip to "standing" for one window and back.

**14. Convert the smoothed label sequence into intervals**: `(activity, start_time, end_time)` pairs. This interval list is the single source of truth every later answer will be computed from — nothing downstream is allowed to compute numbers any other way.

**15. Write the core query functions over this interval list**: total duration of an activity, count of occurrences, onset time of an activity, comparison between two activities' total time. These are plain deterministic functions — no AI involved — which is exactly what makes your answers "grounded" rather than guessed.

---

## Phase 4 — Question-answering interface (Day 5 – Day 7)

**16. Write the fixed-template output formatter first**, matching the brief's exact field order (Answer / Activity-Event / Evidence: Timestamp, Modality, Channel(s) / Explanation). Every other function will call this at the end — build it once, early, so nothing downstream reinvents formatting.

**17. Build a question-intent parser.** Since the exact eval phrasing is unknown, this needs to generalize. Two layers, cheapest first:
   - Rule/keyword matching for obvious cases ("how long", "how many times", "is she", "did she begin").
   - Fall back to your small SLM (Qwen2.5-1.5B or similar, fits fine on your GPU) *only* to classify intent + extract the activity name — e.g. output `{type: "duration", activity: "walking"}`. Critically: the SLM never outputs the final numeric answer itself, it only routes the question to Step 15's functions.

**18. Wire up Tier 1 (identification/verification).** Given a window or recording, call the recognizer, format the answer, leave evidence as N/A per spec. Test manually on a few examples.

**19. Wire up Tier 2 (temporal/quantitative).** Route duration/count/comparison questions to Step 15's functions, fill in Evidence with the real intervals. Test manually.

**20. Wire up Tier 3 (grounding).** For "when did X begin / is there evidence for Y," pull the *exact* interval from the timeline plus the modality/channels that were most informative for that classification (e.g., which features/channels had the strongest signal in that window — already computed in Step 8, so surface it here). Have the SLM turn this into one or two sentences of explanation — but the interval and channels themselves come from data, never from the SLM.

**21. Wire up Tier 4 (open-world reasoning).** This is the one place true semantic reasoning is needed: pattern description ("smooth cyclic motion, no heel-strike spikes → consistent with cycling"). Approach: compute a small set of interpretable signal descriptors per interval (periodicity, smoothness, spike density, variance) — feed *those descriptors plus the interval* to the SLM and ask it to reason in plain language. Again: the SLM explains, it doesn't invent the underlying facts.

---

## Phase 5 — Evaluation harness (Day 7 – Day 9)

**22. Build your own test question set from validation-set recordings** (since the real eval questions are hidden). Cover every tier and every sub-type mentioned in the brief: identification, verification, duration, count, comparison, grounding, open-world. Write down the ground-truth answer + ground-truth interval for each by hand or by construction — this becomes your scoring reference.

**23. Implement the scoring rules exactly as specified**: exact match for categorical, tolerance-based for numeric (with MAE reported too), IoU for temporal/evidence intervals, and the combined "grounded-and-correct" rule for Tier 3. For Tier 4, do a quick rubric self-scoring (1–5) on faithfulness — note in the report that ideally this would be graded by a second human or LLM judge.

**24. Generate Required Figures #1 and #3** (accuracy by question type; accuracy vs. strictness/IoU sweep) from this harness — everything needed is now in place.

**25. Build the robustness test.** Take a handful of test recordings, inject controlled noise / drop random samples / downsample below 25 Hz, and re-run the full pipeline at each corruption level. Plot accuracy vs. corruption level → **Required Figure #5**.

---

## Phase 6 — Efficiency reporting (Day 9, mandatory regardless of bonus)

**26. Measure and report your system's cost**: parameter count and disk size (recognizer + SLM combined), peak memory during inference, and time per query — on whichever machine you actually run it on (GPU box, or CPU-only if simulating the "edge" story; either is fine as long as you state it clearly). This is required for full marks even if the bonus is skipped — it's just reporting, not optimizing.

---

## Phase 7 — Decision point: extra credit (Day 10, only if Phases 1–6 are solid)

**27. If time remains**, produce a second, compressed operating point — the easiest win is 8-bit quantizing the SLM (near-zero effort with `bitsandbytes`/HF) and/or shrinking the recognizer. Re-run Phase 5's accuracy harness on this second version, then plot accuracy vs. cost for both → **Required Figure #4**, completing the Pareto tradeoff story.

**If time is short, skip this entirely** — it's worth up to +10%, but a shaky core system scores far worse than a solid core with no bonus. Don't risk it.

---

## Phase 8 — Finish line (Day 10–12, or the last 2 days of whatever window remains)

**28. Freeze the code, then do a fresh-clone test**: pretend to be a TA, `git clone` into a clean folder, follow only the README, and confirm it runs end-to-end on a sample recording and produces correctly formatted output. Fix anything that breaks — this single step protects the 20% reproducibility score.

**29. Finish the report** (should already be ~70% drafted if sections were written as each phase completed): problem framing, design rationale, results per tier with the 5 figures, efficiency numbers (+ tradeoff curve if attempted), contribution statement (even solo, state that clearly), and AI-tool-use disclosure — be specific about where an SLM was used and where any AI coding assistance was used.

**30. Final repo pass**: check no raw dataset got committed, `.gitignore` is sane, commit messages make sense, TA access is confirmed, and the report PDF is in the repo.

---

## Grading weights, for reference while prioritizing under time pressure

| Dimension | Weight |
|---|---|
| Design and approach | 25% |
| Correctness and reproducibility | 20% |
| Accuracy across the four tasks | 35% |
| Evidence grounding | 20% |
| Efficiency and tradeoff | up to +10% (extra credit) |
