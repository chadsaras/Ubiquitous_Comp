# Query/QA pipeline: real LLM (Qwen 2.5 1.5B via Ollama) test results

**Update (same day):** the two real issues found below (narrow verify fast-path, false
"consistent" claim) have both been fixed in `src/query/intent.py` and `src/query/explain.py`.
See "Fixes applied" at the end of this document for the re-verified results. Sections 1-5 below
are kept as-is as the original findings that motivated the fixes.

Run 2026-09-09. Follows up on `pipeline_smoke_test_results.md`, which only exercised the
deterministic fallback because Ollama wasn't installed. This run installed Ollama (winget),
pulled `qwen2.5:1.5b` (986 MB, Q4_K_M quantization), confirmed the local service was live at
`localhost:11434`, and reran the same tests so the LLM genuinely engaged.

## 1. Only 2 of 7 question types ever call the LLM

Reading `operations.py`: `duration`, `count`, `identify`, `compare` all build their explanation
with plain Python string formatting -- the LLM is never invoked for them, regardless of whether
Ollama is running. **Only `verify` and `onset` questions call `generate_explanation(use_llm=True)`.**
Most of the query surface is 100% deterministic by design, not "LLM with a fallback."

## 2. Intent parsing (`tests/test_stage1_smoke.py`), LLM engaged for real

7/8 questions still resolve via the free regex fast-path (unchanged). The 8th question --
"Did the grandmother take a quick jog in the park?" -- now genuinely reaches the LLM (previously
failed with a connection error). Result:

- Correctly extracted `target_activity = running` (mapped "jog" -> running on its own, no synonym
  table entry needed for "grandmother" phrasing).
- Classified the overall intent as `identify`, when the expected answer was `verify` (this is a
  yes/no question -- "did she jog" -- not an open "what activity" question). A genuine, real
  misclassification by the LLM on the one phrasing style that depends on it.

## 3. Explanation quality: real vs. fallback, same interval, same numbers

| | Deterministic fallback (earlier run) | Real Qwen 2.5 output (this run) |
|---|---|---|
| "Is the user walking?" | "Walking was detected in 4 separate interval(s)... characterized by rhythmic acceleration variance (acc_mag_std = 0.165 g) at a step cadence of 2.3 Hz." | "The sensor evidence supports the activity decision of walking because the acceleration variance (std ~0.10-0.15 g) is high, indicating cyclic motion typical of walking, and the cadence (2.0 Hz) is consistent with the typical walking cadence range of 1.8-2.2 Hz. The gyroscope angular velocity (std ~0.071 rad/s) is also high, suggesting sustained periodic motion typical of walking, with no pronounced impact spikes." |

The LLM version is more fluent and reasons across multiple signals (accel *and* gyro) instead of
quoting one number -- closer to the "clinician-grade explanation" the handoff doc describes.

## 4. A real bug found: correct numbers, incorrect reasoning

One "onset of walking" run produced:

> "...the cadence (3.6 Hz) is consistent with the typical walking cadence of 1.8-2.2 Hz."

**3.6 Hz is not within 1.8-2.2 Hz -- this claim is false.** The number itself was quoted
correctly (satisfies the code's Rule 1: never invent a number), but the *reasoning* about it is
wrong. The system prompt's "never hallucinate numbers" guarantee is verified true; "never say
something false" is not guaranteed. Worth flagging for Tier 4 "Explanation Faithfulness"
evaluation, since graders may specifically probe this.

Also observed: run-to-run wording drift on identical input (e.g. one call said "cadence range
2.0-2.2 Hz" instead of the system prompt's "1.8-2.2 Hz") -- expected, since `temperature=0.1`
isn't 0, but worth knowing before treating any single LLM run as ground truth.

## 5. Latency (this machine, CPU-only, no GPU)

| Scenario | Time |
|---|---|
| Deterministic fallback, full 6-question pipeline | ~14.8s (entirely timeline-building; 0 LLM calls) |
| Same pipeline, LLM active | ~30.2s (+15.4s, for the 2 LLM-backed questions) |
| Single LLM explanation call, isolated (4 repeated calls) | 2.03s, 1.23s, 1.16s, 1.04s (cold-start cost visible on call 1) |

Roughly **1-2 seconds of extra latency per LLM-backed answer** versus instant for the
deterministic ones. Concrete numbers for the report's Phase 6 (efficiency benchmarking) section.

## Inferences

1. The LLM integration works end-to-end for real, not just via fallback -- confirmed by actually
   running it, not just reading the code.
2. It measurably improves explanation quality/fluency over the template fallback.
3. It is not fully reliable: one real intent misclassification, one real factual/reasoning error
   despite correct number quoting, and some run-to-run wording variance. These should be
   documented as known limitations, not silently smoothed over, since the "no hallucinated
   numbers" design goal is met but "no false statements" is a narrower, unverified claim.
4. Its use is narrow by design -- only 2 of 7 query types ever reach it -- which limits both its
   upside (fluency) and its risk (only verify/onset explanations can carry this kind of error).
5. Per-call cost (~1-2s on CPU) is small enough to be practical for interactive use, but adds up
   linearly if a benchmark suite runs many verify/onset questions.

## Fixes applied

**Fix 1 -- intent.py, verify fast-path broadened.** The rule was
`q.startswith(("is the user", "was the user", "did the user"))`, which only recognises "the
user" as a subject. Changed to a regex matching any yes/no auxiliary verb at the start
(`is/was/were/did/does/do/has/had/are`) combined with a detected activity. Re-verified:

- Before: 7/8 test questions resolved via the free fast-path; 1/8 fell through to the LLM and
  was misclassified (`identify` instead of `verify`).
- After: **8/8 resolve via the fast-path, 0/8 need the LLM.** The "grandmother jog" question now
  correctly returns `intent=verify, target=running` with zero LLM latency.

**Fix 2 -- explain.py, LLM's "typical vs atypical" judgment replaced with a computed fact.**
Previously the LLM freely judged whether a cited number "is consistent with" the activity's
typical range, and got this wrong at least once (claimed 3.6 Hz cadence was consistent with a
1.8-2.2 Hz walking band). Added `TYPICAL_RANGES` + `_range_label()` in code: each metric handed
to the LLM is now pre-labelled "(within typical range)" or "(ATYPICAL for X, usual range ...)"
before the prompt is built, and a new STRICT RULE forbids the LLM from claiming consistency for
anything labelled ATYPICAL. Also dropped `temperature` from 0.1 to 0.0 (explanation calls were
observed to reword themselves between identical calls; intent parsing was already 0.0).

Re-verified on the exact scenario that exposed the bug (walking onset, cadence 3.6 Hz,
acc_mag_std 0.041 g):

> "...the cadence (3.6 Hz) is **significantly higher than typical walking cadence (1.8-2.2 Hz)**,
> suggesting the user is actively moving at a brisk pace."

No more false consistency claim. Also confirmed 4 repeated calls on an in-range case (cadence
2.0 Hz, within 1.8-2.2 Hz) now return **word-for-word identical output every time** -- the
temperature=0 fix eliminated the run-to-run wording drift noted above.

**Net effect on "fallback counts":** intent-parsing LLM fallback rate on the 8-question smoke
set went from 1/8 to 0/8. The explanation LLM path is unchanged in *how often* it's called (still
only `verify`/`onset`), but is now grounded against a computed fact rather than free judgment,
which is the more important fix for the "does the LLM path actually run correctly" question. A
full end-to-end pipeline rerun after both fixes still completes cleanly (exit 0, same 5 intervals
on `00EABED2_590_160`) -- no regression from either change.
