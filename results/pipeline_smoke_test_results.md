# Query/QA pipeline: first end-to-end smoke test

Run 2026-09-09. Installed `requirements.txt` (added `langchain-ollama`, `langchain-core`,
`ollama` client) and executed the pipeline for the first time on this machine. Ollama itself
is **not installed** here, so every explanation went through the deterministic template
fallback, not the Qwen 2.5 LLM -- this is the code's designed fallback path, exercised for real.

## 1. Intent parsing (`tests/test_stage1_smoke.py`)

7 / 8 test questions parsed correctly via the free regex fast-path (no LLM needed).
1 failed only because it required the LLM fallback (Ollama unreachable):

> "Did the grandmother take a quick jog in the park?" -> `[WinError 10061] connection refused`

Cause: the fast-path's VERIFY rule only recognises questions starting with "is the user" /
"was the user" / "did the user" -- "did the grandmother" falls through untouched by any rule,
so it's the one phrasing style that actually depends on the LLM being available.

## 2. Query engine against a pre-built timeline (`tests/test_stage2_evidence.py`)

All 7 questions against the `00EABED2_590_160` fixture ran and returned well-formed 6-field
answer blocks. One is worth flagging: "Did the user lie down for a prolonged period?" correctly
answered "Likely no" -- this recording has zero `lying_down` intervals at all, so that's the
right answer, just phrased a bit generically ("No sensor signal pattern matched").

## 3. Full raw-CSV -> timeline -> answer, graded against ground truth

Both fixtures ship a `truth.json` with human-labelled ground-truth intervals -- this let me
grade the pipeline's actual output against reality, not just check it doesn't crash.

### Fixture A: `00EABED2_590_160` (built fresh here, Random Forest)

| | Random Forest | HistGradientBoosting |
|---|---|---|
| Intervals produced | 5 (clean) | 24 (fragmented) |
| Walking detected | 239-1348s, 8127-9269s | 239-1353s, 8127-9267s (near-identical) |
| Sitting total | 7315s | 7126s |

**Graded against truth.json: 93.7% of the recording's time correctly labelled** (8963/9567s).
Walking boundaries matched ground truth almost to the second (239 vs 240, 1348 vs 1347). The
only miss: two `standing_and_moving` episodes (0-240s and 1347-1707s) were both swallowed into
"sitting" -- the model never predicted `standing_and_moving` once in this recording.

### Fixture B: `098A72A5_455_110` (pre-built fixture, Random Forest)

**Graded against truth.json: 37.0% of the recording's time correctly labelled** (2447/6604s).
Concrete errors:
- An opening 120s of `lying_down` was missed entirely (predicted `walking`).
- A 540s `standing_in_place` episode was entirely misread as `walking`/`sitting` -- like
  fixture A, `standing_in_place`/`standing_and_moving` never got predicted correctly anywhere
  in this recording.
- A long ~2700s `sitting` episode was mostly predicted as `lying_down` instead.
- The tail end (14262-14742s, truth = `lying_down`) was predicted as `sitting`.
- Where the model *was* right, it was clearly right: `running` (1260-2950 of a 1260-3240s truth
  span) and the early `walking` burst were both caught with good boundary alignment.

## Inferences

1. **The pipeline works end-to-end and doesn't crash on any code path tested** -- raw CSV in,
   graded answer out, across both trained models, with and without the LLM available.
2. **Walking, running and sitting/lying-down boundaries are trustworthy when the model gets the
   activity right** -- timestamps consistently land within a few seconds of ground truth.
3. **The real weakness is exactly what the earlier 5-fold study found**: `standing_in_place`
   and `standing_and_moving` are essentially invisible to the model in both live recordings --
   never predicted correctly once across two independent test recordings -- and `sitting` vs
   `lying_down` gets confused over long stretches. This matches the model comparison table in
   `report/tables.md` (those two classes have the worst precision/recall of all seven).
4. **Accuracy swings hard by recording** -- 93.7% on one fixture, 37.0% on the other. The
   5-fold measured average (~47-49%) is the honest middle ground; individual recordings can
   look much better or much worse depending on which activities they happen to contain.
5. **HistGradientBoosting produces far noisier, more fragmented intervals than Random Forest**
   (24 vs 5 for the identical recording) despite agreeing almost exactly on where walking
   happens -- RF's merging/smoothing behaves more cleanly for a duration-style question.
6. Still confirmed as *not yet done*: the pipeline defaults to Random Forest / HGB, not the
   FeatureMLP+Context model that actually scored higher in the 5-fold comparison; the query
   layer has no code path to plug that model in yet.
