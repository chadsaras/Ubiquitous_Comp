# Developer Handoff: SLM & Grounded Query Layer Integration

**To:** Person C (Evaluation, Benchmarking & Results Lead)  
**From:** Person B (Query Layer & SLM Integration)  
**Project:** CS60055 Hackathon Challenge 1 — *Ask the Sensors: Grounded Activity Question Answering*  
**Date:** September 9, 2026  
**Status:** Query & SLM Integration Complete (`src/query/`, `src/pipeline.py`)

---

## 1. Executive Summary & Purpose

This document serves as the formal technical handoff for the **Small Language Model (SLM) and Grounded Question-Answering Pipeline** implemented for Track B. 

All core pipeline layers—from raw sensor processing, interval aggregation, hybrid intent parsing, deterministic timeline query execution, to signal-grounded natural-language explanation generation—are **fully implemented, tested, and ready for systematic evaluation**.

Person C can now take ownership of:
1. **Phase 5 (Evaluation Harness):** Constructing benchmark query suites, calculating official scoring metrics (Exact Match, Numeric MAE/Tolerance, Evidence IoU, Combined Grounded Accuracy), and generating **Figures 1, 3, and 5**.
2. **Phase 6 (Efficiency & Benchmarking):** Measuring parameter counts, disk footprints, memory consumption, and per-query latency across CPU/GPU setups to produce **Figure 4**.

---

## 2. Architecture & Design Principles

The query layer follows a strict **three-stage hybrid architecture**. The fundamental design invariant is: **The SLM is never allowed to invent timestamps, durations, counts, or physical metrics.** All factual data is derived deterministically from the aggregated sensor `Timeline`.

```
User Query (Natural Language)
              │
              ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Hybrid Intent Parser (src/query/intent.py)    │
│ • LangChain + Ollama (Qwen 2.5 1.5B) + Regex Fast-Path │
│ • Validated Pydantic Schema: QueryIntent               │
└─────────────────────────────┬──────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Deterministic Query Engine (operations.py)    │
│ • Runs mathematical operations over Timeline Intervals │
│ • Computes exact durations, counts, onsets, comparisons│
│ • Extracts grounded timestamps, modality & channels    │
└─────────────────────────────┬──────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Clinician-Grade Explainer (explain.py)        │
│ • Qwen 2.5 prompted with exact physical statistics:    │
│   cadence (Hz), acc_mag_std (g), gyro_mag_std (rad/s)  │
│ • Deterministic template fallback if LLM offline       │
└─────────────────────────────┬──────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│ Output: Standard 6-Field Answer Block                  │
└────────────────────────────────────────────────────────┘
```

### The 6-Field Output Contract (Contract 2)
Every query produces an instance of `FormattedAnswerBlock` rendered as:
```text
Answer: 120 seconds (2.0 minutes) across 1 episode(s)
Activity/Event: Walking
Evidence:
  Timestamp(s): 35 to 155 (seconds from start)
  Sensor Modality: Accelerometer
  Sensor Channel(s): Acc X/Y/Z, Acc Magnitude
Explanation: Walking was detected across 120 seconds, characterized by rhythmic acceleration variance (acc_mag_std = 0.134 g) at a step cadence of 1.9 Hz.
```

---

## 3. Codebase Map & Key Modules

| Module Path | Primary Responsibility | Key Functions / Classes |
|---|---|---|
| [`src/pipeline.py`](file:///Users/mohan/AI/Hackathon%201/Ubiquitous_Comp/src/pipeline.py) | **Main end-to-end entry point** connecting raw sensor data to final output blocks | `run_pipeline()`, CLI parser |
| [`src/query/engine.py`](file:///Users/mohan/AI/Hackathon%201/Ubiquitous_Comp/src/query/engine.py) | Unified entry point for the query layer | `answer(question, timeline) -> str` |
| [`src/query/intent.py`](file:///Users/mohan/AI/Hackathon%201/Ubiquitous_Comp/src/query/intent.py) | Parses user questions into structured intents with synonym mapping | `parse_intent(question) -> QueryIntent` |
| [`src/query/schema.py`](file:///Users/mohan/AI/Hackathon%201/Ubiquitous_Comp/src/query/schema.py) | Pydantic data contracts for intents and output blocks | `QueryIntent`, `IntentType`, `FormattedAnswerBlock` |
| [`src/query/operations.py`](file:///Users/mohan/AI/Hackathon%201/Ubiquitous_Comp/src/query/operations.py) | Deterministic logic executing calculations over the `Timeline` | `execute_query()`, `handle_identify()`, `handle_duration()`, etc. |
| [`src/query/explain.py`](file:///Users/mohan/AI/Hackathon%201/Ubiquitous_Comp/src/query/explain.py) | LangChain chain prompting Qwen 2.5 with physical sensor statistics | `generate_explanation()` |
| [`src/aggregate/timeline.py`](file:///Users/mohan/AI/Hackathon%201/Ubiquitous_Comp/src/aggregate/timeline.py) | Aggregates window predictions into clean intervals with signal summaries | `build_timeline()`, `Timeline` |

---

## 4. How to Run the Pipeline (CLI & Python API)

### 4.1 Running from Command Line

```bash
# 1. Ask a single question directly from a raw sensor recording:
python src/pipeline.py --recording tests/fixtures/00EABED2_590_160/recording.csv \
                       --question "How long was the user walking?"

# 2. Run a batch of questions from a text file against a precomputed timeline:
python src/pipeline.py --timeline tests/fixtures/00EABED2_590_160/timeline.json \
                       --questions data/eval_questions.txt \
                       --out results/eval_answers.txt

# 3. Choose a specific model (Random Forest or Histogram Gradient Boosting):
python src/pipeline.py --recording tests/fixtures/00EABED2_590_160/recording.csv \
                       --model models/model_rf.joblib \
                       --question "Did the user spend more time walking or sitting?"
```

### 4.2 Programmatic Usage in Python (For Evaluation Scripts)

```python
from pathlib import Path
from src.aggregate.schema import Timeline
from src.aggregate.timeline import build_timeline
from src.query.engine import answer
from src.query.operations import execute_query
from src.query.intent import parse_intent

# Option A: Fast evaluation over pre-cached timeline fixture
timeline = Timeline.from_json("tests/fixtures/00EABED2_590_160/timeline.json")

# Option B: Build timeline from raw recording and model
# timeline = build_timeline("path/to/recording.csv", "models/model_rf.joblib")

question = "Did the user begin walking at any point, and if so, when?"

# 1. Get raw string formatted output:
formatted_text = answer(question, timeline)
print(formatted_text)

# 2. Or get structured Pydantic object for automated grading:
intent = parse_intent(question)
answer_block = execute_query(intent, timeline)

print(answer_block.answer)           # e.g., "Yes, walking began at timestamp 35s"
print(answer_block.activity_event)   # e.g., "Walking"
print(answer_block.timestamps)       # e.g., "35 to 155 (seconds from start)"
print(answer_block.sensor_modality)  # e.g., "Accelerometer"
print(answer_block.sensor_channels)  # e.g., "Acc X/Y/Z, Acc Magnitude"
print(answer_block.explanation)      # e.g., "Walking onset observed at 35s..."
```

---

## 5. Supported Question Tiers & Handlers

The query layer is mapped across the four challenge tiers:

| Tier | Task Type | Example Questions | Handler in `operations.py` | Expected Output Format |
|---|---|---|---|---|
| **Tier 1** | Identification | "What activity is the user doing?" | `handle_identify` | Dominant activity name |
| **Tier 1** | Verification | "Is the user walking?", "Was she running?" | `handle_verify` | "Yes" / "No" + episode count |
| **Tier 2** | Duration | "How long was the user walking?" | `handle_duration` | `X seconds (Y minutes) across N episode(s)` |
| **Tier 2** | Count | "How many times did she walk?" | `handle_count` | `N episode(s)` |
| **Tier 2** | Onset / Localization | "When did the user begin running?" | `handle_onset` | `Yes, <activity> began at timestamp Xs` or `No` |
| **Tier 2** | Comparison | "Did the user spend more time walking or sitting?" | `handle_compare` | `<Act A> (Xs) exceeded <Act B> (Ys)` |
| **Tier 3** | Evidence Grounding | *(Integrated into all queries)* | All handlers | Timestamps, Modality (`Accelerometer` / `Both`), Channels |
| **Tier 4** | Open-World / Semantic | "Did the user engage in prolonged rest?", "Wheeled movement?" | `handle_open_world` | Maps semantic query to canonical sensor patterns |

---

## 6. Blueprint for Person C: Phase 5 (Evaluation Harness)

Person C should implement an automated evaluation suite (`scripts/evaluate_pipeline.py` or under `tests/`):

### 6.1 Benchmark Question Suite Construction
Create a curated test dataset (`data/eval_benchmark.json`) paired with ground-truth labels derived from validation recordings:
- **Identification:** Ground truth is the known dominant label.
- **Verification:** Binary ground truth (True / False).
- **Duration / Count:** Exact duration in seconds and count of intervals.
- **Onset:** Timestamp of the first true episode.
- **Temporal Intervals:** List of `[start, end]` intervals for IoU calculation.

### 6.2 Scoring Rules Implementation
1. **Categorical Answers (Tier 1):** Exact string match / normalized canonical match:
   $$\text{Accuracy}_{\text{cat}} = \mathbb{I}(\hat{y}_{\text{norm}} == y_{\text{norm}})$$
2. **Numeric Answers (Tier 2 Duration / Count):**
   - Tolerance test: $|\hat{v} - v| \le \max(5\text{s}, 0.10 \times v)$
   - Mean Absolute Error (MAE): $\frac{1}{N} \sum |\hat{v}_i - v_i|$
3. **Temporal Evidence Grounding (Tier 3 IoU):**
   $$\text{IoU} = \frac{\text{Area of Overlap}(\hat{I}, I_{\text{true}})}{\text{Area of Union}(\hat{I}, I_{\text{true}})}$$
4. **Combined Grounded Accuracy:**
   $$\text{Score} = \mathbb{I}(\text{Answer is Correct}) \times \mathbb{I}(\text{IoU} \ge \tau) \quad (\text{default } \tau = 0.5)$$
5. **Explanation Faithfulness (Tier 4):**
   - Check whether the numbers cited in `explanation` match the actual values in `interval.summary` within 1% rounding error (0% hallucination verification).

### 6.3 Required Figures to Generate
- **Figure 1 (Accuracy by Question Type):** Bar chart comparing accuracy across Identification, Verification, Duration, Count, Onset, Comparison, and Open-World.
- **Figure 3 (Accuracy vs. IoU Threshold):** Line plot sweeping $\tau \in [0.1, 0.9]$ showing the drop-off in combined accuracy.
- **Figure 5 (Robustness to Corruption):** Inject Gaussian noise to sensor streams, downsample from 25 Hz to 10 Hz, or drop 10%–50% of bursts, and plot end-to-end question accuracy vs. corruption severity.

---

## 7. Blueprint for Person C: Phase 6 (Efficiency Benchmarking)

The report requires explicit resource and latency metrics:

| Metric | Measurement Target | Tool / Methodology |
|---|---|---|
| **Recognizer Size & Params** | `models/model_rf.joblib`, `models/model_hgb.joblib`, `results/model_mlpctx.pt` | `os.path.getsize`, parameter inspection |
| **SLM Size & Params** | Qwen 2.5 1.5B (`~1.54B params`, `~980 MB Q4_K_M` GGUF) | Ollama model info |
| **Peak Memory (RAM / VRAM)** | Peak RAM during timeline generation + SLM generation | `tracemalloc` or `psutil.Process().memory_info().rss` |
| **Latency per Query** | Time to parse intent + execute timeline + generate explanation | `time.perf_counter()` broken down by Stage 1, 2, 3 |
| **Throughput** | Questions answered per second (with batching vs single) | Benchmark loop |

---

## 8. Prerequisites & Environment Setup

1. **Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install langchain langchain-ollama pydantic
   ```
2. **Local SLM (Ollama):**
   Ensure Ollama is running locally with Qwen 2.5 1.5B:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
   *(Note: If Ollama is not running, the system will automatically and gracefully fall back to deterministic regex and template generators without crashing).*

3. **Trained Models:**
   Both `models/model_rf.joblib` and `models/model_hgb.joblib` are tracked and present via Git LFS in the repository.

---

## 9. Contact & Support
If any edge cases arise while building the evaluation harness, Person B is available to extend `operations.py` or tweak intent mappings in `intent.py`. All code is checked into `origin/main`.
