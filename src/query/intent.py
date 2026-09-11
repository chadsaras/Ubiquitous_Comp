"""
Intent and parameter parser using LangChain with ChatOllama (Qwen2.5 1.5B).
Includes a deterministic fast-path and an activity-synonym mapping table.
"""
from __future__ import annotations
import re
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.query.schema import QueryIntent, IntentType, CanonicalActivity

# Canonical activity normalization dictionary
ACTIVITY_SYNONYMS = {
    "walk": "walking", "walking": "walking", "walked": "walking", "stroll": "walking",
    "run": "running", "running": "running", "ran": "running", "jog": "running", "jogging": "running",
    "cycle": "bicycling", "cycling": "bicycling", "bike": "bicycling", "biking": "bicycling",
    "pedal": "bicycling", "pedaling": "bicycling", "bicycling": "bicycling",
    "sit": "sitting", "sitting": "sitting", "sat": "sitting", "seated": "sitting",
    "lie": "lying_down", "lying": "lying_down", "lying down": "lying_down", "lay": "lying_down",
    "sleep": "lying_down", "sleeping": "lying_down", "resting": "lying_down",
    "standing in place": "standing_in_place", "stand in place": "standing_in_place",
    "standing still": "standing_in_place", "still": "standing_in_place",
    "standing and moving": "standing_and_moving", "moving while standing": "standing_and_moving",
    "fidgeting": "standing_and_moving"
}

INTENT_EXTRACTION_SYSTEM_PROMPT = """You are an intent parser for a wearable sensor activity question-answering system.
Given a user's question, classify it into one of the following intent types and extract target activities:

Intents:
- identify: What activity the user is performing (e.g., "What is the user doing?")
- verify: Asking if a specific activity occurred (e.g., "Was the user running?")
- duration: Asking for elapsed time/duration (e.g., "How long did she walk?")
- count: Asking for number of episodes/times (e.g., "How many times did he run?")
- onset: Asking when an activity started/began (e.g., "When did running begin?")
- compare: Comparing durations between two activities (e.g., "More time walking or running?")
- open_world: Abstract/unlabeled behavior (e.g., "prolonged rest", "wheeled or pedal-based movement", "strenuous activity around noon")

The 7 valid canonical classes for target_activity / compare_activity are:
['lying_down', 'sitting', 'standing_in_place', 'standing_and_moving', 'walking', 'running', 'bicycling'].
Map all colloquial synonyms to one of these 7 classes.
"""

_llm_structured_chain = None

def get_structured_llm(model_name: str = "qwen2.5:1.5b"):
    global _llm_structured_chain
    if _llm_structured_chain is None:
        llm = ChatOllama(model=model_name, temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_EXTRACTION_SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        _llm_structured_chain = prompt | llm.with_structured_output(QueryIntent)
    return _llm_structured_chain


def parse_intent_fast_rules(question: str) -> Optional[QueryIntent]:
    """Deterministic rule-based fast path for benchmark questions."""
    q = question.lower().strip()

    # Activity matching helper
    def find_activities():
        found = []
        for term, canonical in ACTIVITY_SYNONYMS.items():
            if re.search(rf"\b{re.escape(term)}\b", q):
                if canonical not in found:
                    found.append(canonical)
        return found

    acts = find_activities()

    # The rules below match cues ANYWHERE in the question rather than anchoring on a prefix.
    # Measured on 32 realistic paraphrases, prefix anchoring routed only 47% correctly: "For how
    # long did she walk?", "Roughly how long was she on a bike?" and "At what point did walking
    # start?" all fell through to the SLM purely because they do not begin with the expected words.
    # Anything the rules miss costs ~2 s of SLM latency and is measurably more likely to be
    # misrouted, so breadth here is worth more than elegance.

    # 1. Compare -- needs two activities plus an explicit comparison cue.
    compare_cue = ("more time" in q or "compare" in q or "exceed" in q or "more than" in q
                   or "greater than" in q or "longer than" in q or " versus " in q or " vs " in q
                   or re.search(r"\bmore\b.*\bor\b", q) or " or " in q)
    if compare_cue and len(acts) >= 2:
        return QueryIntent(intent=IntentType.COMPARE, target_activity=acts[0], compare_activity=acts[1])

    # 2. Count -- "how many" plus an EPISODE noun. Checked before duration so that
    # "how many episodes" and "how many minutes" separate cleanly on the noun, not on word order.
    episode_noun = r"\b(times?|episodes?|bouts?|periods?|occasions?|sessions?|stretches|instances?|walks|runs)\b"
    if (("how many" in q and re.search(episode_noun, q)) or "how often" in q
            or re.search(r"\bcount\b", q) or "number of" in q):
        return QueryIntent(intent=IntentType.COUNT, target_activity=acts[0] if acts else None)

    # 3. Duration. "How much ... did she spend resting?" is the brief's own scenario phrasing.
    # Matches the question form and explicit time nouns -- deliberately NOT the bare word "spend",
    # or the yes/no "Did the user spend a prolonged period resting?" becomes a duration query.
    time_unit = r"\b(minutes?|seconds?|hours?|mins?|hrs?)\b"
    if ("how long" in q or "how much time" in q or q.startswith("how much") or "duration" in q
            or "time spent" in q or "spent doing" in q
            or re.search(r"\btotal\b.*\btime\b", q) or re.search(r"\btime\b.*\btotal\b", q)
            or ("how many" in q and re.search(time_unit, q))
            or re.search(r"\bfor how long\b", q)):
        return QueryIntent(intent=IntentType.DURATION, target_activity=acts[0] if acts else None)

    # 4. Onset -- when something began. "when did/does/do", "at what point", "what time", or any
    # begin/start verb form.
    if (re.search(r"\bwhen\s+(did|does|do|was|were|is)\b", q) or "at what point" in q
            or "what time" in q or "onset" in q
            or re.search(r"\b(begin|begins|began|beginning|start|starts|started|starting|first)\b", q)):
        return QueryIntent(intent=IntentType.ONSET, target_activity=acts[0] if acts else None)

    # 5. Open World keywords. Keep any named activity too: "did she lie down for a prolonged
    # period" must be answered about lying down specifically, not about sedentary time in general,
    # or a long sitting stretch would wrongly satisfy it.
    target = acts[0] if acts else None
    rest_cue = ("prolonged" in q or "rest" in q or "inactivity" in q or "inactive" in q
                or "motionless" in q or "extended period" in q or "long stretch" in q
                or "sedentary" in q or "immobile" in q)
    # Indirect references only. A question that names cycling directly ("Was she cycling?") is a
    # plain verification; open-world is for describing the behaviour without naming the class.
    wheel_cue = "wheel" in q or "pedal" in q or "transport" in q
    strenuous_cue = ("strenuous" in q or "vigorous" in q or "intense" in q or "exertion" in q
                     or "exercise" in q or "exerting" in q or "physically demanding" in q)

    if rest_cue:
        # "rest"/"resting" is generic sedentary behaviour and must not be narrowed to lying down
        # (the synonym table maps "resting" -> lying_down, which would miss a long sitting
        # stretch). Only an explicitly named posture narrows the question.
        posture = ("lie", "lying", "lying down", "lay", "sleep", "sleeping", "sit", "sitting", "seated")
        named_posture = any(re.search(rf"\b{re.escape(t)}\b", q) for t in posture)
        return QueryIntent(intent=IntentType.OPEN_WORLD, semantic_concept="prolonged_rest",
                           target_activity=target if named_posture else None)
    if wheel_cue:
        return QueryIntent(intent=IntentType.OPEN_WORLD, semantic_concept="wheeled_movement",
                           target_activity=target)
    if strenuous_cue:
        return QueryIntent(intent=IntentType.OPEN_WORLD, semantic_concept="strenuous_activity",
                           target_activity=target)

    # 6. Verify -- a yes/no question naming an activity. Subjects vary ("she", "he", a name,
    # "the grandmother"), and the request is often indirect ("Can you confirm...", "Is there any
    # evidence of..."), so the auxiliary-verb opening is only one of several accepted forms.
    if acts and (re.match(r"^(is|was|were|did|does|do|has|had|are|can|could|would|any)\b", q)
                 or "confirm" in q or "any evidence" in q or "any sign" in q
                 or "is there" in q or "was there" in q or "detected" in q):
        return QueryIntent(intent=IntentType.VERIFY, target_activity=acts[0])

    # 7. Identify -- an open "what were they doing" question, in any of its usual forms.
    if (re.search(r"\bwhat\s+(activity|activities)\b", q)
            or re.search(r"\bwhat\s+(is|was|were|are)\b.*\bdoing\b", q)
            or re.search(r"\bwhich\s+activity\b", q)
            or re.search(r"\bwhat\b.*\b(up to|happening|going on)\b", q)
            or re.search(r"\bwhat\b.*\b(kind|type|sort)\s+of\s+(activity|movement|motion)\b", q)
            or re.search(r"\b(tell me|describe)\b.*\b(what|doing|activity)\b", q)):
        return QueryIntent(intent=IntentType.IDENTIFY)

    return None


def parse_intent(question: str, model_name: str = "qwen2.5:1.5b") -> QueryIntent:
    """Hybrid parser: runs fast rule-engine, falls back to Ollama with structured output."""
    rule_intent = parse_intent_fast_rules(question)
    if rule_intent is not None:
        return rule_intent

    # Fallback to LangChain + Ollama Qwen 2.5 1.5B
    chain = get_structured_llm(model_name)
    return chain.invoke({"question": question})
