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

    # 1. Compare
    if "more time" in q or "compare" in q or (" or " in q and len(acts) >= 2):
        if len(acts) >= 2:
            return QueryIntent(intent=IntentType.COMPARE, target_activity=acts[0], compare_activity=acts[1])

    # 2. Duration
    if q.startswith("how long") or "duration" in q or "time spent" in q:
        target = acts[0] if acts else None
        return QueryIntent(intent=IntentType.DURATION, target_activity=target)

    # 3. Count
    if q.startswith("how many times") or "how often" in q or "count" in q or "number of episodes" in q:
        target = acts[0] if acts else None
        return QueryIntent(intent=IntentType.COUNT, target_activity=target)

    # 4. Onset
    if q.startswith("when did") or "onset" in q or "start time" in q or "begin" in q:
        target = acts[0] if acts else None
        return QueryIntent(intent=IntentType.ONSET, target_activity=target)

    # 5. Open World keywords
    if "prolonged" in q or "rest" in q:
        return QueryIntent(intent=IntentType.OPEN_WORLD, semantic_concept="prolonged_rest")
    if "wheeled" in q or "pedal" in q:
        return QueryIntent(intent=IntentType.OPEN_WORLD, semantic_concept="wheeled_movement")
    if "strenuous" in q:
        return QueryIntent(intent=IntentType.OPEN_WORLD, semantic_concept="strenuous_activity")

    # 6. Verify -- any yes/no-style question ("is/was/did/does/were/has ...") naming an
    # activity, not just the literal phrase "the user" (real subjects vary: "she", "he",
    # a name, "the grandmother", etc.)
    if re.match(r"^(is|was|were|did|does|do|has|had|are)\b", q) and acts:
        return QueryIntent(intent=IntentType.VERIFY, target_activity=acts[0])

    # 7. Identify
    if "what activity" in q or "what is the user doing" in q:
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
