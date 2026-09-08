"""
Main entry point for Person B (Query Layer).
Called by run.py:
    def answer(question: str, timeline: Timeline) -> str:
"""
from __future__ import annotations
from src.query.intent import parse_intent
from src.query.operations import execute_query
from src.query.schema import FormattedAnswerBlock


def answer(question: str, timeline) -> str:
    """
    Given a natural language question and a Timeline instance,
    parse the intent with LangChain/Ollama and execute deterministic evidence
    extraction to produce the exact 6-field output block.
    """
    # Stage 1: Hybrid intent & parameter extraction (Pydantic validated)
    intent = parse_intent(question)

    # Stage 2: Deterministic timeline execution & grounded evidence extraction
    answer_block: FormattedAnswerBlock = execute_query(intent, timeline)

    # Format into standard challenge text output
    return answer_block.to_output_string()
