"""
Pydantic schemas for the query layer (Person B).
Guarantees validated, typed structured outputs for query parsing and final answers.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    IDENTIFY = "identify"          # e.g., "What activity is the user performing?"
    VERIFY = "verify"              # e.g., "Is the user running?"
    DURATION = "duration"          # e.g., "How long was the user walking?"
    COUNT = "count"                # e.g., "How many times did the user walk?"
    ONSET = "onset"                # e.g., "When did the user begin running?"
    COMPARE = "compare"            # e.g., "Did the user spend more time walking or running?"
    OPEN_WORLD = "open_world"      # e.g., "Was the user using a wheeled mode of movement?"


class CanonicalActivity(str, Enum):
    LYING_DOWN = "lying_down"
    SITTING = "sitting"
    STANDING_IN_PLACE = "standing_in_place"
    STANDING_AND_MOVING = "standing_and_moving"
    WALKING = "walking"
    RUNNING = "running"
    BICYCLING = "bicycling"


class QueryIntent(BaseModel):
    """Structured representation of the user query parsed by LangChain + Ollama."""
    intent: IntentType = Field(
        description="The primary category/task of the question."
    )
    target_activity: Optional[str] = Field(
        default=None,
        description="The primary target activity mapped to one of the 7 canonical classes: "
                    "lying_down, sitting, standing_in_place, standing_and_moving, walking, running, bicycling. "
                    "None if open-world or generic identification."
    )
    compare_activity: Optional[str] = Field(
        default=None,
        description="The second activity if the intent is COMPARE (e.g., comparing walking vs running)."
    )
    semantic_concept: Optional[str] = Field(
        default=None,
        description="For OPEN_WORLD queries, semantic behavior description: e.g., "
                    "'prolonged_rest', 'wheeled_movement', 'strenuous_activity'."
    )
    time_window_start: Optional[float] = Field(
        default=None,
        description="Start time offset in seconds if question specifies a temporal window (e.g., noon)."
    )
    time_window_end: Optional[float] = Field(
        default=None,
        description="End time offset in seconds if question specifies a temporal window."
    )


class FormattedAnswerBlock(BaseModel):
    """The exact 6-field structured block required by the challenge specification."""
    answer: str = Field(description="Direct answer to the query, or N/A")
    activity_event: str = Field(description="Activity or event, or N/A")
    timestamps: str = Field(description="Time range or ranges (seconds from start), or N/A")
    sensor_modality: str = Field(description="Accelerometer, Gyroscope, Both, or N/A")
    sensor_channels: str = Field(description="Acc X/Y/Z, Gyro X/Y/Z, All, or N/A")
    explanation: str = Field(description="Reasoning grounded in the observed signal, or N/A")

    def to_output_string(self) -> str:
        return (
            f"Answer: {self.answer}\n"
            f"Activity/Event: {self.activity_event}\n"
            f"Evidence:\n"
            f"  Timestamp(s): {self.timestamps}\n"
            f"  Sensor Modality: {self.sensor_modality}\n"
            f"  Sensor Channel(s): {self.sensor_channels}\n"
            f"Explanation: {self.explanation}"
        )
