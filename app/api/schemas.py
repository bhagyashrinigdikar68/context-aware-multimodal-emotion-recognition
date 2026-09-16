from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ModalityAvailabilitySchema(BaseModel):
    text: bool
    audio: bool
    visual: bool
    context: bool


class UtteranceSchema(BaseModel):
    speaker: str
    start_sec: float
    end_sec: float
    text: str
    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    distribution: Dict[str, float]
    modalities_used: ModalityAvailabilitySchema
    attention_weights: Dict[str, float]


class ConversationAnalysisSchema(BaseModel):
    utterances: List[UtteranceSchema]
    diarization_method: str
    num_speakers: int
    duration_sec: float
    is_demo_mode: bool
    dominant_emotion: str
    emotion_distribution: Dict[str, int]


class HealthSchema(BaseModel):
    status: str
    device: str
    demo_mode: bool
