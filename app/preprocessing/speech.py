"""
Module 2: Speech-to-text using OpenAI Whisper, producing a timestamped
transcript. Speaker labels are attached later by `diarization.py`; this
module is purely ASR.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger("preprocessing.speech", settings.LOG_DIR)

_whisper_model = None  # lazy-loaded, cached across calls


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    speaker: Optional[str] = None  # filled in by diarization, else None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper

        logger.info("Loading Whisper model '%s' on %s ...", settings.WHISPER_MODEL_SIZE, settings.DEVICE)
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE, device=settings.DEVICE)
    return _whisper_model


def transcribe(audio_path: str | Path) -> List[TranscriptSegment]:
    """
    Run Whisper on the given WAV/audio file and return timestamped segments.

    Raises a clear error (rather than silently producing garbage) if the
    audio file is missing or empty, so the caller can mark the text modality
    unavailable for that utterance instead of feeding junk downstream.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(f"Audio file missing or empty: {audio_path}")

    model = _get_whisper_model()
    result = model.transcribe(str(audio_path), verbose=False)

    segments = [
        TranscriptSegment(start_sec=seg["start"], end_sec=seg["end"], text=seg["text"].strip())
        for seg in result.get("segments", [])
    ]
    logger.info("Transcribed %d segment(s) from %s", len(segments), audio_path.name)
    return segments


def format_transcript(segments: List[TranscriptSegment]) -> str:
    """Human-readable transcript matching the format shown in the spec."""
    lines = []
    for seg in segments:
        speaker = seg.speaker or "Unknown Speaker"
        lines.append(f"{_fmt_ts(seg.start_sec)} - {_fmt_ts(seg.end_sec)}\n{speaker}:\n\"{seg.text}\"\n")
    return "\n".join(lines)


def _fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"
