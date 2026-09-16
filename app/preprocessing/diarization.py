"""
Module 2 (diarization): assigns anonymous speaker labels ("Speaker 1",
"Speaker 2", ...) to transcript segments.

Primary path: pyannote.audio speaker diarization pipeline (requires a
HuggingFace token with access to the gated pyannote models -- set
HUGGINGFACE_TOKEN in .env).

Fallback path (no token / model unavailable): a simple energy+pause-based
heuristic that alternates speaker turns on sufficiently long silences. This
is clearly weaker and is labeled as such in the returned `method` field --
per Module 2's instruction to clearly label the limitation rather than
pretend it's real diarization.

We never fabricate or guess at real personal identities (Module 13):
speakers are always anonymous indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.config import settings
from app.preprocessing.speech import TranscriptSegment
from app.utils.logging import get_logger

logger = get_logger("preprocessing.diarization", settings.LOG_DIR)


@dataclass
class DiarizationResult:
    segments: List[TranscriptSegment]
    method: str  # "pyannote" | "heuristic_fallback"
    num_speakers: int


def diarize(audio_path: str | Path, transcript_segments: List[TranscriptSegment]) -> DiarizationResult:
    if settings.HUGGINGFACE_TOKEN:
        try:
            return _diarize_pyannote(audio_path, transcript_segments)
        except Exception as exc:
            logger.warning("pyannote diarization failed (%s); falling back to heuristic.", exc)
    else:
        logger.warning(
            "No HUGGINGFACE_TOKEN set -- using heuristic speaker-turn fallback. "
            "This is NOT true diarization; set HUGGINGFACE_TOKEN for pyannote.audio."
        )
    return _diarize_heuristic(transcript_segments)


def _diarize_pyannote(audio_path: str | Path, transcript_segments: List[TranscriptSegment]) -> DiarizationResult:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=settings.HUGGINGFACE_TOKEN
    )
    diarization = pipeline(str(audio_path))

    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    speaker_map = {}

    for seg in transcript_segments:
        mid = (seg.start_sec + seg.end_sec) / 2
        best_speaker = None
        for start, end, spk in turns:
            if start <= mid <= end:
                best_speaker = spk
                break
        if best_speaker is None:
            best_speaker = "unknown"
        if best_speaker not in speaker_map:
            speaker_map[best_speaker] = f"Speaker {len(speaker_map) + 1}"
        seg.speaker = speaker_map[best_speaker]

    logger.info("pyannote diarization found %d speaker(s).", len(speaker_map))
    return DiarizationResult(transcript_segments, method="pyannote", num_speakers=len(speaker_map))


def _diarize_heuristic(transcript_segments: List[TranscriptSegment], pause_threshold_sec: float = 1.2) -> DiarizationResult:
    """
    Alternates speaker label whenever the gap between consecutive utterances
    exceeds `pause_threshold_sec`. This is a coarse proxy for turn-taking and
    will be wrong whenever one speaker pauses mid-turn or two speakers
    respond quickly -- it exists only so the pipeline degrades gracefully,
    not as a diarization substitute.
    """
    current_speaker_idx = 1
    prev_end = None
    for seg in transcript_segments:
        if prev_end is not None and (seg.start_sec - prev_end) > pause_threshold_sec:
            current_speaker_idx = 2 if current_speaker_idx == 1 else 1
        seg.speaker = f"Speaker {current_speaker_idx}"
        prev_end = seg.end_sec

    num_speakers = len({s.speaker for s in transcript_segments})
    return DiarizationResult(transcript_segments, method="heuristic_fallback", num_speakers=num_speakers)
