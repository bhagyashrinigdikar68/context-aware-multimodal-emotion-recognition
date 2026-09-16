from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import ConversationAnalysisSchema, HealthSchema
from app.config import settings
from app.pipeline.inference import EmotionRecognitionPipeline
from app.preprocessing.video import SUPPORTED_VIDEO_EXTS
from app.utils.logging import get_logger

logger = get_logger("api.routes", settings.LOG_DIR)
router = APIRouter()

_pipeline: EmotionRecognitionPipeline | None = None


def get_pipeline() -> EmotionRecognitionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EmotionRecognitionPipeline()
    return _pipeline


@router.get("/health", response_model=HealthSchema)
def health() -> HealthSchema:
    pipeline = get_pipeline()
    return HealthSchema(status="ok", device=settings.DEVICE, demo_mode=pipeline.is_demo_mode)


@router.post("/analyze", response_model=ConversationAnalysisSchema)
async def analyze_conversation(file: UploadFile = File(...)) -> ConversationAnalysisSchema:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTS | {".wav", ".mp3"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    tmp_dir = Path(tempfile.mkdtemp(dir=settings.TMP_DIR))
    try:
        upload_path = tmp_dir / file.filename
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        pipeline = get_pipeline()
        try:
            result = pipeline.analyze_video(upload_path)
        except Exception as exc:
            logger.exception("Pipeline failed on %s", file.filename)
            raise HTTPException(status_code=422, detail=f"Analysis failed: {exc}") from exc

        emotion_counts = Counter(u.emotion for u in result.utterances)
        dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"

        return ConversationAnalysisSchema(
            utterances=[u.__dict__ for u in result.utterances],
            diarization_method=result.diarization_method,
            num_speakers=result.num_speakers,
            duration_sec=result.duration_sec,
            is_demo_mode=result.is_demo_mode,
            dominant_emotion=dominant,
            emotion_distribution=dict(emotion_counts),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
