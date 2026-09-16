"""
Module: end-to-end inference pipeline.

    video/audio file
        -> extract audio, extract frames
        -> ASR (Whisper) -> timestamped transcript
        -> diarization -> anonymous speaker labels
        -> face detection/tracking -> per-utterance face crop (or None)
        -> per-utterance audio features
        -> MultimodalEmotionModel (with context window) -> emotion + confidence + distribution
        -> structured, JSON-serializable result consumed by the API and the dashboard

This module is the only place that orchestrates every preprocessing +
modeling module together; it is intentionally the sole "glue" layer so
each individual module stays independently testable.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from app.config import settings
from app.models.multimodal_model import MultimodalEmotionModel
from app.models.speaker import SpeakerEmbedding
from app.preprocessing import audio as audio_prep
from app.preprocessing import diarization as diarization_prep
from app.preprocessing import face as face_prep
from app.preprocessing import speech as speech_prep
from app.preprocessing import video as video_prep
from app.utils.logging import get_logger
from app.utils.modality_mask import ModalityAvailability, build_mask_tensor

logger = get_logger("pipeline.inference", settings.LOG_DIR)


@dataclass
class UtteranceResult:
    speaker: str
    start_sec: float
    end_sec: float
    text: str
    emotion: str
    confidence: float
    distribution: dict
    modalities_used: dict
    attention_weights: dict


@dataclass
class ConversationResult:
    utterances: List[UtteranceResult] = field(default_factory=list)
    diarization_method: str = "unknown"
    num_speakers: int = 0
    duration_sec: float = 0.0
    is_demo_mode: bool = False


class EmotionRecognitionPipeline:
    """
    Loads (or lazily initializes, in demo mode) the multimodal model and runs
    full conversation analysis.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.device = settings.DEVICE
        self.model = MultimodalEmotionModel().to(self.device)
        self.is_demo_mode = True

        checkpoint_path = checkpoint_path or (settings.MODEL_DIR / "multimodal_best.pt")
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.is_demo_mode = False
            logger.info("Loaded trained checkpoint from %s", checkpoint_path)
        else:
            if not settings.ALLOW_DEMO_MODE:
                raise FileNotFoundError(
                    f"No trained checkpoint found at {checkpoint_path} and ALLOW_DEMO_MODE=false. "
                    "Train a model first (see training/train_multimodal.py) or enable demo mode."
                )
            logger.warning(
                "!!! No trained checkpoint found at %s. Running in DEMO MODE with "
                "RANDOMLY-INITIALIZED weights. Predictions are NOT meaningful and must "
                "never be reported as research results -- this mode exists only to "
                "exercise the UI/pipeline end-to-end. !!!",
                checkpoint_path,
            )

        self.model.eval()
        self.face_detector = face_prep.FaceDetector()

    def analyze_video(self, video_path: str | Path) -> ConversationResult:
        video_path = video_prep.validate_video_file(video_path)
        tmp_dir = Path(tempfile.mkdtemp(dir=settings.TMP_DIR))
        try:
            return self._run(video_path, tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)  # always clean temp files (Module 7)

    def _run(self, video_path: Path, tmp_dir: Path) -> ConversationResult:
        meta = video_prep.get_video_metadata(video_path)
        wav_path = video_prep.extract_audio_track(video_path, tmp_dir / "audio.wav")

        try:
            segments = speech_prep.transcribe(wav_path)
        except Exception as exc:
            logger.error("Transcription failed: %s. Proceeding with empty transcript.", exc)
            segments = []

        diar = diarization_prep.diarize(wav_path, segments)

        waveform = audio_prep.load_waveform(wav_path)

        # face tracking across sampled frames, associated to nearest utterance by timestamp
        frames = list(video_prep.extract_frames(video_path))
        frame_boxes = [self.face_detector.detect(f.image) for f in frames]
        tracks = face_prep.track_faces(frame_boxes)

        results: List[UtteranceResult] = []
        history_buffer: List[dict] = []  # rolling window for context

        for seg in diar.segments:
            availability = ModalityAvailability()

            # ---- text ----
            text = seg.text.strip()
            if not text:
                availability.text = False

            # ---- audio ----
            audio_feat = audio_prep.extract_segment_features(waveform, start_sec=seg.start_sec, end_sec=seg.end_sec)
            if not audio_feat.valid:
                availability.audio = False

            # ---- visual: nearest frame within the utterance window with a tracked face ----
            face_image = self._select_face_for_segment(frames, tracks, seg.start_sec, seg.end_sec)
            if face_image is None:
                availability.visual = False

            speaker_idx = SpeakerEmbedding.parse_speaker_label(seg.speaker or "Speaker 1")

            current = {
                "texts": [text] if availability.text else [""],
                "audio_features": torch.tensor(audio_feat.to_vector()).unsqueeze(0).to(self.device),
                "face_images": self._face_to_tensor(face_image).to(self.device) if face_image is not None else None,
                "speaker_indices": torch.tensor([speaker_idx]).to(self.device),
            }

            history = self._build_history_tensor(history_buffer, speaker_idx, current)
            mask = build_mask_tensor([availability], device=self.device)

            with torch.no_grad():
                output = self.model(current, history=history, modality_mask=mask)

            from app.models.emotion_classifier import EmotionClassifier

            pred_idx, confidence, probs = EmotionClassifier.predict(output["logits"])
            emotion = settings.index_to_label(int(pred_idx.item()))
            distribution = {
                settings.index_to_label(i): float(probs[0, i]) for i in range(settings.num_classes)
            }
            attn = {k: float(v.item()) for k, v in output["attention_weights"].items()}

            results.append(
                UtteranceResult(
                    speaker=seg.speaker or "Unknown",
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    text=text,
                    emotion=emotion,
                    confidence=float(confidence.item()),
                    distribution=distribution,
                    modalities_used=availability.as_dict(),
                    attention_weights=attn,
                )
            )

            history_buffer.append(current)
            if len(history_buffer) > settings.CONTEXT_WINDOW:
                history_buffer.pop(0)

        return ConversationResult(
            utterances=results,
            diarization_method=diar.method,
            num_speakers=diar.num_speakers,
            duration_sec=meta.duration_sec,
            is_demo_mode=self.is_demo_mode,
        )

    # ---- helpers ----

    def _select_face_for_segment(self, frames, tracks, start_sec: float, end_sec: float):
        best_box, best_frame = None, None
        for track in tracks:
            for i, box in enumerate(track.boxes):
                if box is None or i >= len(frames):
                    continue
                ts = frames[i].timestamp_sec
                if start_sec <= ts <= end_sec:
                    best_box, best_frame = box, frames[i]
                    break
            if best_box is not None:
                break
        if best_box is None or best_frame is None:
            return None
        try:
            return face_prep.crop_and_align_face(best_frame.image, best_box)
        except ValueError:
            return None

    def _face_to_tensor(self, face_bgr: np.ndarray) -> torch.Tensor:
        import cv2

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _build_history_tensor(self, history_buffer: List[dict], current_speaker_idx: int, current: dict) -> dict:
        """Stack the rolling history buffer + current utterance into a (B=1, T, ...) history dict."""
        window = history_buffer + [current]
        T = len(window)

        texts_seq = [[step["texts"][0] for step in window]]
        audio_seq = torch.stack([step["audio_features"][0] for step in window]).unsqueeze(0)

        face_list = []
        for step in window:
            if step["face_images"] is not None:
                face_list.append(step["face_images"][0])
            else:
                face_list.append(torch.zeros(3, 224, 224))
        face_seq = torch.stack(face_list).unsqueeze(0)

        speaker_seq = torch.stack([step["speaker_indices"][0] for step in window]).unsqueeze(0)

        return {
            "texts": texts_seq,
            "audio_features": audio_seq.to(self.device),
            "face_images": face_seq.to(self.device),
            "speaker_indices": speaker_seq.to(self.device),
            "padding_mask": None,
        }
