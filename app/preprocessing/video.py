"""
Module 1 (part A): Video preprocessing.

Reads a video file, extracts frames at a configurable FPS, and reports basic
metadata. Face detection/tracking lives in `face.py`; this module only deals
with the raw frame stream so it stays reusable (e.g. for a future non-face
visual signal).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger("preprocessing.video", settings.LOG_DIR)

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi"}


@dataclass
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image: np.ndarray  # BGR, as returned by OpenCV


def validate_video_file(path: str | Path) -> Path:
    """Raise a clear, actionable error if the input file is unusable."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video extension '{p.suffix}'. Supported: {sorted(SUPPORTED_VIDEO_EXTS)}"
        )
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV could not open video file (possibly corrupt): {p}")
    cap.release()
    return p


def get_video_metadata(path: str | Path) -> VideoMetadata:
    p = validate_video_file(path)
    cap = cv2.VideoCapture(str(p))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(p, fps, frame_count, duration, width, height)
    finally:
        cap.release()


def extract_frames(
    path: str | Path,
    target_fps: int | None = None,
) -> Iterator[Frame]:
    """
    Yield frames sampled at `target_fps` (defaults to settings.FRAME_RATE).

    Streams frames one at a time rather than loading the whole video into
    memory, so long conversations don't blow up RAM on a student laptop.
    """
    target_fps = target_fps or settings.FRAME_RATE
    meta = get_video_metadata(path)
    cap = cv2.VideoCapture(str(meta.path))

    if meta.fps <= 0:
        logger.warning("Video reports FPS<=0, falling back to reading every frame.")
        sample_every = 1
        source_fps = target_fps
    else:
        sample_every = max(1, round(meta.fps / target_fps))
        source_fps = meta.fps

    try:
        frame_idx = 0
        yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                timestamp = frame_idx / source_fps
                yield Frame(index=yielded, timestamp_sec=timestamp, image=frame)
                yielded += 1
            frame_idx += 1
    finally:
        cap.release()
    logger.info("Extracted %d frames from %s at ~%d fps", frame_idx, meta.path.name, target_fps)


def extract_audio_track(video_path: str | Path, output_wav: str | Path) -> Path:
    """
    Extract the audio track from a video into a mono 16kHz WAV file.

    Uses moviepy-free ffmpeg-less extraction via OpenCV is not possible for
    audio, so we shell out to ffmpeg (a standard dependency of most video
    pipelines and of `openai-whisper`).
    """
    import subprocess

    video_path = validate_video_file(video_path)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path}:\n{result.stderr[-2000:]}"
        )
    logger.info("Extracted audio track to %s", output_wav)
    return output_wav
