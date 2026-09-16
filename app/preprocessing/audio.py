"""
Module 4 support: acoustic/prosodic feature extraction with librosa.

Produces a fixed-size handcrafted feature vector per utterance (MFCC stats,
pitch, energy, speaking rate proxy, spectral features) that feeds the audio
encoder in `app/models/audio_encoder.py`. Handcrafted features are used as
the *input* representation; the encoder itself is a learned neural network
(Module 10 explicitly asks not to depend only on manual features for the
final representation).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger("preprocessing.audio", settings.LOG_DIR)

SAMPLE_RATE = 16000
N_MFCC = 40


@dataclass
class AudioSegmentFeatures:
    mfcc_mean: np.ndarray      # (N_MFCC,)
    mfcc_std: np.ndarray       # (N_MFCC,)
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    speaking_rate_proxy: float  # onsets per second, correlates with speech rate
    duration_sec: float
    valid: bool = True          # False => modality unavailable for this segment

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.mfcc_mean,
                self.mfcc_std,
                np.array(
                    [
                        self.pitch_mean,
                        self.pitch_std,
                        self.energy_mean,
                        self.energy_std,
                        self.zero_crossing_rate,
                        self.spectral_centroid,
                        self.spectral_bandwidth,
                        self.speaking_rate_proxy,
                    ]
                ),
            ]
        ).astype(np.float32)

    @staticmethod
    def vector_dim() -> int:
        return N_MFCC * 2 + 8

    @staticmethod
    def missing() -> "AudioSegmentFeatures":
        z = np.zeros(N_MFCC, dtype=np.float32)
        return AudioSegmentFeatures(z, z, 0, 0, 0, 0, 0, 0, 0, 0, 0, valid=False)


def load_waveform(path: str | Path, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    import librosa

    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y
    except Exception as exc:
        logger.warning("Could not load audio %s: %s", path, exc)
        return None


def extract_segment_features(
    waveform: np.ndarray, sr: int = SAMPLE_RATE, start_sec: float = 0.0, end_sec: Optional[float] = None
) -> AudioSegmentFeatures:
    """Extract prosodic/spectral features for the [start_sec, end_sec] slice."""
    import librosa

    if waveform is None or len(waveform) == 0:
        return AudioSegmentFeatures.missing()

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr) if end_sec is not None else len(waveform)
    segment = waveform[start_sample:end_sample]

    min_len = int(0.1 * sr)  # need at least 100ms of audio to extract anything meaningful
    if len(segment) < min_len:
        logger.debug("Audio segment too short (%d samples); marking as missing.", len(segment))
        return AudioSegmentFeatures.missing()

    try:
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean, mfcc_std = mfcc.mean(axis=1), mfcc.std(axis=1)

        f0, voiced_flag, _ = librosa.pyin(
            segment, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.nanmean(voiced_f0)) if voiced_f0.size else 0.0
        pitch_std = float(np.nanstd(voiced_f0)) if voiced_f0.size else 0.0

        rms = librosa.feature.rms(y=segment)[0]
        energy_mean, energy_std = float(rms.mean()), float(rms.std())

        zcr = float(librosa.feature.zero_crossing_rate(segment).mean())
        centroid = float(librosa.feature.spectral_centroid(y=segment, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean())

        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration = len(segment) / sr
        speaking_rate = len(onsets) / duration if duration > 0 else 0.0

        return AudioSegmentFeatures(
            mfcc_mean=mfcc_mean.astype(np.float32),
            mfcc_std=mfcc_std.astype(np.float32),
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            energy_std=energy_std,
            zero_crossing_rate=zcr,
            spectral_centroid=centroid,
            spectral_bandwidth=bandwidth,
            speaking_rate_proxy=speaking_rate,
            duration_sec=duration,
            valid=True,
        )
    except Exception as exc:
        logger.warning("Audio feature extraction failed for segment: %s", exc)
        return AudioSegmentFeatures.missing()
