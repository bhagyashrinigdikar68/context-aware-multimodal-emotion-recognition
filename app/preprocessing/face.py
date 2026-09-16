"""
Module 1 (part B): Face detection, cropping, and lightweight tracking.

Uses MediaPipe Face Detection for per-frame detection and a simple
IoU-based tracker to associate detections into per-speaker face tracks
across frames (good enough for short conversational clips; swap in a
stronger tracker/re-ID model for production use).

Design goal: never crash when a face is missing in a frame -- callers get
`None` for that frame's face and the pipeline downstream treats visual
modality as unavailable for that instant, per the missing-modality
requirements (Module 15).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger("preprocessing.face", settings.LOG_DIR)


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    def iou(self, other: "FaceBox") -> float:
        ax1, ay1, ax2, ay2 = self.x, self.y, self.x + self.w, self.y + self.h
        bx1, by1, bx2, by2 = other.x, other.y, other.x + other.w, other.y + other.h
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


@dataclass
class FaceTrack:
    track_id: int
    boxes: List[Optional[FaceBox]] = field(default_factory=list)  # index-aligned with frames

    def last_known_box(self) -> Optional[FaceBox]:
        for b in reversed(self.boxes):
            if b is not None:
                return b
        return None


class FaceDetector:
    """Thin wrapper around MediaPipe Face Detection with a graceful fallback."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self._backend = None
        try:
            import mediapipe as mp

            self._mp_face = mp.solutions.face_detection
            self._detector = self._mp_face.FaceDetection(
                model_selection=1, min_detection_confidence=min_detection_confidence
            )
            self._backend = "mediapipe"
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "MediaPipe unavailable (%s). Falling back to OpenCV Haar cascade "
                "for face detection (lower accuracy).", exc
            )
            import cv2

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            self._backend = "opencv"

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        h, w = image_bgr.shape[:2]
        if self._backend == "mediapipe":
            import cv2

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = self._detector.process(rgb)
            boxes: List[FaceBox] = []
            if result.detections:
                for det in result.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    bw = min(w - x, int(bbox.width * w))
                    bh = min(h - y, int(bbox.height * h))
                    if bw > 0 and bh > 0:
                        boxes.append(FaceBox(x, y, bw, bh, det.score[0] if det.score else 0.0))
            return boxes
        else:
            import cv2

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            detections = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            return [FaceBox(int(x), int(y), int(w_), int(h_), 1.0) for (x, y, w_, h_) in detections]


def crop_and_align_face(image_bgr: np.ndarray, box: FaceBox, out_size: int = 224) -> np.ndarray:
    """Crop the face region with a small margin and resize to a square."""
    import cv2

    h, w = image_bgr.shape[:2]
    margin = int(0.15 * max(box.w, box.h))
    x1 = max(0, box.x - margin)
    y1 = max(0, box.y - margin)
    x2 = min(w, box.x + box.w + margin)
    y2 = min(h, box.y + box.h + margin)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty face crop -- box fell outside frame bounds.")
    return cv2.resize(crop, (out_size, out_size))


def track_faces(frame_boxes: List[List[FaceBox]], iou_threshold: float = 0.3) -> List[FaceTrack]:
    """
    Very simple greedy IoU tracker: associates each frame's detections with
    the closest existing track by IoU, starting a new track when no match is
    found. Sufficient for short clips with a small, mostly-static number of
    speakers; swap for a proper re-ID/tracking model for noisy multi-camera
    footage.
    """
    tracks: List[FaceTrack] = []
    next_id = 0

    for frame_idx, boxes in enumerate(frame_boxes):
        # pad all existing tracks with None for this frame first
        for t in tracks:
            t.boxes.append(None)

        assigned = set()
        for box in boxes:
            best_track, best_iou = None, iou_threshold
            for t in tracks:
                last = t.last_known_box()
                if last is None or t.track_id in assigned:
                    continue
                iou = last.iou(box)
                if iou > best_iou:
                    best_track, best_iou = t, iou
            if best_track is not None:
                best_track.boxes[-1] = box
                assigned.add(best_track.track_id)
            else:
                new_track = FaceTrack(track_id=next_id)
                new_track.boxes = [None] * frame_idx + [box]
                tracks.append(new_track)
                assigned.add(next_id)
                next_id += 1

    logger.info("Tracked %d distinct face(s) across %d frames.", len(tracks), len(frame_boxes))
    return tracks
