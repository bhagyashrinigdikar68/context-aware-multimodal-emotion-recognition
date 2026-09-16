"""
Unit tests for preprocessing modules. These focus on pure-logic pieces
(validation, feature vector shapes, tracking geometry) that don't require a
real video file, GPU, or downloaded model weights, so they run fast and
offline in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.preprocessing.audio import AudioSegmentFeatures, extract_segment_features
from app.preprocessing.face import FaceBox, track_faces
from app.preprocessing.video import validate_video_file


def test_validate_video_file_missing():
    with pytest.raises(FileNotFoundError):
        validate_video_file("/nonexistent/path/video.mp4")


def test_validate_video_file_bad_extension(tmp_path):
    bad_file = tmp_path / "clip.txt"
    bad_file.write_text("not a video")
    with pytest.raises(ValueError):
        validate_video_file(bad_file)


def test_audio_segment_features_missing_on_short_input():
    tiny_waveform = np.zeros(10, dtype=np.float32)  # far shorter than 100ms at 16kHz
    feats = extract_segment_features(tiny_waveform, sr=16000)
    assert feats.valid is False
    assert feats.to_vector().shape[0] == AudioSegmentFeatures.vector_dim()


def test_audio_segment_features_missing_on_empty():
    feats = extract_segment_features(np.array([]), sr=16000)
    assert feats.valid is False


def test_face_box_iou_identical():
    box = FaceBox(0, 0, 10, 10, 0.9)
    assert box.iou(box) == pytest.approx(1.0)


def test_face_box_iou_disjoint():
    a = FaceBox(0, 0, 10, 10, 0.9)
    b = FaceBox(100, 100, 10, 10, 0.9)
    assert a.iou(b) == 0.0


def test_track_faces_single_persistent_face():
    frame_boxes = [
        [FaceBox(10, 10, 50, 50, 0.9)],
        [FaceBox(12, 11, 50, 50, 0.9)],
        [FaceBox(14, 13, 50, 50, 0.9)],
    ]
    tracks = track_faces(frame_boxes)
    assert len(tracks) == 1
    assert len(tracks[0].boxes) == 3
    assert all(b is not None for b in tracks[0].boxes)


def test_track_faces_handles_missing_detection_gracefully():
    frame_boxes = [
        [FaceBox(10, 10, 50, 50, 0.9)],
        [],  # face temporarily undetected
        [FaceBox(15, 14, 50, 50, 0.9)],
    ]
    tracks = track_faces(frame_boxes)
    assert len(tracks) >= 1
    # should not raise, and total tracked frames should equal number of frames processed
    for t in tracks:
        assert len(t.boxes) == 3


def test_track_faces_two_distinct_speakers():
    frame_boxes = [
        [FaceBox(0, 0, 20, 20, 0.9), FaceBox(200, 200, 20, 20, 0.9)],
        [FaceBox(1, 1, 20, 20, 0.9), FaceBox(201, 201, 20, 20, 0.9)],
    ]
    tracks = track_faces(frame_boxes)
    assert len(tracks) == 2
