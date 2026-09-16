"""
Module 5 (dataset prep): validates the MELD raw download and converts it
into a unified per-utterance JSONL manifest (one file per split), including
per-utterance audio extraction and conversation grouping needed for the
context window (Module 6).

Usage:
    python dataset/prepare_meld.py --raw_dir dataset/meld/raw --out_dir dataset/meld/processed

Refuses to run if the raw dataset is missing/incomplete (Module 5's
requirement to validate before training) rather than silently proceeding.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from app.utils.logging import get_logger

logger = get_logger("dataset.prepare_meld")

SPLIT_CONFIG = {
    "train": {"csv": "train_sent_emo.csv", "video_dir": "train_splits"},
    "dev": {"csv": "dev_sent_emo.csv", "video_dir": "dev_splits_complete"},
    "test": {"csv": "test_sent_emo.csv", "video_dir": "output_repeated_splits_test"},
}

# MELD's raw label strings map directly onto settings.EMOTION_LABELS already,
# but we keep an explicit mapping so a mismatch is loud, not silent.
MELD_LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}


def validate_raw_dataset(raw_dir: Path) -> None:
    missing = []
    for split, cfg in SPLIT_CONFIG.items():
        csv_path = raw_dir / split / cfg["csv"]
        video_dir = raw_dir / split / cfg["video_dir"]
        if not csv_path.exists():
            missing.append(str(csv_path))
        if not video_dir.exists():
            missing.append(str(video_dir))
    if missing:
        raise FileNotFoundError(
            "MELD raw dataset is incomplete. Missing:\n  " + "\n  ".join(missing) +
            "\n\nSee dataset/README.md for download instructions. Refusing to "
            "proceed rather than training on a partial/corrupt dataset."
        )


def _clip_filename(row: pd.Series) -> str:
    return f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"


def _extract_utterance_audio(video_path: Path, out_wav: Path) -> bool:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out_wav)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def prepare_split(raw_dir: Path, out_dir: Path, split: str, extract_audio: bool = True) -> dict:
    cfg = SPLIT_CONFIG[split]
    csv_path = raw_dir / split / cfg["csv"]
    video_dir = raw_dir / split / cfg["video_dir"]

    df = pd.read_csv(csv_path)
    label_counts: dict[str, int] = {}
    manifest_path = out_dir / f"{split}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    n_written, n_skipped_missing_video, n_skipped_bad_label = 0, 0, 0

    with open(manifest_path, "w") as f:
        for _, row in df.iterrows():
            raw_label = str(row["Emotion"]).strip().lower()
            if raw_label not in MELD_LABEL_MAP:
                n_skipped_bad_label += 1
                continue

            clip_name = _clip_filename(row)
            video_path = video_dir / clip_name
            if not video_path.exists():
                n_skipped_missing_video += 1
                continue

            audio_path = out_dir / "audio" / split / clip_name.replace(".mp4", ".wav")
            if extract_audio and not audio_path.exists():
                ok = _extract_utterance_audio(video_path, audio_path)
                if not ok:
                    logger.warning("ffmpeg failed on %s; leaving audio_path unset for this utterance.", video_path)
                    audio_path = None

            record = {
                "dialogue_id": str(row["Dialogue_ID"]),
                "utterance_id": str(row["Utterance_ID"]),
                "speaker": f"Speaker {row['Speaker']}" if not str(row["Speaker"]).lower().startswith("speaker") else str(row["Speaker"]),
                "text": str(row["Utterance"]),
                "emotion": MELD_LABEL_MAP[raw_label],
                "video_path": str(video_path),
                "audio_path": str(audio_path) if audio_path else None,
                "start_sec": float(row.get("StartTime", 0) or 0),
                "end_sec": float(row.get("EndTime", 0) or 0),
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1
            label_counts[record["emotion"]] = label_counts.get(record["emotion"], 0) + 1

    logger.info(
        "[%s] wrote %d utterances (skipped: %d missing video, %d bad label). Manifest: %s",
        split, n_written, n_skipped_missing_video, n_skipped_bad_label, manifest_path,
    )
    return {"split": split, "n_written": n_written, "label_counts": label_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=Path("dataset/meld/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("dataset/meld/processed"))
    parser.add_argument("--no_audio", action="store_true", help="Skip per-utterance audio extraction (faster, text-only prep).")
    args = parser.parse_args()

    validate_raw_dataset(args.raw_dir)

    summary = {}
    for split in SPLIT_CONFIG:
        summary[split] = prepare_split(args.raw_dir, args.out_dir, split, extract_audio=not args.no_audio)

    print("\n=== MELD preparation summary ===")
    for split, info in summary.items():
        print(f"{split}: {info['n_written']} utterances")
        for label, count in sorted(info["label_counts"].items(), key=lambda x: -x[1]):
            print(f"    {label:10s} {count}")


if __name__ == "__main__":
    main()
