# Dataset: MELD (Multimodal EmotionLines Dataset)

This project uses **MELD** as its primary research dataset. MELD is **not**
included in this repository (per the project's data-handling requirements) —
you must download it yourself.

## 1. Download

MELD is distributed by the original authors. Two common sources:

- Official repo & paper: https://github.com/declare-lab/MELD
- Raw + processed download (~10GB, video+audio+text): linked from the repo above
  (`MELD.Raw.tar.gz`)

```bash
mkdir -p dataset/meld/raw
cd dataset/meld/raw
# Follow the download link from https://github.com/declare-lab/MELD to obtain
# MELD.Raw.tar.gz, then:
tar -xzf MELD.Raw.tar.gz
```

## 2. Expected folder structure

After extraction, you should have:

```
dataset/meld/raw/
├── train/
│   ├── train_splits/            # .mp4 clips
│   └── train_sent_emo.csv       # utterance text + emotion + speaker + Dialogue_ID
├── dev/
│   ├── dev_splits_complete/
│   └── dev_sent_emo.csv
└── test/
    ├── output_repeated_splits_test/
    └── test_sent_emo.csv
```

The CSVs contain columns including: `Utterance`, `Speaker`, `Emotion`,
`Sentiment`, `Dialogue_ID`, `Utterance_ID`, `Season`, `Episode`, `StartTime`,
`EndTime`.

MELD's emotion label set (`neutral, joy, sadness, anger, surprise, fear,
disgust`) matches `settings.EMOTION_LABELS` by default — no relabeling
needed.

## 3. Prepare / validate

```bash
python dataset/prepare_meld.py --raw_dir dataset/meld/raw --out_dir dataset/meld/processed
```

This script:
- validates that all required files/folders exist before doing anything,
- extracts per-utterance audio clips (via ffmpeg) from each dialogue video,
- builds a unified `processed/{split}.jsonl` manifest with paths + labels + conversation grouping (for context windows),
- reports basic class-balance statistics.

It will **refuse to run** (with a clear error) if `dataset/meld/raw` is
missing or incomplete, rather than silently training on nothing.

## 4. Using a different conversational emotion dataset

The pipeline only requires a JSONL manifest with these fields per utterance:

```json
{
  "dialogue_id": "0",
  "utterance_id": "3",
  "speaker": "Speaker 2",
  "text": "...",
  "emotion": "sadness",
  "video_path": "path/to/clip.mp4",
  "start_sec": 12.4,
  "end_sec": 15.8
}
```

Any dataset (e.g. IEMOCAP, EmoryNLP) can be adapted by writing a small
converter script that emits this schema into `dataset/<name>/processed/*.jsonl`,
then pointing `DATA_DIR` in `.env` at it.
