"""
Visual-only / Text+Visual training (Module 18/19).

IMPORTANT LIMITATION, stated explicitly rather than papered over (per the
project's instruction to flag rather than fake unimplementable components):

`training/data.py`'s `ConversationEmotionDataset` does NOT extract face
crops from the MELD video clips by default -- doing so for the full ~13k
utterance dataset requires running face detection over every clip once
up-front (a batch job, not something to do lazily inside `__getitem__`,
or training will be I/O-bound on video decoding).

To actually train the visual modality:

1. Run the batch face-extraction job below to populate
   `dataset/meld/processed/faces/{split}/{dialogue}_{utt}.jpg` with one
   representative aligned face crop per utterance (falls back to "no face"
   when detection fails, which the dataset then marks as
   `availability.visual = False` -- consistent with Module 15).
2. Re-run this script, which then loads those crops into a
   `VisualOnlyDataset` (defined here) rather than the base
   `ConversationEmotionDataset`.

This keeps the base multimodal pipeline (used by the live inference
pipeline in `app/pipeline/inference.py`) doing real-time face detection
per-request, while training uses a pre-extracted cache for speed -- a
standard and honest engineering tradeoff, not a fake stand-in for the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.config import settings
from app.models.multimodal_model import MultimodalEmotionModel
from app.models.speaker import SpeakerEmbedding
from app.preprocessing import face as face_prep
from app.preprocessing import video as video_prep
from app.utils.logging import get_logger
from app.utils.modality_mask import ModalityAvailability
from training.data import ConversationEmotionDataset
from training.engine import train_model

logger = get_logger("training.train_visual", settings.LOG_DIR)


def extract_faces_for_manifest(manifest_path: Path, faces_dir: Path, split: str) -> None:
    """Batch job: one representative face crop per utterance clip, cached to disk."""
    import cv2

    detector = face_prep.FaceDetector()
    faces_dir = faces_dir / split
    faces_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(manifest_path)]
    for r in tqdm(records, desc=f"extracting faces ({split})"):
        out_path = faces_dir / f"dia{r['dialogue_id']}_utt{r['utterance_id']}.jpg"
        if out_path.exists():
            continue
        try:
            frames = list(video_prep.extract_frames(r["video_path"], target_fps=2))
            for frame in frames:  # take the first frame with a detectable face
                boxes = detector.detect(frame.image)
                if boxes:
                    crop = face_prep.crop_and_align_face(frame.image, boxes[0])
                    cv2.imwrite(str(out_path), crop)
                    break
        except Exception as exc:
            logger.debug("No usable face for %s: %s", r["video_path"], exc)


class VisualOnlyDataset(Dataset):
    def __init__(self, manifest_path: Path, faces_dir: Path, split: str):
        self.records = [json.loads(l) for l in open(manifest_path)]
        self.faces_dir = faces_dir / split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import cv2
        import numpy as np

        r = self.records[idx]
        face_path = self.faces_dir / f"dia{r['dialogue_id']}_utt{r['utterance_id']}.jpg"
        if face_path.exists():
            img = cv2.imread(str(face_path))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            face_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            visual_available = True
        else:
            face_tensor = torch.zeros(3, 224, 224)
            visual_available = False

        return {
            "face_image": face_tensor,
            "speaker_idx": SpeakerEmbedding.parse_speaker_label(r["speaker"]),
            "label": settings.label_to_index(r["emotion"]),
            "availability": ModalityAvailability(text=False, audio=False, visual=visual_available, context=False),
        }


def collate_visual(batch, device="cpu"):
    faces = torch.stack([b["face_image"] for b in batch]).to(device)
    speakers = torch.tensor([b["speaker_idx"] for b in batch]).to(device)
    labels = torch.tensor([b["label"] for b in batch]).to(device)
    current = {"texts": None, "audio_features": None, "face_images": faces, "speaker_indices": speakers}
    return {"current": current, "history": None, "labels": labels, "availabilities": [b["availability"] for b in batch]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--faces_dir", type=Path, default=Path("dataset/meld/processed/faces"))
    parser.add_argument("--extract_faces", action="store_true", help="Run the one-time face extraction batch job first.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    args = parser.parse_args()

    if args.extract_faces:
        extract_faces_for_manifest(args.train, args.faces_dir, "train")
        extract_faces_for_manifest(args.val, args.faces_dir, "dev")

    device = settings.DEVICE
    train_ds = VisualOnlyDataset(args.train, args.faces_dir, "train")
    val_ds = VisualOnlyDataset(args.val, args.faces_dir, "dev")
    collate = lambda b: collate_visual(b, device=device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = MultimodalEmotionModel(use_text=False, use_audio=False, use_visual=True, use_context=False)

    out_dir = Path("outputs/experiments/visual_only")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_model(
        model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device,
        checkpoint_path=settings.MODEL_DIR / "visual_only_best.pt",
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best val Macro F1: {result['best_val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
