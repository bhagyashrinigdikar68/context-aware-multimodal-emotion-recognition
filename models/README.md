# Models Directory

Trained checkpoints are saved here by the training scripts and are
**git-ignored** (see `.gitignore`) -- they are typically hundreds of MB to
GB and don't belong in version control.

## Expected files after training

| File | Produced by |
|---|---|
| `text_only_best.pt` | `training/train_text.py` |
| `text_audio_best.pt` | `training/train_audio.py --mode text_audio` |
| `audio_only_best.pt` | `training/train_audio.py --mode audio_only` |
| `visual_only_best.pt` | `training/train_visual.py` |
| `multimodal_best.pt` | `training/train_multimodal.py` — **this is the checkpoint the live inference pipeline (`app/pipeline/inference.py`) and dashboard load by default.** |
| `ablation_<config>_best.pt` | `training/ablation.py`, one per modality combination |

## Checkpoint format

Each `.pt` file is a `torch.save` dict:

```python
{
    "model_state_dict": <state_dict>,
    "val_f1_macro": <float>,
}
```

Load with:

```python
state = torch.load("models/multimodal_best.pt", map_location=device)
model.load_state_dict(state["model_state_dict"])
```

## No checkpoint yet?

If `multimodal_best.pt` doesn't exist, `EmotionRecognitionPipeline` (and the
dashboard/API built on it) automatically fall back to **DEMO MODE**: the
model architecture runs with randomly-initialized weights so you can
exercise the full UI/pipeline end-to-end, but the emotion predictions are
meaningless and are clearly labeled as such everywhere they're displayed.
Never report demo-mode output as a research result.
