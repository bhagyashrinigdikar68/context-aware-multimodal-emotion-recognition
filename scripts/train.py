"""
Convenience wrapper that runs the full training curriculum in order
(matching the development strategy in the project spec): text-only ->
multimodal (no visual, since faces need pre-extraction) -> full multimodal
with context, then optionally the ablation study.

For fine-grained control, call the individual scripts in `training/`
directly instead.

Usage:
    python scripts/train.py --train dataset/meld/processed/train.jsonl \
                             --val dataset/meld/processed/dev.jsonl \
                             [--run_ablation]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--run_ablation", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    run([py, "training/train_text.py", "--train", args.train, "--val", args.val, "--epochs", str(args.epochs)])
    run([py, "training/train_audio.py", "--train", args.train, "--val", args.val, "--mode", "text_audio", "--epochs", str(args.epochs)])
    run([py, "training/train_multimodal.py", "--train", args.train, "--val", args.val, "--no_visual", "--epochs", str(args.epochs)])

    if args.run_ablation:
        run([py, "training/ablation.py", "--train", args.train, "--val", args.val, "--epochs", str(min(args.epochs, 6))])

    print("\nAll requested training stages complete. See outputs/experiments/ for results.")


if __name__ == "__main__":
    main()
