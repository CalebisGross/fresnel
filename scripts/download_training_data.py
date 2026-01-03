#!/usr/bin/env python3
"""
Download training images from HuggingFace datasets.

Default: LPFF (Large-Pose Flickr Faces) dataset
- 19,590 large-pose face images
- Perfect for 3D reconstruction training
- License: CC-BY-NC-2.0

Usage:
    python scripts/download_training_data.py --count 500
    python scripts/download_training_data.py --dataset ffhq --count 200
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm


def download_lpff(output_dir: Path, count: int):
    """Download images from LPFF dataset."""
    from datasets import load_dataset

    print(f"Loading LPFF dataset from HuggingFace...")
    dataset = load_dataset("onethousand/LPFF", split="train")

    print(f"Downloading {count} images to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset, total=min(count, len(dataset)))):
        if i >= count:
            break
        img = sample['image']
        img.save(output_dir / f"lpff_{i:05d}.jpg")

    print(f"Downloaded {min(count, len(dataset))} images")


def download_ffhq(output_dir: Path, count: int):
    """Download images from FFHQ dataset."""
    from datasets import load_dataset

    print(f"Loading FFHQ dataset from HuggingFace...")
    # Use the thumbnail version for faster downloads
    dataset = load_dataset("nuwandaa/ffhq128", split="train")

    print(f"Downloading {count} images to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset, total=min(count, len(dataset)))):
        if i >= count:
            break
        img = sample['image']
        # Resize to 256x256 for consistency
        img = img.resize((256, 256))
        img.save(output_dir / f"ffhq_{i:05d}.jpg")

    print(f"Downloaded {min(count, len(dataset))} images")


def download_celeba(output_dir: Path, count: int):
    """Download images from CelebA dataset."""
    from datasets import load_dataset

    print(f"Loading CelebA dataset from HuggingFace...")
    dataset = load_dataset("flwrlabs/celeba", split="train")

    print(f"Downloading {count} images to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset, total=min(count, len(dataset)))):
        if i >= count:
            break
        img = sample['image']
        # Resize to 256x256 for consistency
        img = img.resize((256, 256))
        img.save(output_dir / f"celeba_{i:05d}.jpg")

    print(f"Downloaded {min(count, len(dataset))} images")


def main():
    parser = argparse.ArgumentParser(description="Download training images from HuggingFace")
    parser.add_argument('--dataset', type=str, default='lpff',
                        choices=['lpff', 'ffhq', 'celeba'],
                        help='Dataset to download (default: lpff)')
    parser.add_argument('--count', type=int, default=500,
                        help='Number of images to download (default: 500)')
    parser.add_argument('--output_dir', type=str, default='./images/training',
                        help='Output directory (default: ./images/training)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"=== Training Data Download ===")
    print(f"Dataset: {args.dataset}")
    print(f"Count: {args.count}")
    print(f"Output: {output_dir}")
    print()

    if args.dataset == 'lpff':
        download_lpff(output_dir, args.count)
    elif args.dataset == 'ffhq':
        download_ffhq(output_dir, args.count)
    elif args.dataset == 'celeba':
        download_celeba(output_dir, args.count)

    print()
    print("Download complete!")
    print(f"Images saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Run training with: python scripts/train_gaussian_decoder.py --data_dir ./images/training")


if __name__ == "__main__":
    main()
