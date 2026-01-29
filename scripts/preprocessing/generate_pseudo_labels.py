#!/usr/bin/env python3
"""
Generate Pseudo-Labels using Depth Anything V2.

This script runs DA V2 (Small or Large) on a folder of images to generate
depth map "pseudo-labels" that can be used for:

1. Training depth models on your own images (no need for labeled dataset!)
2. Knowledge distillation (train Small to match Large outputs)
3. Creating training data from any image collection

Usage:
    # Generate labels using DA V2 Small (faster, already downloaded)
    python generate_pseudo_labels.py ./my_images --output ./my_data

    # Generate labels using DA V2 Large (better quality, for distillation)
    python generate_pseudo_labels.py ./my_images --output ./my_data --model large

The output folder structure will be:
    my_data/
        images/
            001.jpg -> (copied/linked)
            ...
        depths/
            001.npy
            ...

This can then be used with FolderDepthDataset for training!
"""

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from PIL import Image
import torchvision.transforms as T


def load_depth_anything_v2(model_size: str = 'small', device: torch.device = None):
    """
    Load Depth Anything V2 model from HuggingFace.

    Args:
        model_size: 'small', 'base', 'large', or 'giant'
        device: torch device

    Returns:
        model, processor
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_names = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
        # 'giant': 'depth-anything/Depth-Anything-V2-Giant-hf',  # Very large
    }

    if model_size not in model_names:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_names.keys())}")

    model_name = model_names[model_size]
    print(f"Loading {model_name}...")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)

    if device:
        model = model.to(device)

    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, processor


@torch.no_grad()
def estimate_depth(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
) -> np.ndarray:
    """
    Estimate depth for a single image.

    Args:
        model: DA V2 model
        processor: Image processor
        image: PIL Image
        device: torch device

    Returns:
        Depth map as numpy array (H, W), normalized to [0, 1]
    """
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    outputs = model(**inputs)
    depth = outputs.predicted_depth

    # Post-process
    depth = depth.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth


def process_folder(
    input_dir: Path,
    output_dir: Path,
    model_size: str = 'small',
    max_images: int = None,
    link_images: bool = True,
):
    """
    Process all images in a folder.

    Args:
        input_dir: Folder containing images
        output_dir: Output folder (will create images/ and depths/ subfolders)
        model_size: DA V2 model size
        max_images: Maximum number of images to process
        link_images: Create symlinks instead of copying images
    """
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    model, processor = load_depth_anything_v2(model_size, device)

    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
        image_paths.extend(input_dir.glob(f'*{ext.upper()}'))

    image_paths = sorted(image_paths)

    if max_images:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Found {len(image_paths)} images")

    # Create output directories
    images_dir = output_dir / 'images'
    depths_dir = output_dir / 'depths'
    images_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    for img_path in tqdm(image_paths, desc="Generating depth maps"):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Estimate depth
            depth = estimate_depth(model, processor, image, device)

            # Save depth as numpy
            depth_path = depths_dir / f"{img_path.stem}.npy"
            np.save(depth_path, depth.astype(np.float32))

            # Copy/link image
            out_img_path = images_dir / img_path.name
            if link_images and not out_img_path.exists():
                try:
                    out_img_path.symlink_to(img_path.absolute())
                except OSError:
                    # Symlinks not supported, copy instead
                    shutil.copy(img_path, out_img_path)
            elif not out_img_path.exists():
                shutil.copy(img_path, out_img_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nDone! Created dataset at {output_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Depths: {depths_dir}")
    print(f"  Total: {len(list(depths_dir.glob('*.npy')))} depth maps")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("To use this data for training:")
    print("=" * 60)
    print(f"  Data available at: {output_dir}")


def download_sample_images(output_dir: Path, num_images: int = 100):
    """
    Download sample images from COCO for testing.

    This is useful if you don't have images to train on.
    """
    from datasets import load_dataset

    print(f"Downloading {num_images} sample images from COCO...")

    # Load COCO validation set (small)
    dataset = load_dataset("detection-datasets/coco", split="val")
    dataset = dataset.select(range(min(num_images, len(dataset))))

    images_dir = output_dir / 'raw_images'
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        image = sample['image']
        image.save(images_dir / f"{i:04d}.jpg")

    print(f"Downloaded {len(dataset)} images to {images_dir}")
    return images_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate depth pseudo-labels using Depth Anything V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate labels for your images
    python generate_pseudo_labels.py ./photos --output ./training_data

    # Use larger model for better quality
    python generate_pseudo_labels.py ./photos --output ./training_data --model large

    # Download sample images first
    python generate_pseudo_labels.py --download --output ./training_data
        """
    )

    parser.add_argument('input_dir', type=str, nargs='?',
                        help='Input folder containing images')
    parser.add_argument('--output', '-o', type=str, default='./pseudo_labeled_data',
                        help='Output folder')
    parser.add_argument('--model', '-m', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='DA V2 model size')
    parser.add_argument('--max_images', '-n', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--download', action='store_true',
                        help='Download sample images from COCO first')
    parser.add_argument('--copy', action='store_true',
                        help='Copy images instead of creating symlinks')

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.download:
        # Download sample images
        input_dir = download_sample_images(output_dir, num_images=args.max_images or 100)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return
    else:
        print("Error: Please provide input_dir or use --download")
        parser.print_help()
        return

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        model_size=args.model,
        max_images=args.max_images,
        link_images=not args.copy,
    )


if __name__ == '__main__':
    main()
