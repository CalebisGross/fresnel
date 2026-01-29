#!/usr/bin/env python3
"""
Preprocess training images for Gaussian decoder training.

Extracts:
- DINOv2 features (37x37x384) for each image
- Depth maps (256x256) for each image

Optionally:
- Removes backgrounds using rembg (same as TRELLIS-AMD)
- VLM density maps for semantic-aware training via LM Studio

These are required by train_gaussian_decoder.py which expects precomputed
features in {data_dir}/features/{name}_dinov2.bin and {name}_depth.bin

Usage:
    python scripts/preprocess_training_data.py --data_dir images/training
    python scripts/preprocess_training_data.py --data_dir images/training --remove_background
    python scripts/preprocess_training_data.py --data_dir images/training --use_vlm --vlm_url http://localhost:1234/v1/chat/completions
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Optional rembg for background removal
try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# Optional VLM guidance
try:
    from utils.vlm_guidance import VLMGuidance
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

# ONNX Runtime for inference
import onnxruntime as ort

# Model paths
DEPTH_MODEL = PROJECT_ROOT / "models" / "depth_anything_v2_small.onnx"

# DINOv2 model configurations
DINOV2_CONFIGS = {
    'small': {'path': 'dinov2_small.onnx', 'feature_dim': 384},
    'base': {'path': 'dinov2_base.onnx', 'feature_dim': 768},
    'large': {'path': 'dinov2_large.onnx', 'feature_dim': 1024},
}

def get_dinov2_model_path(model_size: str = 'small') -> Path:
    """Get the DINOv2 ONNX model path for the specified size."""
    if model_size not in DINOV2_CONFIGS:
        raise ValueError(f"Unknown DINOv2 size: {model_size}. Choose from: {list(DINOV2_CONFIGS.keys())}")
    return PROJECT_ROOT / "models" / DINOV2_CONFIGS[model_size]['path']

def get_dinov2_feature_dim(model_size: str = 'small') -> int:
    """Get the feature dimension for the specified DINOv2 model size."""
    return DINOV2_CONFIGS[model_size]['feature_dim']

# DINOv2 constants
DINOV2_INPUT_SIZE = 518
DINOV2_PATCH_SIZE = 14
DINOV2_GRID_SIZE = DINOV2_INPUT_SIZE // DINOV2_PATCH_SIZE  # 37

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def remove_background(image: Image.Image, session=None) -> Image.Image:
    """
    Remove background using rembg (same approach as TRELLIS-AMD).

    Args:
        image: PIL Image to process
        session: Optional rembg session for reuse

    Returns:
        PIL Image with background removed (RGB, black background)
    """
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg not installed. Run: pip install rembg[gpu]")

    # Check if already has alpha with transparency
    if image.mode == 'RGBA':
        alpha = np.array(image)[:, :, 3]
        if not np.all(alpha == 255):
            # Already has transparency, just process it
            output_np = np.array(image).astype(np.float32) / 255.0
            # Premultiply alpha (black background)
            rgb = output_np[:, :, :3] * output_np[:, :, 3:4]
            return Image.fromarray((rgb * 255).astype(np.uint8))

    # Remove background
    image_rgb = image.convert('RGB')
    if session is None:
        session = rembg.new_session('u2net')
    output = rembg.remove(image_rgb, session=session)

    # Find bounding box of foreground
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)

    if len(bbox) == 0:
        # No foreground found, return original with black background
        return image_rgb

    # Get bounding box (y_min, x_min), (y_max, x_max)
    y_min, x_min = bbox.min(axis=0)
    y_max, x_max = bbox.max(axis=0)

    # Center and size with padding
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    size = max(x_max - x_min, y_max - y_min)
    size = int(size * 1.2)  # 20% padding

    # Compute crop box (left, top, right, bottom)
    left = int(center_x - size // 2)
    top = int(center_y - size // 2)
    right = int(center_x + size // 2)
    bottom = int(center_y + size // 2)

    # Clamp to image bounds and adjust to maintain square crop
    h, w = output_np.shape[:2]
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > w:
        left -= (right - w)
        right = w
    if bottom > h:
        top -= (bottom - h)
        bottom = h

    # Ensure valid crop
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    # Crop
    output = output.crop((left, top, right, bottom))

    # Resize to standard size
    output = output.resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.Resampling.LANCZOS)

    # Premultiply alpha (black background)
    output_np = np.array(output).astype(np.float32) / 255.0
    rgb = output_np[:, :, :3] * output_np[:, :, 3:4]

    return Image.fromarray((rgb * 255).astype(np.uint8))


def preprocess_image(image_path: str = None, image: Image.Image = None) -> np.ndarray:
    """
    Load and preprocess image for DINOv2 or Depth Anything V2.

    Both models use the same preprocessing:
    - Resize to 518x518
    - Normalize with ImageNet mean/std
    - Convert to CHW format with batch dimension

    Args:
        image_path: Path to image file (used if image is None)
        image: PIL Image (takes precedence over image_path)

    Returns:
        Preprocessed image array of shape (1, 3, 518, 518)
    """
    if image is None:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.Resampling.BILINEAR)
    else:
        img = image.convert('RGB')
        if img.size != (DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE):
            img = img.resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.Resampling.BILINEAR)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    return img_array


# Backwards-compatible aliases
def preprocess_for_dinov2(image_path: str = None, image: Image.Image = None) -> np.ndarray:
    """Load and preprocess image for DINOv2. Alias for preprocess_image()."""
    return preprocess_image(image_path, image)


def preprocess_for_depth(image_path: str = None, image: Image.Image = None) -> np.ndarray:
    """Load and preprocess image for Depth Anything V2. Alias for preprocess_image()."""
    return preprocess_image(image_path, image)


def extract_dinov2_features(session: ort.InferenceSession, image_path: str, image: Image.Image = None) -> np.ndarray:
    """Extract DINOv2 patch features from image."""
    img_input = preprocess_for_dinov2(image_path, image)

    # Run inference
    outputs = session.run(None, {'pixel_values': img_input})

    # Output shape is (1, 37*37, 384) or (1, 1370, 384) with CLS token
    features = outputs[0][0]  # Remove batch dim

    # If CLS token is included, remove it
    if features.shape[0] == DINOV2_GRID_SIZE * DINOV2_GRID_SIZE + 1:
        features = features[1:]  # Skip CLS token

    # Reshape to (37, 37, 384)
    features = features.reshape(DINOV2_GRID_SIZE, DINOV2_GRID_SIZE, -1)

    return features.astype(np.float32)


def extract_depth(session: ort.InferenceSession, image_path: str, output_size: int = 256, image: Image.Image = None) -> np.ndarray:
    """Extract depth map from image."""
    img_input = preprocess_for_depth(image_path, image)

    # Run inference
    outputs = session.run(None, {'pixel_values': img_input})
    depth = outputs[0][0]  # Remove batch dim, shape depends on model

    # Handle different output shapes
    if len(depth.shape) == 3:
        depth = depth[0]  # Remove channel dim if present

    # Normalize to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)

    # Resize to output size
    depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
    depth_img = depth_img.resize((output_size, output_size), Image.Resampling.BILINEAR)
    depth = np.array(depth_img, dtype=np.float32) / 255.0

    return depth


def preprocess_dataset(
    data_dir: str,
    output_dir: str = None,
    depth_size: int = 256,
    remove_bg: bool = False,
    use_vlm: bool = False,
    vlm_url: str = "http://localhost:1234/v1/chat/completions",
    vlm_grid_size: int = 8,
    dinov2_size: str = 'small'
):
    """Preprocess all images in data_dir."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get DINOv2 model path for selected size
    dinov2_model = get_dinov2_model_path(dinov2_size)
    feature_dim = get_dinov2_feature_dim(dinov2_size)

    # Check models exist
    if not dinov2_model.exists():
        print(f"Error: DINOv2 model not found at {dinov2_model}")
        print(f"Run: python scripts/export/export_dinov2_model.py --size {dinov2_size}")
        sys.exit(1)

    if not DEPTH_MODEL.exists():
        print(f"Error: Depth model not found at {DEPTH_MODEL}")
        print("Run: python scripts/export_depth_model.py")
        sys.exit(1)

    # Check rembg if background removal requested
    if remove_bg and not REMBG_AVAILABLE:
        print("Error: rembg not installed but --remove_background was specified")
        print("Run: pip install rembg[gpu]")
        sys.exit(1)

    # Check VLM if requested
    if use_vlm and not VLM_AVAILABLE:
        print("Error: VLM guidance not available but --use_vlm was specified")
        print("Make sure vlm_guidance.py is in the scripts directory")
        sys.exit(1)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    images = []
    for ext in image_extensions:
        images.extend(data_dir.glob(ext))
        images.extend(data_dir.glob(ext.upper()))
    images = sorted(images)

    if not images:
        print(f"No images found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"DINOv2 model: {dinov2_size} ({feature_dim}-dim features)")
    if remove_bg:
        print("Background removal: ENABLED")
    if use_vlm:
        print(f"VLM guidance: ENABLED (grid_size={vlm_grid_size})")
    print()

    # Load models
    print(f"Loading DINOv2-{dinov2_size} model...")
    dinov2_session = ort.InferenceSession(str(dinov2_model), providers=['CPUExecutionProvider'])

    print("Loading Depth Anything V2 model...")
    depth_session = ort.InferenceSession(str(DEPTH_MODEL), providers=['CPUExecutionProvider'])

    # Initialize rembg session if needed
    rembg_session = None
    if remove_bg:
        print("Loading rembg u2net model...")
        rembg_session = rembg.new_session('u2net')

    # Initialize VLM guidance if requested
    vlm = None
    if use_vlm:
        print(f"Connecting to VLM at {vlm_url}...")
        vlm = VLMGuidance(api_url=vlm_url)
        if not vlm.check_connection():
            print("Warning: Cannot connect to VLM, skipping VLM density extraction")
            print("Make sure LM Studio is running with a VLM model loaded")
            vlm = None
        else:
            print("VLM connected successfully")

    print()
    print("Extracting features and depth maps...")

    # Process images
    skipped = 0
    processed = 0
    vlm_extracted = 0

    # Feature file suffix includes model size for differentiation
    feature_suffix = f"_dinov2_{dinov2_size}.bin" if dinov2_size != 'small' else "_dinov2.bin"

    for img_path in tqdm(images, desc="Preprocessing"):
        name = img_path.stem

        feature_path = output_dir / f"{name}{feature_suffix}"
        depth_path = output_dir / f"{name}_depth.bin"
        vlm_density_path = output_dir / f"{name}_vlm_density.npy"

        # Skip if already processed (check VLM too if enabled)
        all_done = feature_path.exists() and depth_path.exists()
        if vlm and not vlm_density_path.exists():
            all_done = False
        if all_done:
            skipped += 1
            continue

        try:
            # Load and optionally remove background
            processed_img = None
            if remove_bg:
                raw_img = Image.open(img_path)
                processed_img = remove_background(raw_img, rembg_session)

            # Extract DINOv2 features
            if not feature_path.exists():
                features = extract_dinov2_features(dinov2_session, str(img_path), processed_img)
                features.tofile(str(feature_path))

            # Extract depth
            if not depth_path.exists():
                depth = extract_depth(depth_session, str(img_path), depth_size, processed_img)
                depth.tofile(str(depth_path))

            # Extract VLM density map if enabled
            if vlm and not vlm_density_path.exists():
                try:
                    # VLM needs to analyze the same image that training uses
                    # If background removal is enabled, save processed image to temp file
                    if processed_img is not None:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            processed_img.save(tmp.name)
                            density = vlm.get_density_guidance(tmp.name, grid_size=vlm_grid_size)
                            os.unlink(tmp.name)
                    else:
                        density = vlm.get_density_guidance(str(img_path), grid_size=vlm_grid_size)

                    if density is not None:
                        np.save(str(vlm_density_path), density)
                        vlm_extracted += 1
                except Exception as vlm_err:
                    # VLM errors shouldn't stop preprocessing
                    tqdm.write(f"VLM warning for {name}: {vlm_err}")

            processed += 1

        except (IOError, OSError) as e:
            # File read/write errors
            print(f"\nFile error processing {img_path}: {e}")
            continue
        except ValueError as e:
            # Data format errors (corrupted image, etc.)
            print(f"\nData error processing {img_path}: {e}")
            continue
        except RuntimeError as e:
            # ONNX/model errors
            print(f"\nModel error processing {img_path}: {e}")
            continue
        except Exception as e:
            # Unexpected errors - log with full info
            print(f"\nUnexpected error processing {img_path}: {type(e).__name__}: {e}")
            continue

    print()
    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"  Processed: {processed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Total: {len(images)}")
    if remove_bg:
        print(f"  Background removal: enabled")
    if vlm:
        print(f"  VLM density maps: {vlm_extracted}")
    print()
    print(f"Features saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  Run training with:")
    if vlm:
        print(f"    python scripts/train_gaussian_decoder.py --data_dir {data_dir} --use_vlm_guidance")
    else:
        print(f"    python scripts/train_gaussian_decoder.py --data_dir {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess training images for Gaussian decoder")
    parser.add_argument('--data_dir', type=str, default='images/training',
                        help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for features (default: {data_dir}/features)')
    parser.add_argument('--depth_size', type=int, default=256,
                        help='Output depth map size (default: 256)')
    parser.add_argument('--remove_background', action='store_true',
                        help='Remove backgrounds using rembg (requires: pip install rembg[gpu])')
    parser.add_argument('--use_vlm', action='store_true',
                        help='Extract VLM density maps for semantic-aware training (requires LM Studio)')
    parser.add_argument('--vlm_url', type=str, default='http://localhost:1234/v1/chat/completions',
                        help='LM Studio API URL (default: http://localhost:1234/v1/chat/completions)')
    parser.add_argument('--vlm_grid_size', type=int, default=8, choices=[4, 8, 16],
                        help='VLM density grid size (default: 8)')
    parser.add_argument('--dinov2_size', type=str, default='small', choices=['small', 'base', 'large'],
                        help='DINOv2 model size: small (384-dim), base (768-dim), large (1024-dim)')

    args = parser.parse_args()

    print("=" * 60)
    print("Training Data Preprocessing")
    print("=" * 60)
    print()

    preprocess_dataset(
        args.data_dir,
        args.output_dir,
        args.depth_size,
        args.remove_background,
        args.use_vlm,
        args.vlm_url,
        args.vlm_grid_size,
        args.dinov2_size
    )


if __name__ == "__main__":
    main()
