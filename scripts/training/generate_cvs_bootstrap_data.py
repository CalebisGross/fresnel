#!/usr/bin/env python3
"""
Generate Synthetic Multi-View Dataset via Gaussian Decoder Bootstrapping

Takes single images, generates 3D Gaussians via trained decoder, then renders
from multiple viewpoints to create ground-truth multi-view training pairs for CVS.

This is the novel self-supervised approach unique to Fresnel - no external datasets needed!

Usage:
    python scripts/training/generate_cvs_bootstrap_data.py \
        --input_dir images/training \
        --gaussian_decoder checkpoints/gaussian_decoder_exp2.pt \
        --output_dir cvs_training_synthetic \
        --num_poses_per_image 8

Author: Fresnel Research Team
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.gaussian_decoder_models import DirectPatchDecoder
from scripts.models.differentiable_renderer import TileBasedRenderer
from scripts.inference.dinov2_inference import preprocess_image as dinov2_preprocess, IMAGENET_MEAN, IMAGENET_STD

# ============================================================================
# Camera Pose Generation
# ============================================================================

def create_orbit_poses(
    num_poses: int,
    radius: float = 2.0,
    elevation_range: Tuple[float, float] = (-np.pi/6, np.pi/4),
    look_at: np.ndarray = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate camera poses orbiting around an object.

    Args:
        num_poses: Number of poses to generate
        radius: Distance from object center
        elevation_range: (min, max) elevation angles in radians
        look_at: Point to look at (default: origin)

    Returns:
        List of (R, t) tuples (world-to-camera transformations)
    """
    if look_at is None:
        look_at = np.array([0.0, 0.0, 0.0])

    poses = []

    for i in range(num_poses):
        # Azimuth: full circle
        azimuth = 2 * np.pi * i / num_poses

        # Elevation: random within range (varied heights)
        elevation = np.random.uniform(*elevation_range)

        # Camera position in spherical coordinates
        x = radius * np.cos(azimuth) * np.cos(elevation)
        y = radius * np.sin(elevation)
        z = radius * np.sin(azimuth) * np.cos(elevation)
        position = np.array([x, y, z])

        # Look-at rotation matrix
        forward = look_at - position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Up vector (world Y)
        up = np.array([0.0, 1.0, 0.0])

        # Right vector
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Recompute up
        up = np.cross(right, forward)

        # Rotation matrix (world-to-camera)
        R_c2w = np.stack([right, up, -forward], axis=1)
        R = R_c2w.T  # world-to-camera

        # Translation (world-to-camera)
        t = -R @ position

        poses.append((R, t))

    return poses


# ============================================================================
# Feature/Depth Extraction
# ============================================================================

def load_dinov2_model(device='cpu'):
    """Load DINOv2 ONNX model for feature extraction."""
    dinov2_path = PROJECT_ROOT / "models" / "dinov2_small.onnx"

    if not dinov2_path.exists():
        raise FileNotFoundError(
            f"DINOv2 model not found at {dinov2_path}\n"
            "Run: python scripts/export/export_dinov2_model.py"
        )

    import onnxruntime as ort
    session = ort.InferenceSession(
        str(dinov2_path),
        providers=['CPUExecutionProvider']
    )

    return session


def load_depth_model(device='cpu'):
    """Load Depth Anything V2 ONNX model."""
    depth_path = PROJECT_ROOT / "models" / "depth_anything_v2_small.onnx"

    if not depth_path.exists():
        raise FileNotFoundError(
            f"Depth model not found at {depth_path}\n"
            "Run: python scripts/export/export_depth_model.py"
        )

    import onnxruntime as ort
    session = ort.InferenceSession(
        str(depth_path),
        providers=['CPUExecutionProvider']
    )

    return session


def extract_dinov2_features(image_rgb: np.ndarray, dinov2_session) -> torch.Tensor:
    """
    Extract DINOv2 features from RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        dinov2_session: ONNX Runtime session

    Returns:
        (37, 37, 384) feature tensor
    """
    # Preprocess (resize to 518x518, normalize)
    pil_img = Image.fromarray(image_rgb)
    pil_img = pil_img.resize((518, 518), Image.BILINEAR)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0

    # ImageNet normalization
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> NCHW
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    outputs = dinov2_session.run(None, {'pixel_values': img_array})
    patch_tokens = outputs[0][0]  # (1369, 384)

    # Reshape to spatial grid
    features = patch_tokens.reshape(37, 37, 384)

    return torch.from_numpy(features).float()


def extract_depth(image_rgb: np.ndarray, depth_session, target_size=256) -> torch.Tensor:
    """
    Extract depth map from RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        depth_session: ONNX Runtime session
        target_size: Output depth map size

    Returns:
        (1, target_size, target_size) depth tensor [0, 1]
    """
    # Preprocess (resize, normalize)
    pil_img = Image.fromarray(image_rgb)
    pil_img = pil_img.resize((518, 518), Image.BILINEAR)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0

    # HWC -> NCHW
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    outputs = depth_session.run(None, {'pixel_values': img_array})
    depth = outputs[0][0]  # (1, H, W)

    # Resize to target size
    depth_tensor = torch.from_numpy(depth).float()
    # F.interpolate needs (N, C, H, W), so add batch and channel dims
    depth_resized = F.interpolate(
        depth_tensor.unsqueeze(0).unsqueeze(0),  # (H, W) -> (1, 1, H, W)
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # -> (1, target_size, target_size)

    # Normalize to [0, 1]
    depth_min = depth_resized.min()
    depth_max = depth_resized.max()
    depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)

    return depth_normalized


# ============================================================================
# Gaussian Decoder Inference
# ============================================================================

def load_decoder_model(checkpoint_path: str, device: str) -> DirectPatchDecoder:
    """Load trained Gaussian decoder model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model config from checkpoint
    config = checkpoint.get('config', {})

    model = DirectPatchDecoder(
        feature_dim=384,  # DINOv2
        hidden_dims=config.get('hidden_dims', [512, 512, 256, 128]),
        gaussians_per_patch=config.get('gaussians_per_patch', 8),
        dropout=config.get('dropout', 0.1),
        use_fresnel_zones=config.get('use_fresnel_zones', False),
        num_fresnel_zones=config.get('num_fresnel_zones', 8),
        use_edge_aware=config.get('use_edge_aware', False),
        use_phase_output=config.get('use_phase_output', False)
    ).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@torch.no_grad()
def decode_gaussians(
    features: torch.Tensor,
    depth: torch.Tensor,
    decoder: DirectPatchDecoder,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Generate 3D Gaussians from features and depth.

    Args:
        features: (37, 37, 384) DINOv2 features
        depth: (1, H, W) depth map
        decoder: Trained DirectPatchDecoder
        device: torch device

    Returns:
        Dict with positions, scales, rotations, colors, opacities
    """
    # Convert to channel-first format and add batch dimension
    # DirectPatchDecoder expects (B, C, H, W) not (B, H, W, C)
    features = features.permute(2, 0, 1).unsqueeze(0).to(device)  # (37, 37, 384) -> (384, 37, 37) -> (1, 384, 37, 37)
    depth = depth.unsqueeze(0).to(device)  # (1, 1, H, W)

    # Decode
    output = decoder(features, depth)

    return {
        'positions': output['positions'][0],  # (N, 3)
        'scales': output['scales'][0],        # (N, 3)
        'rotations': output['rotations'][0],  # (N, 4) quaternions
        'colors': output['colors'][0],        # (N, 3)
        'opacities': output['opacities'][0],  # (N, 1)
    }


# ============================================================================
# Rendering
# ============================================================================

@torch.no_grad()
def render_view(
    gaussians: Dict[str, torch.Tensor],
    R: np.ndarray,
    t: np.ndarray,
    renderer: TileBasedRenderer,
    image_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render Gaussians from a specific camera pose.

    Args:
        gaussians: Dict of Gaussian parameters
        R: (3, 3) camera rotation (world-to-camera)
        t: (3,) camera translation
        renderer: TileBasedRenderer instance
        image_size: Output image size
        device: torch device

    Returns:
        (rgb, depth) tensors: (3, H, W) and (1, H, W)
    """
    from scripts.models.differentiable_renderer import Camera

    # Create camera with simple perspective projection
    # Assuming focal length = image_size (45 degree FOV approximately)
    focal_length = image_size
    camera = Camera(
        fx=focal_length,
        fy=focal_length,
        cx=image_size / 2,
        cy=image_size / 2,
        width=image_size,
        height=image_size,
        near=0.01,
        far=100.0
    )

    # Set camera view matrix from R and t
    R_torch = torch.from_numpy(R).float().to(device)
    t_torch = torch.from_numpy(t).float().to(device)

    # Create 4x4 view matrix [R | t; 0 0 0 1]
    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, :3] = R_torch
    view_matrix[:3, 3] = t_torch
    camera.set_view(view_matrix)

    # Render (TileBasedRenderer expects (N,) not (N, 1) for opacities)
    rgb, depth = renderer(
        positions=gaussians['positions'],  # (N, 3)
        scales=gaussians['scales'],  # (N, 3)
        rotations=gaussians['rotations'],  # (N, 4)
        colors=gaussians['colors'],  # (N, 3)
        opacities=gaussians['opacities'].squeeze(-1),  # (N,) not (N, 1)
        camera=camera,
        return_depth=True
    )

    # TileBasedRenderer already returns rgb as (3, H, W) and depth as (H, W)
    # Just add channel dimension to depth
    depth = depth.unsqueeze(0)  # (1, H, W)

    return rgb, depth


# ============================================================================
# Main Pipeline
# ============================================================================

def generate_multiview_dataset(
    input_dir: str,
    gaussian_decoder_checkpoint: str,
    output_dir: str,
    num_poses_per_image: int,
    image_size: int,
    device: str,
    visualize: bool = False,
    max_images: int = None
):
    """
    Generate synthetic multi-view dataset.

    Args:
        input_dir: Directory containing training images
        gaussian_decoder_checkpoint: Path to trained decoder checkpoint
        output_dir: Output directory for synthetic dataset
        num_poses_per_image: Number of poses to generate per input
        image_size: Render size
        device: torch device
        visualize: Save visualization grids
        max_images: Limit number of images (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    dinov2_session = load_dinov2_model(device='cpu')
    depth_session = load_depth_model(device='cpu')
    decoder = load_decoder_model(gaussian_decoder_checkpoint, device)
    renderer = TileBasedRenderer(
        image_width=image_size,
        image_height=image_size,
        background=(0.0, 0.0, 0.0),
        max_radius=64,
        use_phase_blending=False
    ).to(device)

    print("Models loaded successfully")

    # Find input images
    input_path = Path(input_dir)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))

    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images")

    # Process each image
    for img_idx, img_file in enumerate(tqdm(image_files, desc="Generating dataset")):
        # Load image
        img = Image.open(img_file).convert('RGB')
        img_rgb = np.array(img)

        # Extract features and depth
        features = extract_dinov2_features(img_rgb, dinov2_session)
        depth = extract_depth(img_rgb, depth_session, target_size=image_size)

        # Generate Gaussians
        gaussians = decode_gaussians(features, depth, decoder, device)

        # Create poses
        poses = create_orbit_poses(num_poses_per_image)

        # Create output directory for this image
        img_output_dir = output_path / f"image_{img_idx:04d}"
        img_output_dir.mkdir(exist_ok=True)

        # Save input data
        img.resize((image_size, image_size), Image.BILINEAR).save(img_output_dir / "input.png")
        np.save(img_output_dir / "input_features.npy", features.numpy())
        np.save(img_output_dir / "input_depth.npy", depth.numpy())

        # Render and save each view
        view_images = []
        for pose_idx, (R, t) in enumerate(poses):
            view_dir = img_output_dir / f"view_{pose_idx:03d}"
            view_dir.mkdir(exist_ok=True)

            # Render
            rgb, depth_rendered = render_view(gaussians, R, t, renderer, image_size, device)

            # Save RGB
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(rgb_np).save(view_dir / "rgb.png")
            view_images.append(rgb_np)

            # Save depth
            np.save(view_dir / "depth.npy", depth_rendered.cpu().numpy())

            # Save pose
            pose_data = {
                'R': R.tolist(),
                't': t.tolist()
            }
            with open(view_dir / "pose.json", 'w') as f:
                json.dump(pose_data, f, indent=2)

        # Optional: Create visualization grid
        if visualize:
            grid_size = int(np.ceil(np.sqrt(num_poses_per_image + 1)))
            cell_size = image_size
            grid = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)

            # Place input
            input_resized = np.array(img.resize((cell_size, cell_size)))
            grid[0:cell_size, 0:cell_size] = input_resized

            # Place views
            for i, view_img in enumerate(view_images):
                row = (i + 1) // grid_size
                col = (i + 1) % grid_size
                grid[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size] = view_img

            Image.fromarray(grid).save(img_output_dir / "grid_visualization.png")

    print(f"\nDataset generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Views per image: {num_poses_per_image}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic multi-view dataset via Gaussian decoder bootstrapping'
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--gaussian_decoder', type=str, required=True,
                        help='Path to trained Gaussian decoder checkpoint (.pt file)')
    parser.add_argument('--output_dir', type=str, default='cvs_training_synthetic',
                        help='Output directory for synthetic dataset')

    parser.add_argument('--num_poses_per_image', type=int, default=8,
                        help='Number of poses to generate per input image')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Render size (square)')

    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Default: auto-detect')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit number of images to process (for testing)')

    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization grids')

    args = parser.parse_args()

    # Detect device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Generate dataset
    generate_multiview_dataset(
        input_dir=args.input_dir,
        gaussian_decoder_checkpoint=args.gaussian_decoder,
        output_dir=args.output_dir,
        num_poses_per_image=args.num_poses_per_image,
        image_size=args.image_size,
        device=device,
        visualize=args.visualize,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
