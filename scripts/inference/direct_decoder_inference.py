#!/usr/bin/env python3
"""
Inference script for trained Direct Decoder.

Loads a trained DirectSLatDecoder and runs inference on samples
from the distillation dataset, saving results as PLY files.
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import numpy as np

from models.direct_slat_decoder import DirectSLatDecoder, count_parameters
from distillation.trellis_dataset import TrellisDistillationDataset


def setup_environment():
    """Set up AMD environment variables."""
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")


def load_model(checkpoint_path: Path, device: str = "cuda") -> DirectSLatDecoder:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint (try both 'config' and 'params' keys)
    config = checkpoint.get('config', checkpoint.get('params', {}))

    # Map auto_tune_v2 param names to model param names
    hidden_dim = config.get('hidden_dim', 512)
    num_gaussians = config.get('num_gaussians_per_voxel', 8)
    occ_threshold = config.get('occ_threshold', config.get('occupancy_threshold', 0.5))

    model = DirectSLatDecoder(
        feature_dim=config.get('feature_dim', 1024),
        hidden_dim=hidden_dim,
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        num_gaussians_per_voxel=num_gaussians,
        max_resolution=config.get('max_resolution', 64),
        dropout=0.0,  # No dropout at inference
        use_checkpoint=False,  # No checkpointing at inference
        predict_occupancy=config.get('predict_occupancy', True),
        occupancy_threshold=occ_threshold,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def save_gaussians_ply(gaussians: np.ndarray, path: Path):
    """
    Save Gaussians to PLY file.

    Format: position(3), scale(3), rotation(4), color(3), opacity(1)
    """
    n_gaussians = len(gaussians)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_gaussians}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
end_header
"""

    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))
        gaussians.astype(np.float32).tofile(f)

    print(f"Saved {n_gaussians} Gaussians to {path}")


def convert_to_ply_format(gaussians: np.ndarray) -> np.ndarray:
    """
    Convert model output to TRELLIS PLY format for visualization.

    Input (model output / training format):
    - Position: [-1, 1] (scaled from TRELLIS world coords)
    - Scale: [0, 1] (actual scale, clamped)
    - Rotation: normalized quaternion
    - Color: [0, 1] RGB
    - Opacity: [0, 1]

    Output (PLY format, matches TRELLIS save_ply):
    - Position: world coords ([-0.5, 0.5] range)
    - Scale: log(actual_scale)
    - Rotation: quaternion
    - Color: SH DC coefficients
    - Opacity: inverse_sigmoid (logits)
    """
    output = gaussians.copy()

    # Position: [-1, 1] -> world coords (divide by 2 to get [-0.5, 0.5])
    output[:, :3] = gaussians[:, :3] / 2.0

    # Scale: actual [0, 1] -> log space
    output[:, 3:6] = np.log(np.clip(gaussians[:, 3:6], 1e-6, 1.0))

    # Rotation: keep as-is
    output[:, 6:10] = gaussians[:, 6:10]

    # Color: [0, 1] RGB -> SH DC coefficients
    # Inverse of: color = SH * C0 + 0.5
    C0 = 0.28209479177387814
    output[:, 10:13] = (gaussians[:, 10:13] - 0.5) / C0

    # Opacity: [0, 1] -> inverse_sigmoid (logits)
    opacity_clipped = np.clip(gaussians[:, 13], 1e-6, 1 - 1e-6)
    output[:, 13] = np.log(opacity_clipped / (1 - opacity_clipped))

    return output


def run_inference(
    model: DirectSLatDecoder,
    dataset: TrellisDistillationDataset,
    sample_idx: int,
    device: str = "cuda",
) -> tuple:
    """Run inference on a single sample."""
    sample = dataset[sample_idx]

    # Prepare inputs
    features = sample['features'].unsqueeze(0).to(device)
    coords = sample['coords'].unsqueeze(0).to(device)
    coord_mask = sample['coord_mask'].unsqueeze(0).to(device)

    # Get ground truth
    target_gaussians = sample['gaussians']
    target_mask = sample['gaussian_mask']

    # Get valid coord count for stats
    n_valid_coords = int(coord_mask[0].sum().item())

    # Run inference with occupancy masking
    with torch.no_grad():
        result = model(features, coords, coord_mask, apply_occupancy_mask=True)

    # Handle occupancy-gated output (returns Dict with list of tensors)
    if isinstance(result, dict):
        pred_gaussians = result['gaussians'][0].cpu().numpy()  # First batch item
        n_pred = result['n_gaussians'][0]

        # Get occupancy stats
        occ_mask = result.get('occupancy_mask')
        if occ_mask is not None:
            n_occupied = int(occ_mask[0].sum().item())
        else:
            n_occupied = n_valid_coords
    else:
        # Fallback for old model without occupancy
        n_pred_per_coord = model.num_gaussians_per_voxel
        n_valid_pred = int(n_valid_coords * n_pred_per_coord)
        pred_gaussians = result[0, :n_valid_pred].cpu().numpy()
        n_occupied = n_valid_coords

    target_gaussians = target_gaussians[target_mask].numpy()

    return pred_gaussians, target_gaussians, sample['sample_name'], n_valid_coords, n_occupied


def main():
    parser = argparse.ArgumentParser(description="Direct Decoder Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fresnel_v2/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/trellis_distillation_diverse",
                        help="Path to distillation dataset")
    parser.add_argument("--output_dir", type=str, default="output/inference",
                        help="Output directory for results")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index to run inference on")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to process")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--max_coords", type=int, default=4000,
                        help="Max coords (should match training)")

    args = parser.parse_args()

    setup_environment()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(Path(args.checkpoint), args.device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Load dataset
    print(f"Loading dataset from {args.data_dir}")
    dataset = TrellisDistillationDataset(
        Path(args.data_dir),
        max_gaussians=50000,
        max_coords=args.max_coords,
    )
    print(f"Dataset size: {len(dataset)}")

    # Run inference on samples
    for i in range(args.num_samples):
        idx = (args.sample_idx + i) % len(dataset)
        print(f"\n{'='*60}")
        print(f"Processing sample {idx}: ", end="")

        pred, target, name, n_valid_coords, n_occupied = run_inference(model, dataset, idx, args.device)
        print(f"{name}")

        # Occupancy stats
        print(f"  Occupied voxels: {n_occupied} / {n_valid_coords} ({100*n_occupied/max(n_valid_coords,1):.1f}%)")
        print(f"  Predicted: {len(pred)} Gaussians ({n_occupied} voxels × {model.num_gaussians_per_voxel} G/voxel)")
        print(f"  Target: {len(target)} Gaussians")

        # Print statistics
        if len(pred) > 0:
            print(f"  Pred position range: [{pred[:,:3].min():.3f}, {pred[:,:3].max():.3f}]")
            print(f"  Pred scale range: [{pred[:,3:6].min():.3f}, {pred[:,3:6].max():.3f}]")
            print(f"  Pred color range: [{pred[:,10:13].min():.3f}, {pred[:,10:13].max():.3f}]")
            print(f"  Pred opacity range: [{pred[:,13].min():.3f}, {pred[:,13].max():.3f}]")
        else:
            print(f"  WARNING: No Gaussians predicted (all voxels marked as unoccupied)")

        # Convert to TRELLIS format and save
        if len(pred) > 0:
            pred_trellis = convert_to_ply_format(pred)
            save_gaussians_ply(pred_trellis, output_dir / f"{name}_pred.ply")
            pred.astype(np.float32).tofile(output_dir / f"{name}_pred.bin")
        else:
            print(f"  Skipping PLY save (no predictions)")

        target_trellis = convert_to_ply_format(target)
        save_gaussians_ply(target_trellis, output_dir / f"{name}_target.ply")
        target.astype(np.float32).tofile(output_dir / f"{name}_target.bin")

    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir}")
    print("Use Fresnel viewer or 3D Gaussian Splatting viewer to visualize PLY files")


if __name__ == "__main__":
    main()
