#!/usr/bin/env python3
"""
Gaussian Decoder inference for Fresnel.

Takes DINOv2 features and depth map, outputs predicted Gaussians.
Uses the trained DirectPatchDecoder (Experiment 2) ONNX model.

Usage:
    python decoder_inference.py features.bin depth.bin output.bin

Output format:
    - Binary file containing N * 14 floats per Gaussian
    - Layout: position(3), scale(3), rotation(4), color(3), opacity(1)
    - Stdout: "num_gaussians" for C++ to read
"""

import sys
import os
import numpy as np
import onnxruntime as ort

# Model path (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gaussian_decoder.onnx")

# Feature grid dimensions (from DINOv2)
FEATURE_DIM = 384
GRID_SIZE = 37

# Depth input size
DEPTH_SIZE = 518


def load_features(features_path: str) -> np.ndarray:
    """
    Load DINOv2 features from binary file.

    Expected format: (37, 37, 384) float32
    Returns: (1, 384, 37, 37) for model input
    """
    features = np.fromfile(features_path, dtype=np.float32)
    features = features.reshape(GRID_SIZE, GRID_SIZE, FEATURE_DIM)
    # HWC -> CHW, add batch
    features = features.transpose(2, 0, 1)
    features = np.expand_dims(features, axis=0)
    return features


def load_depth(depth_path: str, target_size: int = DEPTH_SIZE) -> np.ndarray:
    """
    Load depth map from binary file.

    Returns: (1, 1, H, W) for model input
    """
    # Try to load as raw float32
    depth = np.fromfile(depth_path, dtype=np.float32)

    # Infer dimensions (assume square)
    side = int(np.sqrt(len(depth)))
    if side * side == len(depth):
        depth = depth.reshape(side, side)
    else:
        # Try common sizes
        for s in [518, 512, 256, 384]:
            if s * s == len(depth):
                depth = depth.reshape(s, s)
                break
        else:
            raise ValueError(f"Cannot reshape depth of size {len(depth)}")

    # Resize if needed (simple nearest neighbor)
    if depth.shape[0] != target_size:
        from PIL import Image
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((target_size, target_size), Image.Resampling.NEAREST)
        depth = np.array(depth_img, dtype=np.float32)

    # Add batch and channel dims
    depth = depth[np.newaxis, np.newaxis, :, :]
    return depth


def run_inference(features_path: str, depth_path: str, output_path: str):
    """
    Run Gaussian decoder inference.

    Args:
        features_path: Path to DINOv2 features binary
        depth_path: Path to depth map binary
        output_path: Path to output Gaussians binary
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
        print("Train the decoder first or copy the ONNX model.", file=sys.stderr)
        sys.exit(1)

    # Load inputs
    features = load_features(features_path)
    depth = load_depth(depth_path)

    # Create ONNX Runtime session
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    # Check what inputs the model needs
    input_names = [inp.name for inp in session.get_inputs()]

    # Run inference (DirectPatchDecoder only needs features, not depth)
    if 'depth' in input_names:
        outputs = session.run(None, {
            'features': features,
            'depth': depth
        })
    else:
        # Model doesn't use depth input
        outputs = session.run(None, {
            'features': features
        })

    # Parse outputs
    positions = outputs[0][0]   # (N, 3)
    scales = outputs[1][0]      # (N, 3)
    rotations = outputs[2][0]   # (N, 4)
    colors = outputs[3][0]      # (N, 3)
    opacities = outputs[4][0]   # (N,)

    num_gaussians = positions.shape[0]

    # Pack into output format: 14 floats per Gaussian
    # position(3), scale(3), rotation(4), color(3), opacity(1)
    gaussians = np.zeros((num_gaussians, 14), dtype=np.float32)
    gaussians[:, 0:3] = positions
    gaussians[:, 3:6] = scales
    gaussians[:, 6:10] = rotations
    gaussians[:, 10:13] = colors
    gaussians[:, 13] = opacities

    # Save as raw binary
    gaussians.tofile(output_path)

    # Output count to stdout for C++ to read
    print(f"{num_gaussians}")

    return num_gaussians


def test_novel_views(
    input_image: str,
    checkpoint_path: str,
    output_dir: str,
    num_views: int = 8,
    image_size: int = 256,
    experiment: int = 2,
    n_spiral_points: int = 377,
):
    """
    Test the decoder's ability to produce visible novel views.

    This is a validation checkpoint to ensure the decoder doesn't produce
    dark/black images from novel viewpoints before regenerating CVS data.

    Args:
        input_image: Path to input image
        checkpoint_path: Path to PyTorch checkpoint (.pt)
        output_dir: Directory to save rendered views
        num_views: Number of novel views to generate
        image_size: Render resolution
    """
    import torch
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms

    # Add scripts to path
    script_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(script_dir))

    from models.gaussian_decoder_models import DirectPatchDecoder, FibonacciPatchDecoder
    from models.differentiable_renderer import TileBasedRenderer, FourierGaussianRenderer, Camera

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get('model_state_dict', checkpoint)

    # Check if pose encoding is enabled
    use_pose_encoding = any('pose_encoder' in k for k in model_state.keys())
    print(f"Pose encoding enabled: {use_pose_encoding}")

    # Check for phase output (Fibonacci models with phase blending)
    use_phase_output = any('16:19' in str(k) or 'phase' in k.lower() for k in model_state.keys())
    # Better detection: check output dimension from checkpoint config
    config = checkpoint.get('config', {})
    if isinstance(config, dict):
        use_phase_output = config.get('use_phase_blending', False) or config.get('use_phase_output', False)

    # Create model based on experiment type
    print(f"Experiment type: {experiment}")
    if experiment == 4:
        print(f"Creating FibonacciPatchDecoder with {n_spiral_points} spiral points")
        model = FibonacciPatchDecoder(
            n_spiral_points=n_spiral_points,
            gaussians_per_point=1,
            use_pose_encoding=use_pose_encoding,
            use_phase_output=use_phase_output,
        ).to(device)
    else:
        model = DirectPatchDecoder(
            gaussians_per_patch=4,
            use_pose_encoding=use_pose_encoding,
        ).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # Load and preprocess image
    print(f"Loading image: {input_image}")
    img = Image.open(input_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Try to extract real features using DINOv2, fall back to dummy
    try:
        from transformers import AutoImageProcessor, AutoModel
        print("Extracting DINOv2 features...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        dinov2 = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
        dinov2.eval()

        # Resize image to 518x518 for consistent 37x37 patch grid
        img_resized = img.resize((518, 518))
        inputs = processor(images=img_resized, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dinov2(**inputs, output_hidden_states=True)
            # Get patch features (excluding CLS token)
            features = outputs.last_hidden_state[:, 1:, :]  # (1, num_patches, 384)
            num_patches = features.shape[1]
            grid_size = int(np.sqrt(num_patches))
            print(f"  Feature grid: {grid_size}x{grid_size} = {num_patches} patches")
            features = features.reshape(1, grid_size, grid_size, 384).permute(0, 3, 1, 2)  # (1, 384, H, W)
            # Interpolate to 37x37 if needed
            if grid_size != 37:
                features = torch.nn.functional.interpolate(features, size=(37, 37), mode='bilinear', align_corners=False)

        # Generate simple depth (could use depth estimator for better results)
        print("Using simple gradient depth (for testing)")
        depth = torch.linspace(0.3, 0.7, 518, device=device, dtype=torch.float32).view(1, 1, 1, 518).expand(1, 1, 518, 518)

    except Exception as e:
        print(f"Could not load DINOv2: {e}")
        print("Using random features (results will be noise)")
        features = torch.randn(1, 384, 37, 37, device=device)
        depth = torch.rand(1, 1, 518, 518, device=device) * 0.5 + 0.25

    # Create renderer (FourierGaussianRenderer for experiment 4, TileBasedRenderer otherwise)
    if experiment == 4:
        print("Using FourierGaussianRenderer (vectorized, matches training)")
        renderer = FourierGaussianRenderer(
            image_size, image_size,
            wavelength_r=0.65,
            wavelength_g=0.55,
            wavelength_b=0.45,
            learnable_wavelengths=False,
        )
    else:
        renderer = TileBasedRenderer(image_size, image_size)
    # Camera: fx, fy, cx, cy, width, height
    focal = image_size  # Simple focal length = image size
    camera = Camera(focal, focal, image_size/2, image_size/2, image_size, image_size)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate views at different poses
    print(f"Generating {num_views} novel views...")
    elevation_range = (-30, 45)  # degrees
    azimuth_range = (0, 360)  # degrees

    results = []
    for i in range(num_views):
        # Sample pose
        if i == 0:
            el, az = 0, 0  # Frontal view
        else:
            el = np.random.uniform(*elevation_range)
            az = np.random.uniform(*azimuth_range)

        el_rad = np.radians(el)
        az_rad = np.radians(az)

        print(f"  View {i}: elevation={el:.1f}°, azimuth={az:.1f}°")

        with torch.no_grad():
            # Forward pass with pose (ensure float32, not float64)
            elevation_t = torch.tensor([el_rad], device=device, dtype=torch.float32)
            azimuth_t = torch.tensor([az_rad], device=device, dtype=torch.float32)

            if use_pose_encoding:
                output = model(features, depth, elevation=elevation_t, azimuth=azimuth_t)
            else:
                output = model(features, depth)

            # Render (squeeze batch dimension - renderer expects (N, 3) not (B, N, 3))
            rendered_img = renderer(
                output['positions'][0],      # (N, 3)
                output['scales'][0],         # (N, 3)
                output['rotations'][0],      # (N, 4)
                output['colors'][0],         # (N, 3)
                output['opacities'][0],      # (N,)
                camera
            )

            # Save - renderer returns (H, W, 3) or (3, H, W) tensor directly
            if rendered_img.dim() == 3 and rendered_img.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                img_np = rendered_img.permute(1, 2, 0).cpu().numpy()
            else:
                # Already (H, W, 3)
                img_np = rendered_img.cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            # Compute brightness stats
            brightness = img_np.mean()
            results.append({'view': i, 'elevation': el, 'azimuth': az, 'brightness': brightness})

            # Save image
            view_path = output_path / f"view_{i:02d}_el{el:.0f}_az{az:.0f}.png"
            Image.fromarray(img_np).save(view_path)
            print(f"    Saved: {view_path} (brightness: {brightness:.1f})")

    # Summary
    print("\n" + "=" * 60)
    print("NOVEL VIEW TEST RESULTS")
    print("=" * 60)
    avg_brightness = np.mean([r['brightness'] for r in results])
    min_brightness = min(r['brightness'] for r in results)
    max_brightness = max(r['brightness'] for r in results)
    print(f"Average brightness: {avg_brightness:.1f}")
    print(f"Min brightness: {min_brightness:.1f}")
    print(f"Max brightness: {max_brightness:.1f}")

    if avg_brightness < 30:
        print("\n⚠️  WARNING: Views are very dark! Training may not have fixed the issue.")
        print("    Consider training longer or checking pose encoding.")
    elif avg_brightness < 80:
        print("\n⚠️  CAUTION: Views are somewhat dark. May need more training.")
    else:
        print("\n✅ SUCCESS: Views have good brightness. Ready to regenerate CVS data!")

    print(f"\nResults saved to: {output_path}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gaussian Decoder Inference")
    parser.add_argument('--test_novel_views', action='store_true',
                        help='Test decoder novel view rendering (validation checkpoint)')
    parser.add_argument('--input', '-i', type=str, help='Input image for novel view test')
    parser.add_argument('--checkpoint', '-c', type=str, help='PyTorch checkpoint path')
    parser.add_argument('--output_dir', '-o', type=str, default='decoder_novel_view_test',
                        help='Output directory for test results')
    parser.add_argument('--num_views', type=int, default=8, help='Number of views to generate')
    parser.add_argument('--experiment', type=int, default=2, choices=[2, 4],
                        help='Experiment type: 2=DirectPatchDecoder, 4=FibonacciPatchDecoder')
    parser.add_argument('--n_spiral_points', type=int, default=377,
                        help='Number of spiral points for experiment 4 (Fibonacci)')

    # Legacy positional arguments for binary inference
    parser.add_argument('features_bin', nargs='?', help='Features binary file')
    parser.add_argument('depth_bin', nargs='?', help='Depth binary file')
    parser.add_argument('output_bin', nargs='?', help='Output binary file')

    args = parser.parse_args()

    if args.test_novel_views:
        if not args.input or not args.checkpoint:
            print("Error: --test_novel_views requires --input and --checkpoint")
            sys.exit(1)
        test_novel_views(
            args.input,
            args.checkpoint,
            args.output_dir,
            args.num_views,
            experiment=args.experiment,
            n_spiral_points=args.n_spiral_points,
        )
    elif args.features_bin and args.depth_bin and args.output_bin:
        run_inference(args.features_bin, args.depth_bin, args.output_bin)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
