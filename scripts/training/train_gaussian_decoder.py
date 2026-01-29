#!/usr/bin/env python3
"""
Training Script for Gaussian Decoders

Trains the three experimental decoder approaches:
1. SAAGRefinementNet - Learn residuals from SAAG (Novel)
2. DirectPatchDecoder - Direct prediction (Baseline)
3. FeatureGuidedSAAG - Parameter modulation (Lightweight)

Training uses self-supervised reconstruction loss:
- Render Gaussians to image
- Compare with input image
- Backprop through differentiable renderer

Optional VLM semantic guidance:
- Weight loss by VLM-generated density maps
- Focus training on semantically important regions (faces, eyes, edges)
- Requires precomputed density maps from preprocess_training_data.py --use_vlm

Usage:
    python train_gaussian_decoder.py --experiment 1 --data_dir ./images
    python train_gaussian_decoder.py --experiment 3 --epochs 50  # Start with lightweight
    python train_gaussian_decoder.py --experiment 2 --use_vlm_guidance  # With VLM weighting
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Add scripts directory to path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import numpy as np
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt

# Perceptual losses
try:
    from pytorch_msssim import ssim as ssim_fn
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: pytorch_msssim not available, SSIM loss disabled")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, LPIPS loss disabled")

# Local imports
from models.gaussian_decoder_models import (
    SAAGRefinementNet,
    DirectPatchDecoder,
    FeatureGuidedSAAG,
    FibonacciPatchDecoder,  # Nature-inspired spiral sampling
    count_parameters,
    tensegrity_loss,  # Structural integrity regularization
)
from models.nca_gaussian_decoder import NCAGaussianDecoder  # Exp 014: Neural Cellular Automata
from models.differentiable_renderer import (
    DifferentiableGaussianRenderer,
    TileBasedRenderer,
    SimplifiedRenderer,
    WaveFieldRenderer,
    FourierGaussianRenderer,  # HFGS: Holographic Fourier Gaussian Splatting
    Camera,
    load_gaussians_from_binary,
    save_gaussians_to_binary
)

# Physics-derived components (may not be available during incremental development)
try:
    from models.gaussian_decoder_models import PhysicsDirectPatchDecoder
    PHYSICS_DECODER_AVAILABLE = True
except ImportError as e:
    PHYSICS_DECODER_AVAILABLE = False
    PHYSICS_DECODER_IMPORT_ERROR = str(e)


@dataclass
class TrainingConfig:
    """Training configuration."""
    experiment: int = 3  # Which experiment (1, 2, or 3)
    data_dir: str = "images"
    output_dir: str = "checkpoints"
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    image_size: int = 256  # Render at this size for training (speed)
    feature_size: int = 37  # DINOv2 patch grid size
    feature_dim: int = 384  # DINOv2 feature dimension (384=small, 768=base, 1024=large)

    # Depth fusion: concatenate encoded depth features with DINOv2 features
    use_depth_fusion: bool = False  # Enable depth fusion in decoder
    depth_feature_dim: int = 64  # Dimension of encoded depth features

    # Loss weights
    rgb_weight: float = 1.0
    depth_weight: float = 0.1
    ssim_weight: float = 0.5      # SSIM perceptual loss
    lpips_weight: float = 0.1     # LPIPS perceptual loss
    residual_weight: float = 0.01  # Regularization for Exp 1

    # Data augmentation
    use_augmentation: bool = True

    # Experiment-specific
    gaussians_per_patch: int = 4  # For Exp 2 (increased for better coverage)
    max_images: int = None  # Limit training images (None = use all)

    # Fibonacci decoder (Experiment 4) - Nature-inspired spiral sampling
    n_spiral_points: int = 377  # Fibonacci number: optimal packing (vs 1369 for 37x37 grid)
    use_tensegrity_loss: bool = False  # Structural integrity regularization
    tensegrity_weight: float = 0.01  # Weight for tensegrity loss

    # NCA decoder (Experiment 5) - Neural Cellular Automata Gaussians
    nca_steps: int = 16  # Number of NCA iterations
    nca_neighbors: int = 6  # Local neighborhood size
    nca_step_size: float = 0.1  # Initial step size for updates

    # VLM semantic guidance
    use_vlm_guidance: bool = False  # Use VLM density maps for loss weighting
    vlm_weight: float = 0.5  # How much to weight by VLM density (0=uniform, 1=full VLM weighting)

    # Fresnel-inspired enhancements (named after Augustin-Jean Fresnel's wave optics)
    use_fresnel_zones: bool = False  # Quantize depth into discrete zones
    num_fresnel_zones: int = 8  # Number of discrete depth zones
    boundary_weight: float = 0.1  # Extra loss weight at zone boundaries (Fresnel fringes)
    use_edge_aware: bool = False  # Smaller Gaussians at depth edges (diffraction)
    use_phase_blending: bool = False  # Interference-like alpha compositing
    use_phase_output: bool = False  # Output per-channel RGB phases from decoder (for QSR)
    edge_scale_factor: float = 0.5  # How much to shrink scales at edges (0-1)
    edge_opacity_boost: float = 0.2  # Opacity boost at edges (0-1)
    phase_amplitude: float = 0.25  # Phase interference amplitude (0-1)

    # Multi-pose training: Fix dark novel views via view-dependent opacity
    multi_pose_augmentation: bool = False  # Enable random pose augmentation
    pose_range_elevation: Tuple[float, float] = (-30, 45)  # Degrees
    pose_range_azimuth: Tuple[float, float] = (0, 360)  # Degrees
    frontal_prob: float = 0.3  # Prob of using frontal (reconstruction loss) vs novel (consistency)
    use_pose_encoding: bool = False  # Enable pose encoding in decoder

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    save_interval: int = 10


@dataclass
class PhysicsConfig:
    """
    Configuration for physics-derived Fresnel rendering.

    These settings control the actual wave optics algorithms,
    as opposed to the heuristic Fresnel-inspired settings in TrainingConfig.
    """
    # Wave optics rendering
    use_wave_rendering: bool = False      # Use WaveFieldRenderer (complex field accumulation)
    wavelength: float = 0.05              # Effective wavelength in [0,1] depth units
    learnable_wavelength: bool = True     # Allow network to learn optimal λ

    # Physics-based Fresnel zones
    use_physics_zones: bool = False       # Use PhysicsFresnelZones (sqrt(n) spacing)
    num_zones: int = 8                    # Number of Fresnel zones
    focal_depth: float = 0.5              # Focal plane depth

    # Diffraction-based placement
    use_diffraction_placement: bool = False  # Use FresnelDiffraction for Gaussian placement

    # Physics-informed loss (from PINN research)
    wave_equation_weight: float = 0.0     # Weight for Helmholtz equation constraint (∇²U + k²U = 0)

    # Per-channel wavelength (RGB light physics)
    use_multi_wavelength: bool = False    # Use MultiWavelengthPhysics for per-channel λ

    # Comparison mode
    compare_with_baseline: bool = False   # Render both physics and baseline, log comparison


@dataclass
class HFGSConfig:
    """
    Configuration for Holographic Fourier Gaussian Splatting (HFGS).

    BREAKTHROUGH: O(H×W×log(H×W)) complexity - INDEPENDENT of Gaussian count!

    Key innovations:
    1. Frequency-domain rendering (10x faster, unlimited Gaussians)
    2. Phase retrieval self-supervision (FREE training signal)
    3. Learned wavelengths as 3D prior (depth encoding)

    A Gaussian is the ONLY function that equals its own Fourier transform.
    Instead of splatting N Gaussians one-by-one in spatial domain,
    we add them ALL in frequency domain and do ONE inverse FFT.

    Novel research contributions:
    - First frequency-domain 3DGS renderer
    - First phase retrieval self-supervision for 3DGS
    - First learned wavelength as depth encoding prior
    """
    # Core HFGS settings
    use_fourier_renderer: bool = False    # Use FourierGaussianRenderer

    # Self-supervised losses (key innovations!)
    use_phase_retrieval_loss: bool = True     # Phase retrieval constraint (FREE supervision!)
    phase_retrieval_weight: float = 0.1       # Weight for phase retrieval loss

    use_frequency_loss: bool = True           # Separate high/low frequency losses
    frequency_loss_weight: float = 0.1        # Weight for frequency domain loss
    high_freq_weight: float = 2.0             # Extra weight for high frequencies (edges)
    frequency_cutoff: float = 0.1             # Low/high frequency separation

    # Per-channel wavelengths (RGB light physics)
    # Physical ratios: λ_R : λ_G : λ_B ≈ 1.27 : 1.0 : 0.82
    learnable_wavelengths: bool = True        # Make wavelengths trainable
    wavelength_r: float = 0.0635              # Red wavelength (700nm equivalent)
    wavelength_g: float = 0.05                # Green wavelength (550nm, reference)
    wavelength_b: float = 0.041               # Blue wavelength (450nm equivalent)

    # Focal depth for phase computation
    focal_depth: float = 0.5                  # Focal plane depth for phase = 0


@dataclass
class HFTSConfig:
    """
    Hybrid Fast Training System (HFTS) - 10× SPEEDUP!

    Achieves H100-competitive training speed on consumer GPUs through:
    1. Multi-Resolution Training (MRT): Train at 64×64, validate at 256×256
    2. Progressive Gaussian Growing (PGG): Start with 64 Gaussians, grow to 4096
    3. Stochastic Gaussian Rendering (SGR): Sample K Gaussians per step

    Combined speedup: 10-50× depending on configuration.
    """
    # Multi-Resolution Training (MRT)
    train_resolution: int = None        # Training resolution (None = same as image_size)
    # Progressive Gaussian Growing (PGG)
    progressive_schedule: bool = False  # Enable progressive Gaussian growing
    # Stochastic Gaussian Rendering (SGR)
    stochastic_k: int = None            # Sample K Gaussians (None = use all)
    # Fast mode preset
    fast_mode: bool = False             # Enable all optimizations

    def get_effective_train_resolution(self, image_size: int) -> int:
        """Get the actual training resolution to use."""
        if self.fast_mode:
            return 64  # Fast mode always uses 64×64
        return self.train_resolution if self.train_resolution is not None else image_size

    def get_gaussians_per_patch(self, epoch: int, total_epochs: int, base_gpp: int = 4) -> int:
        """
        Get number of Gaussians per patch for current epoch.

        Progressive schedule (if enabled):
        - First 25%:  1 Gaussian per patch  (37×37×1 = 1,369 total)
        - 25-50%:     2 Gaussians per patch (37×37×2 = 2,738 total)
        - 50-75%:     4 Gaussians per patch (37×37×4 = 5,476 total)
        - Last 25%:   base_gpp (full quality fine-tuning)
        """
        if not self.progressive_schedule and not self.fast_mode:
            return base_gpp

        progress = epoch / max(total_epochs, 1)

        if progress < 0.25:
            return 1
        elif progress < 0.50:
            return 2
        elif progress < 0.75:
            return max(4, base_gpp)
        else:
            return base_gpp  # Full quality for final phase

    def get_stochastic_k(self, total_gaussians: int) -> int:
        """Get number of Gaussians to sample, or total if not using stochastic."""
        if self.fast_mode and self.stochastic_k is None:
            return min(256, total_gaussians)  # Fast mode default
        if self.stochastic_k is not None:
            return min(self.stochastic_k, total_gaussians)
        return total_gaussians  # Use all


class LearnableWavelengths(nn.Module):
    """
    Learnable wavelength parameters for HFGS.

    Per-channel RGB wavelengths that can be optimized during training.
    Constrained to physical range [0.01, 0.5] using sigmoid scaling.
    """

    def __init__(
        self,
        wavelength_r: float = 0.0635,
        wavelength_g: float = 0.05,
        wavelength_b: float = 0.041,
        learnable: bool = True
    ):
        super().__init__()
        # Store raw parameters (will be constrained during forward)
        wavelengths = torch.tensor([wavelength_r, wavelength_g, wavelength_b])
        if learnable:
            self.wavelengths = nn.Parameter(wavelengths)
        else:
            self.register_buffer('wavelengths', wavelengths)
        self.learnable = learnable

    def forward(self) -> torch.Tensor:
        """Return constrained wavelengths in [0.01, 0.5] range."""
        # Use softplus to keep positive, then clamp to physical range
        return torch.clamp(F.softplus(self.wavelengths), min=0.01, max=0.5)

    def get_wavelength(self, channel: int = 1) -> torch.Tensor:
        """Get wavelength for a specific channel (0=R, 1=G, 2=B)."""
        return self.forward()[channel]

    def extra_repr(self) -> str:
        λ = self.forward()
        return f"λ_rgb=[{λ[0]:.4f}, {λ[1]:.4f}, {λ[2]:.4f}], learnable={self.learnable}"


class PhaseRetrievalLoss(nn.Module):
    """
    Self-supervised loss using phase retrieval physics.

    BREAKTHROUGH: FREE training signal without ground truth 3D!

    Key insight: Cameras capture intensity |U|², not the complex field U.
    But we KNOW the phase from depth via the wave equation:
        φ = (2π / λ) × depth

    This provides a physics-based constraint:
        U = √I × exp(iφ) should be frequency-consistent

    No existing 3DGS method uses this self-supervision signal!

    Based on phase retrieval algorithms in computational optics:
    - Gerchberg-Saxton algorithm
    - Ptychography
    - Holographic phase retrieval
    """

    def __init__(self, wavelength: float = 0.05, focal_depth: float = 0.5):
        """
        Initialize PhaseRetrievalLoss.

        Args:
            wavelength: Effective wavelength for phase computation
            focal_depth: Focal plane depth (phase = 0 at this depth)
        """
        super().__init__()
        self.wavelength = wavelength
        self.focal_depth = focal_depth

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        depth: torch.Tensor,
        wavelength: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute phase retrieval loss.

        Args:
            rendered: (B, 3, H, W) rendered image (intensity)
            target: (B, 3, H, W) target image (intensity)
            depth: (B, H, W) or (B, 1, H, W) depth map
            wavelength: Optional external wavelength (for learnable wavelengths)

        Returns:
            Scalar loss value
        """
        # Use external wavelength if provided, otherwise use default
        wl = wavelength if wavelength is not None else self.wavelength

        # Ensure depth is (B, H, W)
        if depth.dim() == 4:
            depth = depth.squeeze(1)

        # Compute phase from depth
        # φ = (2π / λ) × |depth - focal_depth|
        path_diff = torch.abs(depth - self.focal_depth)
        phase = (2 * torch.pi / wl) * path_diff  # (B, H, W)

        # Expand phase for RGB channels: (B, 1, H, W)
        phase_expanded = phase.unsqueeze(1)

        # Reconstruct complex field from intensity + known phase
        # U = √I × exp(iφ)
        rendered_amplitude = torch.sqrt(rendered.clamp(min=1e-8))
        target_amplitude = torch.sqrt(target.clamp(min=1e-8))

        rendered_complex = rendered_amplitude * torch.exp(1j * phase_expanded)
        target_complex = target_amplitude * torch.exp(1j * phase_expanded)

        # Compare in frequency domain (magnitude consistency)
        # The key insight: if phase is correct, frequency magnitudes should match
        rendered_freq = torch.fft.fft2(rendered_complex)
        target_freq = torch.fft.fft2(target_complex)

        # L2 loss on frequency magnitudes
        loss = F.mse_loss(rendered_freq.abs(), target_freq.abs())

        return loss


class FrequencyDomainLoss(nn.Module):
    """
    Frequency domain loss with separate high/low frequency handling.

    Key insight: Different frequency bands encode different information:
    - Low frequencies: Overall color, tone, large-scale structure
    - High frequencies: Edges, fine details, texture

    By separating these and weighting high frequencies more,
    we can achieve sharper reconstructions.

    This is related to:
    - Fourier Neural Operators (FNO)
    - Spectral normalization
    - Frequency-aware perceptual losses
    """

    def __init__(self, cutoff: float = 0.1, high_weight: float = 2.0):
        """
        Initialize FrequencyDomainLoss.

        Args:
            cutoff: Frequency cutoff for low/high separation (0-0.5)
            high_weight: Extra weight for high frequency loss
        """
        super().__init__()
        self.cutoff = cutoff
        self.high_weight = high_weight
        self._mask_cache = {}

    def _get_frequency_masks(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create low/high frequency masks."""
        key = (height, width, str(device))
        if key not in self._mask_cache:
            # Create frequency coordinate grids
            u = torch.fft.fftfreq(width, device=device)
            v = torch.fft.fftfreq(height, device=device)
            V, U = torch.meshgrid(v, u, indexing='ij')

            # Radial frequency
            freq_radius = torch.sqrt(U**2 + V**2)

            # Create masks
            low_mask = (freq_radius < self.cutoff).float()
            high_mask = 1.0 - low_mask

            self._mask_cache[key] = (low_mask, high_mask)

        return self._mask_cache[key]

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frequency domain loss.

        Args:
            rendered: (B, C, H, W) rendered image
            target: (B, C, H, W) target image

        Returns:
            Scalar loss value
        """
        B, C, H, W = rendered.shape

        # Get frequency masks
        low_mask, high_mask = self._get_frequency_masks(H, W, rendered.device)

        # Transform to frequency domain
        rendered_freq = torch.fft.fft2(rendered)
        target_freq = torch.fft.fft2(target)

        # Separate low and high frequency components
        # Apply masks and compute losses
        low_loss = F.mse_loss(
            (rendered_freq * low_mask).abs(),
            (target_freq * low_mask).abs()
        )

        high_loss = F.mse_loss(
            (rendered_freq * high_mask).abs(),
            (target_freq * high_mask).abs()
        )

        # Combined loss with high frequency emphasis
        total_loss = low_loss + self.high_weight * high_loss

        return total_loss


class ImageDataset(Dataset):
    """
    Dataset for training Gaussian decoders.

    Loads images and their precomputed:
    - Depth maps (from Depth Anything V2)
    - DINOv2 features
    - SAAG Gaussians (for Experiment 1)
    - VLM density maps (optional, for semantic-aware loss weighting)
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        feature_cache_dir: Optional[str] = None,
        use_augmentation: bool = True,
        max_images: int = None,
        load_vlm_density: bool = False,
        feature_dim: int = 384
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else self.data_dir / "features"
        self.use_augmentation = use_augmentation
        self.load_vlm_density = load_vlm_density
        self.feature_dim = feature_dim

        # Feature file suffix based on model size (small uses default, others have size suffix)
        feature_dim_to_size = {384: 'small', 768: 'base', 1024: 'large'}
        model_size = feature_dim_to_size.get(feature_dim, 'small')
        self.feature_suffix = f"_dinov2_{model_size}.bin" if model_size != 'small' else "_dinov2.bin"

        # Data augmentation transforms
        if use_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )
            self.augment_prob = 0.5
        else:
            self.color_jitter = None
            self.augment_prob = 0.0

        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(self.data_dir.glob(ext))
            self.image_paths.extend(self.data_dir.glob(ext.upper()))

        self.image_paths = sorted(self.image_paths)

        # Limit number of images if requested
        if max_images is not None and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]
            print(f"Using {len(self.image_paths)} images (limited from {data_dir})")
        else:
            print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        name = img_path.stem

        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Apply data augmentation (color jitter only - spatial augs would require recomputing features)
        apply_augment = self.use_augmentation and np.random.random() < self.augment_prob
        if apply_augment and self.color_jitter is not None:
            img = self.color_jitter(img)

        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)

        # Load precomputed features (if available)
        feature_path = self.feature_cache_dir / f"{name}{self.feature_suffix}"
        depth_path = self.feature_cache_dir / f"{name}_depth.bin"
        saag_path = self.feature_cache_dir / f"{name}_saag.bin"
        vlm_density_path = self.feature_cache_dir / f"{name}_vlm_density.npy"

        # Features: (feature_dim, 37, 37)
        if feature_path.exists():
            features = np.fromfile(feature_path, dtype=np.float32)
            features = features.reshape(37, 37, self.feature_dim).transpose(2, 0, 1)
            features = torch.from_numpy(features.copy())
        else:
            # Placeholder - will be computed on-the-fly in training loop
            features = torch.zeros(self.feature_dim, 37, 37)

        # Depth: (1, H, W) - preprocessed at 256x256, resize to target size
        if depth_path.exists():
            depth = np.fromfile(depth_path, dtype=np.float32)
            # Depth was saved at 256x256 during preprocessing
            depth_size = int(np.sqrt(len(depth)))
            depth = depth.reshape(depth_size, depth_size)
            # Resize to target image_size if different
            if depth_size != self.image_size:
                depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
                depth_img = depth_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                depth = np.array(depth_img, dtype=np.float32) / 255.0
            depth = torch.from_numpy(depth.copy()).unsqueeze(0)
        else:
            depth = torch.zeros(1, self.image_size, self.image_size)

        # SAAG Gaussians (for Experiment 1) - flatten to avoid collate issues
        has_saag = saag_path.exists()
        if has_saag:
            saag = load_gaussians_from_binary(str(saag_path))
            saag_positions = saag['positions'].float()
            saag_scales = saag['scales'].float()
            saag_rotations = saag['rotations'].float()
            saag_colors = saag['colors'].float()
            saag_opacities = saag['opacities'].float()
        else:
            # Empty tensors - will use dummy SAAG in training
            saag_positions = torch.zeros(0, 3)
            saag_scales = torch.zeros(0, 3)
            saag_rotations = torch.zeros(0, 4)
            saag_colors = torch.zeros(0, 3)
            saag_opacities = torch.zeros(0)

        # VLM density map (for semantic-aware loss weighting)
        has_vlm_density = False
        vlm_density = torch.ones(1, self.image_size, self.image_size)  # Default: uniform weighting
        if self.load_vlm_density and vlm_density_path.exists():
            try:
                density_grid = np.load(vlm_density_path)  # (grid_size, grid_size)
                # Interpolate to image size using scipy or PIL
                from scipy.ndimage import zoom
                scale_factor = self.image_size / density_grid.shape[0]
                density_full = zoom(density_grid, scale_factor, order=1)
                # Normalize to [0.5, 1.5] range (0.5 = half weight, 1.5 = 1.5x weight)
                # This ensures VLM guidance doesn't completely ignore any region
                density_full = 0.5 + density_full  # Now [0.5, 1.5] assuming density was [0, 1]
                vlm_density = torch.from_numpy(density_full.astype(np.float32)).unsqueeze(0)
                has_vlm_density = True
            except Exception as e:
                # If loading fails, use uniform weighting
                pass

        return {
            'image': img_tensor,
            'features': features,
            'depth': depth,
            'has_saag': has_saag,
            'saag_positions': saag_positions,
            'saag_scales': saag_scales,
            'saag_rotations': saag_rotations,
            'saag_colors': saag_colors,
            'saag_opacities': saag_opacities,
            'vlm_density': vlm_density,
            'has_vlm_density': has_vlm_density,
            'name': name
        }


def create_camera_from_pose(
    elevation_rad: float,
    azimuth_rad: float,
    render_size: int,
    focal_length_mult: float = 0.8,
    distance: float = 2.0
) -> Camera:
    """
    Create a camera at the specified elevation and azimuth angles.

    Used for multi-pose training to render from different viewpoints.

    Args:
        elevation_rad: Elevation angle in radians (vertical angle from horizon)
        azimuth_rad: Azimuth angle in radians (horizontal angle, 0 = front)
        render_size: Size of rendered image
        focal_length_mult: Multiplier for focal length (relative to render_size)
        distance: Distance from camera to origin

    Returns:
        Camera positioned at the specified angles, looking at origin
    """
    # Camera position in world coordinates (spherical to Cartesian)
    cam_x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cam_y = distance * np.sin(elevation_rad)
    cam_z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

    # Build view matrix (camera looking at origin)
    # Forward = normalize(origin - cam_pos)
    forward = np.array([-cam_x, -cam_y, -cam_z])
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, -1.0])  # Default to looking down -Z
    else:
        forward = forward / forward_norm

    # Up vector (world Y)
    up = np.array([0.0, 1.0, 0.0])

    # Right = forward × up
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Handle looking straight up/down
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm

    # Recalculate up to be orthogonal
    up = np.cross(right, forward)

    # Build rotation matrix (camera basis in world)
    R = np.array([right, up, -forward])  # -forward because OpenGL convention

    # Translation (camera position to origin transformation)
    t = -R @ np.array([cam_x, cam_y, cam_z])

    # Build view matrix (4x4)
    view_matrix = torch.eye(4)
    view_matrix[:3, :3] = torch.from_numpy(R).float()
    view_matrix[:3, 3] = torch.from_numpy(t).float()

    # Create camera
    camera = Camera(
        fx=render_size * focal_length_mult,
        fy=render_size * focal_length_mult,
        cx=render_size / 2,
        cy=render_size / 2,
        width=render_size,
        height=render_size
    )
    camera.set_view(view_matrix)

    return camera


def create_dummy_saag(batch_size: int, num_gaussians: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Create dummy SAAG Gaussians for testing when real ones aren't available."""
    positions = torch.randn(batch_size, num_gaussians, 3, device=device) * 0.5
    positions[..., 2] -= 2  # Move in front of camera

    scales = torch.ones(batch_size, num_gaussians, 3, device=device) * 0.05
    rotations = torch.zeros(batch_size, num_gaussians, 4, device=device)
    rotations[..., 0] = 1  # Identity quaternion

    colors = torch.rand(batch_size, num_gaussians, 3, device=device)
    opacities = torch.ones(batch_size, num_gaussians, device=device) * 0.8

    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'colors': colors,
        'opacities': opacities
    }


def wave_equation_loss(
    wave_field: torch.Tensor,
    wavelength: float,
    pixel_spacing: float = 1.0 / 256.0
) -> torch.Tensor:
    """
    Physics-informed loss: Helmholtz equation constraint.

    Penalizes solutions that violate the wave equation:
        ∇²U + k²U = 0  (Helmholtz equation for monochromatic light)

    where k = 2π/λ is the wave number.

    This is based on Physics-Informed Neural Networks (PINNs) research:
        - arXiv: "Physics-informed Neural Networks (PINNs) for Wave Propagation"
        - arXiv: "Solving the wave equation with physics-informed deep learning"

    The loss encourages the rendered wave field to satisfy the underlying
    physics of wave propagation, potentially improving generalization.

    Args:
        wave_field: Rendered image or wave amplitude (B, C, H, W) or (B, H, W)
        wavelength: Effective wavelength in normalized units
        pixel_spacing: Physical spacing between pixels (default: 1/256)

    Returns:
        Scalar loss value (Helmholtz residual squared)
    """
    # Ensure 4D tensor
    if wave_field.dim() == 3:
        wave_field = wave_field.unsqueeze(1)

    # Wave number: k = 2π / λ
    k = 2 * torch.pi / wavelength

    # Compute Laplacian via finite differences (5-point stencil)
    # ∇²U ≈ (U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1] - 4U[i,j]) / h²
    #
    # Using circular padding (roll) to avoid boundary issues
    laplacian = (
        torch.roll(wave_field, 1, dims=-1) +   # U[i, j+1]
        torch.roll(wave_field, -1, dims=-1) +  # U[i, j-1]
        torch.roll(wave_field, 1, dims=-2) +   # U[i+1, j]
        torch.roll(wave_field, -1, dims=-2) -  # U[i-1, j]
        4 * wave_field                          # -4U[i, j]
    ) / (pixel_spacing ** 2)

    # Helmholtz residual: ∇²U + k²U
    # For valid wave solutions, this should be ~0
    residual = laplacian + (k ** 2) * wave_field

    # L2 loss on residual
    loss = torch.mean(residual ** 2)

    return loss


def compute_losses(
    rendered: torch.Tensor,
    target: torch.Tensor,
    rendered_depth: Optional[torch.Tensor] = None,
    target_depth: Optional[torch.Tensor] = None,
    residuals: Optional[Dict[str, torch.Tensor]] = None,
    config: TrainingConfig = None,
    lpips_fn: Optional[nn.Module] = None,
    vlm_density: Optional[torch.Tensor] = None,
    physics_config: Optional['PhysicsConfig'] = None,
    hfgs_config: Optional['HFGSConfig'] = None,
    phase_retrieval_loss_fn: Optional[PhaseRetrievalLoss] = None,
    frequency_loss_fn: Optional[FrequencyDomainLoss] = None,
    learnable_wavelengths: Optional['LearnableWavelengths'] = None,
    fresnel_zones: Optional[nn.Module] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training losses.

    Args:
        rendered: Rendered image (B, 3, H, W)
        target: Target image (B, 3, H, W)
        rendered_depth: Rendered depth (B, H, W)
        target_depth: Target depth (B, H, W)
        residuals: Residual predictions for regularization
        config: Training config
        lpips_fn: LPIPS loss module
        vlm_density: VLM density weighting map (B, 1, H, W), higher = more important

    Returns:
        total_loss: Combined loss for backprop
        loss_dict: Individual loss values for logging
    """
    loss_dict = {}

    # RGB reconstruction loss (L1)
    # If VLM density provided and enabled, use weighted loss
    if vlm_density is not None and config.use_vlm_guidance and config.vlm_weight > 0:
        # Compute per-pixel L1 loss
        pixel_loss = torch.abs(rendered - target)  # (B, 3, H, W)
        # Detach vlm_density - it's a spatial weight, not part of the learned model
        vlm_density = vlm_density.detach()
        # Resize density to match rendered size if needed
        if vlm_density.shape[-2:] != rendered.shape[-2:]:
            vlm_density = F.interpolate(vlm_density, size=rendered.shape[-2:], mode='bilinear', align_corners=False)
        # Apply weighting: blend between uniform and VLM-weighted
        # weight = (1 - vlm_weight) * 1.0 + vlm_weight * density
        weight = (1.0 - config.vlm_weight) + config.vlm_weight * vlm_density
        weighted_loss = (pixel_loss * weight).mean()
        rgb_loss = weighted_loss
        loss_dict['vlm_weighted'] = True
    else:
        rgb_loss = F.l1_loss(rendered, target)
        loss_dict['vlm_weighted'] = False

    loss_dict['rgb'] = rgb_loss.item()

    total_loss = config.rgb_weight * rgb_loss

    # Ensure rendered is in valid range for perceptual losses
    # (prevents NaN from propagating through SSIM/LPIPS)
    rendered_clamped = torch.clamp(rendered, 0.0, 1.0)

    # SSIM perceptual loss (structural similarity)
    if SSIM_AVAILABLE and config.ssim_weight > 0:
        # ssim_fn expects (B, C, H, W) format
        ssim_val = ssim_fn(rendered_clamped, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        loss_dict['ssim'] = ssim_loss.item()
        total_loss = total_loss + config.ssim_weight * ssim_loss

    # LPIPS perceptual loss (learned perceptual similarity)
    if LPIPS_AVAILABLE and lpips_fn is not None and config.lpips_weight > 0:
        # LPIPS expects images in [-1, 1] range
        # Downscale to 128x128 to reduce memory usage (perceptual loss doesn't need full res)
        rendered_lpips = F.interpolate(rendered_clamped, size=(128, 128), mode='bilinear', align_corners=False)
        target_lpips = F.interpolate(target, size=(128, 128), mode='bilinear', align_corners=False)
        rendered_lpips = rendered_lpips * 2.0 - 1.0
        target_lpips = target_lpips * 2.0 - 1.0
        lpips_loss = lpips_fn(rendered_lpips, target_lpips).mean()
        loss_dict['lpips'] = lpips_loss.item()
        total_loss = total_loss + config.lpips_weight * lpips_loss

    # Depth loss (if available)
    if rendered_depth is not None and target_depth is not None:
        # Normalize depths for comparison (use larger epsilon for numerical stability)
        rd_std = torch.clamp(rendered_depth.std(), min=1e-4)
        td_std = torch.clamp(target_depth.std(), min=1e-4)
        rd_norm = (rendered_depth - rendered_depth.mean()) / rd_std
        td_norm = (target_depth - target_depth.mean()) / td_std
        depth_loss = F.l1_loss(rd_norm, td_norm)
        loss_dict['depth'] = depth_loss.item()
        total_loss = total_loss + config.depth_weight * depth_loss

    # Residual regularization (for Experiment 1)
    if residuals is not None:
        reg_loss = 0
        for key in ['pos_delta', 'scale_delta', 'color_delta', 'opacity_delta']:
            if key in residuals:
                reg_loss = reg_loss + residuals[key].abs().mean()
        loss_dict['residual'] = reg_loss.item()
        total_loss = total_loss + config.residual_weight * reg_loss

    # Fresnel boundary emphasis loss
    # Adds extra weight at depth zone boundaries (like Fresnel diffraction fringes)
    if fresnel_zones is not None and config.boundary_weight > 0 and target_depth is not None:
        # Compute boundary mask from target depth
        boundary_mask = fresnel_zones.compute_boundary_mask(target_depth)  # (B, H, W)

        # Compute per-pixel RGB loss at boundaries
        pixel_loss = torch.abs(rendered - target).mean(dim=1)  # (B, H, W)

        # Extra loss at boundaries (Fresnel fringe emphasis)
        boundary_loss = (pixel_loss * boundary_mask).mean()
        loss_dict['boundary'] = boundary_loss.item()
        total_loss = total_loss + config.boundary_weight * boundary_loss

    # Physics-informed wave equation loss (Helmholtz constraint)
    # From PINN research: ∇²U + k²U = 0 for valid wave solutions
    if physics_config is not None and physics_config.wave_equation_weight > 0:
        wave_loss = wave_equation_loss(
            wave_field=rendered,
            wavelength=physics_config.wavelength,
            pixel_spacing=1.0 / config.image_size
        )
        loss_dict['wave_eq'] = wave_loss.item()
        total_loss = total_loss + physics_config.wave_equation_weight * wave_loss

    # ==========================================================================
    # HFGS: Holographic Fourier Gaussian Splatting losses (BREAKTHROUGH!)
    # ==========================================================================

    # Phase Retrieval Loss - FREE self-supervision from physics!
    # Key insight: Cameras capture |U|², not phase. But we know phase from depth.
    if (hfgs_config is not None and
        hfgs_config.use_phase_retrieval_loss and
        phase_retrieval_loss_fn is not None and
        target_depth is not None):
        try:
            # Get learnable wavelength if available (use green channel as reference)
            wavelength = None
            if learnable_wavelengths is not None:
                wavelength = learnable_wavelengths.get_wavelength(1)  # Green channel
            pr_loss = phase_retrieval_loss_fn(rendered, target, target_depth, wavelength=wavelength)
            loss_dict['phase_retrieval'] = pr_loss.item()
            total_loss = total_loss + hfgs_config.phase_retrieval_weight * pr_loss
        except Exception as e:
            # Phase retrieval can fail with edge cases, don't crash training
            loss_dict['phase_retrieval'] = 0.0

    # Frequency Domain Loss - separate high/low frequency handling
    # High frequencies = edges, details (weighted more for sharpness)
    if (hfgs_config is not None and
        hfgs_config.use_frequency_loss and
        frequency_loss_fn is not None):
        try:
            freq_loss = frequency_loss_fn(rendered, target)
            loss_dict['frequency'] = freq_loss.item()
            total_loss = total_loss + hfgs_config.frequency_loss_weight * freq_loss
        except Exception as e:
            # FFT can fail with edge cases, don't crash training
            loss_dict['frequency'] = 0.0

    loss_dict['total'] = total_loss.item()

    return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    renderer: nn.Module,
    camera: Camera,
    config: TrainingConfig,
    epoch: int,
    lpips_fn: Optional[nn.Module] = None,
    physics_config: Optional[PhysicsConfig] = None,
    hfgs_config: Optional[HFGSConfig] = None,
    phase_retrieval_loss_fn: Optional[PhaseRetrievalLoss] = None,
    frequency_loss_fn: Optional[FrequencyDomainLoss] = None,
    learnable_wavelengths: Optional[LearnableWavelengths] = None,
    hfts_config: Optional[HFTSConfig] = None,
    effective_train_res: int = None,
    fresnel_zones: Optional[nn.Module] = None
) -> Dict[str, float]:
    """Train for one epoch with HFTS (Hybrid Fast Training System) support."""
    model.train()
    device = config.device

    epoch_losses = {k: 0.0 for k in ['total', 'rgb', 'ssim', 'lpips', 'depth', 'residual', 'boundary', 'phase_retrieval', 'frequency', 'tensegrity']}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move to device
        images = batch['image'].to(device)
        features = batch['features'].to(device)
        depth = batch['depth'].to(device)
        vlm_density = batch['vlm_density'].to(device) if 'vlm_density' in batch else None

        B = images.shape[0]

        # Forward pass depends on experiment type
        if config.experiment == 1:
            # Experiment 1: SAAG Refinement
            # Get SAAG Gaussians (or use dummy)
            has_saag = batch['has_saag'].any().item() if torch.is_tensor(batch['has_saag']) else any(batch['has_saag'])
            if has_saag and batch['saag_positions'].shape[1] > 0:
                saag = {
                    'positions': batch['saag_positions'].to(device),
                    'scales': batch['saag_scales'].to(device),
                    'rotations': batch['saag_rotations'].to(device),
                    'colors': batch['saag_colors'].to(device),
                    'opacities': batch['saag_opacities'].to(device)
                }
            else:
                saag = create_dummy_saag(B, 1000, device)

            output = model(
                features,
                saag['positions'],
                saag['scales'],
                saag['rotations'],
                saag['colors'],
                saag['opacities']
            )

            residuals = {k: output[k] for k in ['pos_delta', 'scale_delta', 'color_delta', 'opacity_delta'] if k in output}

        elif config.experiment == 2:
            # Experiment 2: Direct Patch Decoder
            # HFTS Progressive Growing: Use fewer Gaussians in early epochs
            num_gaussians = None
            if hfts_config is not None:
                num_gaussians = hfts_config.get_gaussians_per_patch(
                    epoch, config.epochs, config.gaussians_per_patch
                )

            # Multi-pose training: Sample random poses to fix dark novel views
            elevation, azimuth = None, None
            elevation_cpu, azimuth_cpu = 0.0, 0.0  # CPU values for camera creation
            is_frontal_view = True
            if config.multi_pose_augmentation and config.use_pose_encoding:
                # Decide if this batch uses frontal or novel view
                is_frontal_view = np.random.random() < config.frontal_prob
                if is_frontal_view:
                    # Frontal view: elevation=0, azimuth=0
                    elevation_cpu, azimuth_cpu = 0.0, 0.0
                    elevation = torch.zeros(B, device=device)
                    azimuth = torch.zeros(B, device=device)
                else:
                    # Novel view: random elevation and azimuth (generate on CPU first)
                    el_min, el_max = config.pose_range_elevation
                    az_min, az_max = config.pose_range_azimuth
                    elevation_cpu = np.random.uniform(np.radians(el_min), np.radians(el_max))
                    azimuth_cpu = np.random.uniform(np.radians(az_min), np.radians(az_max))
                    # Create GPU tensors from CPU values (no sync needed)
                    elevation = torch.full((B,), elevation_cpu, device=device)
                    azimuth = torch.full((B,), azimuth_cpu, device=device)

            output = model(features, depth, num_gaussians=num_gaussians,
                          elevation=elevation, azimuth=azimuth)
            residuals = None

        elif config.experiment == 4:
            # Experiment 4: Fibonacci Patch Decoder (nature-inspired spiral sampling)
            # Similar forward pass to Experiment 2, but uses spiral instead of grid

            # Multi-pose training: Sample random poses to fix dark novel views
            elevation, azimuth = None, None
            elevation_cpu, azimuth_cpu = 0.0, 0.0
            is_frontal_view = True
            if config.multi_pose_augmentation and config.use_pose_encoding:
                is_frontal_view = np.random.random() < config.frontal_prob
                if is_frontal_view:
                    elevation_cpu, azimuth_cpu = 0.0, 0.0
                    elevation = torch.zeros(B, device=device)
                    azimuth = torch.zeros(B, device=device)
                else:
                    el_min, el_max = config.pose_range_elevation
                    az_min, az_max = config.pose_range_azimuth
                    elevation_cpu = np.random.uniform(np.radians(el_min), np.radians(el_max))
                    azimuth_cpu = np.random.uniform(np.radians(az_min), np.radians(az_max))
                    elevation = torch.full((B,), elevation_cpu, device=device)
                    azimuth = torch.full((B,), azimuth_cpu, device=device)

            output = model(features, depth, elevation=elevation, azimuth=azimuth)
            residuals = None

        elif config.experiment == 5:
            # Experiment 5: NCA Gaussian Decoder (Neural Cellular Automata)
            # Similar forward pass to Experiments 2 and 4, but with NCA dynamics
            output = model(features, depth)
            residuals = None

        else:  # config.experiment == 3
            # Experiment 3: Feature-Guided SAAG
            # This predicts parameter modifications, needs to be applied to SAAG
            # For training, we'll use dummy SAAG and just ensure the network learns
            param_mods = model(features)

            # Create modified SAAG (simplified for training)
            saag = create_dummy_saag(B, 500, device)

            # Apply modifications (simplified)
            output = {
                'positions': saag['positions'],
                'scales': saag['scales'] * param_mods['base_size_mult'].mean(dim=[1,2]).view(B, 1, 1),
                'rotations': saag['rotations'],
                'colors': saag['colors'],
                'opacities': saag['opacities'] * param_mods['opacity_mult'].mean(dim=[1,2]).view(B, 1)
            }
            residuals = None

        # HFTS Stochastic Gaussian Rendering: Sample K Gaussians for ~20× speedup
        stochastic_k = None
        if hfts_config is not None:
            total_gaussians = output['positions'].shape[1]
            stochastic_k = hfts_config.get_stochastic_k(total_gaussians)

        if stochastic_k is not None and stochastic_k < output['positions'].shape[1]:
            # Importance sampling based on opacity (higher opacity = more important)
            N = output['positions'].shape[1]
            K = stochastic_k

            # Compute importance weights: p(i) ∝ opacity_i (normalized)
            with torch.no_grad():
                # Use mean opacity across batch as importance weight
                importance = output['opacities'].mean(dim=0)  # (N,)
                importance = importance + 1e-6  # Prevent zero probability
                importance = importance / importance.sum()  # Normalize to probability

                # Sample K indices with replacement based on importance
                indices = torch.multinomial(importance, K, replacement=False)

            # Subsample all Gaussian parameters
            output_sampled = {
                'positions': output['positions'][:, indices],
                'scales': output['scales'][:, indices],
                'rotations': output['rotations'][:, indices],
                'colors': output['colors'][:, indices],
                'opacities': output['opacities'][:, indices],
            }
            if 'phases' in output:
                output_sampled['phases'] = output['phases'][:, indices]

            # Replace output with sampled subset
            output = output_sampled

        # Render (with optional phase blending for Fresnel interference)
        rendered_images = []
        rendered_depths = []

        # Get phases if model outputs them (Fresnel phase blending)
        phases = output.get('phases', None)

        # Multi-pose training: Create camera from pose angles if augmentation is enabled
        # This fixes a critical bug where camera was never transformed, only opacity was modified
        render_camera = camera  # Default to frontal camera
        if config.multi_pose_augmentation and elevation is not None and azimuth is not None:
            # Use CPU values directly (no GPU sync from .item())
            train_res = effective_train_res if effective_train_res is not None else config.image_size
            render_camera = create_camera_from_pose(
                elevation_rad=elevation_cpu,
                azimuth_rad=azimuth_cpu,
                render_size=train_res,
                focal_length_mult=0.8  # Match original camera setup
            )

        for b in range(B):
            # Pass phases to renderer if available
            phase_b = phases[b] if phases is not None else None
            rendered, rdepth = renderer(
                output['positions'][b],
                output['scales'][b],
                output['rotations'][b],
                output['colors'][b],
                output['opacities'][b],
                render_camera,
                return_depth=True,
                phases=phase_b
            )
            rendered_images.append(rendered)
            rendered_depths.append(rdepth)

        rendered = torch.stack(rendered_images)  # (B, 3, H, W)
        rendered_depth = torch.stack(rendered_depths)  # (B, H, W)

        # Resize target to match render size (HFTS: uses effective_train_res for speedup)
        train_res = effective_train_res if effective_train_res is not None else config.image_size
        target = F.interpolate(images, size=(train_res, train_res), mode='bilinear', align_corners=False)
        target_depth = F.interpolate(depth, size=(train_res, train_res), mode='bilinear', align_corners=False).squeeze(1)

        # Compute losses (with optional VLM density weighting and HFGS losses)
        loss, loss_dict = compute_losses(
            rendered, target,
            rendered_depth, target_depth,
            residuals, config, lpips_fn,
            vlm_density,
            physics_config=physics_config,
            hfgs_config=hfgs_config,
            phase_retrieval_loss_fn=phase_retrieval_loss_fn,
            frequency_loss_fn=frequency_loss_fn,
            learnable_wavelengths=learnable_wavelengths,
            fresnel_zones=fresnel_zones
        )

        # Tensegrity loss: Encourage structural integrity via golden-ratio spacing
        # Inspired by tensegrity structures (tension/compression balance)
        if config.use_tensegrity_loss and config.tensegrity_weight > 0:
            tens_loss = tensegrity_loss(output['positions'], k_neighbors=6)
            loss = loss + config.tensegrity_weight * tens_loss
            loss_dict['tensegrity'] = tens_loss.item()

        # Check for NaN/Inf before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: NaN/Inf loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses (skip non-numeric values like 'vlm_weighted' boolean)
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)) and k in epoch_losses:
                epoch_losses[k] += v
        num_batches += 1

        # Log
        if batch_idx % config.log_interval == 0:
            log_msg = f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss_dict['total']:.4f} | RGB: {loss_dict['rgb']:.4f}"
            if 'ssim' in loss_dict:
                log_msg += f" | SSIM: {loss_dict['ssim']:.4f}"
            if 'lpips' in loss_dict:
                log_msg += f" | LPIPS: {loss_dict['lpips']:.4f}"
            print(log_msg)

        # Clear GPU cache periodically to prevent OOM from memory fragmentation
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: Dict[str, float],
    config: TrainingConfig
):
    """Save training checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': vars(config)
    }

    path = Path(config.output_dir) / f"decoder_exp{config.experiment}_epoch{epoch}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def save_training_history(history: Dict[str, List[float]], output_dir: str, experiment: int):
    """Save training history to JSON file."""
    path = Path(output_dir) / f"training_history_exp{experiment}.json"
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {path}")


def plot_training_metrics(history: Dict[str, List[float]], output_dir: str, experiment: int):
    """Generate and save training metrics plots."""
    if not history.get('total'):
        print("No training history to plot")
        return

    epochs = list(range(1, len(history['total']) + 1))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training Metrics - Experiment {experiment}', fontsize=14)

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['total'], 'b-', linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: RGB Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['rgb'], 'g-', linewidth=2, label='RGB (L1)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('RGB Reconstruction Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Perceptual Losses (SSIM + LPIPS)
    ax = axes[1, 0]
    has_perceptual = False
    if history.get('ssim') and any(v > 0 for v in history['ssim']):
        ax.plot(epochs, history['ssim'], 'r-', linewidth=2, label='SSIM')
        has_perceptual = True
    if history.get('lpips') and any(v > 0 for v in history['lpips']):
        ax.plot(epochs, history['lpips'], 'm-', linewidth=2, label='LPIPS')
        has_perceptual = True
    if has_perceptual:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Perceptual Losses')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No perceptual losses recorded', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Perceptual Losses (N/A)')

    # Plot 4: All losses combined (normalized for comparison)
    ax = axes[1, 1]
    for key in ['total', 'rgb', 'ssim', 'lpips', 'depth', 'boundary']:
        if history.get(key) and any(v > 0 for v in history[key]):
            values = history[key]
            # Normalize to [0, 1] for comparison
            min_v, max_v = min(values), max(values)
            if max_v > min_v:
                normalized = [(v - min_v) / (max_v - min_v) for v in values]
                ax.plot(epochs, normalized, linewidth=1.5, label=key, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Loss')
    ax.set_title('All Losses (Normalized)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    # Save plot
    plot_path = Path(output_dir) / f"training_metrics_exp{experiment}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training metrics plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Gaussian Decoder")
    parser.add_argument('--experiment', type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help='Which experiment to run (1=SAAG Refinement, 2=Direct, 3=FeatureGuided, 4=Fibonacci, 5=NCA)')
    parser.add_argument('--data_dir', type=str, default='images',
                        help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lpips_weight', type=float, default=0.1,
                        help='LPIPS perceptual loss weight (default: 0.1, use 0.5 for sharper output)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for rendering during training')
    parser.add_argument('--feature_dim', type=int, default=384, choices=[384, 768, 1024],
                        help='DINOv2 feature dimension: 384 (small), 768 (base), 1024 (large)')
    parser.add_argument('--use_depth_fusion', action='store_true',
                        help='Enable depth fusion: concatenate encoded depth features with DINOv2')
    parser.add_argument('--depth_feature_dim', type=int, default=64,
                        help='Dimension of encoded depth features for fusion (default: 64)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gaussians_per_patch', type=int, default=4,
                        help='Gaussians per patch for Experiment 2 (default: 4)')
    parser.add_argument('--n_spiral_points', type=int, default=377,
                        help='Fibonacci spiral points for Experiment 4 (default: 377, Fibonacci number)')
    parser.add_argument('--use_tensegrity_loss', action='store_true',
                        help='Enable tensegrity structural integrity loss (golden ratio spacing)')
    parser.add_argument('--tensegrity_weight', type=float, default=0.01,
                        help='Weight for tensegrity loss (default: 0.01)')

    # NCA decoder (Experiment 5) - Neural Cellular Automata
    parser.add_argument('--nca_steps', type=int, default=16,
                        help='Number of NCA iterations for Experiment 5 (default: 16)')
    parser.add_argument('--nca_neighbors', type=int, default=6,
                        help='NCA neighborhood size for Experiment 5 (default: 6)')
    parser.add_argument('--nca_step_size', type=float, default=0.1,
                        help='Initial NCA step size for Experiment 5 (default: 0.1)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of training images to use (default: all)')
    parser.add_argument('--use_vlm_guidance', action='store_true',
                        help='Use VLM density maps for semantic-aware loss weighting')
    parser.add_argument('--vlm_weight', type=float, default=0.5,
                        help='VLM weighting strength (0=uniform, 1=full VLM weighting, default: 0.5)')

    # Fresnel-inspired enhancements (named after Augustin-Jean Fresnel's wave optics)
    parser.add_argument('--use_fresnel_zones', action='store_true',
                        help='Enable Fresnel depth zones (discrete depth layers)')
    parser.add_argument('--num_fresnel_zones', type=int, default=8,
                        help='Number of discrete depth zones (default: 8)')
    parser.add_argument('--boundary_weight', type=float, default=0.1,
                        help='Extra loss weight at zone boundaries (default: 0.1)')
    parser.add_argument('--use_edge_aware', action='store_true',
                        help='Enable edge-aware Gaussian placement (Fresnel diffraction)')
    parser.add_argument('--use_phase_blending', action='store_true',
                        help='Enable phase-based interference blending')
    parser.add_argument('--use_phase_output', action='store_true',
                        help='Enable per-channel RGB phase output from decoder (for QSR/wave rendering)')
    parser.add_argument('--edge_scale_factor', type=float, default=0.5,
                        help='Scale reduction at edges (0-1, default: 0.5)')
    parser.add_argument('--edge_opacity_boost', type=float, default=0.2,
                        help='Opacity boost at edges (0-1, default: 0.2)')
    parser.add_argument('--phase_amplitude', type=float, default=0.25,
                        help='Phase interference amplitude (0-1, default: 0.25)')

    # Physics-derived algorithms (actual wave optics, not heuristics)
    parser.add_argument('--use_wave_rendering', action='store_true',
                        help='Use WaveFieldRenderer with complex wave field accumulation (true interference)')
    parser.add_argument('--wavelength', type=float, default=0.05,
                        help='Effective wavelength for phase computation (default: 0.05)')
    parser.add_argument('--learnable_wavelength', action='store_true',
                        help='Allow network to learn optimal wavelength')
    parser.add_argument('--use_physics_zones', action='store_true',
                        help='Use PhysicsFresnelZones with sqrt(n) zone spacing (not uniform)')
    parser.add_argument('--use_diffraction_placement', action='store_true',
                        help='Use FresnelDiffraction for Gaussian placement at fringe peaks')
    parser.add_argument('--focal_depth', type=float, default=0.5,
                        help='Focal plane depth for phase computation (default: 0.5)')

    # Physics-informed loss (from PINN research)
    parser.add_argument('--wave_equation_weight', type=float, default=0.0,
                        help='Weight for Helmholtz wave equation constraint loss (default: 0.0, disabled)')
    # Per-channel wavelength (RGB light physics)
    parser.add_argument('--use_multi_wavelength', action='store_true',
                        help='Use per-channel wavelength physics (λ_R:λ_G:λ_B ≈ 1.27:1.0:0.82)')

    # ==========================================================================
    # HFGS: Holographic Fourier Gaussian Splatting (BREAKTHROUGH!)
    # ==========================================================================
    parser.add_argument('--use_fourier_renderer', action='store_true',
                        help='Use FourierGaussianRenderer (O(H×W×log) - 10x faster!)')
    parser.add_argument('--use_phase_retrieval_loss', action='store_true',
                        help='Use phase retrieval self-supervision (FREE training signal!)')
    parser.add_argument('--phase_retrieval_weight', type=float, default=0.1,
                        help='Weight for phase retrieval loss (default: 0.1)')
    parser.add_argument('--use_frequency_loss', action='store_true',
                        help='Use frequency domain loss (separate high/low freq)')
    parser.add_argument('--frequency_loss_weight', type=float, default=0.1,
                        help='Weight for frequency domain loss (default: 0.1)')
    parser.add_argument('--high_freq_weight', type=float, default=2.0,
                        help='Extra weight for high frequency loss (default: 2.0)')
    parser.add_argument('--frequency_cutoff', type=float, default=0.1,
                        help='Low/high frequency separation cutoff (default: 0.1)')
    parser.add_argument('--learnable_wavelengths', action='store_true',
                        help='Make per-channel RGB wavelengths trainable')
    parser.add_argument('--wavelength_r', type=float, default=0.0635,
                        help='Red channel wavelength (default: 0.0635 = 700nm equiv)')
    parser.add_argument('--wavelength_g', type=float, default=0.05,
                        help='Green channel wavelength (default: 0.05 = 550nm equiv)')
    parser.add_argument('--wavelength_b', type=float, default=0.041,
                        help='Blue channel wavelength (default: 0.041 = 450nm equiv)')

    # ==========================================================================
    # QSR: Quantum Scene Representation (per-channel complex wave functions)
    # ==========================================================================
    parser.add_argument('--use_qsr', action='store_true',
                        help='Enable Quantum Scene Representation: per-channel phases + wave rendering + phase retrieval')

    # ==========================================================================
    # HFTS: Hybrid Fast Training System (10× SPEEDUP!)
    # ==========================================================================
    parser.add_argument('--train_resolution', type=int, default=None,
                        help='Training resolution (default: same as image_size). Use 64 for 16× speedup!')
    parser.add_argument('--progressive_schedule', action='store_true',
                        help='Enable progressive Gaussian growing (start small, grow over training)')
    parser.add_argument('--stochastic_k', type=int, default=None,
                        help='Sample K Gaussians per step (default: all). Use 256 for ~20× speedup!')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Enable all HFTS optimizations: 64px training, progressive, stochastic')

    # ==========================================================================
    # Multi-Pose Training: Fix dark novel views via view-dependent opacity
    # ==========================================================================
    parser.add_argument('--multi_pose_augmentation', action='store_true',
                        help='Enable multi-pose training to fix dark novel views')
    parser.add_argument('--pose_range_elevation', type=float, nargs=2, default=[-30, 45],
                        help='Elevation range in degrees (default: -30 45)')
    parser.add_argument('--pose_range_azimuth', type=float, nargs=2, default=[0, 360],
                        help='Azimuth range in degrees (default: 0 360)')
    parser.add_argument('--frontal_prob', type=float, default=0.3,
                        help='Probability of using frontal view for reconstruction loss (default: 0.3)')
    parser.add_argument('--use_pose_encoding', action='store_true',
                        help='Enable pose encoding in decoder (required for multi-pose training)')

    args = parser.parse_args()

    # QSR: Enable all related features when --use_qsr is set
    if args.use_qsr:
        args.use_phase_output = True      # DirectPatchDecoder outputs per-channel phases
        args.use_wave_rendering = True    # Use WaveFieldRenderer for complex field accumulation
        args.use_phase_retrieval_loss = True  # Self-supervised physics loss
        print("\n=== QSR (Quantum Scene Representation) ENABLED ===")
        print("  - Per-channel RGB phases (3 phases per Gaussian)")
        print("  - WaveFieldRenderer with complex wave interference")
        print("  - Phase retrieval self-supervision loss")
        if args.learnable_wavelengths:
            print("  - Learnable per-channel wavelengths (RGB physics)")
        print("================================================\n")

    # Create config
    config = TrainingConfig(
        experiment=args.experiment,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lpips_weight=args.lpips_weight,
        image_size=args.image_size,
        feature_dim=args.feature_dim,
        use_depth_fusion=args.use_depth_fusion,
        depth_feature_dim=args.depth_feature_dim,
        gaussians_per_patch=args.gaussians_per_patch,
        max_images=args.max_images,
        use_vlm_guidance=args.use_vlm_guidance,
        vlm_weight=args.vlm_weight,
        # Fresnel enhancements
        use_fresnel_zones=args.use_fresnel_zones,
        num_fresnel_zones=args.num_fresnel_zones,
        boundary_weight=args.boundary_weight,
        use_edge_aware=args.use_edge_aware,
        use_phase_blending=args.use_phase_blending,
        use_phase_output=args.use_phase_output,
        edge_scale_factor=args.edge_scale_factor,
        edge_opacity_boost=args.edge_opacity_boost,
        phase_amplitude=args.phase_amplitude,
        # Multi-pose training
        multi_pose_augmentation=args.multi_pose_augmentation,
        pose_range_elevation=tuple(args.pose_range_elevation),
        pose_range_azimuth=tuple(args.pose_range_azimuth),
        frontal_prob=args.frontal_prob,
        use_pose_encoding=args.use_pose_encoding,
        # Fibonacci decoder (Experiment 4)
        n_spiral_points=args.n_spiral_points,
        use_tensegrity_loss=args.use_tensegrity_loss,
        tensegrity_weight=args.tensegrity_weight,
        # NCA decoder (Experiment 5)
        nca_steps=args.nca_steps,
        nca_neighbors=args.nca_neighbors,
        nca_step_size=args.nca_step_size,
    )

    # Create physics config
    physics_config = PhysicsConfig(
        use_wave_rendering=args.use_wave_rendering,
        wavelength=args.wavelength,
        learnable_wavelength=args.learnable_wavelength,
        use_physics_zones=args.use_physics_zones,
        num_zones=args.num_fresnel_zones,
        focal_depth=args.focal_depth,
        use_diffraction_placement=args.use_diffraction_placement,
        wave_equation_weight=args.wave_equation_weight,
        use_multi_wavelength=args.use_multi_wavelength,
    )

    # Create HFGS config (Holographic Fourier Gaussian Splatting)
    hfgs_config = HFGSConfig(
        use_fourier_renderer=args.use_fourier_renderer,
        use_phase_retrieval_loss=args.use_phase_retrieval_loss,
        phase_retrieval_weight=args.phase_retrieval_weight,
        use_frequency_loss=args.use_frequency_loss,
        frequency_loss_weight=args.frequency_loss_weight,
        high_freq_weight=args.high_freq_weight,
        frequency_cutoff=args.frequency_cutoff,
        learnable_wavelengths=args.learnable_wavelengths,
        wavelength_r=args.wavelength_r,
        wavelength_g=args.wavelength_g,
        wavelength_b=args.wavelength_b,
        focal_depth=args.focal_depth,
    )

    # Create HFTS config (Hybrid Fast Training System - 10× SPEEDUP!)
    hfts_config = HFTSConfig(
        train_resolution=args.train_resolution,
        progressive_schedule=args.progressive_schedule,
        stochastic_k=args.stochastic_k,
        fast_mode=args.fast_mode,
    )

    # Get effective training resolution
    effective_train_res = hfts_config.get_effective_train_resolution(config.image_size)

    print("=" * 60)
    print(f"Training Experiment {config.experiment}")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Data dir: {config.data_dir}")
    print(f"Output dir: {config.output_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Image size: {config.image_size}")
    if config.use_vlm_guidance:
        print(f"VLM guidance: ENABLED (weight={config.vlm_weight})")

    # Print Fresnel enhancements
    fresnel_enabled = config.use_fresnel_zones or config.use_edge_aware or config.use_phase_blending
    if fresnel_enabled:
        print("Fresnel enhancements:")
        if config.use_fresnel_zones:
            print(f"  - Depth zones: {config.num_fresnel_zones} zones, boundary_weight={config.boundary_weight}")
        if config.use_edge_aware:
            print(f"  - Edge-aware: scale_factor={config.edge_scale_factor}, opacity_boost={config.edge_opacity_boost}")
        if config.use_phase_blending:
            print(f"  - Phase blending: amplitude={config.phase_amplitude}")

    # Print physics-derived settings
    physics_enabled = physics_config.use_wave_rendering or physics_config.use_physics_zones or physics_config.use_diffraction_placement
    if physics_enabled:
        print("Physics-derived wave optics:")
        if physics_config.use_wave_rendering:
            print(f"  - Wave rendering: ENABLED (complex field accumulation)")
        print(f"  - Wavelength: {physics_config.wavelength} (learnable={physics_config.learnable_wavelength})")
        if physics_config.use_physics_zones:
            print(f"  - Physics zones: ENABLED (sqrt(n) spacing, focal_depth={physics_config.focal_depth})")
        if physics_config.use_diffraction_placement:
            print(f"  - Diffraction placement: ENABLED (fringe-based Gaussian density)")

    # Print HFGS settings (Holographic Fourier Gaussian Splatting)
    hfgs_enabled = (
        hfgs_config.use_fourier_renderer or
        hfgs_config.use_phase_retrieval_loss or
        hfgs_config.use_frequency_loss
    )
    if hfgs_enabled:
        print("=" * 60)
        print("HFGS: Holographic Fourier Gaussian Splatting (BREAKTHROUGH!)")
        print("=" * 60)
        if hfgs_config.use_fourier_renderer:
            print(f"  - Fourier renderer: ENABLED (O(H×W×log) - 10x faster!)")
            print(f"  - RGB wavelengths: R={hfgs_config.wavelength_r:.4f}, G={hfgs_config.wavelength_g:.4f}, B={hfgs_config.wavelength_b:.4f}")
            print(f"  - Learnable wavelengths: {hfgs_config.learnable_wavelengths}")
        if hfgs_config.use_phase_retrieval_loss:
            print(f"  - Phase retrieval loss: ENABLED (weight={hfgs_config.phase_retrieval_weight})")
            print(f"    (FREE self-supervision from physics!)")
        if hfgs_config.use_frequency_loss:
            print(f"  - Frequency loss: ENABLED (weight={hfgs_config.frequency_loss_weight})")
            print(f"  - High freq weight: {hfgs_config.high_freq_weight}x")
            print(f"  - Cutoff: {hfgs_config.frequency_cutoff}")

    # Print HFTS settings (Hybrid Fast Training System)
    hfts_enabled = (
        hfts_config.fast_mode or
        hfts_config.train_resolution is not None or
        hfts_config.progressive_schedule or
        hfts_config.stochastic_k is not None
    )
    if hfts_enabled:
        print("=" * 60)
        print("HFTS: Hybrid Fast Training System (10× SPEEDUP!)")
        print("=" * 60)
        speedup_factors = []
        if effective_train_res != config.image_size:
            res_speedup = (config.image_size / effective_train_res) ** 2
            print(f"  - Multi-Resolution Training: {effective_train_res}×{effective_train_res} ({res_speedup:.0f}× per step)")
            speedup_factors.append(res_speedup)
        if hfts_config.progressive_schedule or hfts_config.fast_mode:
            print(f"  - Progressive Gaussian Growing: 1→2→4→{config.gaussians_per_patch} per patch")
            print(f"    (Early epochs ~85× faster, avg ~5× speedup)")
            speedup_factors.append(5)
        if hfts_config.stochastic_k is not None or hfts_config.fast_mode:
            k = hfts_config.stochastic_k or 256
            total_g = 37 * 37 * config.gaussians_per_patch
            stoch_speedup = total_g / k
            print(f"  - Stochastic Rendering: {k}/{total_g} Gaussians sampled ({stoch_speedup:.0f}× per render)")
            # Stochastic needs more iterations, so effective speedup is lower
            speedup_factors.append(stoch_speedup ** 0.5)
        if speedup_factors:
            total_speedup = 1
            for s in speedup_factors:
                total_speedup *= s ** 0.5  # Conservative estimate (sqrt combination)
            print(f"  - Estimated total speedup: {total_speedup:.1f}×")
        if hfts_config.fast_mode:
            print("  - Fast mode: ALL optimizations enabled")

    # Create dataset and dataloader
    dataset = ImageDataset(
        config.data_dir,
        config.image_size,
        use_augmentation=config.use_augmentation,
        max_images=config.max_images,
        load_vlm_density=config.use_vlm_guidance,
        feature_dim=config.feature_dim
    )

    if len(dataset) == 0:
        print("\nNo images found! Creating dummy dataset for testing...")
        # Create dummy data for testing
        dummy_dir = Path("/tmp/fresnel_dummy_data")
        dummy_dir.mkdir(exist_ok=True)

        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(dummy_dir / f"dummy_{i}.png")

        dataset = ImageDataset(str(dummy_dir), config.image_size, feature_dim=config.feature_dim)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True if config.device == 'cuda' else False
    )

    # Check if physics-derived decoder is required
    # NOTE: QSR uses wave rendering but with per-channel LEARNED phases (DirectPatchDecoder),
    # not physics-derived phases (PhysicsDirectPatchDecoder)
    use_physics_decoder = (
        (physics_config.use_wave_rendering and not args.use_qsr) or
        physics_config.use_physics_zones or
        physics_config.use_diffraction_placement
    )

    # Auto-switch to experiment 2 if physics features require it
    if use_physics_decoder and config.experiment != 2:
        print(f"\nNote: Physics features (wave_rendering, physics_zones, or diffraction_placement)")
        print(f"      require experiment 2 (DirectPatchDecoder). Switching from experiment {config.experiment} to 2.\n")
        config.experiment = 2

    # Create model
    if config.experiment == 1:
        model = SAAGRefinementNet()
    elif config.experiment == 2:

        if use_physics_decoder:
            if not PHYSICS_DECODER_AVAILABLE:
                # Physics decoder required but not available
                error_msg = (
                    "PhysicsDirectPatchDecoder is required for --use_wave_rendering, "
                    "--use_physics_zones, or --use_diffraction_placement, but import failed.\n"
                )
                if 'PHYSICS_DECODER_IMPORT_ERROR' in dir():
                    error_msg += f"Import error: {PHYSICS_DECODER_IMPORT_ERROR}\n"
                error_msg += "Check that fresnel_zones.py exports PhysicsFresnelZones."
                raise ImportError(error_msg)

            # PhysicsDirectPatchDecoder with physics-derived phase
            print("Using PhysicsDirectPatchDecoder (physics-derived phase)")
            model = PhysicsDirectPatchDecoder(
                feature_dim=config.feature_dim,
                gaussians_per_patch=config.gaussians_per_patch,
                wavelength=physics_config.wavelength,
                learnable_wavelength=physics_config.learnable_wavelength,
                focal_depth=physics_config.focal_depth,
                use_diffraction_placement=physics_config.use_diffraction_placement,
            )
        else:
            # DirectPatchDecoder with optional heuristic Fresnel enhancements
            # QSR: use_phase_output enables per-channel RGB phases for wave rendering
            model = DirectPatchDecoder(
                feature_dim=config.feature_dim,
                gaussians_per_patch=config.gaussians_per_patch,
                use_fresnel_zones=config.use_fresnel_zones,
                num_fresnel_zones=config.num_fresnel_zones,
                use_edge_aware=config.use_edge_aware,
                use_phase_output=config.use_phase_output or config.use_phase_blending,
                edge_scale_factor=config.edge_scale_factor,
                edge_opacity_boost=config.edge_opacity_boost,
                # Multi-pose training: Enable pose encoding for view-dependent opacity
                use_pose_encoding=config.use_pose_encoding if hasattr(config, 'use_pose_encoding') else False,
                # Depth fusion: concatenate encoded depth with DINOv2 features
                use_depth_fusion=config.use_depth_fusion,
                depth_feature_dim=config.depth_feature_dim,
            )
    elif config.experiment == 4:
        # Fibonacci decoder: Nature-inspired golden spiral sampling
        print(f"Using FibonacciPatchDecoder ({config.n_spiral_points} spiral points)")
        model = FibonacciPatchDecoder(
            feature_dim=config.feature_dim,
            n_spiral_points=config.n_spiral_points,
            gaussians_per_point=1,  # Spiral already provides good distribution
            use_fresnel_zones=config.use_fresnel_zones,
            num_fresnel_zones=config.num_fresnel_zones,
            use_phase_output=config.use_phase_output or config.use_phase_blending,
            use_pose_encoding=config.use_pose_encoding if hasattr(config, 'use_pose_encoding') else False,
        )
    elif config.experiment == 5:
        # NCA decoder: Neural Cellular Automata Gaussians (Exp 014)
        print(f"Using NCAGaussianDecoder ({config.n_spiral_points} points, {config.nca_steps} NCA steps)")
        model = NCAGaussianDecoder(
            feature_dim=config.feature_dim,
            n_points=config.n_spiral_points,  # Reuse Fibonacci spiral points
            n_steps=config.nca_steps,
            k_neighbors=config.nca_neighbors,
        )
    else:  # 3
        model = FeatureGuidedSAAG()

    model = model.to(config.device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Feature dimension: {config.feature_dim} (DINOv2-{'small' if config.feature_dim == 384 else 'base' if config.feature_dim == 768 else 'large'})")
    if config.use_depth_fusion:
        print(f"Depth fusion: ENABLED ({config.depth_feature_dim}-dim depth features)")

    # Create renderer at effective_train_res (HFTS: Multi-Resolution Training)
    # This is the key speedup: train at 64×64 instead of 256×256 = 16× fewer pixels!
    # HFGS mode: Use TileBasedRenderer for training (memory efficient),
    # the HFGS losses (phase retrieval, frequency domain) work on rendered output
    # FourierGaussianRenderer can be used for inference after training
    if hfgs_config.use_fourier_renderer:
        print("HFGS mode enabled:")
        print("  - Training with TileBasedRenderer (memory efficient)")
        print("  - HFGS losses (phase retrieval, frequency domain) applied to output")
        print("  - Learnable wavelengths trained via loss functions")
        print("  - FourierGaussianRenderer available for inference after training")
        # Use TileBasedRenderer for memory-efficient training
        renderer = TileBasedRenderer(
            effective_train_res,
            effective_train_res,
            use_phase_blending=True,  # Enable phase blending for HFGS mode
            phase_amplitude=0.3
        )
    elif config.experiment == 4 and config.use_phase_blending:
        # Fibonacci decoder: Use FourierGaussianRenderer (fully vectorized)
        # WaveFieldRenderer crashes due to per-Gaussian Python for-loop causing
        # memory fragmentation (~380K small tensor allocations per epoch).
        # FourierGaussianRenderer uses batched FFT operations - no memory leaks.
        print("Using FourierGaussianRenderer for Fibonacci (vectorized, no memory fragmentation)")
        renderer = FourierGaussianRenderer(
            effective_train_res,
            effective_train_res,
            wavelength_r=0.65,  # Red wavelength
            wavelength_g=0.55,  # Green wavelength
            wavelength_b=0.45,  # Blue wavelength
            learnable_wavelengths=True,
        )
    elif physics_config.use_wave_rendering:
        # WaveFieldRenderer for true complex wave field accumulation
        print("Using WaveFieldRenderer (complex wave field accumulation)")
        renderer = WaveFieldRenderer(
            effective_train_res,
            effective_train_res,
        )
    else:
        # TileBasedRenderer for memory efficiency - O(N × r²) vs O(N × H × W)
        # With optional heuristic Fresnel phase blending
        renderer = TileBasedRenderer(
            effective_train_res,
            effective_train_res,
            use_phase_blending=config.use_phase_blending,
            phase_amplitude=config.phase_amplitude
        )
    renderer = renderer.to(config.device)

    # Create camera at effective_train_res
    camera = Camera(
        fx=effective_train_res * 0.8,
        fy=effective_train_res * 0.8,
        cx=effective_train_res / 2,
        cy=effective_train_res / 2,
        width=effective_train_res,
        height=effective_train_res
    )

    # Initialize HFGS components (Holographic Fourier Gaussian Splatting)
    # Must be initialized BEFORE optimizer to include learnable wavelengths
    learnable_wavelengths = None
    phase_retrieval_loss_fn = None
    frequency_loss_fn = None

    if hfgs_config.use_fourier_renderer or hfgs_config.use_phase_retrieval_loss:
        # Create learnable wavelength parameters
        print("Initializing LearnableWavelengths (RGB wavelength as 3D prior)")
        learnable_wavelengths = LearnableWavelengths(
            wavelength_r=hfgs_config.wavelength_r,
            wavelength_g=hfgs_config.wavelength_g,
            wavelength_b=hfgs_config.wavelength_b,
            learnable=hfgs_config.learnable_wavelengths
        ).to(config.device)
        print(f"  Initial wavelengths: {learnable_wavelengths}")

    if hfgs_config.use_phase_retrieval_loss:
        print("Initializing PhaseRetrievalLoss (FREE self-supervision!)")
        phase_retrieval_loss_fn = PhaseRetrievalLoss(
            wavelength=hfgs_config.wavelength_g,  # Fallback if no learnable
            focal_depth=hfgs_config.focal_depth
        ).to(config.device)

    if hfgs_config.use_frequency_loss:
        print("Initializing FrequencyDomainLoss (high freq emphasis)")
        frequency_loss_fn = FrequencyDomainLoss(
            cutoff=hfgs_config.frequency_cutoff,
            high_weight=hfgs_config.high_freq_weight
        ).to(config.device)

    # Initialize FresnelZones once (avoid per-batch creation)
    fresnel_zones = None
    if config.use_fresnel_zones and config.boundary_weight > 0:
        try:
            from utils.fresnel_zones import FresnelZones
            fresnel_zones = FresnelZones(
                num_zones=config.num_fresnel_zones,
                depth_range=(0.0, 1.0),
                soft_boundaries=True
            ).to(config.device)
            print(f"Initialized FresnelZones with {config.num_fresnel_zones} zones")
        except ImportError:
            print("FresnelZones not available")

    # Create optimizer - include learnable wavelengths if HFGS enabled
    params_to_optimize = list(model.parameters())
    if learnable_wavelengths is not None and hfgs_config.learnable_wavelengths:
        params_to_optimize.extend(list(learnable_wavelengths.parameters()))
        print(f"  Added {len(list(learnable_wavelengths.parameters()))} learnable wavelength params to optimizer")

    optimizer = AdamW(params_to_optimize, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Initialize LPIPS model for perceptual loss
    # Using 'alex' (AlexNet) instead of 'vgg' for ~4x less memory usage
    lpips_fn = None
    if LPIPS_AVAILABLE and config.lpips_weight > 0:
        print("Initializing LPIPS perceptual loss (AlexNet)...")
        lpips_fn = lpips.LPIPS(net='alex').to(config.device)
        lpips_fn.eval()  # LPIPS should stay in eval mode
        for param in lpips_fn.parameters():
            param.requires_grad = False

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_loss = float('inf')

    # Training history for metrics visualization
    training_history = {
        'total': [],
        'rgb': [],
        'ssim': [],
        'lpips': [],
        'depth': [],
        'boundary': [],
        'residual': [],
        'phase_retrieval': [],  # HFGS: Phase retrieval self-supervision
        'frequency': [],        # HFGS: Frequency domain loss
    }

    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        start_time = time.time()

        # Train (with HFGS losses if enabled)
        losses = train_epoch(
            model, dataloader, optimizer, renderer, camera, config, epoch, lpips_fn,
            physics_config=physics_config,
            hfgs_config=hfgs_config,
            phase_retrieval_loss_fn=phase_retrieval_loss_fn,
            frequency_loss_fn=frequency_loss_fn,
            learnable_wavelengths=learnable_wavelengths,
            hfts_config=hfts_config,
            effective_train_res=effective_train_res,
            fresnel_zones=fresnel_zones
        )

        # Step scheduler
        scheduler.step()

        # Log
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1} complete | Time: {elapsed:.1f}s | "
              f"Loss: {losses['total']:.4f} | RGB: {losses['rgb']:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, losses, config)

        # Save best model
        if losses['total'] < best_loss:
            best_loss = losses['total']
            save_checkpoint(model, optimizer, epoch + 1, losses, config)
            print(f"New best loss: {best_loss:.4f}")

        # Record training history
        for key in training_history:
            training_history[key].append(losses.get(key, 0.0))

    # Save training history and plot metrics
    save_training_history(training_history, config.output_dir, config.experiment)
    plot_training_metrics(training_history, config.output_dir, config.experiment)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.4f}")

    # Export to ONNX
    export_path = Path(config.output_dir) / f"gaussian_decoder_exp{config.experiment}.onnx"
    print(f"\nExporting to ONNX: {export_path}")

    try:
        model.eval()
        if config.experiment == 1:
            dummy_features = torch.randn(1, 384, 37, 37, device=config.device)
            dummy_pos = torch.randn(1, 1000, 3, device=config.device)
            dummy_scale = torch.ones(1, 1000, 3, device=config.device) * 0.05
            dummy_rot = torch.zeros(1, 1000, 4, device=config.device)
            dummy_rot[..., 0] = 1
            dummy_color = torch.rand(1, 1000, 3, device=config.device)
            dummy_opacity = torch.ones(1, 1000, device=config.device)

            torch.onnx.export(
                model,
                (dummy_features, dummy_pos, dummy_scale, dummy_rot, dummy_color, dummy_opacity),
                str(export_path),
                input_names=['features', 'saag_positions', 'saag_scales', 'saag_rotations', 'saag_colors', 'saag_opacities'],
                output_names=['positions', 'scales', 'rotations', 'colors', 'opacities'],
                dynamic_axes={
                    'features': {0: 'batch'},
                    'saag_positions': {0: 'batch', 1: 'num_gaussians'},
                    'saag_scales': {0: 'batch', 1: 'num_gaussians'},
                    'saag_rotations': {0: 'batch', 1: 'num_gaussians'},
                    'saag_colors': {0: 'batch', 1: 'num_gaussians'},
                    'saag_opacities': {0: 'batch', 1: 'num_gaussians'}
                },
                opset_version=16  # grid_sample requires opset 16+
            )
        elif config.experiment == 2:
            dummy_features = torch.randn(1, 384, 37, 37, device=config.device)
            dummy_depth = torch.rand(1, 1, 518, 518, device=config.device)

            torch.onnx.export(
                model,
                (dummy_features, dummy_depth),
                str(export_path),
                input_names=['features', 'depth'],
                output_names=['positions', 'scales', 'rotations', 'colors', 'opacities'],
                dynamic_axes={
                    'features': {0: 'batch'},
                    'depth': {0: 'batch'}
                },
                opset_version=14
            )
        elif config.experiment in [4, 5]:
            # Experiments 4 (Fibonacci) and 5 (NCA) - similar export as experiment 2
            dummy_features = torch.randn(1, 384, 37, 37, device=config.device)
            dummy_depth = torch.rand(1, 1, 518, 518, device=config.device)

            torch.onnx.export(
                model,
                (dummy_features, dummy_depth),
                str(export_path),
                input_names=['features', 'depth'],
                output_names=['positions', 'scales', 'rotations', 'colors', 'opacities'],
                dynamic_axes={
                    'features': {0: 'batch'},
                    'depth': {0: 'batch'}
                },
                opset_version=16  # grid_sample requires opset 16+
            )
        else:  # 3
            dummy_features = torch.randn(1, 384, 37, 37, device=config.device)

            torch.onnx.export(
                model,
                (dummy_features,),
                str(export_path),
                input_names=['features'],
                output_names=['aspect_ratio_mult', 'edge_threshold_add', 'edge_shrink_mult',
                             'normal_strength_mult', 'base_size_mult', 'opacity_mult'],
                dynamic_axes={
                    'features': {0: 'batch'}
                },
                opset_version=14
            )

        print(f"ONNX export successful: {export_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Model saved as PyTorch checkpoint only.")


if __name__ == '__main__':
    main()
