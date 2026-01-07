#!/usr/bin/env python3
"""
Gaussian Decoder Models

Three experimental approaches for learned Gaussian prediction:
1. SAAGRefinementNet - Learn residuals from SAAG initialization (Novel)
2. DirectPatchDecoder - Predict Gaussians directly from features (Baseline)
3. FeatureGuidedSAAG - Learn to modulate SAAG parameters spatially (Lightweight)

All models take DINOv2 features (37x37x384) and depth as input.

Fresnel-Inspired Enhancements:
- Fresnel Depth Zones: Organize Gaussians into discrete depth layers
- Edge-Aware Placement: More/smaller Gaussians at depth discontinuities (like diffraction fringes)
- Phase Blending: Optional interference-like alpha compositing

Named after Augustin-Jean Fresnel (1788-1827), whose wave optics theory
inspires our approach to organizing and blending Gaussian primitives.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

# Add scripts directory to path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import Fresnel utilities
try:
    from utils.fresnel_zones import FresnelZones, FresnelEdgeDetector, PhysicsFresnelZones
    FRESNEL_AVAILABLE = True
    PHYSICS_FRESNEL_AVAILABLE = True
except ImportError:
    FRESNEL_AVAILABLE = False
    PHYSICS_FRESNEL_AVAILABLE = False

# FresnelDiffraction may not exist yet during incremental development
try:
    from utils.fresnel_zones import FresnelDiffraction
    DIFFRACTION_AVAILABLE = True
except ImportError:
    DIFFRACTION_AVAILABLE = False


def rotation_6d_to_quaternion(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to quaternion.

    6D representation from "On the Continuity of Rotation Representations
    in Neural Networks" - Zhou et al. CVPR 2019

    Args:
        rot_6d: (..., 6) 6D rotation

    Returns:
        quat: (..., 4) quaternion (w, x, y, z)
    """
    # Extract two vectors
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:6]

    # Gram-Schmidt orthogonalization with numerical stability
    # Use explicit eps=1e-6 (default 1e-12 is too small)
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2_raw = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    # Ensure b2_raw has non-zero norm before normalizing
    b2_raw = b2_raw + 1e-8 * torch.randn_like(b2_raw).sign()
    b2 = F.normalize(b2_raw, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    # Regularize degenerate cross products (when b1 || b2)
    b3_norm = b3.norm(dim=-1, keepdim=True)
    b3 = torch.where(b3_norm < 1e-6, torch.tensor([0., 0., 1.], device=b3.device), b3)
    b3 = F.normalize(b3, dim=-1, eps=1e-6)

    # Build rotation matrix
    R = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)

    # Convert rotation matrix to quaternion using ONNX-compatible operations
    # Uses torch.where instead of boolean mask indexing for ONNX compatibility
    # Based on: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    R00, R01, R02 = R_flat[:, 0, 0], R_flat[:, 0, 1], R_flat[:, 0, 2]
    R10, R11, R12 = R_flat[:, 1, 0], R_flat[:, 1, 1], R_flat[:, 1, 2]
    R20, R21, R22 = R_flat[:, 2, 0], R_flat[:, 2, 1], R_flat[:, 2, 2]

    trace = R00 + R11 + R22

    # Compute all four cases (avoids boolean indexing for ONNX compatibility)
    # Case 1: trace > 0
    s1 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2
    qw1 = 0.25 * s1
    qx1 = (R21 - R12) / s1
    qy1 = (R02 - R20) / s1
    qz1 = (R10 - R01) / s1

    # Case 2: R00 > R11 and R00 > R22
    s2 = torch.sqrt(torch.clamp(1.0 + R00 - R11 - R22, min=1e-10)) * 2
    qw2 = (R21 - R12) / s2
    qx2 = 0.25 * s2
    qy2 = (R01 + R10) / s2
    qz2 = (R02 + R20) / s2

    # Case 3: R11 > R22
    s3 = torch.sqrt(torch.clamp(1.0 + R11 - R00 - R22, min=1e-10)) * 2
    qw3 = (R02 - R20) / s3
    qx3 = (R01 + R10) / s3
    qy3 = 0.25 * s3
    qz3 = (R12 + R21) / s3

    # Case 4: else (R22 is largest)
    s4 = torch.sqrt(torch.clamp(1.0 + R22 - R00 - R11, min=1e-10)) * 2
    qw4 = (R10 - R01) / s4
    qx4 = (R02 + R20) / s4
    qy4 = (R12 + R21) / s4
    qz4 = 0.25 * s4

    # Select case using torch.where (ONNX-compatible)
    cond1 = trace > 0
    cond2 = (R00 > R11) & (R00 > R22)
    cond3 = R11 > R22

    # Nested selection: case1 -> case2 -> case3 -> case4
    qw = torch.where(cond1, qw1, torch.where(cond2, qw2, torch.where(cond3, qw3, qw4)))
    qx = torch.where(cond1, qx1, torch.where(cond2, qx2, torch.where(cond3, qx3, qx4)))
    qy = torch.where(cond1, qy1, torch.where(cond2, qy2, torch.where(cond3, qy3, qy4)))
    qz = torch.where(cond1, qz1, torch.where(cond2, qz2, torch.where(cond3, qz3, qz4)))

    quat = torch.stack([qw, qx, qy, qz], dim=-1)

    # Normalize with explicit eps for numerical stability
    quat = F.normalize(quat, dim=-1, eps=1e-6)

    return quat.reshape(*batch_shape, 4)


class MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureInterpolator(nn.Module):
    """
    Bilinear interpolation of feature grid at arbitrary positions.

    Given a feature grid (B, C, H, W) and positions (B, N, 2),
    returns features at those positions via bilinear sampling.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,      # (B, C, H, W)
        positions: torch.Tensor,      # (B, N, 2) in [0, 1] normalized coords
    ) -> torch.Tensor:
        """
        Sample features at positions.

        Args:
            features: (B, C, H, W) feature grid
            positions: (B, N, 2) normalized positions [0, 1]

        Returns:
            sampled: (B, N, C) sampled features
        """
        B, C, H, W = features.shape
        N = positions.shape[1]

        # Convert [0, 1] to [-1, 1] for grid_sample
        grid = positions * 2 - 1  # (B, N, 2)

        # grid_sample expects (B, H_out, W_out, 2)
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)

        # Sample
        sampled = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # (B, C, N, 1)

        # Reshape to (B, N, C)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)

        return sampled


# =============================================================================
# Experiment 1: Learned SAAG Refinement (Novel Approach)
# =============================================================================

class SAAGRefinementNet(nn.Module):
    """
    Learn residuals from SAAG initialization.

    Key insight: SAAG provides excellent geometric priors. Instead of predicting
    everything from scratch, we learn to refine SAAG's outputs.

    Architecture:
    1. Sample DINOv2 features at each Gaussian position
    2. Concatenate with SAAG Gaussian params
    3. MLP predicts residuals
    4. Output = SAAG + scaled residuals
    """

    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dims: list = [256, 128],
        residual_scale: float = 0.1,  # Scale residuals to keep close to init
        dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.residual_scale = residual_scale

        # Feature interpolator
        self.interpolator = FeatureInterpolator()

        # Input: features (384) + position (3) + scale (3) + rotation (4) + color (3) + opacity (1)
        # Total: 398
        input_dim = feature_dim + 14

        # Output: position_delta (3) + scale_delta (3) + rotation_6d (6) + color_delta (3) + opacity_delta (1)
        # Using 6D rotation for continuous representation
        output_dim = 16

        self.mlp = MLP(input_dim, hidden_dims, output_dim, dropout)

        # Learned scale for residuals per output type
        self.pos_scale = nn.Parameter(torch.tensor(0.05))
        self.scale_scale = nn.Parameter(torch.tensor(0.1))
        self.color_scale = nn.Parameter(torch.tensor(0.1))
        self.opacity_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        features: torch.Tensor,           # (B, 384, 37, 37) DINOv2 features
        saag_positions: torch.Tensor,      # (B, N, 3) SAAG Gaussian positions
        saag_scales: torch.Tensor,         # (B, N, 3) SAAG scales
        saag_rotations: torch.Tensor,      # (B, N, 4) SAAG quaternions (w,x,y,z)
        saag_colors: torch.Tensor,         # (B, N, 3) SAAG colors
        saag_opacities: torch.Tensor,      # (B, N) SAAG opacities
        image_size: Tuple[int, int] = (518, 518),  # Original image size
        intrinsics: Optional[torch.Tensor] = None   # (B, 4) [fx, fy, cx, cy]
    ) -> Dict[str, torch.Tensor]:
        """
        Refine SAAG Gaussians using learned residuals.

        Returns:
            Dict with refined Gaussian parameters
        """
        B, N = saag_positions.shape[:2]
        device = features.device

        # Project 3D positions to 2D for feature sampling
        # Simple projection: x/z, y/z (assuming camera at origin)
        pos_2d = saag_positions[..., :2] / (saag_positions[..., 2:3].clamp(min=0.1))

        # Normalize to [0, 1] based on assumed field of view
        # Assuming positions are roughly in [-2, 2] after normalization
        pos_2d_norm = (pos_2d + 2) / 4  # Map [-2, 2] to [0, 1]
        pos_2d_norm = pos_2d_norm.clamp(0, 1)

        # Sample features at Gaussian positions
        sampled_features = self.interpolator(features, pos_2d_norm)  # (B, N, 384)

        # Concatenate all inputs
        inputs = torch.cat([
            sampled_features,
            saag_positions,
            saag_scales,
            saag_rotations,
            saag_colors,
            saag_opacities.unsqueeze(-1)
        ], dim=-1)  # (B, N, 398)

        # Predict residuals
        residuals = self.mlp(inputs)  # (B, N, 16)

        # Parse and apply residuals
        pos_delta = residuals[..., 0:3] * self.pos_scale * self.residual_scale
        scale_delta = residuals[..., 3:6] * self.scale_scale * self.residual_scale
        rot_6d = residuals[..., 6:12]  # Raw 6D rotation
        color_delta = residuals[..., 12:15] * self.color_scale * self.residual_scale
        opacity_delta = residuals[..., 15:16] * self.opacity_scale * self.residual_scale

        # Apply position residual
        refined_positions = saag_positions + pos_delta

        # Apply scale residual (multiplicative via exp to keep positive)
        refined_scales = saag_scales * torch.exp(scale_delta)

        # Apply rotation residual
        # Convert 6D to quaternion, then compose with SAAG rotation
        rot_delta_quat = rotation_6d_to_quaternion(rot_6d)

        # Quaternion multiplication: q_result = q_delta * q_saag
        # This applies the delta rotation first
        refined_rotations = self._quaternion_multiply(rot_delta_quat, saag_rotations)
        refined_rotations = F.normalize(refined_rotations, dim=-1)

        # Apply color residual (additive, clamped)
        refined_colors = torch.clamp(saag_colors + color_delta, 0, 1)

        # Apply opacity residual (additive, clamped)
        refined_opacities = torch.clamp(saag_opacities + opacity_delta.squeeze(-1), 0, 1)

        return {
            'positions': refined_positions,
            'scales': refined_scales,
            'rotations': refined_rotations,
            'colors': refined_colors,
            'opacities': refined_opacities,
            # Also return residuals for regularization
            'pos_delta': pos_delta,
            'scale_delta': scale_delta,
            'color_delta': color_delta,
            'opacity_delta': opacity_delta
        }

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply two quaternions.

        q1 * q2 applies q2 first, then q1.
        Format: (w, x, y, z)
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)


# =============================================================================
# Experiment 2: Direct Patch Decoder (Baseline)
# =============================================================================

class DirectPatchDecoder(nn.Module):
    """
    Predict Gaussians directly from DINOv2 patch features.

    Each 14x14 patch in the original image (37x37 grid) predicts K Gaussians.
    Simple baseline without SAAG initialization.

    Total Gaussians: 37 * 37 * K = 1369 * K

    Fresnel-Inspired Enhancements:
    - Fresnel Depth Zones: Quantize depth into discrete layers for better discontinuity handling
    - Edge-Aware Placement: Smaller, denser Gaussians at depth edges (like diffraction fringes)
    - Phase Output: Optional phase parameter for interference-like blending
    """

    def __init__(
        self,
        feature_dim: int = 384,
        gaussians_per_patch: int = 8,  # Increased for better detail coverage
        hidden_dims: list = [512, 512, 256, 128],  # Deeper network for more capacity
        dropout: float = 0.1,
        # Fresnel enhancement options
        use_fresnel_zones: bool = False,
        num_fresnel_zones: int = 8,
        use_edge_aware: bool = False,
        use_phase_output: bool = False,
        edge_scale_factor: float = 0.5,   # How much to shrink scales at edges
        edge_opacity_boost: float = 0.2,  # How much to boost opacity at edges
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.gaussians_per_patch = gaussians_per_patch

        # Fresnel enhancement flags
        self.use_fresnel_zones = use_fresnel_zones
        self.use_edge_aware = use_edge_aware
        self.use_phase_output = use_phase_output
        self.edge_scale_factor = edge_scale_factor
        self.edge_opacity_boost = edge_opacity_boost

        # Output per Gaussian: position (3) + scale (3) + rotation_6d (6) + color (3) + opacity (1) = 16
        # Optionally add phase (1) for interference blending = 17
        output_per_gaussian = 17 if use_phase_output else 16
        self.output_per_gaussian = output_per_gaussian
        output_dim = gaussians_per_patch * output_per_gaussian

        # Per-patch MLP
        self.mlp = MLP(feature_dim, hidden_dims, output_dim, dropout)

        # Learned initial depth offset
        self.depth_offset = nn.Parameter(torch.tensor(-2.0))  # Start behind camera

        # Initialize Fresnel zones if enabled
        if use_fresnel_zones and FRESNEL_AVAILABLE:
            self.fresnel_zones = FresnelZones(
                num_zones=num_fresnel_zones,
                depth_range=(0.0, 1.0),
                soft_boundaries=True
            )
        else:
            self.fresnel_zones = None

        # Initialize edge detector if enabled
        if use_edge_aware:
            self.edge_detector = FresnelEdgeDetector(
                in_channels=1,
                hidden_channels=16,
                use_depth_gradients=True
            ) if FRESNEL_AVAILABLE else self._create_simple_edge_detector()
        else:
            self.edge_detector = None

    def _create_simple_edge_detector(self) -> nn.Module:
        """Fallback edge detector if FresnelEdgeDetector not available."""
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        features: torch.Tensor,           # (B, 384, 37, 37) DINOv2 features
        depth: Optional[torch.Tensor] = None,  # (B, 1, H, W) depth map
        image_size: Tuple[int, int] = (518, 518),
        num_gaussians: Optional[int] = None  # HFTS: Override gaussians_per_patch for progressive growing
    ) -> Dict[str, torch.Tensor]:
        """
        Predict Gaussians from features with optional Fresnel enhancements.

        Fresnel-inspired features:
        - Depth zone quantization for better discontinuity handling
        - Edge-aware scale/opacity modulation (like diffraction fringes)
        - Optional phase output for interference blending

        HFTS Progressive Growing:
        - Pass num_gaussians to use fewer Gaussians per patch during early training
        - This provides significant speedup (1 Gaussian = 4× faster than 4 Gaussians)

        Returns:
            Dict with Gaussian parameters (and optionally 'phases', 'edge_strength')
        """
        B, C, H, W = features.shape  # (B, 384, 37, 37)
        # HFTS: Use num_gaussians if provided, otherwise use full capacity
        K = min(num_gaussians, self.gaussians_per_patch) if num_gaussians is not None else self.gaussians_per_patch
        device = features.device
        output_dim = self.output_per_gaussian

        # Flatten spatial dims for per-patch processing
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*37*37, 384)

        # Predict Gaussian params (always predict full capacity, then slice)
        outputs = self.mlp(features_flat)  # (B*37*37, full_K*output_dim)
        outputs = outputs.reshape(B, H, W, self.gaussians_per_patch, output_dim)  # (B, 37, 37, full_K, output_dim)

        # HFTS Progressive Growing: Only use first K Gaussians
        if K < self.gaussians_per_patch:
            outputs = outputs[..., :K, :]  # (B, 37, 37, K, output_dim)

        # Parse outputs
        raw_pos = outputs[..., 0:3]        # (B, 37, 37, K, 3)
        raw_scale = outputs[..., 3:6]       # (B, 37, 37, K, 3)
        rot_6d = outputs[..., 6:12]         # (B, 37, 37, K, 6)
        raw_color = outputs[..., 12:15]     # (B, 37, 37, K, 3)
        raw_opacity = outputs[..., 15:16]   # (B, 37, 37, K, 1)

        # Optional phase output for interference blending
        if self.use_phase_output and output_dim >= 17:
            raw_phase = outputs[..., 16:17]  # (B, 37, 37, K, 1)
        else:
            raw_phase = None

        # Create position grid
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )  # Each (37, 37)

        # Base positions from grid
        base_x = x_grid.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, K)  # (B, 37, 37, K)
        base_y = y_grid.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, K)

        # =========================================================================
        # FRESNEL DEPTH ZONES: Quantize depth to discrete layers
        # =========================================================================
        edge_strength = None  # Will be computed if edge-aware is enabled

        if depth is not None:
            # Resize depth to patch grid
            depth_grid = F.interpolate(depth, (H, W), mode='bilinear', align_corners=False)

            # Compute edge strength before depth processing (for edge-aware placement)
            if self.use_edge_aware and self.edge_detector is not None:
                # Edge detector expects (B, 1, H, W)
                depth_for_edges = F.interpolate(depth, (H, W), mode='bilinear', align_corners=False)
                edge_strength = self.edge_detector(depth_for_edges)  # (B, 1, H, W)

            # Apply Fresnel zone quantization if enabled
            if self.use_fresnel_zones and self.fresnel_zones is not None:
                # Quantize to zone centers for discrete depth layers
                depth_squeezed = depth_grid.squeeze(1)  # (B, H, W)
                zone_centers = self.fresnel_zones.get_zone_centers_for_depth(depth_squeezed)
                depth_grid = zone_centers.unsqueeze(1)  # (B, 1, H, W)

            depth_grid = depth_grid.squeeze(1).unsqueeze(-1).expand(-1, -1, -1, K)  # (B, 37, 37, K)
            base_z = self.depth_offset + depth_grid * (-2)  # Scale depth to reasonable range
        else:
            base_z = torch.full((B, H, W, K), self.depth_offset.item(), device=device)

        # Add predicted offsets to base positions (increased freedom for better positioning)
        positions = torch.stack([
            base_x + raw_pos[..., 0] * 0.25,  # More freedom to reposition
            base_y + raw_pos[..., 1] * 0.25,
            base_z + raw_pos[..., 2] * 0.25
        ], dim=-1)  # (B, 37, 37, K, 3)

        # Scale: ensure positive via softplus (increased range for larger Gaussians)
        # Clamp input to prevent exp() overflow in softplus (exp(x) overflows for x > ~88)
        raw_scale_clamped = torch.clamp(raw_scale, min=-10, max=20)
        scales = F.softplus(raw_scale_clamped + 1.0) * 0.15  # (B, 37, 37, K, 3)
        # Clamp final scales to prevent extreme values causing covariance issues
        scales = torch.clamp(scales, min=1e-6, max=2.0)

        # Rotation: 6D to quaternion
        rotations = rotation_6d_to_quaternion(rot_6d)  # (B, 37, 37, K, 4)

        # Color: sigmoid to [0, 1]
        colors = torch.sigmoid(raw_color)  # (B, 37, 37, K, 3)

        # Opacity: sigmoid to [0, 1]
        opacities = torch.sigmoid(raw_opacity).squeeze(-1)  # (B, 37, 37, K)

        # =========================================================================
        # FRESNEL EDGE-AWARE PLACEMENT: Modify scales/opacities at depth edges
        # Like diffraction fringes - concentrate detail at boundaries
        # =========================================================================
        if self.use_edge_aware and edge_strength is not None:
            # Expand edge strength to match Gaussian dimensions
            edge_expanded = edge_strength.squeeze(1).unsqueeze(-1)  # (B, H, W, 1)
            edge_expanded = edge_expanded.expand(-1, -1, -1, K)  # (B, H, W, K)

            # Shrink scales at edges (finer detail at boundaries)
            # edge_scale_factor=0.5 means scales are reduced by up to 50% at strong edges
            scale_modifier = 1.0 - self.edge_scale_factor * edge_expanded.unsqueeze(-1)
            scales = scales * scale_modifier  # (B, 37, 37, K, 3)

            # Boost opacity at edges (sharper boundaries)
            opacity_modifier = self.edge_opacity_boost * edge_expanded
            opacities = torch.clamp(opacities + opacity_modifier, 0, 1)  # (B, 37, 37, K)

        # =========================================================================
        # PHASE OUTPUT: For Huygens-Fresnel interference blending
        # =========================================================================
        phases = None
        if raw_phase is not None:
            # Phase in [0, 1] representing 0 to 2*pi
            phases = torch.sigmoid(raw_phase).squeeze(-1)  # (B, 37, 37, K)

        # Flatten spatial dims
        N = H * W * K
        positions = positions.reshape(B, N, 3)
        scales = scales.reshape(B, N, 3)
        rotations = rotations.reshape(B, N, 4)
        colors = colors.reshape(B, N, 3)
        opacities = opacities.reshape(B, N)

        result = {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities
        }

        # Include optional outputs
        if phases is not None:
            result['phases'] = phases.reshape(B, N)

        if edge_strength is not None:
            result['edge_strength'] = edge_strength  # (B, 1, H, W) - useful for visualization/debugging

        return result


# =============================================================================
# Physics-Derived Direct Patch Decoder
# =============================================================================

class PhysicsDirectPatchDecoder(nn.Module):
    """
    DirectPatchDecoder with physics-derived phase computation.

    Instead of predicting arbitrary phases or using heuristics, this decoder
    computes phases from optical path lengths using the wave equation:

        φ = (2π / λ) × path_length

    Key differences from DirectPatchDecoder:
    1. Phase is COMPUTED from z-position, not predicted
    2. Uses PhysicsFresnelZones for proper sqrt(n) zone spacing
    3. Optionally uses FresnelDiffraction for edge placement

    This connects the neural network to actual wave optics physics.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        gaussians_per_patch: int = 8,
        hidden_dims: list = [512, 512, 256, 128],
        dropout: float = 0.1,
        wavelength: float = 0.05,
        learnable_wavelength: bool = True,
        focal_depth: float = 0.5,
        use_diffraction_placement: bool = False,
    ):
        """
        Initialize PhysicsDirectPatchDecoder.

        Args:
            feature_dim: DINOv2 feature dimension (384)
            gaussians_per_patch: Number of Gaussians per patch
            hidden_dims: MLP hidden layer dimensions
            dropout: Dropout rate
            wavelength: Initial wavelength for phase computation
            learnable_wavelength: If True, wavelength is learnable
            focal_depth: Focal plane depth for phase computation
            use_diffraction_placement: If True, use Fresnel diffraction for edge placement
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.gaussians_per_patch = gaussians_per_patch
        self.wavelength = wavelength
        self.use_diffraction_placement = use_diffraction_placement

        # Output per Gaussian: position (3) + scale (3) + rotation_6d (6) + color (3) + opacity (1) = 16
        # Note: NO phase output - it's computed from physics!
        output_per_gaussian = 16
        self.output_per_gaussian = output_per_gaussian
        output_dim = gaussians_per_patch * output_per_gaussian

        # Per-patch MLP (same as DirectPatchDecoder)
        self.mlp = MLP(feature_dim, hidden_dims, output_dim, dropout)

        # Learned initial depth offset
        self.depth_offset = nn.Parameter(torch.tensor(-2.0))

        # Physics components
        if PHYSICS_FRESNEL_AVAILABLE:
            self.fresnel_zones = PhysicsFresnelZones(
                wavelength=wavelength,
                focal_depth=focal_depth,
                learnable_wavelength=learnable_wavelength,
            )
        else:
            self.fresnel_zones = None
            print("Warning: PhysicsFresnelZones not available, phases will be zero")

        # Optional diffraction component for edge placement
        if use_diffraction_placement and DIFFRACTION_AVAILABLE:
            self.diffraction = FresnelDiffraction(wavelength=wavelength)
        else:
            self.diffraction = None

    def forward(
        self,
        features: torch.Tensor,           # (B, 384, 37, 37) DINOv2 features
        depth: Optional[torch.Tensor] = None,  # (B, 1, H, W) depth map
        image_size: Tuple[int, int] = (518, 518)
    ) -> Dict[str, torch.Tensor]:
        """
        Predict Gaussians with physics-derived phases.

        The key difference from DirectPatchDecoder is that phases are
        COMPUTED from z-positions using the wave equation, not predicted.

        Returns:
            Dict with Gaussian parameters including physics-derived 'phases'
        """
        B, C, H, W = features.shape  # (B, 384, 37, 37)
        K = self.gaussians_per_patch
        device = features.device
        output_dim = self.output_per_gaussian

        # Flatten spatial dims for per-patch processing
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Predict Gaussian params (excluding phase)
        outputs = self.mlp(features_flat)
        outputs = outputs.reshape(B, H, W, K, output_dim)

        # Parse outputs (same as DirectPatchDecoder, but no phase)
        raw_pos = outputs[..., 0:3]
        raw_scale = outputs[..., 3:6]
        rot_6d = outputs[..., 6:12]
        raw_color = outputs[..., 12:15]
        raw_opacity = outputs[..., 15:16]

        # Create position grid
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )

        base_x = x_grid.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, K)
        base_y = y_grid.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, K)

        # Depth handling
        if depth is not None:
            depth_grid = F.interpolate(depth, (H, W), mode='bilinear', align_corners=False)
            depth_grid = depth_grid.squeeze(1).unsqueeze(-1).expand(-1, -1, -1, K)
            base_z = self.depth_offset + depth_grid * (-2)
        else:
            base_z = torch.full((B, H, W, K), self.depth_offset.item(), device=device)

        # Compute positions
        positions = torch.stack([
            base_x + raw_pos[..., 0] * 0.25,
            base_y + raw_pos[..., 1] * 0.25,
            base_z + raw_pos[..., 2] * 0.25
        ], dim=-1)

        # Scale, rotation, color, opacity (same as DirectPatchDecoder)
        scales = F.softplus(raw_scale + 1.0) * 0.15
        rotations = rotation_6d_to_quaternion(rot_6d)
        colors = torch.sigmoid(raw_color)
        opacities = torch.sigmoid(raw_opacity).squeeze(-1)

        # =========================================================================
        # PHYSICS-DERIVED PHASE (the key difference!)
        # Phase = (2π / λ) × path_length_from_camera
        # =========================================================================
        if self.fresnel_zones is not None:
            # Get z-positions (depth)
            z_positions = positions[..., 2]  # (B, H, W, K)

            # Normalize z to [0, 1] range for phase computation
            z_min = z_positions.min()
            z_max = z_positions.max()
            z_normalized = (z_positions - z_min) / (z_max - z_min + 1e-8)

            # Compute phase from depth using wave equation
            phases = self.fresnel_zones.depth_to_phase(z_normalized)

            # Wrap to [0, 2π]
            phases = phases % (2 * torch.pi)
        else:
            # Fallback: zero phase
            phases = torch.zeros_like(positions[..., 0])

        # Flatten spatial dims
        N = H * W * K
        positions = positions.reshape(B, N, 3)
        scales = scales.reshape(B, N, 3)
        rotations = rotations.reshape(B, N, 4)
        colors = colors.reshape(B, N, 3)
        opacities = opacities.reshape(B, N)
        phases = phases.reshape(B, N)

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
            'phases': phases,  # Physics-derived!
        }


# =============================================================================
# Diffractive Layer (D²NN Inspired)
# =============================================================================

class DiffractiveLayer(nn.Module):
    """
    Learnable diffractive surface inspired by D²NN (Diffractive Deep Neural Networks).

    From: "All-Optical Machine Learning Using Diffractive Deep Neural Networks"
          Lin et al., Science 2018

    Each spatial location has a trainable complex transmission coefficient:
        t(x, y) = A(x, y) × exp(i × φ(x, y))

    Where:
        A: Amplitude modulation [0, 1]
        φ: Phase modulation [0, 2π]

    When applied to a wave field, the output is:
        U_out = U_in × t

    D²NN achieved 98% MNIST accuracy with just learnable diffraction.
    This adds expressive power to our wave-based rendering.
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int = 3,  # RGB
        init_amplitude: float = 0.5,
        init_phase_scale: float = 0.1,  # Small random init for stability
    ):
        """
        Initialize DiffractiveLayer.

        Args:
            height: Spatial height of the diffractive surface
            width: Spatial width of the diffractive surface
            num_channels: Number of color channels (default 3 for RGB)
            init_amplitude: Initial amplitude value (before sigmoid)
            init_phase_scale: Scale for random phase initialization
        """
        super().__init__()

        self.height = height
        self.width = width
        self.num_channels = num_channels

        # Learnable amplitude modulation (raw, before sigmoid)
        # Shape: (num_channels, height, width) for per-channel control
        # or (1, height, width) for shared across channels
        self.amplitude_raw = nn.Parameter(
            torch.full((num_channels, height, width), init_amplitude)
        )

        # Learnable phase modulation
        # Initialized with small random values for stability
        self.phase = nn.Parameter(
            torch.randn(num_channels, height, width) * init_phase_scale
        )

    def get_transmission(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the amplitude and phase components of the transmission.

        Returns:
            amplitude: (C, H, W) in [0, 1]
            phase: (C, H, W) in [0, 2π]
        """
        amplitude = torch.sigmoid(self.amplitude_raw)
        # Wrap phase to [0, 2π]
        phase_wrapped = self.phase % (2 * torch.pi)
        return amplitude, phase_wrapped

    def forward(
        self,
        wave_field: torch.Tensor,  # (H, W, C, 2) or (B, H, W, C, 2) - last dim is [real, imag]
    ) -> torch.Tensor:
        """
        Apply diffractive modulation to incoming wave field.

        Complex multiplication: U_out = U_in × t
        Where t = A × exp(iφ) = A×cos(φ) + i×A×sin(φ)

        Args:
            wave_field: Complex wave field with shape (H, W, C, 2) or (B, H, W, C, 2)
                        Last dimension is [real, imaginary]

        Returns:
            Modulated wave field with same shape as input
        """
        # Get transmission coefficients
        amplitude, phase = self.get_transmission()

        # Compute complex transmission: t = A × exp(iφ)
        t_real = amplitude * torch.cos(phase)  # (C, H, W)
        t_imag = amplitude * torch.sin(phase)  # (C, H, W)

        # Reshape for broadcasting: (H, W, C)
        t_real = t_real.permute(1, 2, 0)  # (H, W, C)
        t_imag = t_imag.permute(1, 2, 0)  # (H, W, C)

        # Handle batched input
        has_batch = wave_field.dim() == 5
        if has_batch:
            t_real = t_real.unsqueeze(0)  # (1, H, W, C)
            t_imag = t_imag.unsqueeze(0)  # (1, H, W, C)

        # Extract real and imaginary parts of input
        u_real = wave_field[..., 0]  # (..., H, W, C)
        u_imag = wave_field[..., 1]  # (..., H, W, C)

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_real = u_real * t_real - u_imag * t_imag
        out_imag = u_real * t_imag + u_imag * t_real

        # Stack back to complex representation
        return torch.stack([out_real, out_imag], dim=-1)

    def forward_complex(
        self,
        wave_field: torch.Tensor,  # (H, W, C) or (B, H, W, C) complex tensor
    ) -> torch.Tensor:
        """
        Apply diffractive modulation using native complex tensors.

        Alternative interface for torch.cfloat tensors.

        Args:
            wave_field: Complex wave field (H, W, C) or (B, H, W, C)

        Returns:
            Modulated complex wave field
        """
        amplitude, phase = self.get_transmission()

        # Build complex transmission coefficient
        t_complex = amplitude * torch.exp(1j * phase)  # (C, H, W)
        t_complex = t_complex.permute(1, 2, 0)  # (H, W, C)

        # Handle batched input
        if wave_field.dim() == 4:
            t_complex = t_complex.unsqueeze(0)  # (1, H, W, C)

        return wave_field * t_complex

    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for the diffractive layer.

        Encourages:
        1. Smooth amplitude (total variation)
        2. Smooth phase (total variation)
        3. Moderate amplitude values (not too transparent or opaque)

        Returns:
            Scalar regularization loss
        """
        amplitude, phase = self.get_transmission()

        # Total variation for smoothness
        amp_tv = (
            (amplitude[:, 1:, :] - amplitude[:, :-1, :]).abs().mean() +
            (amplitude[:, :, 1:] - amplitude[:, :, :-1]).abs().mean()
        )

        phase_tv = (
            (phase[:, 1:, :] - phase[:, :-1, :]).abs().mean() +
            (phase[:, :, 1:] - phase[:, :, :-1]).abs().mean()
        )

        # Encourage amplitude near 0.5 (moderate transmission)
        amp_center = ((amplitude - 0.5) ** 2).mean()

        return 0.01 * amp_tv + 0.01 * phase_tv + 0.001 * amp_center


class MultiscaleDiffractiveLayer(nn.Module):
    """
    Multi-scale diffractive layer with pyramid structure.

    Applies diffractive modulation at multiple spatial scales,
    allowing both coarse and fine control over wave propagation.
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int = 3,
        num_scales: int = 3,  # Number of pyramid levels
    ):
        super().__init__()

        self.num_scales = num_scales

        # Create diffractive layers at each scale
        self.layers = nn.ModuleList()
        for i in range(num_scales):
            scale_h = height // (2 ** i)
            scale_w = width // (2 ** i)
            if scale_h < 4 or scale_w < 4:
                break
            self.layers.append(
                DiffractiveLayer(scale_h, scale_w, num_channels)
            )

        self.actual_scales = len(self.layers)

    def forward(
        self,
        wave_field: torch.Tensor,  # (H, W, C, 2) - last dim is [real, imag]
    ) -> torch.Tensor:
        """
        Apply multi-scale diffractive modulation.

        Upsamples coarse modulations to match input resolution.
        """
        H, W = wave_field.shape[:2]
        C = wave_field.shape[2]
        device = wave_field.device

        # Accumulate modulated fields
        result = wave_field.clone()

        for i, layer in enumerate(self.layers):
            # Get layer's native resolution
            layer_h, layer_w = layer.height, layer.width

            if i == 0:
                # First layer: direct application at full resolution
                result = layer(result)
            else:
                # Coarser layers: downsample, modulate, upsample
                # Downsample wave field
                result_down = F.interpolate(
                    result.permute(2, 3, 0, 1),  # (C, 2, H, W)
                    size=(layer_h, layer_w),
                    mode='bilinear',
                    align_corners=False
                ).permute(2, 3, 0, 1)  # (h, w, C, 2)

                # Apply modulation
                result_down = layer(result_down)

                # Upsample back
                result_up = F.interpolate(
                    result_down.permute(2, 3, 0, 1),  # (C, 2, h, w)
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).permute(2, 3, 0, 1)  # (H, W, C, 2)

                # Blend with existing result (residual connection)
                weight = 1.0 / (i + 1)  # Decrease influence of coarser scales
                result = result * (1 - weight) + result_up * weight

        return result

    def regularization_loss(self) -> torch.Tensor:
        """Combined regularization from all scales."""
        total = torch.tensor(0.0, device=self.layers[0].amplitude_raw.device)
        for layer in self.layers:
            total = total + layer.regularization_loss()
        return total / len(self.layers)


# =============================================================================
# Experiment 3: Feature-Guided SAAG (Lightweight)
# =============================================================================

class FeatureGuidedSAAG(nn.Module):
    """
    Learn to spatially modulate SAAG hyperparameters based on features.

    Keep the SAAG algorithm intact, but let the network predict
    per-patch parameter modifications.

    Extremely lightweight - just learns to tune existing algorithm.

    Predicted params per patch:
    - aspect_ratio_mult: Multiplier for SAAG aspect ratio
    - edge_threshold_add: Additive adjustment to edge threshold
    - edge_shrink_mult: Multiplier for edge shrink factor
    - normal_strength_mult: Multiplier for normal alignment strength
    - base_size_mult: Multiplier for base Gaussian size
    - opacity_mult: Multiplier for opacity
    """

    def __init__(
        self,
        feature_dim: int = 384,
        num_params: int = 6,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_params = num_params

        # Very simple network: just two linear layers
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_params)
        )

        # Initialize to predict zeros (no modification)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        features: torch.Tensor,  # (B, 384, 37, 37) DINOv2 features
    ) -> Dict[str, torch.Tensor]:
        """
        Predict per-patch SAAG parameter modifications.

        Returns:
            Dict with parameter maps (B, 37, 37)
        """
        B, C, H, W = features.shape

        # Flatten to per-patch
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Predict params
        params = self.net(features_flat)  # (B*H*W, 6)
        params = params.reshape(B, H, W, self.num_params)

        # Parse and apply activation functions
        return {
            # Multiplicative params use 1 + tanh(x) * scale to stay near 1
            'aspect_ratio_mult': 1.0 + torch.tanh(params[..., 0]) * 0.5,    # [0.5, 1.5]
            'edge_threshold_add': torch.tanh(params[..., 1]) * 0.1,         # [-0.1, 0.1]
            'edge_shrink_mult': 1.0 + torch.tanh(params[..., 2]) * 0.3,     # [0.7, 1.3]
            'normal_strength_mult': 1.0 + torch.tanh(params[..., 3]) * 0.3, # [0.7, 1.3]
            'base_size_mult': 1.0 + torch.tanh(params[..., 4]) * 0.5,       # [0.5, 1.5]
            'opacity_mult': 1.0 + torch.tanh(params[..., 5]) * 0.3,         # [0.7, 1.3]
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    B = 2  # Batch size
    N = 1000  # Number of SAAG Gaussians

    # Create dummy inputs
    features = torch.randn(B, 384, 37, 37, device=device)
    depth = torch.rand(B, 1, 518, 518, device=device)

    # SAAG inputs for refinement model
    saag_positions = torch.randn(B, N, 3, device=device) * 0.5
    saag_positions[..., 2] -= 2  # Move in front of camera
    saag_scales = torch.ones(B, N, 3, device=device) * 0.05
    saag_rotations = torch.zeros(B, N, 4, device=device)
    saag_rotations[..., 0] = 1  # Identity quaternion
    saag_colors = torch.rand(B, N, 3, device=device)
    saag_opacities = torch.ones(B, N, device=device) * 0.8

    print("=" * 60)
    print("Testing Experiment 1: SAAGRefinementNet")
    print("=" * 60)
    model1 = SAAGRefinementNet().to(device)
    print(f"Parameters: {count_parameters(model1):,}")

    output1 = model1(
        features, saag_positions, saag_scales, saag_rotations,
        saag_colors, saag_opacities
    )

    print(f"Output positions shape: {output1['positions'].shape}")
    print(f"Output scales shape: {output1['scales'].shape}")
    print(f"Output rotations shape: {output1['rotations'].shape}")
    print(f"Output colors shape: {output1['colors'].shape}")
    print(f"Output opacities shape: {output1['opacities'].shape}")

    # Test backward pass
    loss = output1['positions'].mean() + output1['colors'].mean()
    loss.backward()
    print("Backward pass: OK")

    print("\n" + "=" * 60)
    print("Testing Experiment 2: DirectPatchDecoder")
    print("=" * 60)
    model2 = DirectPatchDecoder(gaussians_per_patch=2).to(device)
    print(f"Parameters: {count_parameters(model2):,}")

    output2 = model2(features, depth)

    N2 = 37 * 37 * 2  # 2738 Gaussians
    print(f"Expected N: {N2}")
    print(f"Output positions shape: {output2['positions'].shape}")
    print(f"Output scales shape: {output2['scales'].shape}")
    print(f"Output rotations shape: {output2['rotations'].shape}")
    print(f"Output colors shape: {output2['colors'].shape}")
    print(f"Output opacities shape: {output2['opacities'].shape}")

    # Test backward pass
    loss = output2['positions'].mean() + output2['colors'].mean()
    loss.backward()
    print("Backward pass: OK")

    print("\n" + "=" * 60)
    print("Testing Experiment 3: FeatureGuidedSAAG")
    print("=" * 60)
    model3 = FeatureGuidedSAAG().to(device)
    print(f"Parameters: {count_parameters(model3):,}")

    output3 = model3(features)

    for key, val in output3.items():
        print(f"  {key}: shape {val.shape}, range [{val.min().item():.3f}, {val.max().item():.3f}]")

    # Test backward pass
    loss = sum(v.mean() for v in output3.values())
    loss.backward()
    print("Backward pass: OK")

    print("\n" + "=" * 60)
    print("Parameter comparison:")
    print("=" * 60)
    print(f"Experiment 1 (SAAGRefinementNet): {count_parameters(model1):,} params")
    print(f"Experiment 2 (DirectPatchDecoder): {count_parameters(model2):,} params")
    print(f"Experiment 3 (FeatureGuidedSAAG): {count_parameters(model3):,} params")
