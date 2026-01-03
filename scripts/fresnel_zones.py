#!/usr/bin/env python3
"""
Fresnel Zones - Physics-Inspired Depth Organization for Gaussian Splatting

This module implements concepts from Augustin-Jean Fresnel's wave optics theory
for organizing 3D Gaussian primitives in depth space.

Key Concepts:
    - Fresnel Zones: Discrete depth layers that organize Gaussians hierarchically
    - Zone Boundaries: Depth discontinuities where more detail is needed (like diffraction fringes)
    - Wave Superposition: Multiple Gaussians combine like wavelet contributions

The core insight is that just as Fresnel zones organize wave contributions by path length,
we can organize Gaussians by depth to improve rendering quality at discontinuities.

Usage:
    from fresnel_zones import FresnelZones

    fresnel = FresnelZones(num_zones=8)
    zone_idx = fresnel.quantize_depth(depth_tensor)
    boundary_mask = fresnel.compute_boundary_mask(depth_tensor)

    # In loss computation
    boundary_weight = fresnel.get_boundary_weight(depth_tensor, base_weight=1.0, boundary_boost=2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union


class FresnelZones(nn.Module):
    """
    Discrete depth zone management inspired by Fresnel zone plates.

    Fresnel zones in optics are concentric regions where light waves contribute
    alternately constructively and destructively. We adapt this concept for
    depth-based Gaussian organization:

    - Zones provide hierarchical depth structure (coarse-to-fine)
    - Zone boundaries indicate depth discontinuities (silhouettes, edges)
    - Boundary emphasis improves reconstruction of depth edges

    Attributes:
        num_zones: Number of discrete depth zones
        depth_range: Tuple of (min_depth, max_depth) for normalization
        zone_boundaries: Tensor of zone boundary positions
        zone_centers: Tensor of zone center positions
        boundary_threshold: Distance threshold for boundary detection
    """

    def __init__(
        self,
        num_zones: int = 8,
        depth_range: Tuple[float, float] = (0.0, 1.0),
        boundary_threshold: float = 0.02,
        soft_boundaries: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize FresnelZones.

        Args:
            num_zones: Number of discrete depth zones (default: 8)
            depth_range: (min, max) depth values for zone computation
            boundary_threshold: Distance from boundary to be considered "at boundary"
            soft_boundaries: If True, use soft/differentiable boundary detection
            device: Torch device for tensors
        """
        super().__init__()

        self.num_zones = num_zones
        self.depth_range = depth_range
        self.boundary_threshold = boundary_threshold
        self.soft_boundaries = soft_boundaries

        # Compute zone boundaries (N+1 boundaries for N zones)
        boundaries = torch.linspace(
            depth_range[0], depth_range[1], num_zones + 1
        )
        self.register_buffer('zone_boundaries', boundaries)

        # Compute zone centers
        centers = (boundaries[:-1] + boundaries[1:]) / 2
        self.register_buffer('zone_centers', centers)

        # Zone width for soft boundary computation
        zone_width = (depth_range[1] - depth_range[0]) / num_zones
        self.register_buffer('zone_width', torch.tensor(zone_width))

        # Learnable boundary emphasis (can be trained)
        self.boundary_emphasis = nn.Parameter(torch.ones(num_zones + 1))

    def quantize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous depth values to discrete zone indices.

        This assigns each depth value to its corresponding Fresnel zone,
        similar to how Fresnel zone plates discretize optical path lengths.

        Args:
            depth: Tensor of depth values, any shape

        Returns:
            Tensor of zone indices (0 to num_zones-1), same shape as input
        """
        # Clamp to valid range
        depth_clamped = torch.clamp(depth, self.depth_range[0], self.depth_range[1])

        # Use bucketize for efficient zone assignment
        # Note: bucketize returns index of bucket, boundaries[1:-1] gives interior boundaries
        zone_idx = torch.bucketize(depth_clamped, self.zone_boundaries[1:-1])

        return zone_idx

    def get_zone_centers_for_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Get the center depth value for each input depth's zone.

        This "snaps" depth values to their zone centers, creating the
        discrete layered structure characteristic of Fresnel zone organization.

        Args:
            depth: Tensor of depth values

        Returns:
            Tensor of zone center values, same shape as input
        """
        zone_idx = self.quantize_depth(depth)

        # Gather zone centers for each depth value
        # Handle arbitrary input shapes
        original_shape = zone_idx.shape
        flat_idx = zone_idx.flatten()
        flat_centers = self.zone_centers[flat_idx]

        return flat_centers.view(original_shape)

    def compute_boundary_mask(
        self,
        depth: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Detect pixels near zone boundaries (like Fresnel diffraction fringes).

        In Fresnel diffraction, intensity peaks occur at zone boundaries.
        We use this to identify depth discontinuities where more Gaussian
        detail is needed.

        Args:
            depth: Tensor of depth values
            threshold: Override default boundary threshold

        Returns:
            Boolean tensor (or soft mask if soft_boundaries=True)
        """
        if threshold is None:
            threshold = self.boundary_threshold

        # Compute distance to nearest boundary
        # Shape: (*depth.shape, num_boundaries)
        depth_expanded = depth.unsqueeze(-1)
        distances = torch.abs(depth_expanded - self.zone_boundaries)

        # Minimum distance to any boundary
        min_distance, nearest_boundary = distances.min(dim=-1)

        if self.soft_boundaries:
            # Soft mask using sigmoid for differentiability
            # Lower distance = higher boundary strength
            sharpness = 10.0 / threshold  # Controls transition sharpness
            boundary_mask = torch.sigmoid(sharpness * (threshold - min_distance))
        else:
            # Hard mask
            boundary_mask = (min_distance < threshold).float()

        return boundary_mask

    def get_boundary_weight(
        self,
        depth: torch.Tensor,
        base_weight: float = 1.0,
        boundary_boost: float = 2.0
    ) -> torch.Tensor:
        """
        Compute per-pixel loss weights with boundary emphasis.

        This implements the "Fresnel fringe" concept: pixels at depth
        discontinuities (zone boundaries) receive higher training weight,
        similar to how diffraction fringes concentrate energy at edges.

        Args:
            depth: Tensor of depth values
            base_weight: Weight for non-boundary pixels
            boundary_boost: Additional weight multiplier at boundaries

        Returns:
            Tensor of weights, same shape as depth
        """
        boundary_mask = self.compute_boundary_mask(depth)

        # Linear interpolation between base and boosted weight
        weights = base_weight + boundary_mask * (boundary_boost - base_weight)

        return weights

    def compute_zone_gradients(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude within each zone for edge detection.

        High gradients within a zone indicate depth discontinuities
        that should receive more Gaussian density (like diffraction fringes).

        Args:
            depth: Depth tensor of shape (B, 1, H, W) or (B, H, W)

        Returns:
            Gradient magnitude tensor
        """
        # Ensure 4D tensor
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)

        # Compute gradients
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)

        # Gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        return gradient_magnitude.squeeze(1)

    def get_adaptive_density(
        self,
        depth: torch.Tensor,
        min_density: float = 0.5,
        max_density: float = 2.0
    ) -> torch.Tensor:
        """
        Compute adaptive Gaussian density based on depth zone and boundaries.

        This implements the "Fresnel lens efficiency" concept: more Gaussians
        where detail is needed (boundaries, foreground) and fewer where not.

        Args:
            depth: Depth tensor
            min_density: Minimum density multiplier
            max_density: Maximum density multiplier (at boundaries)

        Returns:
            Density multiplier tensor
        """
        # Base density from zone (foreground gets more density)
        zone_idx = self.quantize_depth(depth).float()
        zone_factor = 1.0 - (zone_idx / self.num_zones) * 0.3  # Foreground boost

        # Boundary boost
        boundary_mask = self.compute_boundary_mask(depth)

        # Combine: zone factor + boundary emphasis
        density = zone_factor * (min_density + boundary_mask * (max_density - min_density))

        return density

    def interpolate_across_zones(
        self,
        depth: torch.Tensor,
        zone_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate features across zones for smooth transitions.

        This implements wave superposition: features from adjacent zones
        are blended based on proximity to zone boundaries.

        Args:
            depth: Depth tensor (B, H, W)
            zone_features: Per-zone features (num_zones, C) or (B, num_zones, C)

        Returns:
            Interpolated features (B, H, W, C) or (B, C, H, W)
        """
        # Compute zone indices and fractional position within zone
        depth_normalized = (depth - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
        depth_scaled = depth_normalized * self.num_zones

        # Lower and upper zone indices
        zone_low = torch.floor(depth_scaled).long().clamp(0, self.num_zones - 1)
        zone_high = (zone_low + 1).clamp(0, self.num_zones - 1)

        # Interpolation weight (how close to upper zone)
        alpha = (depth_scaled - zone_low.float()).unsqueeze(-1)

        # Handle different zone_features shapes
        if zone_features.dim() == 2:
            # (num_zones, C) -> expand for batch
            features_low = zone_features[zone_low]
            features_high = zone_features[zone_high]
        else:
            # (B, num_zones, C) -> gather per batch
            B = zone_features.shape[0]
            features_low = torch.gather(
                zone_features, 1,
                zone_low.view(B, -1, 1).expand(-1, -1, zone_features.shape[-1])
            ).view(*zone_low.shape, -1)
            features_high = torch.gather(
                zone_features, 1,
                zone_high.view(B, -1, 1).expand(-1, -1, zone_features.shape[-1])
            ).view(*zone_high.shape, -1)

        # Linear interpolation (wave superposition)
        interpolated = (1 - alpha) * features_low + alpha * features_high

        return interpolated

    def get_zone_encoding(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Create a learnable zone encoding for each depth value.

        Returns a one-hot or soft encoding of the zone, useful as
        additional input features to the decoder.

        Args:
            depth: Depth tensor of any shape

        Returns:
            Zone encoding tensor with extra dimension of size num_zones
        """
        zone_idx = self.quantize_depth(depth)

        if self.soft_boundaries:
            # Soft encoding based on distance to each zone center
            depth_expanded = depth.unsqueeze(-1)
            distances = torch.abs(depth_expanded - self.zone_centers)

            # Convert distances to soft assignments (closer = higher weight)
            # Use softmax over negative distances
            encoding = F.softmax(-distances / self.zone_width, dim=-1)
        else:
            # Hard one-hot encoding
            encoding = F.one_hot(zone_idx, num_classes=self.num_zones).float()

        return encoding

    def forward(
        self,
        depth: torch.Tensor,
        return_all: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        Main forward pass - compute all Fresnel zone information.

        Args:
            depth: Input depth tensor
            return_all: If True, return dict with all computed values

        Returns:
            Zone indices, or dict with zone_idx, centers, boundary_mask, etc.
        """
        zone_idx = self.quantize_depth(depth)

        if not return_all:
            return zone_idx

        return {
            'zone_idx': zone_idx,
            'zone_centers': self.get_zone_centers_for_depth(depth),
            'boundary_mask': self.compute_boundary_mask(depth),
            'boundary_weight': self.get_boundary_weight(depth),
            'zone_encoding': self.get_zone_encoding(depth),
            'adaptive_density': self.get_adaptive_density(depth),
            'gradient_magnitude': self.compute_zone_gradients(depth.unsqueeze(1) if depth.dim() == 3 else depth)
        }

    def extra_repr(self) -> str:
        return (
            f"num_zones={self.num_zones}, "
            f"depth_range={self.depth_range}, "
            f"boundary_threshold={self.boundary_threshold}, "
            f"soft_boundaries={self.soft_boundaries}"
        )


class FresnelEdgeDetector(nn.Module):
    """
    Learned edge detector inspired by Fresnel diffraction patterns.

    In Fresnel diffraction, edges create characteristic intensity patterns
    with bright/dark fringes. This module learns to detect depth edges
    that benefit from increased Gaussian density.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        use_depth_gradients: bool = True
    ):
        """
        Initialize edge detector.

        Args:
            in_channels: Number of input channels (1 for depth, more for features)
            hidden_channels: Hidden layer channels
            use_depth_gradients: If True, concatenate depth gradients as input
        """
        super().__init__()

        self.use_depth_gradients = use_depth_gradients

        # If using gradients, input has 3 channels (depth + grad_x + grad_y)
        actual_in = in_channels + 2 if use_depth_gradients else in_channels

        self.conv1 = nn.Conv2d(actual_in, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

        self.activation = nn.ReLU(inplace=True)

        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Detect edges in depth map.

        Args:
            depth: Depth tensor (B, 1, H, W)

        Returns:
            Edge strength tensor (B, 1, H, W), values in [0, 1]
        """
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        if self.use_depth_gradients:
            # Compute depth gradients
            grad_x = F.conv2d(depth, self.sobel_x, padding=1)
            grad_y = F.conv2d(depth, self.sobel_y, padding=1)

            # Concatenate as input
            x = torch.cat([depth, grad_x, grad_y], dim=1)
        else:
            x = depth

        # Convolutional layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))

        return x


def visualize_fresnel_zones(
    depth: torch.Tensor,
    fresnel: FresnelZones,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization of Fresnel zone organization.

    Args:
        depth: Depth tensor (H, W)
        fresnel: FresnelZones instance
        save_path: Optional path to save visualization

    Returns:
        RGB visualization as numpy array
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    # Ensure 2D
    if depth.dim() > 2:
        depth = depth.squeeze()

    depth_np = depth.cpu().numpy()

    # Get zone information
    with torch.no_grad():
        info = fresnel(depth, return_all=True)
        zone_idx = info['zone_idx'].cpu().numpy()
        boundary_mask = info['boundary_mask'].cpu().numpy()

    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Original depth
    ax = axes[0, 0]
    im = ax.imshow(depth_np, cmap='viridis')
    ax.set_title('Original Depth')
    plt.colorbar(im, ax=ax)

    # 2. Zone assignment (colored by zone)
    ax = axes[0, 1]
    im = ax.imshow(zone_idx, cmap='tab10', vmin=0, vmax=fresnel.num_zones-1)
    ax.set_title(f'Fresnel Zones ({fresnel.num_zones} zones)')
    plt.colorbar(im, ax=ax, ticks=range(fresnel.num_zones))

    # 3. Zone boundaries (Fresnel fringes)
    ax = axes[1, 0]
    im = ax.imshow(boundary_mask, cmap='hot')
    ax.set_title('Zone Boundaries (Fresnel Fringes)')
    plt.colorbar(im, ax=ax)

    # 4. Combined visualization
    ax = axes[1, 1]
    # Create HSV image: hue from zone, saturation from boundary
    hue = zone_idx / fresnel.num_zones
    saturation = 0.5 + 0.5 * boundary_mask
    value = 0.8 + 0.2 * boundary_mask
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    ax.imshow(rgb)
    ax.set_title('Combined (Zone Color + Boundary Emphasis)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Fresnel zone visualization to {save_path}")

    # Convert figure to numpy array
    fig.canvas.draw()
    rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return rgb_array


# Test/demo code
if __name__ == "__main__":
    print("=" * 60)
    print("Fresnel Zones - Demo")
    print("=" * 60)

    # Create instance
    fresnel = FresnelZones(num_zones=8)
    print(f"\nCreated: {fresnel}")

    # Test with synthetic depth map
    H, W = 64, 64

    # Create depth with clear zones (ramp)
    y = torch.linspace(0, 1, H).view(-1, 1).expand(H, W)
    x = torch.linspace(0, 1, W).view(1, -1).expand(H, W)

    # Circular depth (like looking at a face)
    depth = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    depth = 1.0 - torch.clamp(depth * 2, 0, 1)  # Invert: center is close

    print(f"\nTest depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Get all zone information
    info = fresnel(depth, return_all=True)

    print(f"\nZone indices shape: {info['zone_idx'].shape}")
    print(f"Unique zones: {torch.unique(info['zone_idx']).tolist()}")
    print(f"Boundary mask range: [{info['boundary_mask'].min():.3f}, {info['boundary_mask'].max():.3f}]")
    print(f"Adaptive density range: [{info['adaptive_density'].min():.3f}, {info['adaptive_density'].max():.3f}]")

    # Test boundary weight computation
    weights = fresnel.get_boundary_weight(depth, base_weight=1.0, boundary_boost=2.0)
    print(f"Boundary weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Test zone encoding
    encoding = fresnel.get_zone_encoding(depth)
    print(f"Zone encoding shape: {encoding.shape}")  # Should be (H, W, num_zones)

    # Create visualization
    print("\nGenerating visualization...")
    try:
        viz = visualize_fresnel_zones(depth, fresnel, "fresnel_zones_demo.png")
        print(f"Visualization shape: {viz.shape}")
    except Exception as e:
        print(f"Visualization failed (matplotlib may not be available): {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
