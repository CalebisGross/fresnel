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


class PhysicsFresnelZones(nn.Module):
    """
    Physics-derived Fresnel zone computation based on optical path differences.

    Unlike the heuristic FresnelZones class which uses uniform depth slicing,
    this implementation uses the actual Fresnel zone plate formula:

        r_n = √(n × λ × f)

    where:
        r_n = radius of nth zone boundary
        λ = wavelength (learnable or fixed)
        f = focal distance
        n = zone number (1, 2, 3, ...)

    Key physics insights:
    1. Real Fresnel zones are NOT uniformly spaced - inner zones are wider
    2. Each zone contributes with alternating phase (±π)
    3. Phase is derived from optical path length: φ = (2π/λ) × path_length

    Named after Augustin-Jean Fresnel (1788-1827).
    """

    def __init__(
        self,
        num_zones: int = 8,
        wavelength: float = 0.05,      # Effective wavelength in depth units [0,1]
        focal_depth: float = 0.5,       # Focal plane depth
        learnable_wavelength: bool = True,
        wavelength_min: float = 0.01,   # Minimum wavelength (prevents divergence)
        wavelength_max: float = 0.5,    # Maximum wavelength
    ):
        """
        Initialize PhysicsFresnelZones.

        Args:
            num_zones: Number of Fresnel zones
            wavelength: Effective wavelength in normalized depth units [0, 1]
            focal_depth: Depth of the focal plane (where path difference = 0)
            learnable_wavelength: If True, wavelength is a learnable parameter
            wavelength_min: Minimum allowed wavelength (risk mitigation)
            wavelength_max: Maximum allowed wavelength (risk mitigation)
        """
        super().__init__()

        self.num_zones = num_zones
        self.focal_depth = focal_depth
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max

        # Learnable or fixed wavelength
        if learnable_wavelength:
            # Use raw parameter, constrain in forward pass
            self._wavelength_raw = nn.Parameter(torch.tensor(wavelength))
        else:
            self.register_buffer('_wavelength_raw', torch.tensor(wavelength))

        self.learnable_wavelength = learnable_wavelength

    @property
    def wavelength(self) -> torch.Tensor:
        """Get wavelength with constraints applied."""
        # Constrain to [wavelength_min, wavelength_max] to prevent divergence
        return torch.clamp(
            torch.abs(self._wavelength_raw),
            self.wavelength_min,
            self.wavelength_max
        )

    def compute_zone_boundaries(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute zone boundaries using Fresnel zone plate formula.

        r_n = sqrt(n × λ × f)

        Unlike uniform spacing, this produces wider inner zones and
        narrower outer zones - the true physics of Fresnel zones.

        Args:
            device: Device for output tensor

        Returns:
            Tensor of shape (num_zones + 1,) with zone boundaries in [0, 1]
        """
        if device is None:
            device = self._wavelength_raw.device

        λ = self.wavelength
        f = self.focal_depth

        # Zone indices 0 to num_zones
        n = torch.arange(self.num_zones + 1, device=device, dtype=torch.float32)

        # Fresnel zone radii: r_n = sqrt(n × λ × f)
        r_n = torch.sqrt(n * λ * f)

        # Normalize to [0, 1] range
        r_n = r_n / (r_n[-1] + 1e-8)

        return r_n

    def get_zone_index(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Get zone index for each depth value.

        Uses physics-based boundaries rather than uniform slicing.

        Args:
            depth: Tensor of depth values in [0, 1]

        Returns:
            Tensor of zone indices (0 to num_zones-1)
        """
        boundaries = self.compute_zone_boundaries(depth.device)

        # bucketize assigns to bucket based on boundaries
        zone_idx = torch.bucketize(depth, boundaries[1:-1])
        zone_idx = torch.clamp(zone_idx, 0, self.num_zones - 1)

        return zone_idx

    def get_zone_phase(self, zone_idx: torch.Tensor) -> torch.Tensor:
        """
        Get phase for each zone (alternating pattern).

        In Fresnel zone plates:
        - Odd zones: phase = 0 (constructive contribution)
        - Even zones: phase = π (destructive, but we use the sign flip)

        This alternating pattern is KEY to Fresnel zone physics!

        Args:
            zone_idx: Tensor of zone indices

        Returns:
            Tensor of phase values (0 or π)
        """
        # Alternating: zone 0 -> 0, zone 1 -> π, zone 2 -> 0, ...
        return (zone_idx % 2).float() * torch.pi

    def compute_path_difference(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute optical path difference from focal plane.

        In wave optics, path difference determines phase:
        Δpath = |depth - focal_depth|

        Args:
            depth: Tensor of depth values

        Returns:
            Tensor of path differences (always positive)
        """
        return torch.abs(depth - self.focal_depth)

    def depth_to_phase(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert depth to optical phase using wave equation.

        φ = (2π / λ) × path_length

        This is the REAL physics - phase directly from path length!

        Args:
            depth: Tensor of depth values

        Returns:
            Tensor of phase values in radians
        """
        λ = self.wavelength
        path_diff = self.compute_path_difference(depth)

        # Wave equation: φ = (2π / λ) × d
        phase = (2 * torch.pi / λ) * path_diff

        return phase

    def forward(
        self,
        depth: torch.Tensor,
        return_all: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        Compute physics-based Fresnel zone information.

        Args:
            depth: Input depth tensor
            return_all: If True, return dict with all computed values

        Returns:
            Phase values, or dict with zone_idx, phase, boundaries, etc.
        """
        phase = self.depth_to_phase(depth)

        if not return_all:
            return phase

        zone_idx = self.get_zone_index(depth)

        return {
            'phase': phase,
            'zone_idx': zone_idx,
            'zone_phase': self.get_zone_phase(zone_idx),
            'path_difference': self.compute_path_difference(depth),
            'boundaries': self.compute_zone_boundaries(depth.device),
            'wavelength': self.wavelength,
        }

    def extra_repr(self) -> str:
        return (
            f"num_zones={self.num_zones}, "
            f"wavelength={self.wavelength.item():.4f}, "
            f"focal_depth={self.focal_depth}, "
            f"learnable={self.learnable_wavelength}"
        )


class MultiWavelengthPhysics(nn.Module):
    """
    Per-channel wavelength physics for RGB light.

    Real light has different wavelengths per color channel:
        - Red:   ~700nm
        - Green: ~550nm
        - Blue:  ~450nm

    Physical ratios: λ_R : λ_G : λ_B ≈ 1.27 : 1.0 : 0.82

    This enables:
        - Chromatic aberration effects (color fringing at edges)
        - More realistic interference patterns per channel
        - Wavelength-dependent diffraction behavior

    Based on research from:
        - Nature: "Towards real-time photorealistic 3D holography with deep neural networks"
        - arXiv: "Complex-Valued Holographic Radiance Fields"
    """

    # Physical wavelength ratios (normalized to green)
    WAVELENGTH_RATIO_R = 700.0 / 550.0  # ~1.27
    WAVELENGTH_RATIO_G = 1.0
    WAVELENGTH_RATIO_B = 450.0 / 550.0  # ~0.82

    def __init__(
        self,
        base_wavelength: float = 0.05,
        learnable: bool = True,
        use_physical_ratios: bool = True,
        wavelength_min: float = 0.01,
        wavelength_max: float = 0.5,
        focal_depth: float = 0.5,
    ):
        """
        Initialize MultiWavelengthPhysics.

        Args:
            base_wavelength: Base wavelength (used for green channel)
            learnable: If True, wavelengths are learnable parameters
            use_physical_ratios: If True, initialize with physical RGB ratios
            wavelength_min: Minimum allowed wavelength
            wavelength_max: Maximum allowed wavelength
            focal_depth: Focal plane depth for path difference computation
        """
        super().__init__()

        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.focal_depth = focal_depth
        self.learnable = learnable

        # Initialize wavelengths with physical ratios if requested
        if use_physical_ratios:
            init_r = base_wavelength * self.WAVELENGTH_RATIO_R
            init_g = base_wavelength * self.WAVELENGTH_RATIO_G
            init_b = base_wavelength * self.WAVELENGTH_RATIO_B
        else:
            init_r = init_g = init_b = base_wavelength

        if learnable:
            self._wavelength_r_raw = nn.Parameter(torch.tensor(init_r))
            self._wavelength_g_raw = nn.Parameter(torch.tensor(init_g))
            self._wavelength_b_raw = nn.Parameter(torch.tensor(init_b))
        else:
            self.register_buffer('_wavelength_r_raw', torch.tensor(init_r))
            self.register_buffer('_wavelength_g_raw', torch.tensor(init_g))
            self.register_buffer('_wavelength_b_raw', torch.tensor(init_b))

    def _constrain_wavelength(self, wavelength_raw: torch.Tensor) -> torch.Tensor:
        """Apply constraints to prevent wavelength divergence."""
        return torch.clamp(
            torch.abs(wavelength_raw),
            self.wavelength_min,
            self.wavelength_max
        )

    @property
    def wavelength_r(self) -> torch.Tensor:
        """Red channel wavelength (longest)."""
        return self._constrain_wavelength(self._wavelength_r_raw)

    @property
    def wavelength_g(self) -> torch.Tensor:
        """Green channel wavelength (reference)."""
        return self._constrain_wavelength(self._wavelength_g_raw)

    @property
    def wavelength_b(self) -> torch.Tensor:
        """Blue channel wavelength (shortest)."""
        return self._constrain_wavelength(self._wavelength_b_raw)

    @property
    def wavelengths(self) -> torch.Tensor:
        """All wavelengths as (3,) tensor [R, G, B]."""
        return torch.stack([self.wavelength_r, self.wavelength_g, self.wavelength_b])

    def compute_path_difference(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute optical path difference from focal plane.

        Args:
            depth: Depth tensor of any shape

        Returns:
            Path difference tensor (same shape as input)
        """
        return torch.abs(depth - self.focal_depth)

    def depth_to_phase_rgb(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute per-channel phase based on wavelength.

        φ_c = (2π / λ_c) × path_length

        Different wavelengths produce different phase accumulation rates,
        leading to chromatic aberration and wavelength-dependent interference.

        Args:
            depth: Depth tensor of shape (...) - any shape

        Returns:
            Phase tensor of shape (..., 3) with [R, G, B] phases
        """
        path_diff = self.compute_path_difference(depth)

        # Phase for each channel: φ = (2π / λ) × d
        phase_r = (2 * torch.pi / self.wavelength_r) * path_diff
        phase_g = (2 * torch.pi / self.wavelength_g) * path_diff
        phase_b = (2 * torch.pi / self.wavelength_b) * path_diff

        # Stack to get (..., 3) tensor
        phases = torch.stack([phase_r, phase_g, phase_b], dim=-1)

        return phases

    def depth_to_phase_single(self, depth: torch.Tensor, channel: str = 'g') -> torch.Tensor:
        """
        Compute phase for a single channel.

        Args:
            depth: Depth tensor
            channel: 'r', 'g', or 'b'

        Returns:
            Phase tensor (same shape as depth)
        """
        path_diff = self.compute_path_difference(depth)

        if channel.lower() == 'r':
            wavelength = self.wavelength_r
        elif channel.lower() == 'g':
            wavelength = self.wavelength_g
        elif channel.lower() == 'b':
            wavelength = self.wavelength_b
        else:
            raise ValueError(f"Unknown channel: {channel}. Use 'r', 'g', or 'b'.")

        return (2 * torch.pi / wavelength) * path_diff

    def get_chromatic_dispersion(self) -> torch.Tensor:
        """
        Compute chromatic dispersion (wavelength spread).

        Returns:
            Dispersion value (λ_r - λ_b) / λ_g
        """
        return (self.wavelength_r - self.wavelength_b) / self.wavelength_g

    def forward(
        self,
        depth: torch.Tensor,
        return_all: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        Compute per-channel phases.

        Args:
            depth: Input depth tensor
            return_all: If True, return dict with additional info

        Returns:
            RGB phases tensor, or dict with phases and wavelength info
        """
        phases = self.depth_to_phase_rgb(depth)

        if not return_all:
            return phases

        return {
            'phases': phases,
            'phase_r': phases[..., 0],
            'phase_g': phases[..., 1],
            'phase_b': phases[..., 2],
            'wavelength_r': self.wavelength_r,
            'wavelength_g': self.wavelength_g,
            'wavelength_b': self.wavelength_b,
            'wavelengths': self.wavelengths,
            'chromatic_dispersion': self.get_chromatic_dispersion(),
        }

    def extra_repr(self) -> str:
        return (
            f"λ_r={self.wavelength_r.item():.4f}, "
            f"λ_g={self.wavelength_g.item():.4f}, "
            f"λ_b={self.wavelength_b.item():.4f}, "
            f"learnable={self.learnable}"
        )


class FresnelDiffraction(nn.Module):
    """
    Physics-based Fresnel diffraction pattern computation.

    At depth discontinuities (edges), Fresnel diffraction creates characteristic
    intensity fringes. This class computes these patterns using the actual
    Fresnel integrals:

        C(w) = ∫₀ʷ cos(πt²/2) dt  (Fresnel cosine integral)
        S(w) = ∫₀ʷ sin(πt²/2) dt  (Fresnel sine integral)

    The diffraction intensity is:
        I(w) = |C(w) + 0.5|² + |S(w) + 0.5|²

    where w = x × √(2 / (λ × z)) is the Fresnel parameter.

    Key insight: Intensity peaks occur at specific distances from edges,
    not uniformly! This determines optimal Gaussian placement.

    Applications:
    - compute_edge_density(): Where to place more Gaussians
    - get_fringe_positions(): Exact fringe peak locations
    """

    def __init__(
        self,
        wavelength: float = 0.05,
        num_fringe_samples: int = 16,
        lut_size: int = 1000,
        lut_max_w: float = 5.0,
    ):
        """
        Initialize FresnelDiffraction.

        Args:
            wavelength: Effective wavelength for diffraction computation
            num_fringe_samples: Number of fringe positions to compute
            lut_size: Size of Fresnel integral lookup table
            lut_max_w: Maximum w value for LUT
        """
        super().__init__()

        self.wavelength = wavelength
        self.num_samples = num_fringe_samples
        self.lut_size = lut_size
        self.lut_max_w = lut_max_w

        # Build Fresnel integral lookup table
        self._build_fresnel_lut()

    def _build_fresnel_lut(self):
        """
        Build lookup table for Fresnel integrals C(w) and S(w).

        Uses numerical integration (cumulative sum approximation).
        """
        # Sample points for LUT
        w = torch.linspace(0, self.lut_max_w, self.lut_size)

        # Numerical integration using trapezoidal rule
        dt = w[1] - w[0]
        t = torch.linspace(0, self.lut_max_w, self.lut_size)

        # C(w) = ∫₀ʷ cos(πt²/2) dt
        cos_integrand = torch.cos(torch.pi * t ** 2 / 2)
        C = torch.cumsum(cos_integrand, dim=0) * dt

        # S(w) = ∫₀ʷ sin(πt²/2) dt
        sin_integrand = torch.sin(torch.pi * t ** 2 / 2)
        S = torch.cumsum(sin_integrand, dim=0) * dt

        # Register as buffers (not parameters)
        # Use _lut suffix to avoid conflict with fresnel_C/fresnel_S methods
        self.register_buffer('_w_lut', w)
        self.register_buffer('_C_lut', C)
        self.register_buffer('_S_lut', S)

    def _interp_lut(self, w: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation from lookup table.

        Args:
            w: Fresnel parameter values to look up
            lut: Lookup table (fresnel_C or fresnel_S)

        Returns:
            Interpolated values
        """
        # Clamp to valid range
        w_clamped = torch.clamp(w, 0, self._w_lut[-1])

        # Compute interpolation indices
        idx_float = w_clamped / self._w_lut[-1] * (len(lut) - 1)
        idx_low = idx_float.long()
        idx_high = (idx_low + 1).clamp(max=len(lut) - 1)
        frac = idx_float - idx_low.float()

        # Linear interpolation
        return lut[idx_low] * (1 - frac) + lut[idx_high] * frac

    def fresnel_C(self, w: torch.Tensor) -> torch.Tensor:
        """
        Compute Fresnel cosine integral C(w) = ∫₀ʷ cos(πt²/2) dt.

        Args:
            w: Fresnel parameter values

        Returns:
            C(w) values
        """
        return self._interp_lut(w, self._C_lut)

    def fresnel_S(self, w: torch.Tensor) -> torch.Tensor:
        """
        Compute Fresnel sine integral S(w) = ∫₀ʷ sin(πt²/2) dt.

        Args:
            w: Fresnel parameter values

        Returns:
            S(w) values
        """
        return self._interp_lut(w, self._S_lut)

    def fresnel_intensity(self, w: torch.Tensor) -> torch.Tensor:
        """
        Compute Fresnel diffraction intensity at parameter w.

        I(w) = |C(w) + 0.5|² + |S(w) + 0.5|²

        The +0.5 accounts for the geometric shadow contribution
        (unobstructed wave has C=S=0.5 at infinity).

        Args:
            w: Fresnel parameter values

        Returns:
            Intensity values (normalized around 0.5 for geometric edge)
        """
        C = self._interp_lut(w, self._C_lut)
        S = self._interp_lut(w, self._S_lut)

        # Intensity formula with geometric shadow contribution
        I = (C + 0.5) ** 2 + (S + 0.5) ** 2

        return I

    def compute_fresnel_parameter(
        self,
        distance_from_edge: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Fresnel parameter w from distance and depth.

        w = distance × √(2 / (λ × z))

        Args:
            distance_from_edge: Distance from edge (can be signed)
            depth: Depth values

        Returns:
            Fresnel parameter w
        """
        λ = self.wavelength
        z = depth.clamp(min=0.1)  # Avoid division issues

        w = torch.abs(distance_from_edge) * torch.sqrt(2 / (λ * z))

        return w

    def compute_edge_density(
        self,
        depth: torch.Tensor,           # (B, 1, H, W)
        edge_mask: torch.Tensor,       # (B, 1, H, W) binary edges
        distance_from_edge: torch.Tensor,  # (B, 1, H, W) signed distance
    ) -> torch.Tensor:
        """
        Compute Gaussian density based on Fresnel diffraction pattern.

        Returns a density map showing where to place more Gaussians.
        Peaks occur at diffraction fringe locations, not uniformly at edges.

        Args:
            depth: Depth tensor
            edge_mask: Binary mask of edge pixels
            distance_from_edge: Signed distance from nearest edge

        Returns:
            Density map (B, 1, H, W) with higher values at fringe peaks
        """
        # Compute Fresnel parameter
        w = self.compute_fresnel_parameter(distance_from_edge, depth)

        # Get diffraction intensity pattern
        intensity = self.fresnel_intensity(w)

        # Density is modulated by intensity and masked to edge regions
        density = intensity * edge_mask

        return density

    def get_fringe_positions(self, depth_at_edge: float) -> torch.Tensor:
        """
        Return the x-positions of diffraction fringes relative to edge.

        Fringe maxima occur approximately at:
            w_n ≈ √(2n + 0.5) for n = 0, 1, 2, ...

        These are the OPTIMAL positions for Gaussian placement!

        Args:
            depth_at_edge: Depth value at the edge

        Returns:
            Tensor of fringe positions (distances from edge)
        """
        λ = self.wavelength

        # Fringe maxima approximation
        n = torch.arange(self.num_samples, device=self._w_lut.device)
        w_n = torch.sqrt(2 * n + 0.5)

        # Convert back to physical distance
        # w = x × √(2 / (λ × z))  →  x = w × √(λ × z / 2)
        x_n = w_n * torch.sqrt(torch.tensor(λ * depth_at_edge / 2))

        return x_n

    def forward(
        self,
        depth: torch.Tensor,
        edge_mask: torch.Tensor,
        distance_from_edge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffraction-based density map.

        Args:
            depth: Depth tensor (B, 1, H, W)
            edge_mask: Edge mask (B, 1, H, W)
            distance_from_edge: Distance field (B, 1, H, W)

        Returns:
            Density map for Gaussian placement
        """
        return self.compute_edge_density(depth, edge_mask, distance_from_edge)

    def extra_repr(self) -> str:
        return (
            f"wavelength={self.wavelength}, "
            f"num_samples={self.num_samples}, "
            f"lut_size={self.lut_size}"
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
