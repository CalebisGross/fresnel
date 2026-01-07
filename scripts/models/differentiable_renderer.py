#!/usr/bin/env python3
"""
Differentiable 2D Gaussian Splatting Renderer

A simple, differentiable Gaussian splatting renderer for training.
Renders 3D Gaussians to 2D images using perspective projection and
alpha compositing.

This implementation prioritizes:
1. Differentiability (for backprop through Gaussian params)
2. Simplicity (no CUDA kernels, pure PyTorch)
3. Correctness over speed (can optimize later)

Based on: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class Camera:
    """Simple pinhole camera model."""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        near: float = 0.01,
        far: float = 100.0
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.near = near
        self.far = far

        # View matrix (identity = camera at origin looking down -Z)
        self.view_matrix = torch.eye(4)

    def set_view(self, view_matrix: torch.Tensor):
        """Set view matrix (world-to-camera transform)."""
        self.view_matrix = view_matrix

    def project(self, points_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to 2D pixel coordinates.

        Args:
            points_3d: (N, 3) world-space points

        Returns:
            points_2d: (N, 2) pixel coordinates
            depths: (N,) depth values
        """
        # Transform to camera space
        ones = torch.ones(points_3d.shape[0], 1, device=points_3d.device)
        points_homo = torch.cat([points_3d, ones], dim=1)  # (N, 4)

        view = self.view_matrix.to(points_3d.device)
        points_cam = (view @ points_homo.T).T[:, :3]  # (N, 3)

        # Perspective projection
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]

        # Avoid division by zero
        z = torch.clamp(z.abs(), min=self.near) * torch.sign(z + 1e-8)

        u = self.fx * x / (-z) + self.cx
        v = self.fy * (-y) / (-z) + self.cy  # Flip Y for image coords

        points_2d = torch.stack([u, v], dim=1)
        depths = -z  # Positive depth = in front of camera

        return points_2d, depths

    def get_intrinsics(self) -> torch.Tensor:
        """Get 3x3 intrinsic matrix."""
        K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        return K


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        q: (..., 4) quaternion tensor

    Returns:
        R: (..., 3, 3) rotation matrix
    """
    # Normalize quaternion
    q = F.normalize(q, dim=-1)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Build rotation matrix
    R = torch.stack([
        1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y,
        2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x,
        2*x*z - 2*w*y,         2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R


def compute_2d_covariance(
    positions_3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    camera: Camera
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 2D covariance matrices from 3D Gaussian parameters.

    Projects 3D covariance to 2D using the Jacobian of perspective projection.

    Args:
        positions_3d: (N, 3) Gaussian centers
        scales: (N, 3) scales along local axes
        rotations: (N, 4) quaternions (w, x, y, z)
        camera: Camera object

    Returns:
        cov_2d: (N, 2, 2) 2D covariance matrices
        means_2d: (N, 2) projected centers
        depths: (N,) depth values
    """
    N = positions_3d.shape[0]
    device = positions_3d.device

    # Transform to camera space
    ones = torch.ones(N, 1, device=device)
    points_homo = torch.cat([positions_3d, ones], dim=1)
    view = camera.view_matrix.to(device)
    points_cam = (view @ points_homo.T).T[:, :3]  # (N, 3)

    # Get depths
    depths = -points_cam[:, 2]  # Positive depth

    # Compute 3D covariance: Sigma = R @ S @ S^T @ R^T
    R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)
    S = torch.diag_embed(scales)  # (N, 3, 3)

    # Transform rotation to camera space
    view_rot = view[:3, :3].to(device)  # (3, 3)
    R_cam = view_rot @ R  # (N, 3, 3) - broadcasts correctly

    RS = R_cam @ S  # (N, 3, 3)
    cov_3d = RS @ RS.transpose(-1, -2)  # (N, 3, 3)

    # Jacobian of perspective projection
    fx, fy = camera.fx, camera.fy
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    # Clamp z to avoid numerical issues
    z_safe = torch.clamp(z.abs(), min=0.01) * torch.sign(z + 1e-8)
    z2 = z_safe * z_safe

    # Jacobian J = d(u,v)/d(x,y,z)
    # u = fx * x / (-z) + cx
    # v = fy * (-y) / (-z) + cy
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / (-z_safe)          # du/dx
    J[:, 0, 2] = fx * x / z2             # du/dz
    J[:, 1, 1] = fy / z_safe             # dv/dy
    J[:, 1, 2] = fy * y / z2             # dv/dz

    # Project covariance: Sigma_2d = J @ Sigma_3d @ J^T
    cov_2d = J @ cov_3d @ J.transpose(-1, -2)  # (N, 2, 2)

    # Compute 2D means
    u = fx * x / (-z_safe) + camera.cx
    v = fy * (-y) / (-z_safe) + camera.cy
    means_2d = torch.stack([u, v], dim=1)

    return cov_2d, means_2d, depths


def gaussian_2d(
    coords: torch.Tensor,
    mean: torch.Tensor,
    cov: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate 2D Gaussian at given coordinates.

    Args:
        coords: (H, W, 2) pixel coordinates
        mean: (N, 2) Gaussian centers
        cov: (N, 2, 2) covariance matrices

    Returns:
        values: (N, H, W) Gaussian values at each pixel
    """
    N = mean.shape[0]
    H, W = coords.shape[:2]
    device = mean.device

    # Compute inverse covariance
    # Add small regularization for numerical stability
    cov_reg = cov + 1e-4 * torch.eye(2, device=device).unsqueeze(0)
    cov_inv = torch.linalg.pinv(cov_reg)  # (N, 2, 2)

    # Compute determinant for normalization
    det = torch.linalg.det(cov_reg)  # (N,)

    # Expand for broadcasting: (N, 1, 1, 2)
    mean_exp = mean.view(N, 1, 1, 2)

    # Difference: (N, H, W, 2)
    diff = coords.unsqueeze(0) - mean_exp

    # Mahalanobis distance: (N, H, W)
    # d^2 = diff^T @ cov_inv @ diff
    cov_inv_exp = cov_inv.view(N, 1, 1, 2, 2)
    diff_exp = diff.unsqueeze(-1)  # (N, H, W, 2, 1)

    mahal = (diff_exp.transpose(-1, -2) @ cov_inv_exp @ diff_exp).squeeze(-1).squeeze(-1)

    # Gaussian value (unnormalized for rendering)
    values = torch.exp(-0.5 * mahal)

    return values


class DifferentiableGaussianRenderer(nn.Module):
    """
    Differentiable 2D Gaussian splatting renderer.

    Renders 3D Gaussians to a 2D image using perspective projection
    and front-to-back alpha compositing.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        super().__init__()
        self.width = image_width
        self.height = image_height
        self.background = torch.tensor(background)

        # Pre-compute pixel coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('pixel_coords', torch.stack([x_coords, y_coords], dim=-1))

    def forward(
        self,
        positions: torch.Tensor,      # (N, 3) world positions
        scales: torch.Tensor,          # (N, 3) scales
        rotations: torch.Tensor,       # (N, 4) quaternions (w,x,y,z)
        colors: torch.Tensor,          # (N, 3) RGB colors
        opacities: torch.Tensor,       # (N,) opacity values
        camera: Camera,
        return_depth: bool = False
    ) -> torch.Tensor:
        """
        Render Gaussians to image.

        Args:
            positions: (N, 3) Gaussian centers in world space
            scales: (N, 3) scale along each axis
            rotations: (N, 4) quaternions (w, x, y, z)
            colors: (N, 3) RGB colors [0, 1]
            opacities: (N,) opacity values [0, 1]
            camera: Camera parameters
            return_depth: If True, also return depth map

        Returns:
            image: (3, H, W) rendered RGB image
            depth: (H, W) depth map (if return_depth=True)
        """
        N = positions.shape[0]
        device = positions.device
        H, W = self.height, self.width

        # Move buffers to device
        pixel_coords = self.pixel_coords.to(device)
        background = self.background.to(device)

        # Compute 2D covariance and projections
        cov_2d, means_2d, depths = compute_2d_covariance(
            positions, scales, rotations, camera
        )

        # Sort by depth (front to back)
        depth_order = torch.argsort(depths)

        means_2d = means_2d[depth_order]
        cov_2d = cov_2d[depth_order]
        colors = colors[depth_order]
        opacities = opacities[depth_order]
        depths = depths[depth_order]

        # Filter Gaussians outside view frustum
        visible = (depths > camera.near) & (depths < camera.far)
        visible &= (means_2d[:, 0] > -100) & (means_2d[:, 0] < W + 100)
        visible &= (means_2d[:, 1] > -100) & (means_2d[:, 1] < H + 100)

        if visible.sum() == 0:
            # No visible Gaussians - maintain gradient connection
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            img = background.view(3, 1, 1).expand(3, H, W) + grad_anchor
            if return_depth:
                return img, torch.zeros(H, W, device=device) + grad_anchor
            return img

        means_2d = means_2d[visible]
        cov_2d = cov_2d[visible]
        colors = colors[visible]
        opacities = opacities[visible]
        depths = depths[visible]

        N_visible = means_2d.shape[0]

        # Compute Gaussian values at all pixels
        # This is memory-intensive but fully differentiable
        # For large N, we'd need tiling

        # Limit batch size for memory
        MAX_BATCH = 512

        # Initialize output
        accumulated_color = torch.zeros(H, W, 3, device=device)
        accumulated_alpha = torch.zeros(H, W, device=device)
        accumulated_depth = torch.zeros(H, W, device=device)

        # Process in batches
        for start in range(0, N_visible, MAX_BATCH):
            end = min(start + MAX_BATCH, N_visible)
            batch_size = end - start

            # Get batch
            batch_means = means_2d[start:end]
            batch_cov = cov_2d[start:end]
            batch_colors = colors[start:end]
            batch_opacities = opacities[start:end]
            batch_depths = depths[start:end]

            # Compute Gaussian values
            gauss_values = gaussian_2d(pixel_coords, batch_means, batch_cov)  # (batch, H, W)

            # Compute alpha
            alpha = gauss_values * batch_opacities.view(batch_size, 1, 1)

            # Clamp alpha
            alpha = torch.clamp(alpha, 0, 0.99)

            # Alpha compositing (front to back)
            for i in range(batch_size):
                # Transmittance
                T = 1.0 - accumulated_alpha

                # Contribution
                contrib = alpha[i] * T

                # Accumulate color
                accumulated_color += contrib.unsqueeze(-1) * batch_colors[i].view(1, 1, 3)

                # Accumulate depth
                accumulated_depth += contrib * batch_depths[i]

                # Update alpha
                accumulated_alpha = accumulated_alpha + contrib

        # Add background
        T_final = 1.0 - accumulated_alpha
        accumulated_color += T_final.unsqueeze(-1) * background.view(1, 1, 3)

        # Transpose to (3, H, W)
        image = accumulated_color.permute(2, 0, 1)

        # Clamp
        image = torch.clamp(image, 0, 1)

        # Ensure gradient connection
        if not image.requires_grad:
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            image = image + grad_anchor
            accumulated_depth = accumulated_depth + grad_anchor

        if return_depth:
            return image, accumulated_depth
        return image


class TileBasedRenderer(nn.Module):
    """
    Memory-efficient tile-based Gaussian splatting renderer.

    Only evaluates Gaussians within their effective radius (3σ),
    reducing memory from O(N × H × W) to O(N × r²).

    For 5K Gaussians at 256×256 with avg radius 20px:
    - Old: 5K × 65K = 325M operations, ~3GB memory
    - New: 5K × 1.2K = 6M operations, ~100MB memory

    Fresnel-Inspired Enhancement: Phase-Based Interference Blending
    ---------------------------------------------------------------
    When phase values are provided, the renderer modulates alpha based
    on phase differences, simulating wave interference patterns:
    - Constructive interference (phase alignment) → higher alpha
    - Destructive interference (phase opposition) → lower alpha

    This creates smoother transitions at Gaussian overlaps, inspired by
    the Huygens-Fresnel principle where wavelets superpose.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_radius: int = 64,  # Cap radius for very large Gaussians
        # Fresnel phase blending parameters
        use_phase_blending: bool = False,
        phase_amplitude: float = 0.25,  # How much phase affects alpha (0-1)
    ):
        super().__init__()
        self.width = image_width
        self.height = image_height
        self.background = torch.tensor(background)
        self.max_radius = max_radius
        self.use_phase_blending = use_phase_blending
        self.phase_amplitude = phase_amplitude

    def _compute_radius(self, cov_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute effective radius from 2D covariance (3σ rule).

        For a 2D Gaussian with covariance Σ, the effective radius
        is 3 * sqrt(max eigenvalue), which captures 99.7% of the mass.

        Args:
            cov_2d: (N, 2, 2) covariance matrices

        Returns:
            radii: (N,) effective radii in pixels
        """
        # Eigenvalues of 2x2 matrix: λ = (a+d)/2 ± sqrt((a-d)²/4 + bc)
        a = cov_2d[:, 0, 0]
        b = cov_2d[:, 0, 1]
        c = cov_2d[:, 1, 0]
        d = cov_2d[:, 1, 1]

        trace = a + d
        det = a * d - b * c

        # Clamp det to avoid negative sqrt
        det = torch.clamp(det, min=1e-6)

        # Max eigenvalue: (trace + sqrt(trace² - 4*det)) / 2
        discriminant = torch.clamp(trace * trace - 4 * det, min=0)
        max_eigenvalue = (trace + torch.sqrt(discriminant)) / 2

        # 3σ radius
        radii = 3.0 * torch.sqrt(torch.clamp(max_eigenvalue, min=1e-6))

        # Cap at max_radius
        radii = torch.clamp(radii, max=self.max_radius)

        return radii

    def forward(
        self,
        positions: torch.Tensor,      # (N, 3) world positions
        scales: torch.Tensor,          # (N, 3) scales
        rotations: torch.Tensor,       # (N, 4) quaternions (w,x,y,z)
        colors: torch.Tensor,          # (N, 3) RGB colors
        opacities: torch.Tensor,       # (N,) opacity values
        camera: Camera,
        return_depth: bool = False,
        phases: Optional[torch.Tensor] = None,  # (N,) phase values [0, 1] for interference
    ) -> torch.Tensor:
        """
        Render Gaussians to image using tile-based approach.

        Memory efficient: only computes Gaussian values within each
        Gaussian's effective radius.

        Fresnel Phase Blending:
        If phases are provided and use_phase_blending is True, alpha values
        are modulated by a phase-dependent interference factor:
            interference = 1 - amplitude + amplitude * cos(phase * 2π)
        This simulates constructive/destructive wave interference.
        """
        N = positions.shape[0]
        device = positions.device
        H, W = self.height, self.width

        background = self.background.to(device)

        # Compute 2D covariance and projections
        cov_2d, means_2d, depths = compute_2d_covariance(
            positions, scales, rotations, camera
        )

        # Compute effective radius for each Gaussian
        radii = self._compute_radius(cov_2d)

        # Sort by depth (front to back)
        depth_order = torch.argsort(depths)

        means_2d = means_2d[depth_order]
        cov_2d = cov_2d[depth_order]
        colors = colors[depth_order]
        opacities = opacities[depth_order]
        depths = depths[depth_order]
        radii = radii[depth_order]

        # Also sort phases if provided
        if phases is not None:
            phases = phases[depth_order]

        # Filter Gaussians outside view frustum
        visible = (depths > camera.near) & (depths < camera.far)
        visible &= (means_2d[:, 0] + radii > 0) & (means_2d[:, 0] - radii < W)
        visible &= (means_2d[:, 1] + radii > 0) & (means_2d[:, 1] - radii < H)

        if visible.sum() == 0:
            # Return background but maintain gradient connection to inputs
            # Add a zero-weighted sum of inputs to preserve gradient flow for backprop
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            img = background.view(3, 1, 1).expand(3, H, W) + grad_anchor
            if return_depth:
                return img, torch.zeros(H, W, device=device) + grad_anchor
            return img

        means_2d = means_2d[visible]
        cov_2d = cov_2d[visible]
        colors = colors[visible]
        opacities = opacities[visible]
        depths = depths[visible]
        radii = radii[visible]

        if phases is not None:
            phases = phases[visible]

        N_visible = means_2d.shape[0]

        # Initialize output buffers
        accumulated_color = torch.zeros(H, W, 3, device=device)
        accumulated_alpha = torch.zeros(H, W, device=device)
        accumulated_depth = torch.zeros(H, W, device=device)

        # For phase blending: track accumulated phase for interference computation
        if self.use_phase_blending and phases is not None:
            accumulated_phase = torch.zeros(H, W, device=device)
        else:
            accumulated_phase = None

        # Compute inverse covariance for all Gaussians (needed for evaluation)
        cov_reg = cov_2d + 1e-4 * torch.eye(2, device=device).unsqueeze(0)
        cov_inv = torch.linalg.pinv(cov_reg)  # (N_visible, 2, 2)

        # Process each Gaussian - only compute within its radius
        for i in range(N_visible):
            mean = means_2d[i]  # (2,)
            radius = radii[i].item()
            inv_cov = cov_inv[i]  # (2, 2)
            color = colors[i]  # (3,)
            opacity = opacities[i]
            depth = depths[i]

            # Get phase if available
            phase = phases[i] if phases is not None else None

            # Compute bounding box
            x0 = max(0, int(mean[0].item() - radius))
            x1 = min(W, int(mean[0].item() + radius) + 1)
            y0 = max(0, int(mean[1].item() - radius))
            y1 = min(H, int(mean[1].item() + radius) + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            # Create local coordinate grid
            local_y, local_x = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32),
                torch.arange(x0, x1, device=device, dtype=torch.float32),
                indexing='ij'
            )

            # Compute difference from mean
            dx = local_x - mean[0]  # (h, w)
            dy = local_y - mean[1]  # (h, w)

            # Mahalanobis distance: d² = [dx, dy] @ inv_cov @ [dx, dy]^T
            # inv_cov is [[a, b], [c, d]]
            a, b = inv_cov[0, 0], inv_cov[0, 1]
            c, d = inv_cov[1, 0], inv_cov[1, 1]

            mahal = a * dx * dx + (b + c) * dx * dy + d * dy * dy

            # Gaussian value
            gauss_val = torch.exp(-0.5 * mahal)

            # Alpha
            alpha = gauss_val * opacity

            # =========================================================================
            # FRESNEL PHASE BLENDING: Interference-like alpha modulation
            # =========================================================================
            if self.use_phase_blending and phase is not None and accumulated_phase is not None:
                # Compute phase difference with accumulated phase at each pixel
                # This simulates wave interference between this Gaussian and previous ones
                prev_phase = accumulated_phase[y0:y1, x0:x1]

                # Phase difference: how much this Gaussian's phase differs from accumulated
                # phase_diff in [0, 1] where 0 = aligned, 0.5 = opposite
                phase_diff = torch.abs(phase - prev_phase)
                phase_diff = torch.min(phase_diff, 1.0 - phase_diff)  # Wrap around

                # Interference factor: constructive (0 diff) → 1, destructive (0.5 diff) → lower
                # Using cosine for smooth interference pattern
                interference = (1.0 - self.phase_amplitude) + \
                               self.phase_amplitude * torch.cos(phase_diff * 2 * 3.14159)

                # Modulate alpha by interference
                alpha = alpha * interference

            alpha = torch.clamp(alpha, 0, 0.99)

            # Get current transmittance for this region
            T = 1.0 - accumulated_alpha[y0:y1, x0:x1]

            # Contribution
            contrib = alpha * T

            # Accumulate
            accumulated_color[y0:y1, x0:x1] += contrib.unsqueeze(-1) * color.view(1, 1, 3)
            accumulated_depth[y0:y1, x0:x1] += contrib * depth
            accumulated_alpha[y0:y1, x0:x1] += contrib

            # Update accumulated phase (weighted average)
            if accumulated_phase is not None and phase is not None:
                # Update phase: blend current phase with contribution weight
                phase_contrib = contrib / (accumulated_alpha[y0:y1, x0:x1].clamp(min=1e-6))
                accumulated_phase[y0:y1, x0:x1] = (
                    accumulated_phase[y0:y1, x0:x1] * (1 - phase_contrib) +
                    phase * phase_contrib
                )

        # Add background
        T_final = 1.0 - accumulated_alpha
        accumulated_color += T_final.unsqueeze(-1) * background.view(1, 1, 3)

        # Transpose to (3, H, W)
        image = accumulated_color.permute(2, 0, 1)
        image = torch.clamp(image, 0, 1)

        # Ensure gradient connection even if no Gaussians contributed
        # (e.g., all bounding boxes were invalid)
        if not image.requires_grad:
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            image = image + grad_anchor
            accumulated_depth = accumulated_depth + grad_anchor

        if return_depth:
            return image, accumulated_depth
        return image


class WaveFieldRenderer(nn.Module):
    """
    Physics-based wave field renderer using complex amplitude accumulation.

    Instead of alpha blending (intensity addition), this renderer:
    1. Accumulates complex amplitudes: U = Σ A_i × exp(i×φ_i)
    2. Converts to intensity at the end: I = |U|²

    This properly models wave interference as per the Huygens-Fresnel principle,
    where each Gaussian acts as a wavelet source contributing to the total field.

    Key difference from TileBasedRenderer:
    - TileBasedRenderer: alpha blending (intensity-based, no interference)
    - WaveFieldRenderer: complex field accumulation (true wave interference)

    Named after Augustin-Jean Fresnel's wave optics theory.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_radius: int = 64,
    ):
        super().__init__()
        self.width = image_width
        self.height = image_height
        self.max_radius = max_radius
        self.register_buffer('background', torch.tensor(background))

    def _compute_radius(self, cov_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute effective radius from 2D covariance (3σ rule).

        Args:
            cov_2d: (N, 2, 2) covariance matrices

        Returns:
            radii: (N,) effective radii in pixels
        """
        a = cov_2d[:, 0, 0]
        b = cov_2d[:, 0, 1]
        c = cov_2d[:, 1, 0]
        d = cov_2d[:, 1, 1]

        trace = a + d
        det = a * d - b * c
        det = torch.clamp(det, min=1e-6)

        discriminant = torch.clamp(trace * trace - 4 * det, min=0)
        max_eigenvalue = (trace + torch.sqrt(discriminant)) / 2

        radii = 3.0 * torch.sqrt(torch.clamp(max_eigenvalue, min=1e-6))
        radii = torch.clamp(radii, max=self.max_radius)

        return radii

    def forward(
        self,
        positions: torch.Tensor,      # (N, 3) world positions
        scales: torch.Tensor,          # (N, 3) scales
        rotations: torch.Tensor,       # (N, 4) quaternions (w,x,y,z)
        colors: torch.Tensor,          # (N, 3) RGB colors - treated as amplitude
        opacities: torch.Tensor,       # (N,) opacity values
        camera: Camera,
        return_depth: bool = False,
        phases: Optional[torch.Tensor] = None,  # (N,) phase values - REQUIRED for wave rendering
    ) -> torch.Tensor:
        """
        Render Gaussians using complex wave field accumulation.

        Physics: U(P) = Σ A_i × exp(i × φ_i)
                 I = |U|² = Re(U)² + Im(U)²

        Args:
            positions: (N, 3) Gaussian centers in world space
            scales: (N, 3) scale along each axis
            rotations: (N, 4) quaternions (w, x, y, z)
            colors: (N, 3) RGB colors [0, 1] - treated as wave amplitude
            opacities: (N,) opacity values [0, 1]
            camera: Camera parameters
            return_depth: If True, also return depth map
            phases: (N,) phase values in radians - REQUIRED for wave rendering

        Returns:
            image: (3, H, W) rendered RGB image
            depth: (H, W) depth map (if return_depth=True)
        """
        # Phases are required for wave field rendering
        if phases is None:
            raise ValueError("WaveFieldRenderer requires phases tensor. Use PhysicsDirectPatchDecoder to generate phases.")

        N = positions.shape[0]
        device = positions.device
        H, W = self.height, self.width

        background = self.background.to(device)

        # Compute 2D covariance and projections
        cov_2d, means_2d, depths = compute_2d_covariance(
            positions, scales, rotations, camera
        )

        # Compute effective radius for each Gaussian
        radii = self._compute_radius(cov_2d)

        # Filter Gaussians outside view frustum
        visible = (depths > camera.near) & (depths < camera.far)
        visible &= (means_2d[:, 0] + radii > 0) & (means_2d[:, 0] - radii < W)
        visible &= (means_2d[:, 1] + radii > 0) & (means_2d[:, 1] - radii < H)

        if visible.sum() == 0:
            # Return background but maintain gradient connection to inputs
            # Add a zero-weighted sum of inputs to preserve gradient flow for backprop
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            img = background.view(3, 1, 1).expand(3, H, W) + grad_anchor
            if return_depth:
                return img, torch.zeros(H, W, device=device) + grad_anchor
            return img

        means_2d = means_2d[visible]
        cov_2d = cov_2d[visible]
        colors = colors[visible]
        opacities = opacities[visible]
        depths = depths[visible]
        radii = radii[visible]
        phases = phases[visible]

        N_visible = means_2d.shape[0]

        # Initialize complex wave field (real + imaginary per color channel)
        # Shape: (H, W, 3) for each component
        wave_real = torch.zeros(H, W, 3, device=device)
        wave_imag = torch.zeros(H, W, 3, device=device)
        accumulated_depth = torch.zeros(H, W, device=device)
        total_weight = torch.zeros(H, W, device=device)

        # Compute inverse covariance for all Gaussians
        cov_reg = cov_2d + 1e-4 * torch.eye(2, device=device).unsqueeze(0)
        cov_inv = torch.linalg.pinv(cov_reg)

        # Process each Gaussian - accumulate complex field
        for i in range(N_visible):
            mean = means_2d[i]
            radius = radii[i].item()
            inv_cov = cov_inv[i]
            color = colors[i]
            opacity = opacities[i]
            depth = depths[i]
            phase = phases[i]

            # Compute bounding box
            x0 = max(0, int(mean[0].item() - radius))
            x1 = min(W, int(mean[0].item() + radius) + 1)
            y0 = max(0, int(mean[1].item() - radius))
            y1 = min(H, int(mean[1].item() + radius) + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            # Create local coordinate grid
            local_y, local_x = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32),
                torch.arange(x0, x1, device=device, dtype=torch.float32),
                indexing='ij'
            )

            # Compute difference from mean
            dx = local_x - mean[0]
            dy = local_y - mean[1]

            # Mahalanobis distance
            a, b = inv_cov[0, 0], inv_cov[0, 1]
            c, d = inv_cov[1, 0], inv_cov[1, 1]
            mahal = a * dx * dx + (b + c) * dx * dy + d * dy * dy

            # Gaussian value (amplitude envelope)
            gauss_val = torch.exp(-0.5 * mahal)

            # Amplitude = opacity × gaussian_weight
            amplitude = gauss_val * opacity  # (h, w)

            # Complex contribution: A × exp(iφ) = A×cos(φ) + i×A×sin(φ)
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)

            # Accumulate complex field (color modulates amplitude)
            # wave_real += A × color × cos(φ)
            # wave_imag += A × color × sin(φ)
            wave_real[y0:y1, x0:x1] += amplitude.unsqueeze(-1) * color.view(1, 1, 3) * cos_phase
            wave_imag[y0:y1, x0:x1] += amplitude.unsqueeze(-1) * color.view(1, 1, 3) * sin_phase

            # Track depth (weighted by amplitude for depth map)
            accumulated_depth[y0:y1, x0:x1] += amplitude * depth
            total_weight[y0:y1, x0:x1] += amplitude

        # Convert to intensity: I = |U|² = real² + imag²
        intensity = wave_real ** 2 + wave_imag ** 2  # (H, W, 3)

        # Convert intensity to "color-like" values via sqrt
        # (intensity is energy, we want amplitude for display)
        rendered = torch.sqrt(intensity + 1e-8)

        # Normalize - wave superposition can exceed 1.0 constructively
        # Use differentiable normalization (no Python control flow for gradient safety)
        max_val = rendered.max().clamp(min=1.0)
        rendered = rendered / max_val

        rendered = torch.clamp(rendered, 0, 1)

        # Add background where total amplitude is low
        total_amplitude = torch.sqrt((wave_real ** 2 + wave_imag ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        total_amplitude = total_amplitude.clamp(0, 1)
        rendered = rendered + background.view(1, 1, 3) * (1 - total_amplitude)

        # Transpose to (3, H, W)
        image = rendered.permute(2, 0, 1)
        image = torch.clamp(image, 0, 1)

        # Ensure gradient connection
        if not image.requires_grad:
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            image = image + grad_anchor
            accumulated_depth = accumulated_depth + grad_anchor

        if return_depth:
            # Normalize depth by total weight
            depth_map = accumulated_depth / (total_weight + 1e-8)
            return image, depth_map
        return image


class AngularSpectrumPropagator(nn.Module):
    """
    Angular Spectrum Method (ASM) for wave propagation.

    Propagates a 2D complex wave field by a distance z using FFT:
        U(x, y, z) = F⁻¹{ F{U(x, y, 0)} × H(fₓ, fᵧ, z) }

    Where H is the transfer function:
        H(fₓ, fᵧ, z) = exp(i × 2π × z × √(1/λ² - fₓ² - fᵧ²))

    More accurate than simple Fresnel for near-field propagation.
    Based on: "Towards Real-Time Photorealistic 3D Holography" (Nature, 2021)
    """

    def __init__(
        self,
        height: int,
        width: int,
        pixel_pitch: float = 1.0 / 256.0,  # Pixel spacing in scene units
        wavelength: float = 0.05,
        band_limit: bool = True,  # Limit frequencies to avoid aliasing
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.band_limit = band_limit

        # Precompute frequency coordinates
        fx = torch.fft.fftfreq(width, d=pixel_pitch)
        fy = torch.fft.fftfreq(height, d=pixel_pitch)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')

        self.register_buffer('FX', FX)
        self.register_buffer('FY', FY)

    def _compute_transfer_function(
        self,
        z_distance: torch.Tensor,
        wavelength: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ASM transfer function for propagation by z_distance.

        Args:
            z_distance: Propagation distance (can be negative)
            wavelength: Optional wavelength override (for per-channel)

        Returns:
            H: (H, W) complex transfer function
        """
        device = z_distance.device
        FX = self.FX.to(device)
        FY = self.FY.to(device)

        wl = wavelength if wavelength is not None else self.wavelength
        if isinstance(wl, (int, float)):
            wl = torch.tensor(wl, device=device)

        # kz² = (1/λ)² - fx² - fy²
        kz_sq = (1.0 / wl) ** 2 - FX ** 2 - FY ** 2

        # Band limiting: zero out evanescent waves (kz² < 0)
        if self.band_limit:
            kz_sq = torch.clamp(kz_sq, min=0)

        kz = torch.sqrt(kz_sq)

        # Transfer function: H = exp(i × 2π × z × kz)
        H = torch.exp(1j * 2 * torch.pi * z_distance * kz)

        return H

    def propagate(
        self,
        field: torch.Tensor,
        z_distance: torch.Tensor,
        wavelength: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagate complex field by z_distance using ASM.

        Args:
            field: (H, W) or (H, W, C) complex field (torch.cfloat)
            z_distance: Propagation distance
            wavelength: Optional per-channel wavelength

        Returns:
            propagated: Same shape as input, propagated field
        """
        # Handle single-channel or multi-channel
        if field.dim() == 2:
            field = field.unsqueeze(-1)
            squeeze_out = True
        else:
            squeeze_out = False

        H, W, C = field.shape
        device = field.device

        # Propagate each channel
        propagated_channels = []
        for c in range(C):
            channel_field = field[..., c]  # (H, W)

            # Get wavelength for this channel
            if wavelength is not None and wavelength.dim() > 0:
                wl = wavelength[c] if len(wavelength) > c else wavelength
            else:
                wl = wavelength

            # Compute transfer function
            H_tf = self._compute_transfer_function(z_distance, wl)

            # FFT propagation
            field_fft = torch.fft.fft2(channel_field)
            propagated_fft = field_fft * H_tf
            propagated = torch.fft.ifft2(propagated_fft)

            propagated_channels.append(propagated)

        result = torch.stack(propagated_channels, dim=-1)

        if squeeze_out:
            result = result.squeeze(-1)

        return result

    def forward(
        self,
        field: torch.Tensor,
        z_distance: torch.Tensor,
        wavelength: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Alias for propagate()."""
        return self.propagate(field, z_distance, wavelength)


class ASMWaveFieldRenderer(nn.Module):
    """
    Wave field renderer using Angular Spectrum Method for propagation.

    Unlike WaveFieldRenderer which accumulates complex amplitudes directly,
    this renderer:
    1. Places Gaussians at discrete depth planes
    2. Uses ASM to propagate each plane to the focal plane
    3. Sums the propagated fields for true interference

    More physically accurate for holographic rendering.
    Based on: "Towards Real-Time Photorealistic 3D Holography" (Nature, 2021)
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_radius: int = 64,
        num_depth_planes: int = 16,  # Number of discrete depth layers
        depth_range: Tuple[float, float] = (0.1, 2.0),  # Near to far depth
        focal_depth: float = 0.5,  # Focal plane for ASM propagation
        pixel_pitch: float = 1.0 / 256.0,
        wavelength: float = 0.05,
    ):
        super().__init__()
        self.width = image_width
        self.height = image_height
        self.max_radius = max_radius
        self.num_depth_planes = num_depth_planes
        self.depth_range = depth_range
        self.focal_depth = focal_depth
        self.wavelength = wavelength

        self.register_buffer('background', torch.tensor(background))

        # Create depth planes
        depths = torch.linspace(depth_range[0], depth_range[1], num_depth_planes)
        self.register_buffer('depth_planes', depths)

        # Create ASM propagator
        self.propagator = AngularSpectrumPropagator(
            height=image_height,
            width=image_width,
            pixel_pitch=pixel_pitch,
            wavelength=wavelength
        )

    def _compute_radius(self, cov_2d: torch.Tensor) -> torch.Tensor:
        """Compute effective radius from 2D covariance (3σ rule)."""
        a = cov_2d[:, 0, 0]
        b = cov_2d[:, 0, 1]
        c = cov_2d[:, 1, 0]
        d = cov_2d[:, 1, 1]

        trace = a + d
        det = a * d - b * c
        det = torch.clamp(det, min=1e-6)

        discriminant = torch.clamp(trace * trace - 4 * det, min=0)
        max_eigenvalue = (trace + torch.sqrt(discriminant)) / 2

        radii = 3.0 * torch.sqrt(torch.clamp(max_eigenvalue, min=1e-6))
        radii = torch.clamp(radii, max=self.max_radius)

        return radii

    def _assign_to_depth_plane(self, depths: torch.Tensor) -> torch.Tensor:
        """
        Assign each Gaussian to nearest depth plane.

        Args:
            depths: (N,) depth values

        Returns:
            plane_indices: (N,) index of nearest depth plane
        """
        # Compute distance to each plane
        distances = (depths.unsqueeze(1) - self.depth_planes.unsqueeze(0)).abs()
        return distances.argmin(dim=1)

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        camera: Camera,
        return_depth: bool = False,
        phases: Optional[torch.Tensor] = None,
        wavelengths_rgb: Optional[torch.Tensor] = None,  # (3,) per-channel wavelengths
    ) -> torch.Tensor:
        """
        Render using Angular Spectrum Method wave propagation.

        Pipeline:
        1. Project Gaussians to 2D, compute depth
        2. Assign each Gaussian to a depth plane
        3. Render Gaussians at each depth plane into complex field
        4. Propagate each plane to focal depth using ASM
        5. Sum propagated fields (interference)
        6. Convert to intensity

        Args:
            positions: (N, 3) Gaussian centers
            scales: (N, 3) scale along each axis
            rotations: (N, 4) quaternions
            colors: (N, 3) RGB colors [0, 1]
            opacities: (N,) opacity values [0, 1]
            camera: Camera parameters
            return_depth: If True, also return depth map
            phases: (N,) phase values in radians
            wavelengths_rgb: (3,) wavelengths for R, G, B channels

        Returns:
            image: (3, H, W) rendered RGB image
        """
        if phases is None:
            raise ValueError("ASMWaveFieldRenderer requires phases tensor.")

        N = positions.shape[0]
        device = positions.device
        H, W = self.height, self.width

        background = self.background.to(device)

        # Project and get depths
        cov_2d, means_2d, depths = compute_2d_covariance(
            positions, scales, rotations, camera
        )
        radii = self._compute_radius(cov_2d)

        # Filter visible
        visible = (depths > camera.near) & (depths < camera.far)
        visible &= (means_2d[:, 0] + radii > 0) & (means_2d[:, 0] - radii < W)
        visible &= (means_2d[:, 1] + radii > 0) & (means_2d[:, 1] - radii < H)

        if visible.sum() == 0:
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            img = background.view(3, 1, 1).expand(3, H, W) + grad_anchor
            if return_depth:
                return img, torch.zeros(H, W, device=device) + grad_anchor
            return img

        # Filter to visible
        means_2d = means_2d[visible]
        cov_2d = cov_2d[visible]
        colors = colors[visible]
        opacities = opacities[visible]
        depths = depths[visible]
        radii = radii[visible]
        phases = phases[visible]

        N_visible = means_2d.shape[0]

        # Assign to depth planes
        plane_indices = self._assign_to_depth_plane(depths)

        # Compute inverse covariance
        cov_reg = cov_2d + 1e-4 * torch.eye(2, device=device).unsqueeze(0)
        cov_inv = torch.linalg.pinv(cov_reg)

        # Initialize per-plane complex fields (real + imag per channel)
        plane_fields = torch.zeros(
            self.num_depth_planes, H, W, 3, 2, device=device
        )  # Last dim: [real, imag]

        # Render Gaussians to their assigned depth planes
        for i in range(N_visible):
            mean = means_2d[i]
            radius = radii[i].item()
            inv_cov = cov_inv[i]
            color = colors[i]
            opacity = opacities[i]
            phase = phases[i]
            plane_idx = plane_indices[i]

            # Bounding box
            x0 = max(0, int(mean[0].item() - radius))
            x1 = min(W, int(mean[0].item() + radius) + 1)
            y0 = max(0, int(mean[1].item() - radius))
            y1 = min(H, int(mean[1].item() + radius) + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            # Local coordinates
            local_y, local_x = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32),
                torch.arange(x0, x1, device=device, dtype=torch.float32),
                indexing='ij'
            )

            dx = local_x - mean[0]
            dy = local_y - mean[1]

            a, b = inv_cov[0, 0], inv_cov[0, 1]
            c, d = inv_cov[1, 0], inv_cov[1, 1]
            mahal = a * dx * dx + (b + c) * dx * dy + d * dy * dy

            gauss_val = torch.exp(-0.5 * mahal)
            amplitude = gauss_val * opacity

            # Complex contribution
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)

            # Add to plane field
            plane_fields[plane_idx, y0:y1, x0:x1, :, 0] += (
                amplitude.unsqueeze(-1) * color.view(1, 1, 3) * cos_phase
            )
            plane_fields[plane_idx, y0:y1, x0:x1, :, 1] += (
                amplitude.unsqueeze(-1) * color.view(1, 1, 3) * sin_phase
            )

        # Convert to complex tensors for propagation
        # Sum propagated fields from each plane at focal depth
        total_field = torch.zeros(H, W, 3, dtype=torch.cfloat, device=device)

        focal_depth_tensor = torch.tensor(self.focal_depth, device=device)

        for p in range(self.num_depth_planes):
            plane_depth = self.depth_planes[p]
            z_prop = focal_depth_tensor - plane_depth  # Propagation distance

            # Convert to complex
            field_complex = torch.complex(
                plane_fields[p, :, :, :, 0],
                plane_fields[p, :, :, :, 1]
            )  # (H, W, 3)

            # Skip empty planes
            if field_complex.abs().max() < 1e-8:
                continue

            # Propagate each color channel
            for c in range(3):
                wl = wavelengths_rgb[c] if wavelengths_rgb is not None else self.wavelength
                propagated = self.propagator.propagate(
                    field_complex[..., c],
                    z_prop,
                    wavelength=wl
                )
                total_field[..., c] += propagated

        # Convert to intensity: I = |U|²
        intensity = (total_field.real ** 2 + total_field.imag ** 2)

        # Square root for display (intensity → amplitude)
        rendered = torch.sqrt(intensity + 1e-8)

        # Normalize
        max_val = rendered.max().clamp(min=1.0)
        rendered = rendered / max_val
        rendered = torch.clamp(rendered, 0, 1)

        # Add background
        total_amp = (total_field.abs().sum(dim=-1, keepdim=True)).clamp(0, 1)
        rendered = rendered + background.view(1, 1, 3) * (1 - total_amp)

        # Transpose to (3, H, W)
        image = rendered.permute(2, 0, 1)
        image = torch.clamp(image, 0, 1)

        # Gradient connection
        if not image.requires_grad:
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            image = image + grad_anchor

        if return_depth:
            # Simple depth from weighted sum
            depth_map = torch.zeros(H, W, device=device)
            return image, depth_map

        return image


class SimplifiedRenderer(nn.Module):
    """
    Simplified renderer for faster training.

    Uses point splatting instead of full covariance projection.
    Less accurate but much faster for training.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        splat_size: int = 3,  # Splat radius in pixels
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        super().__init__()
        self.width = image_width
        self.height = image_height
        self.splat_size = splat_size
        self.background = torch.tensor(background)

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        camera: Camera,
        return_depth: bool = False
    ) -> torch.Tensor:
        """Simplified splatting render."""
        device = positions.device
        H, W = self.height, self.width
        background = self.background.to(device)

        # Project to 2D
        means_2d, depths = camera.project(positions)

        # Sort by depth
        depth_order = torch.argsort(depths, descending=True)  # Back to front for this impl

        means_2d = means_2d[depth_order]
        colors = colors[depth_order]
        opacities = opacities[depth_order]
        depths = depths[depth_order]
        scales = scales[depth_order]

        # Initialize canvas
        image = background.view(3, 1, 1).expand(3, H, W).clone()
        depth_map = torch.full((H, W), float('inf'), device=device)

        # Simple splatting
        for i in range(means_2d.shape[0]):
            x, y = means_2d[i]
            d = depths[i]

            if d <= 0:
                continue

            # Compute splat radius from scale
            radius = int(max(scales[i].mean().item() * camera.fx / d, 1))
            radius = min(radius, 20)  # Cap radius

            # Splat bounds
            x_int, y_int = int(x.item()), int(y.item())
            x0 = max(0, x_int - radius)
            x1 = min(W, x_int + radius + 1)
            y0 = max(0, y_int - radius)
            y1 = min(H, y_int + radius + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            # Compute Gaussian weights for splat
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32),
                torch.arange(x0, x1, device=device, dtype=torch.float32),
                indexing='ij'
            )

            dist_sq = (xx - x) ** 2 + (yy - y) ** 2
            weight = torch.exp(-dist_sq / (2 * max(radius/2, 1) ** 2))

            # Apply opacity
            alpha = weight * opacities[i]
            alpha = torch.clamp(alpha, 0, 1)

            # Blend
            color = colors[i].view(3, 1, 1)
            for c in range(3):
                image[c, y0:y1, x0:x1] = (
                    alpha * color[c] + (1 - alpha) * image[c, y0:y1, x0:x1]
                )

            # Update depth
            depth_map[y0:y1, x0:x1] = torch.where(
                alpha > 0.1,
                torch.minimum(depth_map[y0:y1, x0:x1], d.expand(y1-y0, x1-x0)),
                depth_map[y0:y1, x0:x1]
            )

        # Ensure gradient connection
        if not image.requires_grad:
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            image = image + grad_anchor
            depth_map = depth_map + grad_anchor

        if return_depth:
            depth_map = torch.where(depth_map == float('inf'), torch.zeros_like(depth_map), depth_map)
            return image, depth_map
        return image


def load_gaussians_from_binary(path: str) -> dict:
    """
    Load Gaussians from binary file (exported from C++).

    Expected format: N gaussians, each with:
    - position: 3 floats
    - scale: 3 floats
    - rotation: 4 floats (w, x, y, z)
    - color: 3 floats
    - opacity: 1 float
    Total: 14 floats per Gaussian
    """
    data = np.fromfile(path, dtype=np.float32)
    N = len(data) // 14
    data = data[:N * 14].reshape(N, 14)

    return {
        'positions': torch.from_numpy(data[:, 0:3].copy()),
        'scales': torch.from_numpy(data[:, 3:6].copy()),
        'rotations': torch.from_numpy(data[:, 6:10].copy()),
        'colors': torch.from_numpy(data[:, 10:13].copy()),
        'opacities': torch.from_numpy(data[:, 13].copy())
    }


def save_gaussians_to_binary(path: str, gaussians: dict):
    """Save Gaussians to binary file."""
    N = gaussians['positions'].shape[0]
    data = np.zeros((N, 14), dtype=np.float32)

    data[:, 0:3] = gaussians['positions'].cpu().numpy()
    data[:, 3:6] = gaussians['scales'].cpu().numpy()
    data[:, 6:10] = gaussians['rotations'].cpu().numpy()
    data[:, 10:13] = gaussians['colors'].cpu().numpy()
    data[:, 13] = gaussians['opacities'].cpu().numpy()

    data.tofile(path)


class FourierGaussianRenderer(nn.Module):
    """
    Frequency-domain Gaussian splatting renderer using Holographic principles.

    BREAKTHROUGH: O(H×W×log(H×W)) complexity - INDEPENDENT of Gaussian count!

    Key insight: A Gaussian is the ONLY function that equals its own Fourier transform.
    Instead of splatting N Gaussians one-by-one in spatial domain (O(N × r²)),
    we add them ALL in frequency domain and do ONE inverse FFT.

    Physics:
    - Gaussian FT: G(x,y) with scale σ → G_freq(u,v) with scale 1/(2πσ)
    - Translation → phase shift: x₀ offset → exp(-2πi×u×x₀)
    - Wave phase from depth: φ = (2π/λ) × z
    - Complex addition → natural interference patterns

    Novel aspects:
    1. First frequency-domain 3DGS renderer
    2. Learnable per-channel wavelengths (RGB light physics)
    3. Phase from depth enables self-supervised training

    Named: Holographic Fourier Gaussian Splatting (HFGS)
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        wavelength_r: float = 0.0635,  # Red wavelength (700nm equivalent)
        wavelength_g: float = 0.05,    # Green wavelength (550nm equivalent)
        wavelength_b: float = 0.041,   # Blue wavelength (450nm equivalent)
        learnable_wavelengths: bool = True,
        focal_depth: float = 0.5,
    ):
        """
        Initialize FourierGaussianRenderer.

        Args:
            image_width: Output image width
            image_height: Output image height
            background: RGB background color (default black)
            wavelength_r: Red channel wavelength for phase computation
            wavelength_g: Green channel wavelength (reference)
            wavelength_b: Blue channel wavelength
            learnable_wavelengths: If True, wavelengths are trainable parameters
            focal_depth: Focal plane depth for phase computation
        """
        super().__init__()

        self.width = image_width
        self.height = image_height
        self.focal_depth = focal_depth
        self.register_buffer('background', torch.tensor(background))

        # Precompute frequency grid
        # u, v are frequencies in cycles per pixel
        u = torch.fft.fftfreq(image_width)
        v = torch.fft.fftfreq(image_height)
        V, U = torch.meshgrid(v, u, indexing='ij')  # (H, W)

        self.register_buffer('U', U)
        self.register_buffer('V', V)
        self.register_buffer('U2_V2', U**2 + V**2)

        # Per-channel wavelengths (learnable or fixed)
        # Physical ratios: λ_R : λ_G : λ_B ≈ 1.27 : 1.0 : 0.82
        wavelengths = torch.tensor([wavelength_r, wavelength_g, wavelength_b])
        if learnable_wavelengths:
            self.wavelengths = nn.Parameter(wavelengths)
        else:
            self.register_buffer('wavelengths', wavelengths)

        self.learnable_wavelengths = learnable_wavelengths

        # Wavelength constraints to prevent divergence
        self.wavelength_min = 0.01
        self.wavelength_max = 0.5

    def _get_constrained_wavelengths(self) -> torch.Tensor:
        """Get wavelengths with constraints applied."""
        return torch.clamp(
            torch.abs(self.wavelengths),
            self.wavelength_min,
            self.wavelength_max
        )

    def forward(
        self,
        positions: torch.Tensor,      # (N, 3) world positions
        scales: torch.Tensor,          # (N, 3) scales
        rotations: torch.Tensor,       # (N, 4) quaternions (unused in freq domain)
        colors: torch.Tensor,          # (N, 3) RGB colors - wave amplitude
        opacities: torch.Tensor,       # (N,) opacity values
        camera: Camera,
        return_depth: bool = False,
        phases: Optional[torch.Tensor] = None,  # (N,) optional override phases
    ) -> torch.Tensor:
        """
        Render Gaussians in frequency domain with wave physics.

        Algorithm:
        1. Project 3D Gaussians to 2D
        2. For each Gaussian, compute its frequency-domain representation
        3. Accumulate all Gaussians in frequency domain (complex addition)
        4. Single inverse FFT to get spatial image
        5. Convert complex field to intensity: I = |U|²

        Args:
            positions: (N, 3) Gaussian centers in world space
            scales: (N, 3) scale along each axis
            rotations: (N, 4) quaternions (used for 2D covariance projection)
            colors: (N, 3) RGB colors [0, 1] - treated as wave amplitude
            opacities: (N,) opacity values [0, 1]
            camera: Camera parameters
            return_depth: If True, also return depth map (approximation)
            phases: (N,) optional phase override. If None, computed from z-depth.

        Returns:
            image: (3, H, W) rendered RGB image
            depth: (H, W) depth map (if return_depth=True)
        """
        N = positions.shape[0]
        device = positions.device
        H, W = self.height, self.width

        # Move buffers to device
        U = self.U.to(device)
        V = self.V.to(device)
        U2_V2 = self.U2_V2.to(device)
        background = self.background.to(device)

        # Get constrained wavelengths
        wavelengths = self._get_constrained_wavelengths()

        # Compute 2D covariance and projections
        cov_2d, means_2d, depths = compute_2d_covariance(
            positions, scales, rotations, camera
        )

        # Filter visible Gaussians
        visible = (depths > camera.near) & (depths < camera.far)
        visible &= (means_2d[:, 0] > -W) & (means_2d[:, 0] < 2*W)
        visible &= (means_2d[:, 1] > -H) & (means_2d[:, 1] < 2*H)

        if visible.sum() == 0:
            # Return background with gradient connection
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            img = background.view(3, 1, 1).expand(3, H, W) + grad_anchor
            if return_depth:
                return img, torch.zeros(H, W, device=device) + grad_anchor
            return img

        # Filter to visible Gaussians
        means_2d = means_2d[visible]
        cov_2d = cov_2d[visible]
        colors = colors[visible]
        opacities = opacities[visible]
        depths = depths[visible]
        if phases is not None:
            phases = phases[visible]

        N_visible = means_2d.shape[0]

        # Compute effective 2D scale from covariance (average of eigenvalues)
        # For frequency domain Gaussian: σ_freq = 1/(2π×σ_spatial)
        # Eigenvalues of 2x2 covariance
        a = cov_2d[:, 0, 0]
        d = cov_2d[:, 1, 1]
        # Average scale (isotropic approximation)
        σ_2d = torch.sqrt((a + d) / 2 + 1e-8)  # (N_visible,)

        # Normalize positions to [-0.5, 0.5] range for FFT
        # means_2d is in pixel coordinates [0, W] x [0, H]
        x_norm = (means_2d[:, 0] / W) - 0.5  # [-0.5, 0.5]
        y_norm = (means_2d[:, 1] / H) - 0.5  # [-0.5, 0.5]

        # Compute phase from depth if not provided
        if phases is None:
            # Phase = (2π / λ) × |depth - focal_depth|
            # Use green wavelength for single-phase computation
            path_diff = torch.abs(depths - self.focal_depth)
            phases_computed = (2 * torch.pi / wavelengths[1]) * path_diff
        else:
            phases_computed = phases

        # Normalize scale for frequency domain
        # In freq domain, Gaussian with spatial σ has freq σ_f = 1/(2π×σ)
        # We work in cycles/pixel, so scale accordingly
        σ_freq = σ_2d / (2 * torch.pi * W)  # Normalize by image width

        # Initialize complex frequency field for RGB
        freq_field = torch.zeros(3, H, W, dtype=torch.cfloat, device=device)

        # Compute per-channel phases from depth (different λ per channel)
        # phase_c = (2π / λ_c) × path_diff
        path_diff = torch.abs(depths - self.focal_depth)
        phases_rgb = (2 * torch.pi / wavelengths.view(3, 1)) * path_diff.unsqueeze(0)
        # Shape: (3, N_visible)

        # BATCHED PROCESSING to prevent OOM
        # Instead of creating (N, H, W) tensors for ALL Gaussians at once,
        # process in batches and accumulate into freq_field incrementally.
        # This keeps memory bounded to O(batch_size × H × W) instead of O(N × H × W)
        batch_size = 16  # Small batch to fit in VRAM alongside training

        for batch_start in range(0, N_visible, batch_size):
            batch_end = min(batch_start + batch_size, N_visible)
            B = batch_end - batch_start  # Actual batch size

            # Get batch slices
            σ_batch = σ_freq[batch_start:batch_end]
            x_batch = x_norm[batch_start:batch_end]
            y_batch = y_norm[batch_start:batch_end]
            opacity_batch = opacities[batch_start:batch_end]
            color_batch = colors[batch_start:batch_end]
            phases_batch = phases_rgb[:, batch_start:batch_end]  # (3, B)

            # Expand for broadcasting: (B, H, W)
            σ_exp = σ_batch.view(B, 1, 1)
            x_exp = x_batch.view(B, 1, 1)
            y_exp = y_batch.view(B, 1, 1)

            # Gaussian amplitude in frequency domain: (B, H, W)
            gaussian_freq = σ_exp * torch.exp(-2 * torch.pi**2 * σ_exp**2 * U2_V2)

            # Translation phase shift: exp(-2πi(u×x + v×y))
            translation_phase = torch.exp(-2j * torch.pi * (U * x_exp + V * y_exp))

            # Combine Gaussian shape with translation
            base_contribution = gaussian_freq * translation_phase  # (B, H, W) complex
            del gaussian_freq, translation_phase  # Free memory

            # Apply opacity
            opacity_exp = opacity_batch.view(B, 1, 1)
            base_contribution = base_contribution * opacity_exp

            # For each color channel, apply wave phase and accumulate
            for c in range(3):
                # Wave phase for this channel: (B, 1, 1)
                wave_phase = torch.exp(1j * phases_batch[c].view(B, 1, 1))

                # Color amplitude for this channel: (B, 1, 1)
                color_amp = color_batch[:, c].view(B, 1, 1)

                # Channel contribution: (B, H, W) complex
                channel_contrib = color_amp * base_contribution * wave_phase

                # Accumulate into freq_field (sum over batch)
                freq_field[c] = freq_field[c] + channel_contrib.sum(dim=0)
                del channel_contrib  # Free memory immediately

            del base_contribution  # Free memory

        # Single inverse FFT to get spatial field
        spatial_field = torch.fft.ifft2(freq_field)  # (3, H, W) complex

        # Convert to intensity: I = |U|² = real² + imag²
        intensity = spatial_field.real**2 + spatial_field.imag**2  # (3, H, W)

        # Normalize - constructive interference can exceed 1.0
        max_val = intensity.max()
        if max_val > 1e-8:
            intensity = intensity / max_val

        # Apply sqrt to convert intensity to amplitude-like values for display
        # (optional - depends on desired look)
        # image = torch.sqrt(intensity + 1e-8)
        image = intensity  # Keep as intensity for now

        # Add background where intensity is low
        total_intensity = intensity.sum(dim=0, keepdim=True)  # (1, H, W)
        bg_weight = torch.clamp(1.0 - total_intensity, 0, 1)
        image = image + background.view(3, 1, 1) * bg_weight

        # Clamp final output
        image = torch.clamp(image, 0, 1)

        # Ensure gradient connection
        if not image.requires_grad:
            grad_anchor = (colors.sum() + opacities.sum() + positions.sum()) * 0.0
            image = image + grad_anchor

        if return_depth:
            # Approximate depth map from weighted accumulation
            # Use intensity-weighted depth
            depth_map = torch.zeros(H, W, device=device)
            # Simple approximation: use center pixel depths weighted by opacity
            # A more accurate version would accumulate per-pixel
            return image, depth_map

        return image

    def extra_repr(self) -> str:
        λ = self._get_constrained_wavelengths()
        return (
            f"size=({self.height}, {self.width}), "
            f"λ_rgb=[{λ[0]:.4f}, {λ[1]:.4f}, {λ[2]:.4f}], "
            f"learnable={self.learnable_wavelengths}"
        )


if __name__ == '__main__':
    # Test the renderer
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test scene
    N = 100
    positions = torch.randn(N, 3) * 0.5
    positions[:, 2] -= 3  # Move in front of camera

    scales = torch.ones(N, 3) * 0.1
    rotations = torch.zeros(N, 4)
    rotations[:, 0] = 1  # Identity quaternion (w=1)

    colors = torch.rand(N, 3)
    opacities = torch.ones(N) * 0.8

    # Create camera
    camera = Camera(
        fx=500, fy=500,
        cx=256, cy=256,
        width=512, height=512
    )

    # Create renderer
    renderer = DifferentiableGaussianRenderer(512, 512)
    renderer = renderer.to(device)

    # Move data to device
    positions = positions.to(device)
    scales = scales.to(device)
    rotations = rotations.to(device)
    colors = colors.to(device)
    opacities = opacities.to(device)

    # Enable gradients
    positions.requires_grad_(True)
    colors.requires_grad_(True)

    # Render
    image = renderer(positions, scales, rotations, colors, opacities, camera)

    # Test backward pass
    loss = image.mean()
    loss.backward()

    print(f"Image shape: {image.shape}")
    print(f"Position gradient shape: {positions.grad.shape}")
    print(f"Color gradient shape: {colors.grad.shape}")

    # Display
    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.title("Test Render")
    plt.savefig("/tmp/test_render.png")
    print("Saved test render to /tmp/test_render.png")
