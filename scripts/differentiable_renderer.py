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
    cov_inv = torch.linalg.inv(cov_reg)  # (N, 2, 2)

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
            # No visible Gaussians
            img = background.view(3, 1, 1).expand(3, H, W)
            if return_depth:
                return img, torch.zeros(H, W, device=device)
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
            img = background.view(3, 1, 1).expand(3, H, W)
            if return_depth:
                return img, torch.zeros(H, W, device=device)
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
        cov_inv = torch.linalg.inv(cov_reg)  # (N_visible, 2, 2)

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

        if return_depth:
            return image, accumulated_depth
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
