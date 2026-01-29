#!/usr/bin/env python3
"""
Neural Cellular Automata Gaussian Decoder (NCA-GS)

Experiment 014: Instead of feed-forward Gaussian prediction, treat Gaussians
as cells in a cellular automaton that update their parameters based on
learned local rules over N timesteps.

Hypothesis: Feed-forward decoders produce "blobby" outputs because they
predict all parameters in one shot without iterative refinement. NCA dynamics
can achieve emergent global structure from local interactions.

References:
- Mordvintsev et al. (2020). "Growing Neural Cellular Automata"
- distill.pub/2020/growing-ca/
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

# Import utilities from existing decoder module
from models.gaussian_decoder_models import (
    fibonacci_spiral_positions,
    rotation_6d_to_quaternion,
    MLP,
)


class NCAGaussianDecoder(nn.Module):
    """
    Gaussian decoder using Neural Cellular Automata dynamics.

    Instead of one-shot prediction, Gaussians are initialized from features
    then iteratively refined through learned local update rules. Each Gaussian
    "cell" perceives its k-nearest neighbors and updates its state accordingly.

    Architecture:
    1. init_state: Features -> Initial Gaussian state (B, N, state_dim)
    2. NCA loop (n_steps iterations):
       - Find k-nearest neighbors based on position
       - Perceive: self_state + neighbor_states -> hidden
       - Update: hidden -> delta
       - Apply: state += step_size * delta (with stochastic masking)
    3. parse_state: Final state -> Gaussian parameters dict

    Key design choices:
    - 377 Fibonacci spiral points (from Exp 013, O(N²) tractable)
    - k=6 neighbors (standard NCA)
    - Stochastic update masking (p=0.5) for training stability
    - Learnable step size
    - 6D rotation representation (continuous, differentiable)
    """

    def __init__(
        self,
        feature_dim: int = 384,
        n_points: int = 377,      # Fibonacci spiral points
        n_steps: int = 16,         # NCA iterations
        k_neighbors: int = 6,      # Local neighborhood size
        hidden_dim: int = 128,     # Hidden dimension in NCA networks
        update_prob: float = 0.5,  # Stochastic update probability
        # State dimensions: pos(3) + scale(3) + rot_6d(6) + color(3) + opacity(1) = 16
        state_dim: int = 16,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_points = n_points
        self.n_steps = n_steps
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.update_prob = update_prob
        self.state_dim = state_dim

        # Pre-compute spiral coordinates
        spiral_x, spiral_y = fibonacci_spiral_positions(n_points, torch.device('cpu'))
        self.register_buffer('spiral_x', spiral_x)
        self.register_buffer('spiral_y', spiral_y)

        # Learned depth offset (from FibonacciPatchDecoder)
        self.depth_offset = nn.Parameter(torch.tensor(-2.0))

        # =========================================================================
        # INITIAL STATE NETWORK
        # =========================================================================
        # Predicts initial Gaussian state from DINOv2 features
        # Input: sampled features (384D) -> Output: initial state (state_dim)
        self.init_state_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim)
        )

        # =========================================================================
        # NCA PERCEPTION NETWORK
        # =========================================================================
        # Perceives self + k neighbors, outputs hidden representation
        # Input: self_state (state_dim) + neighbor_states (k * state_dim)
        perception_input_dim = state_dim * (k_neighbors + 1)
        self.perception = nn.Sequential(
            nn.Linear(perception_input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # =========================================================================
        # NCA UPDATE RULE NETWORK
        # =========================================================================
        # Computes state delta from perceived neighborhood
        # Input: perceived hidden -> Output: state delta
        self.update_rule = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim)
        )

        # Initialize update rule output to near-zero (residual learning)
        nn.init.zeros_(self.update_rule[-1].weight)
        nn.init.zeros_(self.update_rule[-1].bias)

        # Learnable step size (controls update magnitude)
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        features: torch.Tensor,              # (B, 384, 37, 37) DINOv2 features
        depth: Optional[torch.Tensor] = None,  # (B, 1, H, W) depth map
        image_size: Tuple[int, int] = (518, 518),
        num_gaussians: Optional[int] = None,   # For compatibility, ignored
        elevation: Optional[torch.Tensor] = None,
        azimuth: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,         # Override default step count
        return_trajectory: bool = False,       # Return intermediate states for visualization
    ) -> Dict[str, torch.Tensor]:
        """
        Predict Gaussians using NCA dynamics.

        Args:
            features: DINOv2 feature grid (B, 384, 37, 37)
            depth: Optional depth map (B, 1, H, W)
            image_size: Target image size (for position scaling)
            n_steps: Override number of NCA iterations
            return_trajectory: If True, return list of intermediate states

        Returns:
            Dict with 'positions', 'scales', 'rotations', 'colors', 'opacities'
            If return_trajectory, also includes 'trajectory' list
        """
        n_steps = n_steps if n_steps is not None else self.n_steps
        B, C, H, W = features.shape
        device = features.device
        N = self.n_points

        # =========================================================================
        # SAMPLE FEATURES AT SPIRAL POSITIONS
        # =========================================================================
        spiral_coords = torch.stack([self.spiral_x, self.spiral_y], dim=-1)  # (N, 2)
        spiral_coords = spiral_coords.view(1, 1, N, 2).expand(B, -1, -1, -1)  # (B, 1, N, 2)

        sampled_features = F.grid_sample(
            features, spiral_coords,
            mode='bilinear', padding_mode='border', align_corners=True
        )  # (B, C, 1, N)
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)  # (B, N, C)

        # =========================================================================
        # SAMPLE DEPTH AT SPIRAL POSITIONS (for position initialization)
        # =========================================================================
        if depth is not None:
            depth_sampled = F.grid_sample(
                depth, spiral_coords,
                mode='bilinear', padding_mode='border', align_corners=True
            )  # (B, 1, 1, N)
            depth_sampled = depth_sampled.squeeze(1).squeeze(1)  # (B, N)
        else:
            depth_sampled = torch.zeros(B, N, device=device)

        # =========================================================================
        # INITIALIZE STATE FROM FEATURES
        # =========================================================================
        # Flatten for network
        features_flat = sampled_features.reshape(B * N, C)
        initial_state = self.init_state_net(features_flat)  # (B*N, state_dim)
        state = initial_state.reshape(B, N, self.state_dim)  # (B, N, state_dim)

        # Initialize positions from spiral + depth
        # State layout: [pos(3), scale(3), rot_6d(6), color(3), opacity(1)]
        base_x = self.spiral_x.view(1, N).expand(B, -1)  # (B, N)
        base_y = self.spiral_y.view(1, N).expand(B, -1)
        base_z = self.depth_offset + depth_sampled * (-2)  # (B, N)

        # Override initial position predictions with spiral-based positions
        # Z is LOCKED to depth - NCA can only refine X/Y, not depth (Exp 015 fix)
        state = state.clone()
        state[..., 0] = base_x + state[..., 0].detach() * 0.15  # X: small offset
        state[..., 1] = base_y + state[..., 1].detach() * 0.15  # Y: small offset
        state[..., 2] = base_z  # Z: locked to depth

        # =========================================================================
        # NCA ITERATION LOOP
        # =========================================================================
        trajectory = [state.detach().clone()] if return_trajectory else None

        for step in range(n_steps):
            state = self._nca_step(state)
            if return_trajectory:
                trajectory.append(state.detach().clone())

        # =========================================================================
        # PARSE FINAL STATE INTO GAUSSIAN PARAMETERS
        # =========================================================================
        result = self._parse_state(state)

        if return_trajectory:
            result['trajectory'] = trajectory

        return result

    def _nca_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Single NCA update step.

        Args:
            state: Current state (B, N, state_dim)

        Returns:
            Updated state (B, N, state_dim)
        """
        B, N, D = state.shape

        # =========================================================================
        # FIND K-NEAREST NEIGHBORS (based on position)
        # =========================================================================
        positions = state[..., :3]  # (B, N, 3)
        dists = torch.cdist(positions, positions)  # (B, N, N)

        # Get k+1 nearest (includes self at distance 0), exclude self
        # Using topk with largest=False gives smallest distances
        _, neighbor_idx = dists.topk(self.k_neighbors + 1, dim=-1, largest=False)
        neighbor_idx = neighbor_idx[..., 1:]  # (B, N, k) - exclude self

        # =========================================================================
        # GATHER NEIGHBOR STATES
        # =========================================================================
        neighbor_states = self._gather_neighbors(state, neighbor_idx)  # (B, N, k*D)

        # =========================================================================
        # PERCEPTION: Process self + neighbors
        # =========================================================================
        perception_input = torch.cat([state, neighbor_states], dim=-1)  # (B, N, (k+1)*D)
        perception_flat = perception_input.reshape(B * N, -1)
        perceived = self.perception(perception_flat)  # (B*N, hidden)

        # =========================================================================
        # UPDATE RULE: Compute delta
        # =========================================================================
        delta = self.update_rule(perceived)  # (B*N, D)
        delta = delta.reshape(B, N, D)

        # =========================================================================
        # STOCHASTIC UPDATE MASK (NCA stability trick)
        # =========================================================================
        if self.training:
            # Random mask: each cell has update_prob chance of updating
            mask = (torch.rand(B, N, 1, device=state.device) < self.update_prob).float()
            delta = delta * mask

        # =========================================================================
        # APPLY UPDATE
        # =========================================================================
        new_state = state + self.step_size * delta

        return new_state

    def _gather_neighbors(
        self,
        state: torch.Tensor,  # (B, N, D)
        neighbor_idx: torch.Tensor  # (B, N, k)
    ) -> torch.Tensor:
        """
        Gather neighbor states using indices.

        Args:
            state: Full state tensor (B, N, D)
            neighbor_idx: Indices of neighbors (B, N, k)

        Returns:
            Neighbor states flattened (B, N, k*D)
        """
        B, N, D = state.shape
        k = neighbor_idx.shape[-1]

        # Expand state for gathering: (B, N, k, D)
        # We want to gather from dimension 1 (the N dimension)
        idx_expanded = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, k, D)

        # Expand state to match
        state_expanded = state.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, D)

        # Gather neighbors
        # idx_expanded[b, n, ki, d] tells us which point to get for batch b, point n, neighbor ki
        neighbors = torch.gather(
            state.unsqueeze(1).expand(-1, N, -1, -1),  # (B, N, N, D)
            dim=2,  # gather along the N dimension
            index=idx_expanded
        )  # (B, N, k, D)

        # Flatten neighbor dimension
        return neighbors.reshape(B, N, k * D)

    def _parse_state(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert raw state tensor to Gaussian parameter dict.

        State layout: [pos(3), scale(3), rot_6d(6), color(3), opacity(1)] = 16

        Args:
            state: (B, N, 16)

        Returns:
            Dict with positions, scales, rotations, colors, opacities
        """
        B, N, D = state.shape

        # Extract components
        positions = state[..., 0:3]  # (B, N, 3)
        raw_scale = state[..., 3:6]  # (B, N, 3)
        rot_6d = state[..., 6:12]    # (B, N, 6)
        raw_color = state[..., 12:15]  # (B, N, 3)
        raw_opacity = state[..., 15:16]  # (B, N, 1)

        # Scale: ensure positive via softplus
        raw_scale_clamped = torch.clamp(raw_scale, min=-10, max=20)
        scales = F.softplus(raw_scale_clamped + 1.0) * 0.15
        scales = torch.clamp(scales, min=1e-6, max=2.0)

        # Rotation: 6D to quaternion
        rotations = rotation_6d_to_quaternion(rot_6d)  # (B, N, 4)

        # Color: sigmoid to [0, 1]
        colors = torch.sigmoid(raw_color)  # (B, N, 3)

        # Opacity: sigmoid to [0, 1]
        opacities = torch.sigmoid(raw_opacity).squeeze(-1)  # (B, N)

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = NCAGaussianDecoder(
        n_points=377,
        n_steps=16,
        k_neighbors=6,
    ).to(device)

    print(f"NCA Gaussian Decoder")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Points: {model.n_points}")
    print(f"  NCA steps: {model.n_steps}")
    print(f"  Neighbors: {model.k_neighbors}")

    # Create dummy inputs
    B = 2
    features = torch.randn(B, 384, 37, 37, device=device)
    depth = torch.rand(B, 1, 518, 518, device=device)

    print("\nRunning forward pass...")

    # Forward with trajectory
    output = model(features, depth, return_trajectory=True)

    print(f"\nOutput shapes:")
    print(f"  positions: {output['positions'].shape}")
    print(f"  scales: {output['scales'].shape}")
    print(f"  rotations: {output['rotations'].shape}")
    print(f"  colors: {output['colors'].shape}")
    print(f"  opacities: {output['opacities'].shape}")
    print(f"  trajectory length: {len(output['trajectory'])}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = output['positions'].sum() + output['scales'].sum()
    loss.backward()
    print(f"  step_size grad: {model.step_size.grad}")
    print(f"  init_state_net[0].weight grad norm: {model.init_state_net[0].weight.grad.norm():.6f}")
    print(f"  update_rule[-1].weight grad norm: {model.update_rule[-1].weight.grad.norm():.6f}")

    # Test different step counts
    print("\nTesting step ablation:")
    for n in [1, 4, 8, 16]:
        out = model(features, depth, n_steps=n)
        pos_std = out['positions'].std().item()
        print(f"  n_steps={n:2d}: position std = {pos_std:.4f}")

    print("\nAll tests passed!")
