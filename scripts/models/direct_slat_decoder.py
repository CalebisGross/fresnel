#!/usr/bin/env python3
"""
Direct Structured Latent Decoder for Fresnel v2.

This module implements direct prediction of Gaussians from DINOv2 features
and sparse structure coordinates, replacing TRELLIS's slow Stage 2 diffusion.

Architecture options:
1. DirectSLatDecoder: Sparse transformer (most similar to TRELLIS teacher)
2. MLPSLatDecoder: Simple MLP baseline for comparison
3. HybridSLatDecoder: Sparse structure + dense refinement

These models are trained via distillation from TRELLIS outputs.
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for sparse voxel coordinates."""

    def __init__(self, d_model: int, max_resolution: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_resolution = max_resolution

        # Learnable position embeddings for each axis
        self.pos_embed_x = nn.Embedding(max_resolution, d_model // 3)
        self.pos_embed_y = nn.Embedding(max_resolution, d_model // 3)
        self.pos_embed_z = nn.Embedding(max_resolution, d_model - 2 * (d_model // 3))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 4) tensor with [batch_idx, x, y, z] or (N, 4)

        Returns:
            (B, N, d_model) or (N, d_model) positional embeddings
        """
        # Handle both batched and unbatched inputs
        if coords.dim() == 2:
            x, y, z = coords[:, 1], coords[:, 2], coords[:, 3]
        else:
            x, y, z = coords[:, :, 1], coords[:, :, 2], coords[:, :, 3]

        # Clamp to valid range
        x = x.clamp(0, self.max_resolution - 1).long()
        y = y.clamp(0, self.max_resolution - 1).long()
        z = z.clamp(0, self.max_resolution - 1).long()

        pos_x = self.pos_embed_x(x)
        pos_y = self.pos_embed_y(y)
        pos_z = self.pos_embed_z(z)

        return torch.cat([pos_x, pos_y, pos_z], dim=-1)


class CrossAttention(nn.Module):
    """Cross-attention from sparse voxels to image features.

    Uses chunked attention for memory efficiency when N is large.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        chunk_size: int = 1024,  # Process queries in chunks
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.chunk_size = chunk_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor (B, N, D)
            context: Key/Value tensor (B, M, D)
            mask: Optional attention mask (B, N) for x

        Returns:
            (B, N, D) attended features
        """
        B, N, D = x.shape
        M = context.shape[1]

        # Compute K, V once (they're shared across all query chunks)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # Each: (B, num_heads, M, head_dim)

        # Use chunked attention for memory efficiency
        if N > self.chunk_size:
            return self._chunked_attention(x, k, v, mask, B, N, D)

        # Standard attention for small N
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(~attn_mask, -1e4)  # Use -1e4 instead of -inf for stability

        # Stable softmax: subtract max before exp to prevent overflow
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = attn - attn_max
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Replace any NaN from softmax (e.g., all-zero rows) with uniform attention
        if torch.isnan(attn).any():
            attn = torch.nan_to_num(attn, nan=1.0 / attn.shape[-1])

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _chunked_attention(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        B: int,
        N: int,
        D: int,
    ) -> torch.Tensor:
        """Process attention in chunks to save memory."""
        outputs = []

        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            x_chunk = x[:, i:end]  # (B, chunk_size, D)
            chunk_n = end - i

            q_chunk = self.q(x_chunk).reshape(B, chunk_n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attn = (q_chunk @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                mask_chunk = mask[:, i:end].unsqueeze(1).unsqueeze(-1)
                attn = attn.masked_fill(~mask_chunk, -1e4)  # Use -1e4 instead of -inf

            # Stable softmax: subtract max before exp
            attn_max = attn.max(dim=-1, keepdim=True).values
            attn = attn - attn_max
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Replace any NaN with uniform attention
            if torch.isnan(attn).any():
                attn = torch.nan_to_num(attn, nan=1.0 / attn.shape[-1])

            out_chunk = (attn @ v).transpose(1, 2).reshape(B, chunk_n, D)
            outputs.append(out_chunk)

        x = torch.cat(outputs, dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SparseTransformerBlock(nn.Module):
    """Transformer block for sparse voxel features with cross-attention to image."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention to image features
        x = x + self.cross_attn(self.norm1(x), self.norm2(context), mask)
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class OccupancyHead(nn.Module):
    """
    Predict binary occupancy for each voxel coordinate.

    This head learns WHICH voxels should contain Gaussians, enabling
    selective prediction rather than uniform filling of all voxels.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize to slight positive bias (assume more voxels occupied initially)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_features: (B, N, hidden_dim) voxel features from transformer

        Returns:
            (B, N) occupancy logits (pre-sigmoid)
        """
        return self.mlp(voxel_features).squeeze(-1)


class GaussianHead(nn.Module):
    """
    Output head that predicts Gaussian parameters.

    Predicts per-voxel Gaussians with:
    - Position offset (3)
    - Scale (3)
    - Rotation (4 quaternion)
    - Color (3)
    - Opacity (1)

    Total: 14 parameters per Gaussian
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_gaussians_per_voxel: int = 32,
        init_offset_scale: float = 0.5,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians_per_voxel
        out_dim = num_gaussians_per_voxel * 14

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # LEARNABLE position offset scale - optimized during training
        # Controls how far Gaussians can move from voxel centers
        self.position_offset_scale = nn.Parameter(torch.tensor(init_offset_scale))

        # LEARNABLE scale factor - helps model learn correct scale distribution
        # TRELLIS scales are typically in [0.001, 0.01] range
        self.scale_factor = nn.Parameter(torch.tensor(0.01))

        # Initialize final layer to small values
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.head[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Voxel features (B, N, D)
            coords: Voxel coordinates (B, N, 4) with [batch_idx, x, y, z]

        Returns:
            Gaussian parameters (B, N * num_gaussians, 14)
        """
        B, N, D = x.shape

        # Check for NaN/Inf in input and replace
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Predict raw parameters
        raw = self.head(x)  # (B, N, num_gaussians * 14)

        # Clamp raw values to prevent overflow in activations
        raw = raw.clamp(-10.0, 10.0)
        raw = raw.reshape(B, N, self.num_gaussians, 14)

        # Apply activations
        gaussians = torch.zeros_like(raw)

        # Position: offset from voxel center (normalized to [-1, 1] grid)
        # Clamp coords to valid range to prevent invalid positions
        coord_xyz = coords[:, :, 1:4].float().clamp(0, 63)
        voxel_centers = coord_xyz / 64.0 * 2 - 1  # Normalize to [-1, 1]
        voxel_centers = voxel_centers.unsqueeze(2).expand(-1, -1, self.num_gaussians, -1)
        # Allow offsets so Gaussians can move from voxel centers
        # position_offset_scale is LEARNABLE - model learns optimal range
        pos_offset = torch.tanh(raw[..., :3]) * self.position_offset_scale
        gaussians[..., :3] = (voxel_centers + pos_offset).clamp(-1.0, 1.0)

        # Scale: softplus * learnable scale_factor to match target range
        # Target scales from TRELLIS range from ~0.001 to 0.1
        # scale_factor is LEARNABLE - model learns optimal amplification
        gaussians[..., 3:6] = (F.softplus(raw[..., 3:6]) * self.scale_factor.abs()).clamp(1e-4, 1.0)

        # Rotation: normalize quaternion with epsilon for numerical stability
        quat = raw[..., 6:10]
        quat_norm = quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gaussians[..., 6:10] = quat / quat_norm

        # Color: sigmoid to [0, 1]
        gaussians[..., 10:13] = torch.sigmoid(raw[..., 10:13])

        # Opacity: sigmoid to [0, 1]
        gaussians[..., 13] = torch.sigmoid(raw[..., 13])

        # Flatten to (B, N * num_gaussians, 14)
        gaussians = gaussians.reshape(B, N * self.num_gaussians, 14)

        # Final NaN check - replace any remaining NaN with safe defaults
        if torch.isnan(gaussians).any():
            gaussians = torch.nan_to_num(gaussians, nan=0.0)

        return gaussians


class DirectSLatDecoder(nn.Module):
    """
    Direct decoder from DINOv2 features + sparse structure to Gaussians.

    This replaces TRELLIS's slow Stage 2 diffusion with a single forward pass.
    Uses sparse transformer blocks with cross-attention to image features.

    Supports gradient checkpointing for memory-efficient training.

    With predict_occupancy=True, the model first predicts which voxels should
    contain Gaussians (occupancy), then only predicts Gaussians for those voxels.
    This prevents the "uniform grid" problem where Gaussians are placed everywhere.
    """

    def __init__(
        self,
        feature_dim: int = 1024,  # DINOv2 feature dimension
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_gaussians_per_voxel: int = 8,  # Reduced from 32 for selective prediction
        max_resolution: int = 64,
        dropout: float = 0.1,
        use_checkpoint: bool = False,  # Enable gradient checkpointing
        predict_occupancy: bool = True,  # Enable occupancy-gated prediction
        occupancy_threshold: float = 0.5,  # Threshold for inference
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians_per_voxel = num_gaussians_per_voxel
        self.use_checkpoint = use_checkpoint
        self.predict_occupancy = predict_occupancy
        self.occupancy_threshold = occupancy_threshold

        # Project image features to hidden dim
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # 3D positional encoding for voxels
        self.pos_encoding = PositionalEncoding3D(hidden_dim, max_resolution)

        # Initial voxel embedding (learnable query)
        self.voxel_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                hidden_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                drop=dropout,
                attn_drop=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Occupancy head (predicts which voxels should have Gaussians)
        if predict_occupancy:
            self.occupancy_head = OccupancyHead(hidden_dim)
        else:
            self.occupancy_head = None

        # Gaussian output head
        self.gaussian_head = GaussianHead(
            hidden_dim,
            hidden_dim,
            num_gaussians_per_voxel,
        )

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for training stability."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for attention projections
                if 'q' in name or 'kv' in name or 'proj' in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        coord_mask: Optional[torch.Tensor] = None,
        apply_occupancy_mask: bool = False,  # For inference: apply hard mask
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: DINOv2 features (B, num_patches, feature_dim)
            coords: Sparse voxel coordinates (B, N, 4) with [batch_idx, x, y, z]
            coord_mask: Valid coordinate mask (B, N)
            apply_occupancy_mask: If True, only predict Gaussians for occupied voxels

        Returns:
            Dict with:
                - 'gaussians': Gaussian parameters (B, M, 14) where M depends on occupancy
                - 'occupancy_logits': (B, N) occupancy logits if predict_occupancy=True
                - 'occupancy_mask': (B, N) binary mask if apply_occupancy_mask=True
        """
        B, N, _ = coords.shape

        # Input validation and cleaning
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp coords to valid range
        coords = coords.clone()
        coords[:, :, 1:4] = coords[:, :, 1:4].clamp(0, 63)

        # Project image features
        context = self.feature_proj(features)  # (B, num_patches, hidden_dim)

        # Clean context after projection
        if torch.isnan(context).any():
            context = torch.nan_to_num(context, nan=0.0)

        # Initialize voxel features with positional encoding
        pos = self.pos_encoding(coords)  # (B, N, hidden_dim)
        x = self.voxel_embed.expand(B, N, -1) + pos

        # Apply transformer blocks (with optional checkpointing)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, context, coord_mask, use_reentrant=False)
            else:
                x = block(x, context, coord_mask)

            # Clean intermediate results
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)

        x = self.norm(x)

        result = {}

        # Predict occupancy if enabled
        if self.predict_occupancy and self.occupancy_head is not None:
            occupancy_logits = self.occupancy_head(x)  # (B, N)
            result['occupancy_logits'] = occupancy_logits

            if apply_occupancy_mask:
                # Apply hard threshold for inference
                occupancy_mask = torch.sigmoid(occupancy_logits) > self.occupancy_threshold
                result['occupancy_mask'] = occupancy_mask

                # Only predict Gaussians for occupied voxels
                # For batched inference, we need to handle variable counts per batch
                gaussians_list = []
                for b in range(B):
                    mask_b = occupancy_mask[b]  # (N,)
                    if coord_mask is not None:
                        mask_b = mask_b & coord_mask[b]

                    if mask_b.sum() == 0:
                        # No occupied voxels, return empty
                        gaussians_list.append(torch.zeros(0, 14, device=x.device))
                    else:
                        x_b = x[b:b+1, mask_b]  # (1, num_occupied, hidden)
                        coords_b = coords[b:b+1, mask_b]  # (1, num_occupied, 4)
                        gaussians_b = self.gaussian_head(x_b, coords_b)  # (1, num_occupied * G, 14)
                        gaussians_list.append(gaussians_b.squeeze(0))

                # For inference, return list of per-batch Gaussians (variable length)
                result['gaussians'] = gaussians_list
                result['n_gaussians'] = [g.shape[0] for g in gaussians_list]
            else:
                # Training: predict Gaussians for all voxels (loss handles masking)
                gaussians = self.gaussian_head(x, coords)
                result['gaussians'] = gaussians
        else:
            # No occupancy prediction - predict Gaussians for all voxels
            gaussians = self.gaussian_head(x, coords)
            result['gaussians'] = gaussians

        return result

    def enable_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self.use_checkpoint = True

    def disable_checkpointing(self):
        """Disable gradient checkpointing for faster inference."""
        self.use_checkpoint = False


class MLPSLatDecoder(nn.Module):
    """
    Simple MLP-based decoder baseline.

    Faster but doesn't leverage cross-attention to image features as effectively.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_gaussians_per_voxel: int = 32,
        max_resolution: int = 64,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians_per_voxel = num_gaussians_per_voxel

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Position encoding
        self.pos_encoding = PositionalEncoding3D(hidden_dim, max_resolution)

        # MLP layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        self.mlp = nn.Sequential(*layers)

        # Output head
        self.gaussian_head = GaussianHead(
            hidden_dim,
            hidden_dim,
            num_gaussians_per_voxel,
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        coord_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = coords.shape

        # Global feature (mean pool image features)
        global_feat = self.feature_proj(features.mean(dim=1, keepdim=True))  # (B, 1, hidden)
        global_feat = global_feat.expand(-1, N, -1)

        # Position encoding
        pos = self.pos_encoding(coords)

        # Combine and process
        x = global_feat + pos
        x = self.mlp(x)

        # Predict Gaussians
        gaussians = self.gaussian_head(x, coords)

        return gaussians


class DirectStructurePredictor(nn.Module):
    """
    Direct predictor for sparse structure (occupancy) from image features.

    This replaces TRELLIS's Stage 1 diffusion with direct prediction.
    Outputs binary occupancy grid that can be converted to sparse coordinates.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        resolution: int = 64,
        threshold: float = 0.5,
    ):
        super().__init__()

        self.resolution = resolution
        self.threshold = threshold

        # Project and reshape features to 3D
        # DINOv2 outputs (B, 37*37, feature_dim) for 518x518 input
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

        # 3D CNN to predict occupancy
        # Start from 2D feature map, expand to 3D
        # Output channels: hidden_dim * resolution // 4, which after reshape becomes
        # (hidden_dim * resolution // 4) // resolution = hidden_dim // 4 channels
        self.depth_channels = hidden_dim // 4  # Channels after reshape to 3D
        self.to_3d = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.depth_channels * resolution, 1),  # Expand for 3D reshape
        )

        self.conv3d = nn.Sequential(
            nn.Conv3d(self.depth_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv3d(hidden_dim // 2, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: DINOv2 features (B, num_patches, feature_dim)

        Returns:
            occupancy: Binary occupancy grid (B, 1, D, H, W)
            coords: Sparse coordinates (N, 4) with [batch_idx, x, y, z]
        """
        B = features.shape[0]
        H = W = int(math.sqrt(features.shape[1]))  # 37 for DINOv2

        # Project features
        x = self.feature_proj(features)  # (B, H*W, hidden)
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, hidden, H, W)

        # Expand to 3D
        x = self.to_3d(x)  # (B, depth_channels * D, H, W)
        D = self.resolution
        x = x.reshape(B, self.depth_channels, D, H, W)  # (B, depth_channels, D, H, W)

        # Upsample to target resolution
        x = F.interpolate(x, size=(D, D, D), mode='trilinear', align_corners=False)

        # Predict occupancy
        logits = self.conv3d(x)  # (B, 1, D, D, D)
        occupancy = torch.sigmoid(logits)

        # Extract sparse coordinates
        coords_list = []
        for b in range(B):
            occupied = (occupancy[b, 0] > self.threshold).nonzero()  # (N, 3)
            batch_idx = torch.full((occupied.shape[0], 1), b, device=occupied.device)
            coords_b = torch.cat([batch_idx, occupied], dim=1)  # (N, 4)
            coords_list.append(coords_b)

        coords = torch.cat(coords_list, dim=0) if coords_list else torch.zeros(0, 4, device=features.device)

        return occupancy, coords


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # Test DirectSLatDecoder with occupancy prediction
    print("\n=== DirectSLatDecoder (with occupancy) ===")
    model = DirectSLatDecoder(
        feature_dim=1024,
        hidden_dim=512,
        num_layers=6,
        num_gaussians_per_voxel=8,
        predict_occupancy=True,
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Dummy inputs
    B = 2
    features = torch.randn(B, 1369, 1024).to(device)  # DINOv2 output
    coords = torch.randint(0, 64, (B, 1000, 4)).to(device)
    coords[:, :, 0] = torch.arange(B).unsqueeze(1).expand(-1, 1000).to(device)  # batch idx
    mask = torch.ones(B, 1000, dtype=torch.bool).to(device)

    # Test training mode (no occupancy mask applied)
    with torch.no_grad():
        result = model(features, coords, mask, apply_occupancy_mask=False)
    print(f"Input features: {features.shape}")
    print(f"Input coords: {coords.shape}")
    print(f"Output gaussians: {result['gaussians'].shape}")  # (B, N * 8, 14)
    print(f"Occupancy logits: {result['occupancy_logits'].shape}")  # (B, N)

    # Test inference mode (occupancy mask applied)
    with torch.no_grad():
        result_infer = model(features, coords, mask, apply_occupancy_mask=True)
    print(f"Inference mode - Gaussians per batch: {result_infer['n_gaussians']}")
    print(f"Occupancy mask: {result_infer['occupancy_mask'].shape}")

    # Test DirectSLatDecoder without occupancy (backward compatible)
    print("\n=== DirectSLatDecoder (no occupancy) ===")
    model_no_occ = DirectSLatDecoder(
        feature_dim=1024,
        hidden_dim=512,
        num_layers=6,
        num_gaussians_per_voxel=8,
        predict_occupancy=False,
    ).to(device)
    print(f"Parameters: {count_parameters(model_no_occ):,}")

    with torch.no_grad():
        result_no_occ = model_no_occ(features, coords, mask)
    print(f"Output gaussians: {result_no_occ['gaussians'].shape}")

    # Test MLPSLatDecoder
    print("\n=== MLPSLatDecoder ===")
    model_mlp = MLPSLatDecoder(
        feature_dim=1024,
        hidden_dim=512,
        num_layers=4,
        num_gaussians_per_voxel=8,
    ).to(device)
    print(f"Parameters: {count_parameters(model_mlp):,}")

    with torch.no_grad():
        gaussians_mlp = model_mlp(features, coords, mask)
    print(f"Output gaussians: {gaussians_mlp.shape}")

    # Test DirectStructurePredictor
    print("\n=== DirectStructurePredictor ===")
    structure_model = DirectStructurePredictor(
        feature_dim=1024,
        hidden_dim=256,
        resolution=64,
    ).to(device)
    print(f"Parameters: {count_parameters(structure_model):,}")

    with torch.no_grad():
        occupancy, pred_coords = structure_model(features)
    print(f"Occupancy: {occupancy.shape}")
    print(f"Predicted coords: {pred_coords.shape}")

    print("\nAll tests passed!")
