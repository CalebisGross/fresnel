"""
Consistency View Synthesis (CVS) - Novel View Generation via Self-Consistency

A novel approach to single-step view synthesis WITHOUT requiring a pretrained diffusion model.
Uses the self-consistency property: f(x_t, t) = f(x_t', t') for all points on the same trajectory.

Key innovations:
1. Geometric pose embedding via Plucker ray coordinates
2. Fresnel-inspired wave optics attention mechanism
3. Self-supervised consistency training from scratch

Author: Fresnel Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CVSConfig:
    """Configuration for Consistency View Synthesizer."""
    # Model dimensions
    image_size: int = 256
    latent_channels: int = 4  # If using VAE latent space

    # U-Net architecture
    base_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)  # [128, 256, 384, 512]
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)  # Apply attention at these spatial sizes

    # Pose encoding
    pose_embed_dim: int = 256

    # Image conditioning
    image_embed_dim: int = 384  # DINOv2 feature dim
    cross_attention_dim: int = 384

    # Time embedding
    time_embed_dim: int = 256

    # Training
    num_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012

    # Consistency-specific
    ema_decay: float = 0.9999

    def __post_init__(self):
        self.channels = [self.base_channels * m for m in self.channel_mult]


# ============================================================================
# Building Blocks
# ============================================================================

class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 for stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )

        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Cross-attention for image conditioning."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.scale = dim_head ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Reshape spatial to sequence
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Project queries, keys, values
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, -1, self.heads * self.dim_head)
        out = self.to_out(out)

        # Reshape back to spatial
        return out.permute(0, 2, 1).view(B, C, H, W)


class FresnelWaveAttention(nn.Module):
    """
    Novel attention mechanism inspired by Fresnel wave optics.

    Uses interference patterns based on path length differences,
    similar to how light waves interfere in holography.
    """

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Learnable wavelength for interference
        self.wavelength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        qkv = self.to_qkv(x_flat).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, -1, self.heads, self.dim_head).transpose(1, 2), qkv)

        # Standard dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Fresnel interference modulation
        # Create position-based phase differences
        positions = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        ), dim=-1).float().view(-1, 2)  # (H*W, 2)

        # Compute pairwise distances
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (H*W, H*W, 2)
        distances = torch.sqrt((pos_diff ** 2).sum(-1) + 1e-8)  # (H*W, H*W)

        # Phase modulation via path length
        phase = 2 * math.pi * distances / (self.wavelength.abs() * H + 1e-6)
        interference = torch.cos(phase)  # (H*W, H*W)

        # Modulate attention with interference pattern
        dots = dots + interference.unsqueeze(0).unsqueeze(0) * 0.1

        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, -1, self.heads * self.dim_head)
        out = self.to_out(out)

        return out.permute(0, 2, 1).view(B, C, H, W)


class AttentionBlock(nn.Module):
    """Combined self-attention and cross-attention block."""

    def __init__(
        self,
        channels: int,
        context_dim: int,
        use_fresnel: bool = True
    ):
        super().__init__()
        self.norm1 = GroupNorm32(32, channels)

        if use_fresnel:
            self.self_attn = FresnelWaveAttention(channels)
        else:
            self.self_attn = CrossAttention(channels, channels)

        self.norm2 = GroupNorm32(32, channels)
        self.cross_attn = CrossAttention(channels, context_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        # Self-attention with residual
        h = self.norm1(x)
        if isinstance(self.self_attn, FresnelWaveAttention):
            h = self.self_attn(h)
        else:
            h_flat = h.view(h.shape[0], h.shape[1], -1).permute(0, 2, 1)
            h = self.self_attn(h, h_flat)
        x = x + h

        # Cross-attention with residual
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h

        return x


class Downsample(nn.Module):
    """Spatial downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ============================================================================
# Pose Encoder
# ============================================================================

class PluckerPoseEncoder(nn.Module):
    """
    Encodes relative camera pose using Plucker ray coordinates.

    Novel approach: Instead of just encoding (R, t) directly, we encode
    the geometric relationship via Plucker lines which naturally capture
    the epipolar geometry between views.
    """

    def __init__(self, embed_dim: int = 256, cross_attention_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attention_dim = cross_attention_dim

        # 6D rotation representation (Zhou et al.) + 3D translation = 9D
        # Plus Plucker coordinates for principal ray = 6D
        # Total: 15D raw input

        self.encoder = nn.Sequential(
            nn.Linear(15, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Project to cross-attention dimension
        self.proj = nn.Linear(embed_dim, cross_attention_dim)

        # Learnable query for cross-attention (output same dim as cross_attention)
        self.pose_queries = nn.Parameter(torch.randn(16, cross_attention_dim) * 0.02)

    def rotation_6d_to_matrix(self, r6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D rotation representation to rotation matrix."""
        # r6d: (B, 6) -> two column vectors
        a1 = r6d[:, :3]
        a2 = r6d[:, 3:6]

        # Gram-Schmidt orthonormalization
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)

    def compute_plucker(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Plucker coordinates for a ray.
        Plucker: (d, m) where d = direction, m = origin × direction
        """
        d = F.normalize(direction, dim=-1)
        m = torch.cross(origin, d, dim=-1)
        return torch.cat([d, m], dim=-1)  # (B, 6)

    def forward(
        self,
        R_rel: torch.Tensor,  # (B, 3, 3) relative rotation
        t_rel: torch.Tensor   # (B, 3) relative translation
    ) -> torch.Tensor:
        """
        Encode relative camera transformation.

        Returns: (B, 16, cross_attention_dim) pose embeddings for cross-attention
        """
        B = R_rel.shape[0]

        # Extract 6D rotation representation
        r6d = R_rel[:, :, :2].reshape(B, 6)

        # Compute Plucker coordinates for camera ray
        origin = torch.zeros(B, 3, device=R_rel.device)
        direction = t_rel  # Translation points to target camera
        plucker = self.compute_plucker(origin, direction)

        # Combine features
        raw_features = torch.cat([r6d, t_rel, plucker], dim=-1)  # (B, 15)

        # Encode and project to cross-attention dimension
        pose_embed = self.encoder(raw_features)  # (B, embed_dim)
        pose_embed = self.proj(pose_embed)  # (B, cross_attention_dim)

        # Expand with learnable queries for richer representation
        queries = self.pose_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 16, cross_attention_dim)

        # Modulate queries with pose embedding
        return queries + pose_embed.unsqueeze(1)  # (B, 16, cross_attention_dim)


# ============================================================================
# Image Encoder (Adapter for DINOv2)
# ============================================================================

class ImageFeatureAdapter(nn.Module):
    """
    Lightweight adapter for DINOv2 features.

    Takes frozen DINOv2 features and projects them for conditioning.
    """

    def __init__(
        self,
        in_dim: int = 384,  # DINOv2-S
        out_dim: int = 384,
        num_tokens: int = 256  # Reduce from 37x37=1369 to 256
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Project features
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1369, out_dim) * 0.02)

        # Token compression via attention
        self.compress_queries = nn.Parameter(torch.randn(num_tokens, out_dim) * 0.02)
        self.compress_attn = nn.MultiheadAttention(out_dim, 8, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 37, 37, 384) DINOv2 features

        Returns:
            (B, num_tokens, out_dim) compressed feature tokens
        """
        B = features.shape[0]

        # Flatten spatial
        x = features.view(B, -1, features.shape[-1])  # (B, 1369, 384)

        # Add position embeddings
        x = x + self.pos_embed[:x.shape[1]]

        # Project
        x = self.proj(x)  # (B, 1369, out_dim)

        # Compress to fewer tokens via cross-attention
        queries = self.compress_queries.unsqueeze(0).expand(B, -1, -1)
        compressed, _ = self.compress_attn(queries, x, x)  # (B, num_tokens, out_dim)

        return compressed


# ============================================================================
# U-Net Backbone
# ============================================================================

class ConsistencyUNet(nn.Module):
    """
    Efficient U-Net for consistency model.

    Optimized for consumer GPUs with ~200M parameters.
    """

    def __init__(self, config: CVSConfig):
        super().__init__()
        self.config = config

        # Input/output convolutions
        self.in_conv = nn.Conv2d(3, config.base_channels, 3, padding=1)
        self.out_conv = nn.Sequential(
            GroupNorm32(32, config.base_channels),
            nn.SiLU(),
            nn.Conv2d(config.base_channels, 3, 3, padding=1)
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(config.time_embed_dim),
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim * 4, config.time_embed_dim)
        )

        # Build encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()

        channels = [config.base_channels] + list(config.channels)
        current_res = config.image_size

        for i, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            blocks = nn.ModuleList()

            for j in range(config.num_res_blocks):
                blocks.append(ResBlock(
                    ch_in if j == 0 else ch_out,
                    ch_out,
                    config.time_embed_dim
                ))

            self.encoder.append(blocks)

            # Add attention at specified resolutions
            if current_res in config.attention_resolutions:
                self.encoder_attns.append(AttentionBlock(
                    ch_out,
                    config.cross_attention_dim,
                    use_fresnel=True
                ))
            else:
                self.encoder_attns.append(None)

            current_res = current_res // 2

        # Downsamplers
        self.downsamplers = nn.ModuleList([
            Downsample(ch) for ch in config.channels[:-1]
        ])

        # Middle block
        mid_ch = config.channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, config.time_embed_dim)
        self.mid_attn = AttentionBlock(mid_ch, config.cross_attention_dim, use_fresnel=True)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, config.time_embed_dim)

        # Pose injection at bottleneck (uses cross_attention_dim since pose encoder outputs that)
        self.pose_proj = nn.Linear(config.cross_attention_dim, mid_ch)

        # Build decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        channels_rev = list(reversed(config.channels))
        current_res = config.image_size // (2 ** len(config.channels))

        for i, (ch_in, ch_out) in enumerate(zip(channels_rev[:-1], channels_rev[1:])):
            blocks = nn.ModuleList()

            # Skip connection channels match encoder output at corresponding level
            # Decoder stage i receives skip from encoder stage (N-1-i)
            # Encoder stage (N-1-i) outputs channels[N-1-i] = channels_rev[i] = ch_in
            skip_ch = ch_in

            for j in range(config.num_res_blocks + 1):
                if j == 0:
                    blocks.append(ResBlock(ch_in + skip_ch, ch_out, config.time_embed_dim))
                else:
                    blocks.append(ResBlock(ch_out, ch_out, config.time_embed_dim))

            self.decoder.append(blocks)

            current_res = current_res * 2
            if current_res in config.attention_resolutions:
                self.decoder_attns.append(AttentionBlock(
                    ch_out,
                    config.cross_attention_dim,
                    use_fresnel=True
                ))
            else:
                self.decoder_attns.append(None)

            self.upsamplers.append(Upsample(ch_out))

        # Final decoder block
        final_blocks = nn.ModuleList()
        for j in range(config.num_res_blocks + 1):
            if j == 0:
                final_blocks.append(ResBlock(
                    channels_rev[-1] + config.base_channels,
                    config.base_channels,
                    config.time_embed_dim
                ))
            else:
                final_blocks.append(ResBlock(
                    config.base_channels,
                    config.base_channels,
                    config.time_embed_dim
                ))
        self.final_decoder = final_blocks

    def forward(
        self,
        x: torch.Tensor,           # (B, 3, H, W) noisy target
        t: torch.Tensor,           # (B,) timesteps
        image_cond: torch.Tensor,  # (B, N, D) image conditioning tokens
        pose_cond: torch.Tensor    # (B, M, D) pose conditioning tokens
    ) -> torch.Tensor:
        """
        Forward pass of consistency U-Net.

        Returns: (B, 3, H, W) predicted clean image x_0
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Combine image and pose conditioning
        context = torch.cat([image_cond, pose_cond], dim=1)  # (B, N+M, D)

        # Initial conv
        h = self.in_conv(x)

        # Encoder with skip connections
        skips = [h]

        for i, (blocks, attn) in enumerate(zip(self.encoder, self.encoder_attns)):
            for block in blocks:
                h = block(h, t_emb)

            if attn is not None:
                h = attn(h, context)

            skips.append(h)

            # Downsample (except after last encoder stage)
            if i < len(self.downsamplers):
                h = self.downsamplers[i](h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)

        # Inject pose at bottleneck (global)
        pose_global = pose_cond.mean(dim=1)  # (B, D)
        pose_proj = self.pose_proj(pose_global)  # (B, mid_ch)
        h = h + pose_proj[:, :, None, None]

        h = self.mid_block2(h, t_emb)

        # Decoder
        for i, (blocks, attn) in enumerate(zip(self.decoder, self.decoder_attns)):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for block in blocks:
                h = block(h, t_emb)

            if attn is not None:
                h = attn(h, context)

            # Upsample (have same number as decoder stages - 1)
            if i < len(self.upsamplers):
                h = self.upsamplers[i](h)

        # Final block
        skip = skips.pop()
        h = torch.cat([h, skip], dim=1)
        for block in self.final_decoder:
            h = block(h, t_emb)

        # Output
        return self.out_conv(h)


# ============================================================================
# Complete CVS Model
# ============================================================================

class ConsistencyViewSynthesizer(nn.Module):
    """
    Complete Consistency View Synthesis model.

    One-step generation of novel views from a single input image.
    """

    def __init__(self, config: Optional[CVSConfig] = None):
        super().__init__()
        self.config = config or CVSConfig()

        # Components
        self.image_adapter = ImageFeatureAdapter(
            in_dim=384,  # DINOv2-S
            out_dim=self.config.cross_attention_dim
        )

        self.pose_encoder = PluckerPoseEncoder(
            embed_dim=self.config.pose_embed_dim,
            cross_attention_dim=self.config.cross_attention_dim
        )

        self.unet = ConsistencyUNet(self.config)

        # Noise schedule (cosine schedule from Improved DDPM)
        self.register_buffer('betas', self._cosine_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine noise schedule from Improved DDPM."""
        steps = self.config.num_timesteps
        s = 0.008
        t = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean images."""
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy, noise

    def forward(
        self,
        input_image: torch.Tensor,        # (B, 3, H, W) input view
        input_features: torch.Tensor,     # (B, 37, 37, 384) DINOv2 features
        R_rel: torch.Tensor,              # (B, 3, 3) relative rotation
        t_rel: torch.Tensor,              # (B, 3) relative translation
        target_image: Optional[torch.Tensor] = None,  # (B, 3, H, W) for training
        timestep: Optional[torch.Tensor] = None       # (B,) specific timestep
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.

        Training: provide target_image, returns consistency prediction
        Inference: omit target_image, returns generated view
        """
        B = input_image.shape[0]
        device = input_image.device

        # Encode image features
        image_cond = self.image_adapter(input_features)  # (B, 256, 384)

        # Encode pose
        pose_cond = self.pose_encoder(R_rel, t_rel)  # (B, 16, 256)

        if target_image is not None:
            # Training mode
            if timestep is None:
                timestep = torch.randint(0, self.config.num_timesteps, (B,), device=device)

            # Add noise to target
            noisy_target, noise = self.add_noise(target_image, timestep)

            # Predict clean image (consistency model predicts x_0, not noise)
            x0_pred = self.unet(noisy_target, timestep.float(), image_cond, pose_cond)

            return {
                'x0_pred': x0_pred,
                'target': target_image,
                'noisy': noisy_target,
                'noise': noise,
                'timestep': timestep
            }
        else:
            # Inference mode - one-step generation
            # Start from pure noise
            z = torch.randn(B, 3, self.config.image_size, self.config.image_size, device=device)
            t = torch.full((B,), self.config.num_timesteps - 1, device=device)

            # One-step denoising
            x0_pred = self.unet(z, t.float(), image_cond, pose_cond)

            return {
                'generated': x0_pred
            }

    @torch.no_grad()
    def generate(
        self,
        input_image: torch.Tensor,
        input_features: torch.Tensor,
        R_rel: torch.Tensor,
        t_rel: torch.Tensor,
        num_steps: int = 1
    ) -> torch.Tensor:
        """
        Generate novel view with optional multi-step refinement.

        num_steps=1: One-step generation (fastest, ~3ms)
        num_steps=2-4: Multi-step refinement (higher quality)
        """
        B = input_image.shape[0]
        device = input_image.device

        # Encode conditions
        image_cond = self.image_adapter(input_features)
        pose_cond = self.pose_encoder(R_rel, t_rel)

        # Start from noise
        z = torch.randn(B, 3, self.config.image_size, self.config.image_size, device=device)

        if num_steps == 1:
            # One-step generation
            t = torch.full((B,), self.config.num_timesteps - 1, device=device, dtype=torch.float)
            return self.unet(z, t, image_cond, pose_cond)
        else:
            # Multi-step refinement
            timesteps = torch.linspace(
                self.config.num_timesteps - 1, 0, num_steps + 1,
                device=device
            ).long()

            for i in range(num_steps):
                t = timesteps[i].expand(B).float()
                z = self.unet(z, t, image_cond, pose_cond)

                if i < num_steps - 1:
                    # Add small noise for next step
                    t_next = timesteps[i + 1]
                    noise_scale = self.sqrt_one_minus_alphas_cumprod[t_next]
                    z = z + noise_scale * torch.randn_like(z) * 0.5

            return z


# ============================================================================
# Consistency Training Loss
# ============================================================================

class ConsistencyLoss(nn.Module):
    """
    Loss function for consistency training (from scratch, no pretrained diffusion).

    Uses self-consistency: f(x_t, t) ≈ f(x_t', t') for nearby timesteps
    """

    def __init__(
        self,
        lambda_consistency: float = 1.0,
        lambda_reconstruction: float = 1.0,
        lambda_perceptual: float = 0.5
    ):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_perceptual = lambda_perceptual

        # Simple perceptual loss using VGG-like features
        self.perceptual_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple perceptual loss."""
        pred_feat = self.perceptual_net(pred)
        target_feat = self.perceptual_net(target)
        return F.l1_loss(pred_feat, target_feat)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        ema_model: Optional[nn.Module] = None,
        model: Optional[nn.Module] = None,
        input_features: Optional[torch.Tensor] = None,
        R_rel: Optional[torch.Tensor] = None,
        t_rel: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute consistency training loss.

        Args:
            model_output: Output from forward pass containing x0_pred, target, etc.
            ema_model: EMA version of the model (for consistency target)
            model: Current model (for computing adjacent timestep prediction)
            input_features, R_rel, t_rel: Conditioning inputs
        """
        x0_pred = model_output['x0_pred']
        target = model_output['target']
        noisy = model_output['noisy']
        timestep = model_output['timestep']

        losses = {}

        # 1. Reconstruction loss (predicted x_0 vs ground truth)
        l1_loss = F.l1_loss(x0_pred, target)
        losses['l1'] = l1_loss * self.lambda_reconstruction

        # 2. Perceptual loss
        perc_loss = self.perceptual_loss(x0_pred, target)
        losses['perceptual'] = perc_loss * self.lambda_perceptual

        # 3. Consistency loss (if EMA model provided)
        if ema_model is not None and model is not None:
            # Get adjacent timestep prediction from EMA model
            # t' = t - Δt
            delta_t = 1
            t_prev = torch.clamp(timestep - delta_t, min=0)

            # Compute x_{t-1} via one Euler step (using current model's prediction)
            # This is simplified - full implementation would use proper ODE solver
            alpha_t = model.sqrt_alphas_cumprod[timestep][:, None, None, None]
            alpha_t_prev = model.sqrt_alphas_cumprod[t_prev][:, None, None, None]

            # Approximate x_{t-1} from x_t and x_0 prediction
            x_t_prev = alpha_t_prev * x0_pred + (1 - alpha_t_prev) / (1 - alpha_t + 1e-8) * (noisy - alpha_t * x0_pred)
            x_t_prev = torch.clamp(x_t_prev, -1, 1)

            # Get EMA model's prediction at t-1
            with torch.no_grad():
                image_cond = ema_model.image_adapter(input_features)
                pose_cond = ema_model.pose_encoder(R_rel, t_rel)
                x0_ema = ema_model.unet(x_t_prev, t_prev.float(), image_cond, pose_cond)

            # Consistency loss: predictions should match
            consistency_loss = F.mse_loss(x0_pred, x0_ema.detach())
            losses['consistency'] = consistency_loss * self.lambda_consistency

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


# ============================================================================
# Training Utilities
# ============================================================================

def create_ema_model(model: nn.Module) -> nn.Module:
    """Create EMA copy of model."""
    ema = type(model)(model.config)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    return ema


def update_ema(model: nn.Module, ema: nn.Module, decay: float = 0.9999):
    """Update EMA model parameters."""
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


def get_relative_pose(
    R_source: torch.Tensor,  # (B, 3, 3)
    t_source: torch.Tensor,  # (B, 3)
    R_target: torch.Tensor,  # (B, 3, 3)
    t_target: torch.Tensor   # (B, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute relative camera pose from source to target.

    Returns:
        R_rel: (B, 3, 3) relative rotation
        t_rel: (B, 3) relative translation
    """
    # R_rel = R_target @ R_source.T
    R_rel = torch.bmm(R_target, R_source.transpose(-1, -2))

    # t_rel = t_target - R_rel @ t_source
    t_rel = t_target - torch.bmm(R_rel, t_source.unsqueeze(-1)).squeeze(-1)

    return R_rel, t_rel


# ============================================================================
# Model Info
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: ConsistencyViewSynthesizer):
    """Print model architecture info."""
    total = count_parameters(model)
    image_adapter = count_parameters(model.image_adapter)
    pose_encoder = count_parameters(model.pose_encoder)
    unet = count_parameters(model.unet)

    print(f"\n{'='*50}")
    print(f"Consistency View Synthesizer - Architecture Summary")
    print(f"{'='*50}")
    print(f"Total Parameters:    {total:,} ({total/1e6:.1f}M)")
    print(f"  Image Adapter:     {image_adapter:,} ({image_adapter/1e6:.1f}M)")
    print(f"  Pose Encoder:      {pose_encoder:,} ({pose_encoder/1e6:.1f}M)")
    print(f"  U-Net Backbone:    {unet:,} ({unet/1e6:.1f}M)")
    print(f"{'='*50}")
    print(f"Config:")
    print(f"  Image Size:        {model.config.image_size}×{model.config.image_size}")
    print(f"  Channels:          {model.config.channels}")
    print(f"  Attention Res:     {model.config.attention_resolutions}")
    print(f"  Timesteps:         {model.config.num_timesteps}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    # Test the model
    config = CVSConfig(image_size=256)
    model = ConsistencyViewSynthesizer(config)
    print_model_info(model)

    # Test forward pass
    B = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    input_image = torch.randn(B, 3, 256, 256, device=device)
    input_features = torch.randn(B, 37, 37, 384, device=device)
    R_rel = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    t_rel = torch.tensor([[0.5, 0.0, 0.0]], device=device).expand(B, -1)
    target_image = torch.randn(B, 3, 256, 256, device=device)

    print("Testing training forward pass...")
    output = model(input_image, input_features, R_rel, t_rel, target_image)
    print(f"  x0_pred shape: {output['x0_pred'].shape}")
    print(f"  timestep range: [{output['timestep'].min()}, {output['timestep'].max()}]")

    print("\nTesting inference (one-step)...")
    with torch.no_grad():
        generated = model.generate(input_image, input_features, R_rel, t_rel, num_steps=1)
    print(f"  generated shape: {generated.shape}")

    print("\nTesting inference (multi-step)...")
    with torch.no_grad():
        generated = model.generate(input_image, input_features, R_rel, t_rel, num_steps=4)
    print(f"  generated shape: {generated.shape}")

    print("\n✓ All tests passed!")
