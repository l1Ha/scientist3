"""Cartan-DEC-MiAE model.

A clean, standalone PyTorch implementation of the method described in
"Cartan Moving Frames and Discrete Exterior Calculus for Self-Supervised
Protein Structure Representation" (ScientistTwo, 2026).

The implementation keeps the asymmetric masked-autoencoder skeleton:

* a sparse, gauge-invariant encoder sees only visible residues and is biased by
  curvature inner products and connection heat transport;
* a lightweight Cartan-Darboux decoder predicts connection 1-forms, curvature
  2-forms, soldering forms and anchor-relative geometry;
* predicted forms are integrated along a sequence path back to 3D frames and
  coordinates, so the encoder cannot solve the task by local interpolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry import (
    backbone_features,
    connection_forms,
    connection_heat_kernel,
    connection_inner_product,
    contact_adjacency,
    frames_from_backbone,
    geometry_targets,
    integrate_sequence_paths,
    nearest_visible_anchor,
    so3_exp,
    triangle_curvature,
)
from .losses import cartan_loss
from .masking import make_mask


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings for residue indices."""

    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, positions: Tensor) -> Tensor:
        # positions: [B, L] (long)
        args = positions.float()[..., None] * self.freqs.to(positions.device)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding[..., : self.dim]


class RBFExpansion(nn.Module):
    """Fixed Gaussian radial basis expansion."""

    def __init__(self, n_bins: int = 16, min_val: float = 0.0, max_val: float = 1.0) -> None:
        super().__init__()
        centres = torch.linspace(min_val, max_val, n_bins)
        self.register_buffer("centres", centres, persistent=False)
        self.width = (max_val - min_val) / max(n_bins - 1, 1)

    def forward(self, values: Tensor) -> Tensor:
        return torch.exp(
            -((values[..., None] - self.centres.to(values.device)) ** 2)
            / (self.width**2 + 1e-8)
        )


class AttentionBias(nn.Module):
    """Additive geometric attention logits.

    ``curvature`` is the normalised inner product ``Tr(Omega_i Omega_j^T)``;
    ``heat`` is the connection heat-transport affinity; ``distance`` is an
    optional Euclidean RBF.  Each channel has a learnable per-head scale.
    """

    def __init__(
        self,
        n_heads: int,
        use_curvature: bool = True,
        use_heat: bool = True,
        use_distance: bool = False,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.use_curvature = use_curvature
        self.use_heat = use_heat
        self.use_distance = use_distance

        self.curvature_scale = nn.Parameter(torch.zeros(n_heads))
        self.heat_scale = nn.Parameter(torch.zeros(n_heads))
        self.distance_scale = nn.Parameter(torch.zeros(n_heads))

        self.heat_rbf = RBFExpansion(16, 0.0, 1.0)
        self.distance_rbf = RBFExpansion(16, 0.0, 32.0)
        self.heat_mlp = nn.Sequential(
            nn.Linear(16, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.distance_mlp = nn.Sequential(
            nn.Linear(16, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        curvature_pair: Optional[Tensor],
        heat_pair: Optional[Tensor],
        distance_pair: Optional[Tensor],
    ) -> Optional[Tensor]:
        bias = None
        if self.use_curvature and curvature_pair is not None:
            # [B, L, L] -> [B, 1, L, L]
            channel = curvature_pair[:, None]
            bias = self.curvature_scale[None, :, None, None] * channel
        if self.use_heat and heat_pair is not None:
            channel = self.heat_mlp(self.heat_rbf(heat_pair)).squeeze(-1)[:, None]
            term = self.heat_scale[None, :, None, None] * channel
            bias = term if bias is None else bias + term
        if self.use_distance and distance_pair is not None:
            channel = self.distance_mlp(self.distance_rbf(distance_pair)).squeeze(-1)[:, None]
            term = self.distance_scale[None, :, None, None] * channel
            bias = term if bias is None else bias + term
        return bias


class GeometricAttention(nn.Module):
    """Multi-head self-attention with an optional additive pair bias."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        key_mask: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias
        logits = logits.masked_fill(
            ~key_mask[:, None, None, :], torch.finfo(logits.dtype).min
        )
        attn = torch.softmax(logits, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        out = self.out(out)
        return out * key_mask[..., None].to(out.dtype)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with optional geometric attention bias."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        bias_module: Optional[AttentionBias] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GeometricAttention(dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.bias_module = bias_module

    def forward(
        self,
        x: Tensor,
        key_mask: Tensor,
        curvature_pair: Optional[Tensor] = None,
        heat_pair: Optional[Tensor] = None,
        distance_pair: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        if bias is None and self.bias_module is not None:
            bias = self.bias_module(curvature_pair, heat_pair, distance_pair)
        x = x + self.dropout(self.attn(self.norm1(x), key_mask, bias))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


def _gather_pair(pair: Tensor, indices: Tensor) -> Tensor:
    """Gather a ``[B, L, L]`` matrix on both axes with ``[B, K]`` indices."""
    b, k = indices.shape
    rows = pair.gather(1, indices[:, :, None].expand(b, k, pair.shape[-1]))
    return rows.gather(2, indices[:, None, :].expand(b, k, k))


# ---------------------------------------------------------------------------
# Encoder / decoder
# ---------------------------------------------------------------------------


class CartanEncoder(nn.Module):
    """Sparse gauge-invariant encoder over visible residues."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 512,
        depth: int = 12,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_curvature_bias: bool = True,
        use_heat_bias: bool = True,
        use_distance_bias: bool = False,
    ) -> None:
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.position = SinusoidalPositionalEmbedding(d_model)
        self.bias_module = (
            AttentionBias(
                n_heads,
                use_curvature=use_curvature_bias,
                use_heat=use_heat_bias,
                use_distance=use_distance_bias,
            )
            if (use_curvature_bias or use_heat_bias or use_distance_bias)
            else None
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout=dropout,
                    bias_module=self.bias_module,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        features: Tensor,
        visible: Tensor,
        visible_index: Tensor,
        curvature_pair: Tensor,
        heat_pair: Tensor,
        distance_pair: Tensor,
    ) -> Tensor:
        b, length, _ = features.shape
        k = visible_index.shape[1]

        gather_features = features.gather(
            1, visible_index[..., None].expand(b, k, features.shape[-1])
        )
        gather_visible = visible.gather(1, visible_index)
        x = self.feature_proj(gather_features)
        x = x + self.position(visible_index)
        x = x * gather_visible[..., None].to(x.dtype)

        curvature_vis = _gather_pair(curvature_pair, visible_index)
        heat_vis = _gather_pair(heat_pair, visible_index)
        distance_vis = _gather_pair(distance_pair, visible_index)

        # The geometric bias is shared across layers and computed once.
        bias = None
        if self.bias_module is not None:
            bias = self.bias_module(curvature_vis, heat_vis, distance_vis)

        for block in self.blocks:
            x = block(x, gather_visible, bias=bias)

        x = self.norm(x)

        out = torch.zeros(b, length, x.shape[-1], dtype=x.dtype, device=x.device)
        out.scatter_(
            1,
            visible_index[..., None].expand(b, k, x.shape[-1]),
            x * gather_visible[..., None].to(x.dtype),
        )
        return out


class CartanDecoder(nn.Module):
    """Lightweight Cartan-Darboux decoder."""

    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 256,
        depth: int = 3,
        n_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(decoder_dim))
        self.position = SinusoidalPositionalEmbedding(decoder_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_dim, n_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.omega_head = nn.Linear(decoder_dim, 3)
        self.curvature_head = nn.Linear(decoder_dim, 3)
        self.theta_head = nn.Linear(decoder_dim, 3)
        self.position_head = nn.Linear(decoder_dim, 3)
        self.rotation_head = nn.Linear(decoder_dim, 6)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        encoder_output: Tensor,
        valid: Tensor,
        masked: Tensor,
        positions: Tensor,
    ) -> dict[str, Tensor]:
        b, length, _ = encoder_output.shape
        x = self.embed(encoder_output)
        mask_tokens = self.mask_token.view(1, 1, -1).expand(b, length, -1)
        x = torch.where(masked[..., None], mask_tokens, x)
        x = x + self.position(positions)
        x = x * valid[..., None].to(x.dtype)

        for block in self.blocks:
            x = block(x, valid)

        x = self.norm(x)
        return {
            "omega": self.omega_head(x),
            "curvature": self.curvature_head(x),
            "theta": self.theta_head(x),
            "position": self.position_head(x),
            "rotation": self.rotation_head(x),
        }


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@dataclass
class CartanDECConfig:
    """Configuration for :class:`CartanDEC`."""

    feature_dim: int = 11
    d_model: int = 512
    depth: int = 12
    n_heads: int = 8
    dropout: float = 0.0
    decoder_dim: int = 256
    decoder_depth: int = 3
    decoder_heads: int = 8

    contact_radius: float = 8.0
    max_neighbors: Optional[int] = None
    heat_tau: float = 0.5
    max_exact_heat_residues: int = 192

    mask_strategy: str = "span"
    mask_ratio: float = 0.5
    span_length: int = 10
    mask_temperature: float = 1.0
    hard_geometry_mask: bool = False

    use_curvature_bias: bool = True
    use_heat_bias: bool = True
    use_distance_bias: bool = False

    lambda_theta: float = 1.0
    lambda_position: float = 1.0
    lambda_rotation: float = 1.0
    lambda_omega: float = 1.0
    lambda_curvature: float = 1.0
    lambda_integration: float = 1.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class CartanDEC(nn.Module):
    """Cartan-DEC masked invariant autoencoder."""

    def __init__(self, config: Optional[CartanDECConfig] = None) -> None:
        super().__init__()
        self.config = config or CartanDECConfig()
        cfg = self.config
        self.encoder = CartanEncoder(
            feature_dim=cfg.feature_dim,
            d_model=cfg.d_model,
            depth=cfg.depth,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            use_curvature_bias=cfg.use_curvature_bias,
            use_heat_bias=cfg.use_heat_bias,
            use_distance_bias=cfg.use_distance_bias,
        )
        self.decoder = CartanDecoder(
            encoder_dim=cfg.d_model,
            decoder_dim=cfg.decoder_dim,
            depth=cfg.decoder_depth,
            n_heads=cfg.decoder_heads,
            dropout=cfg.dropout,
        )

    # -- geometry ---------------------------------------------------------

    def compute_geometry(self, coords: Tensor, valid: Tensor) -> dict[str, Tensor]:
        frames, translations = frames_from_backbone(coords, valid=valid)
        adj = contact_adjacency(
            translations,
            valid=valid,
            radius=self.config.contact_radius,
            max_neighbors=self.config.max_neighbors,
        )
        omega_contact = connection_forms(frames, adj)
        curvature, curvature_energy = triangle_curvature(frames, adj, omega_contact)
        heat = connection_heat_kernel(
            frames,
            adj,
            tau=self.config.heat_tau,
            max_exact_residues=self.config.max_exact_heat_residues,
        )
        features = backbone_features(coords, frames, valid=valid)
        curvature_pair = connection_inner_product(curvature)
        distance = torch.cdist(translations, translations)
        targets = geometry_targets(
            frames,
            translations,
            adj,
            omega_contact,
            curvature=curvature,
            curvature_energy=curvature_energy,
        )

        return {
            "frames": frames,
            "translations": translations,
            "adjacency": adj,
            "omega_contact": omega_contact,
            "curvature": curvature,
            "curvature_energy": curvature_energy,
            "heat": heat,
            "features": features,
            "curvature_pair": curvature_pair,
            "distance": distance,
            "targets": targets,
        }

    # -- encoding ---------------------------------------------------------

    def _visible_index(self, visible: Tensor) -> Tensor:
        counts = visible.sum(dim=-1)
        k = max(int(counts.max().item()) if counts.numel() else 1, 1)
        order = torch.argsort(visible.to(torch.int8), dim=-1, descending=True)
        return order[:, :k]

    def encode(
        self,
        coords: Tensor,
        valid: Optional[Tensor] = None,
        geometry: Optional[dict[str, Tensor]] = None,
    ) -> Tensor:
        """Encode all valid residues (no masking) for probing."""
        if valid is None:
            valid = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
        geom = geometry or self.compute_geometry(coords, valid)
        visible = valid
        visible_index = self._visible_index(visible)
        return self.encoder(
            geom["features"],
            visible,
            visible_index,
            geom["curvature_pair"],
            geom["heat"],
            geom["distance"],
        )

    def pool(self, token_features: Tensor, valid: Tensor) -> Tensor:
        """Mean + max pooling used by the frozen-encoder probe."""
        mask = valid[..., None].to(token_features.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (token_features * mask).sum(dim=1) / denom
        neg_inf = torch.finfo(token_features.dtype).min
        maxed = (token_features.masked_fill(~valid[..., None], neg_inf)).max(dim=1).values
        maxed = torch.where(valid.any(dim=1, keepdim=True), maxed, torch.zeros_like(maxed))
        return torch.cat([mean, maxed], dim=-1)

    def extract_features(self, coords: Tensor, valid: Optional[Tensor] = None) -> Tensor:
        encoded = self.encode(coords, valid)
        if valid is None:
            valid = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
        return self.pool(encoded, valid)

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        coords: Tensor,
        valid: Optional[Tensor] = None,
        masked: Optional[Tensor] = None,
        mask_ratio: Optional[float] = None,
        mask_strategy: Optional[str] = None,
        plddt: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_geometry: bool = False,
    ) -> dict:
        if coords.dim() != 4:
            raise ValueError("coords must have shape [B, L, 3, 3] (N, CA, C)")
        b, length = coords.shape[:2]
        if valid is None:
            valid = torch.ones(b, length, dtype=torch.bool, device=coords.device)
        if plddt is not None:
            # Keep the hook for future use; the current feature set is purely
            # geometric, so pLDDT is accepted but not concatenated.
            pass

        geom = self.compute_geometry(coords, valid)
        if masked is None:
            masked = make_mask(
                valid,
                strategy=mask_strategy or self.config.mask_strategy,
                ratio=self.config.mask_ratio if mask_ratio is None else mask_ratio,
                span_length=self.config.span_length,
                defect=geom["curvature_energy"],
                temperature=self.config.mask_temperature,
                hard=self.config.hard_geometry_mask,
                generator=generator,
            )
        else:
            masked = masked & valid

        visible = valid & ~masked
        visible_index = self._visible_index(visible)
        anchor = nearest_visible_anchor(visible, valid)

        encoder_output = self.encoder(
            geom["features"],
            visible,
            visible_index,
            geom["curvature_pair"],
            geom["heat"],
            geom["distance"],
        )
        positions = torch.arange(length, device=coords.device).view(1, length).expand(b, length)
        predictions = self.decoder(encoder_output, valid, masked, positions)

        # Integrate predicted forms from anchors back to 3D geometry.
        integrated_frames, integrated_trans = integrate_sequence_paths(
            predictions["omega"],
            predictions["theta"],
            geom["frames"],
            geom["translations"],
            anchor,
            masked,
            valid,
        )

        anchor_frames = geom["frames"].gather(
            1, anchor[..., None, None].expand(b, length, 3, 3)
        )
        anchor_trans = geom["translations"].gather(
            1, anchor[..., None].expand(b, length, 3)
        )

        target_position = torch.einsum(
            "blij,blj->bli",
            anchor_frames.transpose(-1, -2),
            geom["translations"] - anchor_trans,
        )
        target_rotation = anchor_frames.transpose(-1, -2) @ geom["frames"]

        integrated_position = torch.einsum(
            "blij,blj->bli",
            anchor_frames.transpose(-1, -2),
            integrated_trans - anchor_trans,
        )
        integrated_rotation = anchor_frames.transpose(-1, -2) @ integrated_frames

        targets = {
            "theta": geom["targets"]["theta"],
            "omega": geom["targets"]["omega"],
            "curvature": geom["targets"]["curvature"],
            "position": target_position,
            "rotation": target_rotation,
        }

        loss, components = cartan_loss(
            predictions,
            targets,
            masked=masked,
            valid=valid,
            integrated_position=integrated_position,
            integrated_rotation=integrated_rotation,
            config=self.config,
        )

        result = {
            "loss": loss,
            "loss_components": components,
            "predictions": predictions,
            "targets": targets,
            "masked": masked,
            "visible": visible,
            "anchor": anchor,
            "integrated_frames": integrated_frames,
            "integrated_translations": integrated_trans,
            "integrated_position": integrated_position,
            "integrated_rotation": integrated_rotation,
        }
        if return_geometry:
            result["geometry"] = geom
        return result


def build_model(**kwargs) -> CartanDEC:
    """Convenience constructor with keyword overrides."""
    return CartanDEC(CartanDECConfig(**kwargs))


# ---------------------------------------------------------------------------
# Rotation utilities used by losses and probes
# ---------------------------------------------------------------------------


def rotation_6d_to_matrix(vector: Tensor) -> Tensor:
    """Convert a 6D rotation representation to a proper rotation matrix."""
    a1, a2 = vector[..., :3], vector[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """First two columns of a rotation matrix, flattened to 6D."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)
