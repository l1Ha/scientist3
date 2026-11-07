"""Mask-centric losses for Cartan-DEC-MiAE.

The total objective is

``L = L_theta + L_pos + L_rot + L_omega + L_Omega + L_integration``

where every term is evaluated only on masked residues.  ``L_integration``
compares the forms integrated back to 3D geometry with the anchor-relative
ground truth, which is what forces the encoder to learn long-range transport.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over the last dimension, restricted to ``mask``."""
    if mask.sum() == 0:
        return prediction.sum() * 0.0
    error = (prediction - target).pow(2).mean(dim=-1)
    return (error * mask.to(error.dtype)).sum() / mask.to(error.dtype).sum().clamp_min(1.0)


def rotation_6d_to_matrix(vector: Tensor) -> Tensor:
    """Convert a 6D rotation representation to a rotation matrix."""
    a1, a2 = vector[..., :3], vector[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_loss(prediction_6d: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Geodesic-style rotation loss, averaged over masked residues."""
    if mask.sum() == 0:
        return prediction_6d.sum() * 0.0
    predicted = rotation_6d_to_matrix(prediction_6d)
    # trace(R_pred^T R_gt) = sum elementwise product.
    trace = (predicted * target).sum(dim=(-1, -2))
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    error = 1.0 - cosine
    return (error * mask.to(error.dtype)).sum() / mask.to(error.dtype).sum().clamp_min(1.0)


def matrix_rotation_loss(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Rotation loss when both inputs are matrices."""
    if mask.sum() == 0:
        return predicted.sum() * 0.0
    trace = (predicted * target).sum(dim=(-1, -2))
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    error = 1.0 - cosine
    return (error * mask.to(error.dtype)).sum() / mask.to(error.dtype).sum().clamp_min(1.0)


def cartan_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    masked: Tensor,
    valid: Tensor,
    integrated_position: Optional[Tensor] = None,
    integrated_rotation: Optional[Tensor] = None,
    config: Optional[object] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the mask-centric Cartan-DEC objective."""
    mask = masked & valid
    cfg = config

    def weight(name: str, default: float) -> float:
        return float(getattr(cfg, name, default)) if cfg is not None else default

    l_theta = masked_mse(predictions["theta"], targets["theta"], mask)
    l_omega = masked_mse(predictions["omega"], targets["omega"], mask)
    l_curvature_vector = masked_mse(
        predictions["curvature"], targets["curvature"], mask
    )
    curvature_energy_pred = predictions["curvature"].norm(dim=-1)
    curvature_energy_target = targets["curvature"].norm(dim=-1)
    l_curvature_energy = masked_mse(
        curvature_energy_pred[..., None], curvature_energy_target[..., None], mask
    )
    l_curvature = l_curvature_vector + 0.5 * l_curvature_energy

    # Positions are measured in Angstrom; divide by a fixed scale so the
    # position term is numerically comparable to the rotation/form terms.
    position_scale = 10.0
    l_position = masked_mse(
        predictions["position"] / position_scale,
        targets["position"] / position_scale,
        mask,
    )
    l_rotation = rotation_loss(predictions["rotation"], targets["rotation"], mask)

    if integrated_position is not None:
        l_integration_position = masked_mse(
            integrated_position / position_scale,
            targets["position"] / position_scale,
            mask,
        )
    else:
        l_integration_position = predictions["position"].sum() * 0.0

    if integrated_rotation is not None:
        l_integration_rotation = matrix_rotation_loss(
            integrated_rotation, targets["rotation"], mask
        )
    else:
        l_integration_rotation = predictions["rotation"].sum() * 0.0

    l_integration = l_integration_position + l_integration_rotation

    total = (
        weight("lambda_theta", 1.0) * l_theta
        + weight("lambda_position", 1.0) * l_position
        + weight("lambda_rotation", 1.0) * l_rotation
        + weight("lambda_omega", 1.0) * l_omega
        + weight("lambda_curvature", 1.0) * l_curvature
        + weight("lambda_integration", 1.0) * l_integration
    )

    components = {
        "theta": l_theta,
        "position": l_position,
        "rotation": l_rotation,
        "omega": l_omega,
        "curvature": l_curvature,
        "integration_position": l_integration_position,
        "integration_rotation": l_integration_rotation,
        "total": total,
    }
    return total, components
