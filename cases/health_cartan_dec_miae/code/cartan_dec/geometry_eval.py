"""Deterministic masked-geometry reconstruction metrics.

Fold classification on the synthetic smoke benchmark saturates quickly, so the
case also reports how well a frozen encoder/decoder reconstructs held-out
backbone geometry under a *fixed* mask.  These metrics directly measure the
Cartan transport objectives and are therefore the primary endpoints of the
bundled case; the linear/MLP probe remains a secondary sanity check.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .masking import make_mask
from .model import matrix_to_rotation_6d, rotation_6d_to_matrix


def _masked_mean(values: Tensor, mask: Tensor) -> tuple[float, int]:
    selected = values[mask & torch.isfinite(values)]
    if selected.numel() == 0:
        return 0.0, 0
    return float(selected.sum()), int(selected.numel())


def masked_geometry_metrics(
    model,
    loader,
    device: torch.device,
    mask_seed: int = 1234,
    mask_ratio: float = 0.5,
    mask_strategy: str = "span",
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Reconstruct held-out backbone geometry under a deterministic mask.

    The same mask seed is used for every variant, so differences between
    variants come from the learned representation rather than mask luck.
    """

    model.eval()
    position_sse = 0.0
    position_count = 0
    integrated_sse = 0.0
    integrated_count = 0
    rotation_sum = 0.0
    rotation_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            coords = batch["coords"].to(device)
            valid = batch["valid"].to(device)
            generator = torch.Generator(device=valid.device)
            generator.manual_seed(mask_seed + batch_index)
            masked = make_mask(
                valid,
                strategy=mask_strategy,
                ratio=mask_ratio,
                span_length=10,
                generator=generator,
            )
            output = model(coords, valid, masked=masked)
            targets = output["targets"]
            mask = output["masked"] & valid

            position_error = (
                output["predictions"]["position"] - targets["position"]
            ).pow(2).sum(dim=-1)
            integrated_error = (
                output["integrated_position"] - targets["position"]
            ).pow(2).sum(dim=-1)
            position_sse += _masked_mean(position_error, mask)[0]
            position_count += int(mask.sum())

            integrated_sse += _masked_mean(integrated_error, mask)[0]
            integrated_count += int(mask.sum())

            predicted_rotation = rotation_6d_to_matrix(
                output["predictions"]["rotation"]
            )
            relative = predicted_rotation.transpose(-1, -2) @ targets["rotation"]
            trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
            angle = torch.acos(cosine) * (180.0 / math.pi)
            rotation_sum += _masked_mean(angle, mask)[0]
            rotation_count += int(mask.sum())

    return {
        "position_rmse": math.sqrt(position_sse / max(position_count, 1)),
        "integrated_position_rmse": math.sqrt(
            integrated_sse / max(integrated_count, 1)
        ),
        "rotation_error_deg": rotation_sum / max(rotation_count, 1),
        "masked_residues": position_count,
    }
