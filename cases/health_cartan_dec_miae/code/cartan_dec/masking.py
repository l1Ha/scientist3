"""Masking strategies for Cartan-DEC-MiAE pretraining.

All functions return a boolean mask with shape ``[B, L]`` where ``True`` means
"masked / to be reconstructed".  At least one valid residue per sample is kept
visible so the encoder always receives some context.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _per_row_valid(valid: Tensor) -> list[Tensor]:
    return [torch.nonzero(valid[b], as_tuple=False).flatten() for b in range(valid.shape[0])]


def random_masking(
    valid: Tensor,
    ratio: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Uniform random masking."""
    masked = torch.zeros_like(valid)
    ratio = float(min(max(ratio, 0.0), 1.0))
    for b, indices in enumerate(_per_row_valid(valid)):
        n = indices.numel()
        if n <= 1:
            continue
        n_mask = min(max(int(round(n * ratio)), 1), n - 1)
        perm = torch.randperm(n, generator=generator, device=valid.device)
        masked[b, indices[perm[:n_mask]]] = True
    return masked


def span_masking(
    valid: Tensor,
    ratio: float = 0.5,
    span_length: int = 10,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Contiguous span masking along the sequence.

    The number of spans is chosen so the expected masked fraction is close to
    ``ratio``; overlaps are counted once.  Extra masked positions are removed
    randomly to hit an exact per-sample budget.
    """
    masked = torch.zeros_like(valid)
    ratio = float(min(max(ratio, 0.0), 1.0))
    span_length = max(int(span_length), 1)

    for b, indices in enumerate(_per_row_valid(valid)):
        n = indices.numel()
        if n <= 1:
            continue
        n_mask = min(max(int(round(n * ratio)), 1), n - 1)
        n_spans = max(1, int(round(n_mask / span_length)))
        starts = torch.randint(
            0, max(n - span_length + 1, 1), (n_spans,), generator=generator, device=valid.device
        )
        chosen: set[int] = set()
        for start in starts.tolist():
            for offset in range(span_length):
                if start + offset < n:
                    chosen.add(start + offset)
        chosen_list = sorted(chosen)
        if len(chosen_list) > n_mask:
            order = torch.randperm(len(chosen_list), generator=generator, device=valid.device)
            chosen_list = [chosen_list[i] for i in order[:n_mask].tolist()]
        masked[b, indices[torch.tensor(chosen_list, device=valid.device, dtype=torch.long)]] = True

    return masked


def geometry_masking(
    valid: Tensor,
    defect: Tensor,
    ratio: float = 0.5,
    span_length: int = 6,
    temperature: float = 1.0,
    hard: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Mask spans centred on high curvature / holonomy-defect residues.

    ``hard=True`` selects the top-k defect residues; ``hard=False`` samples
    centres from a Boltzmann distribution with temperature ``temperature``.
    This mirrors the paper's "holonomy-curvature masking" ablation, which is
    useful as a controlled baseline even though the paper finds it does not
    beat contiguous span masking at scale.
    """
    masked = torch.zeros_like(valid)
    ratio = float(min(max(ratio, 0.0), 1.0))
    span_length = max(int(span_length), 1)

    for b, indices in enumerate(_per_row_valid(valid)):
        n = indices.numel()
        if n <= 1:
            continue
        n_mask = min(max(int(round(n * ratio)), 1), n - 1)
        n_spans = max(1, int(round(n_mask / span_length)))
        scores = defect[b, indices].float()
        if scores.numel() == 0:
            continue

        if hard:
            centres = torch.topk(scores, k=min(n_spans, n)).indices
        else:
            logits = scores / max(temperature, 1e-6)
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)
            centres = torch.multinomial(
                probs,
                num_samples=min(n_spans, n),
                replacement=False,
                generator=generator,
            )

        chosen: set[int] = set()
        for centre in centres.tolist():
            start = max(0, centre - span_length // 2)
            end = min(n, start + span_length)
            for offset in range(start, end):
                chosen.add(offset)

        chosen_list = sorted(chosen)
        if len(chosen_list) > n_mask:
            order = torch.randperm(len(chosen_list), generator=generator, device=valid.device)
            chosen_list = [chosen_list[i] for i in order[:n_mask].tolist()]
        masked[b, indices[torch.tensor(chosen_list, device=valid.device, dtype=torch.long)]] = True

    return masked


def make_mask(
    valid: Tensor,
    strategy: str = "span",
    ratio: float = 0.5,
    span_length: int = 10,
    defect: Optional[Tensor] = None,
    temperature: float = 1.0,
    hard: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Dispatch to a masking strategy."""
    strategy = strategy.lower()
    if strategy in {"random", "uniform"}:
        return random_masking(valid, ratio=ratio, generator=generator)
    if strategy in {"span", "contiguous"}:
        return span_masking(
            valid, ratio=ratio, span_length=span_length, generator=generator
        )
    if strategy in {"geometry", "curvature", "holonomy"}:
        if defect is None:
            raise ValueError("geometry masking requires a per-residue defect tensor")
        return geometry_masking(
            valid,
            defect,
            ratio=ratio,
            span_length=span_length,
            temperature=temperature,
            hard=hard,
            generator=generator,
        )
    raise ValueError(f"Unknown masking strategy: {strategy}")
