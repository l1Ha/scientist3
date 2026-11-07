"""Datasets and collation for Cartan-DEC-MiAE.

The synthetic dataset is intentionally small and dependency-free: it lets the
full pretraining + probing pipeline run on a laptop in a few minutes.  For real
experiments use :class:`TEDBenchDataset`, which wraps the public
`TEDBench/ted` and `TEDBench/cath` Hugging Face datasets.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), eps)


def _make_backbone(ca: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place N and C atoms around a C-alpha trace."""
    length = ca.shape[0]
    coords = np.zeros((length, 3, 3), dtype=np.float32)
    coords[:, 1, :] = ca

    # A smoothly varying perpendicular direction keeps the synthetic frames
    # from collapsing onto a single orientation.
    base_u = _normalize(rng.normal(size=3))
    for i in range(length):
        if i == 0:
            tangent = _normalize(ca[1] - ca[0]) if length > 1 else np.array([1.0, 0.0, 0.0])
        elif i == length - 1:
            tangent = _normalize(ca[-1] - ca[-2])
        else:
            tangent = _normalize(ca[i + 1] - ca[i - 1])

        u = base_u - float(np.dot(base_u, tangent)) * tangent
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
        u = _normalize(u)
        # Add a small per-residue twist.
        angle = 0.15 * i
        v = np.cross(tangent, u)
        u = math.cos(angle) * u + math.sin(angle) * v

        coords[i, 0, :] = ca[i] - 1.46 * tangent + 0.90 * u  # N
        coords[i, 2, :] = ca[i] + 1.52 * tangent + 0.90 * u  # C

    return coords


def _helix(length: int, rng: np.random.Generator) -> np.ndarray:
    radius = 2.3 + rng.normal(scale=0.05)
    rise = 1.5
    turn = math.radians(100.0)
    theta = np.arange(length) * turn
    ca = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), rise * np.arange(length)],
        axis=-1,
    )
    ca += rng.normal(scale=0.08, size=ca.shape)
    return ca


def _sheet(length: int, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(length) * 3.3
    y = np.where(np.arange(length) % 2 == 0, 1.0, -1.0) * 2.1
    z = np.sin(np.arange(length) * 0.15) * 0.7
    ca = np.stack([x, y, z], axis=-1)
    ca += rng.normal(scale=0.08, size=ca.shape)
    return ca


def _coil(length: int, rng: np.random.Generator) -> np.ndarray:
    ca = np.zeros((length, 3), dtype=np.float64)
    direction = _normalize(rng.normal(size=3))
    for i in range(1, length):
        # Persistence plus a random turn keeps the chain non-collapsed.
        direction = _normalize(direction + 0.65 * rng.normal(size=3))
        ca[i] = ca[i - 1] + 3.8 * direction
    return ca


def _globule(length: int, rng: np.random.Generator) -> np.ndarray:
    ca = np.zeros((length, 3), dtype=np.float64)
    direction = _normalize(rng.normal(size=3))
    radius = 3.0 * (length ** (1.0 / 3.0))
    for i in range(1, length):
        direction = _normalize(direction + 0.9 * rng.normal(size=3))
        step = 3.8 * direction
        candidate = ca[i - 1] + step
        # Soft confinement to a sphere.
        if np.linalg.norm(candidate) > radius:
            candidate = candidate - 1.8 * candidate / max(np.linalg.norm(candidate), 1e-6)
        ca[i] = candidate
    return ca


class SyntheticBackboneDataset(Dataset):
    """A small synthetic fold-classification benchmark.

    Four morphology families (helix, sheet, coil, globule) produce distinct
    local geometry while sharing the same backbone representation.  It is not a
    scientific benchmark; it is a smoke test for the full pipeline.
    """

    def __init__(
        self,
        n_samples: int = 256,
        n_classes: int = 4,
        min_length: int = 60,
        max_length: int = 110,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.samples: list[dict] = []
        rng = np.random.default_rng(seed)
        generators = [_helix, _sheet, _coil, _globule]
        for i in range(n_samples):
            label = i % n_classes
            length = int(rng.integers(min_length, max_length + 1))
            generator = generators[label % len(generators)]
            ca = generator(length, rng)
            coords = _make_backbone(ca.astype(np.float64), rng)
            self.samples.append(
                {
                    "coords": torch.from_numpy(coords),
                    "label": torch.tensor(label, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class TEDBenchDataset(Dataset):
    """Optional wrapper around the public TEDBench Hugging Face datasets.

    Example:
        >>> ds = TEDBenchDataset("TEDBench/ted", split="train")
        >>> sample = ds[0]
        >>> sample["coords"].shape
        torch.Size([L, 3, 3])
    """

    def __init__(
        self,
        repo_id: str = "TEDBench/ted",
        split: str = "train",
        max_length: Optional[int] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TEDBenchDataset requires the `datasets` package: pip install datasets"
            ) from exc

        self.dataset = load_dataset(repo_id, split=split)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        coords = torch.tensor(sample["coords"], dtype=torch.float32)
        if self.max_length is not None:
            coords = coords[: self.max_length]
        return {
            "coords": coords,
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
        }


def collate_backbones(batch: list[dict]) -> dict[str, Tensor]:
    """Pad variable-length backbones and build the validity mask."""
    max_length = max(item["coords"].shape[0] for item in batch)
    batch_size = len(batch)
    coords = torch.zeros(batch_size, max_length, 3, 3, dtype=torch.float32)
    valid = torch.zeros(batch_size, max_length, dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])

    for i, item in enumerate(batch):
        length = item["coords"].shape[0]
        coords[i, :length] = item["coords"]
        valid[i, :length] = True

    return {"coords": coords, "valid": valid, "label": labels}


def random_rotation(coords: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """Apply a random global rotation to each backbone (augmentation)."""
    # QR-based random rotation.
    a = torch.randn(coords.shape[0], 3, 3, generator=generator, device=coords.device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    return coords @ q.transpose(-1, -2)


def make_dataset(
    name: str = "synthetic",
    n_samples: int = 256,
    n_classes: int = 4,
    seed: int = 0,
    repo_id: str = "TEDBench/ted",
    split: str = "train",
    max_length: Optional[int] = None,
    min_length: int = 60,
    synthetic_max_length: int = 110,
) -> Dataset:
    """Factory used by the training/evaluation entry points."""
    if name == "synthetic":
        return SyntheticBackboneDataset(
            n_samples=n_samples,
            n_classes=n_classes,
            seed=seed,
            min_length=min_length,
            max_length=synthetic_max_length,
        )
    if name in {"tedbench", "ted", "hf"}:
        return TEDBenchDataset(repo_id=repo_id, split=split, max_length=max_length)
    raise ValueError(f"Unknown dataset: {name}")
