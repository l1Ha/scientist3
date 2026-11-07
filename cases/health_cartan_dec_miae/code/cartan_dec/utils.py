"""Small utilities shared by training and evaluation."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class AverageMeter:
    """Running mean tracker for loss components."""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.count = 0

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self.total[key] = self.total.get(key, 0.0) + float(value) * n
        self.count += n

    def averages(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.total.items()}

    def reset(self) -> None:
        self.total.clear()
        self.count = 0


class CSVLogger:
    """Append metric dictionaries to a CSV file with a stable header."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self.header is None:
            self.header = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.header)
                writer.writeheader()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.header)
            writer.writerow({key: row.get(key, "") for key in self.header})


def save_json(path: str | os.PathLike, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | os.PathLike, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
