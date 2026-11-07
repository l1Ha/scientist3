"""Frozen-encoder probing for fold classification.

The paper evaluates every pretraining condition with an identical probe: mean
and max pooled token latents feed a shared MLP head.  We follow that protocol
and additionally expose a linear probe for quick sanity checks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

try:  # scikit-learn is optional for training but required for metrics
    from sklearn.metrics import accuracy_score as _sk_accuracy
    from sklearn.metrics import f1_score as _sk_f1
except ImportError:  # pragma: no cover
    _sk_accuracy = None
    _sk_f1 = None


def extract_features(model, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    """Run a frozen model over a loader and pool residue latents."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            coords = batch["coords"].to(device)
            valid = batch["valid"].to(device)
            pooled = model.extract_features(coords, valid)
            features.append(pooled.cpu())
            labels.append(batch["label"].cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


class MLPProbe(nn.Module):
    """Shared MLP probe used by the paper's frozen-encoder evaluation."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _metrics(predictions: Tensor, labels: Tensor) -> dict[str, float]:
    pred = predictions.argmax(dim=-1).cpu().numpy()
    true = labels.cpu().numpy()
    if _sk_accuracy is not None:
        return {
            "accuracy": float(_sk_accuracy(true, pred)),
            "macro_f1": float(_sk_f1(true, pred, average="macro", zero_division=0)),
        }
    accuracy = float((predictions.argmax(-1).cpu() == labels.cpu()).float().mean())
    return {"accuracy": accuracy, "macro_f1": float("nan")}


def train_probe(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    num_classes: int,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 512,
    dropout: float = 0.1,
    label_smoothing: float = 0.1,
    device: Optional[torch.device] = None,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[MLPProbe, dict[str, float]]:
    """Train the shared MLP probe and return test metrics."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)

    # Standardise using training statistics only.
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    train_ds = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    probe = MLPProbe(
        input_dim=train_features.shape[-1],
        num_classes=num_classes,
        hidden=hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = probe(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * y.numel()
            total += y.numel()
        scheduler.step()
        if verbose and (epoch + 1) % max(epochs // 5, 1) == 0:
            print(f"  probe epoch {epoch + 1:3d}/{epochs}  loss={total_loss / max(total, 1):.4f}")

    probe.eval()
    with torch.no_grad():
        test_logits = probe(test_features.to(device))
        train_logits = probe(train_features.to(device))
    metrics = _metrics(test_logits, test_labels.to(device))
    metrics["train_accuracy"] = float(
        (train_logits.argmax(-1).cpu() == train_labels).float().mean()
    )
    return probe, metrics


def linear_probe(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
) -> dict[str, float]:
    """Optional multinomial logistic-regression probe."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise ImportError("linear_probe requires scikit-learn") from exc

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = ((train_features - mean) / std).numpy()
    x_test = ((test_features - mean) / std).numpy()

    classifier = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
    )
    classifier.fit(x_train, train_labels.numpy())
    predictions = classifier.predict(x_test)
    if _sk_accuracy is not None:
        return {
            "accuracy": float(_sk_accuracy(test_labels.numpy(), predictions)),
            "macro_f1": float(
                _sk_f1(test_labels.numpy(), predictions, average="macro", zero_division=0)
            ),
        }
    return {"accuracy": float((predictions == test_labels.numpy()).mean()), "macro_f1": float("nan")}
