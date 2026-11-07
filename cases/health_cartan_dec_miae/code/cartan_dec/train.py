"""Pretraining entry point for Cartan-DEC-MiAE.

Examples
--------
Quick synthetic smoke test (CPU friendly)::

    python -m cartan_dec.train --dataset synthetic --samples 128 --epochs 3 \
        --d-model 128 --depth 4 --decoder-dim 128 --decoder-depth 2 \
        --out runs/quickstart

Full TEDBench pretraining (requires the `datasets` package and a GPU)::

    python -m cartan_dec.train --dataset tedbench --repo-id TEDBench/ted \
        --epochs 5 --batch-size 32 --d-model 512 --depth 12 \
        --out runs/tedbench
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import collate_backbones, make_dataset
from .geometry_eval import masked_geometry_metrics
from .model import CartanDEC, CartanDECConfig
from .probe import extract_features, train_probe
from .utils import AverageMeter, CSVLogger, resolve_device, save_checkpoint, save_json, set_seed


@dataclass
class TrainConfig:
    # data
    dataset: str = "synthetic"
    samples: int = 256
    test_samples: int = 96
    n_classes: int = 4
    seed: int = 0
    repo_id: str = "TEDBench/ted"
    train_split: str = "train"
    test_repo_id: str = "TEDBench/ted"
    test_split: str = "test"
    max_length: Optional[int] = None
    min_length: int = 60
    max_synthetic_length: int = 110

    # optimisation
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 100
    grad_clip: float = 1.0
    steps_per_epoch: int = 0
    num_workers: int = 0
    device: str = "auto"

    # model
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
    use_curvature_bias: bool = True
    use_heat_bias: bool = True
    use_distance_bias: bool = False

    # masking / loss
    mask_strategy: str = "span"
    mask_ratio: float = 0.5
    span_length: int = 10
    mask_temperature: float = 1.0
    hard_geometry_mask: bool = False
    lambda_theta: float = 1.0
    lambda_position: float = 1.0
    lambda_rotation: float = 1.0
    lambda_omega: float = 1.0
    lambda_curvature: float = 1.0
    lambda_integration: float = 1.0

    # logging / probe
    out: str = "runs/cartan_dec"
    log_every: int = 10
    save_every: int = 0
    probe: bool = True
    probe_epochs: int = 50
    probe_batch_size: int = 128
    probe_hidden: int = 512
    skip_probe: bool = False


def build_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config


def build_model_config(config: TrainConfig) -> CartanDECConfig:
    return CartanDECConfig(
        d_model=config.d_model,
        depth=config.depth,
        n_heads=config.n_heads,
        dropout=config.dropout,
        decoder_dim=config.decoder_dim,
        decoder_depth=config.decoder_depth,
        decoder_heads=config.decoder_heads,
        contact_radius=config.contact_radius,
        max_neighbors=config.max_neighbors,
        heat_tau=config.heat_tau,
        max_exact_heat_residues=config.max_exact_heat_residues,
        mask_strategy=config.mask_strategy,
        mask_ratio=config.mask_ratio,
        span_length=config.span_length,
        mask_temperature=config.mask_temperature,
        hard_geometry_mask=config.hard_geometry_mask,
        use_curvature_bias=config.use_curvature_bias,
        use_heat_bias=config.use_heat_bias,
        use_distance_bias=config.use_distance_bias,
        lambda_theta=config.lambda_theta,
        lambda_position=config.lambda_position,
        lambda_rotation=config.lambda_rotation,
        lambda_omega=config.lambda_omega,
        lambda_curvature=config.lambda_curvature,
        lambda_integration=config.lambda_integration,
    )


def _datasets(config: TrainConfig):
    if config.dataset == "synthetic":
        train = make_dataset(
            "synthetic",
            n_samples=config.samples,
            n_classes=config.n_classes,
            seed=config.seed,
            min_length=config.min_length,
            synthetic_max_length=config.max_synthetic_length,
        )
        test = make_dataset(
            "synthetic",
            n_samples=config.test_samples,
            n_classes=config.n_classes,
            seed=config.seed + 10_000,
            min_length=config.min_length,
            synthetic_max_length=config.max_synthetic_length,
        )
        return train, test

    train = make_dataset(
        "tedbench",
        repo_id=config.repo_id,
        split=config.train_split,
        max_length=config.max_length,
    )
    test = make_dataset(
        "tedbench",
        repo_id=config.test_repo_id,
        split=config.test_split,
        max_length=config.max_length,
    )
    # Infer the label cardinality when the Hugging Face dataset exposes it.
    try:
        label_feature = train.dataset.features["label"]
        if hasattr(label_feature, "num_classes"):
            config.n_classes = int(label_feature.num_classes)
    except Exception:
        pass
    return train, test


def train(config: TrainConfig) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", asdict(config))

    train_dataset, test_dataset = _datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_backbones,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_backbones,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_backbones,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_config = build_model_config(config)
    model = CartanDEC(model_config).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device}  parameters={n_parameters / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = config.steps_per_epoch or len(train_loader)
    total_steps = max(steps_per_epoch * config.epochs, 1)

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return max(step, 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger = CSVLogger(out_dir / "losses.csv")
    meter = AverageMeter()
    global_step = 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        meter.reset()
        for step, batch in enumerate(train_loader, start=1):
            if config.steps_per_epoch and step > config.steps_per_epoch:
                break
            coords = batch["coords"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(
                    coords,
                    valid,
                    mask_ratio=config.mask_ratio,
                    mask_strategy=config.mask_strategy,
                )
                loss = output["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}: {loss}")

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            scheduler.step()

            components = {key: float(value.detach()) for key, value in output["loss_components"].items()}
            meter.update(components)
            global_step += 1

            if config.log_every and global_step % config.log_every == 0:
                averages = meter.averages()
                print(
                    f"epoch {epoch:3d} step {global_step:6d} "
                    f"loss={averages.get('total', float('nan')):.4f} "
                    f"pos={averages.get('position', float('nan')):.4f} "
                    f"omega={averages.get('omega', float('nan')):.4f} "
                    f"curv={averages.get('curvature', float('nan')):.4f}"
                )
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    **averages,
                }
                logger.log(row)
                meter.reset()

        if config.save_every and epoch % config.save_every == 0:
            save_checkpoint(out_dir / f"checkpoint_epoch{epoch}.pt", model, optimizer, epoch, asdict(config))

    save_checkpoint(out_dir / "last.pt", model, optimizer, config.epochs, asdict(config))
    elapsed = time.time() - start
    print(f"pretraining finished in {elapsed / 60:.1f} min")

    results: dict = {
        "config": asdict(config),
        "parameters": n_parameters,
        "pretrain_minutes": elapsed / 60.0,
    }

    if config.probe and not config.skip_probe:
        print("extracting frozen features...")
        train_features, train_labels = extract_features(model, eval_train_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)
        print("training probe...")
        _, metrics = train_probe(
            train_features,
            train_labels,
            test_features,
            test_labels,
            num_classes=config.n_classes,
            epochs=config.probe_epochs,
            batch_size=config.probe_batch_size,
            hidden=config.probe_hidden,
            device=device,
            seed=config.seed,
            verbose=True,
        )
        results["probe"] = metrics
        print(f"probe: {metrics}")

    print("evaluating masked geometry reconstruction...")
    results["reconstruction"] = masked_geometry_metrics(
        model, test_loader, device, mask_seed=1234
    )
    print(f"reconstruction: {results['reconstruction']}")
    save_json(out_dir / "metrics.json", results)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain Cartan-DEC-MiAE")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "tedbench"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--n-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--test-repo-id", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-synthetic-length", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--decoder-dim", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--contact-radius", type=float, default=None)
    parser.add_argument("--max-neighbors", type=int, default=None)
    parser.add_argument("--heat-tau", type=float, default=None)
    parser.add_argument("--max-exact-heat-residues", type=int, default=None)
    parser.add_argument("--no-curvature-bias", action="store_true")
    parser.add_argument("--no-heat-bias", action="store_true")
    parser.add_argument("--distance-bias", action="store_true")

    parser.add_argument("--mask-strategy", default=None, choices=["span", "random", "geometry"])
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--span-length", type=int, default=None)
    parser.add_argument("--mask-temperature", type=float, default=None)
    parser.add_argument("--hard-geometry-mask", action="store_true")
    parser.add_argument("--lambda-theta", type=float, default=None)
    parser.add_argument("--lambda-position", type=float, default=None)
    parser.add_argument("--lambda-rotation", type=float, default=None)
    parser.add_argument("--lambda-omega", type=float, default=None)
    parser.add_argument("--lambda-curvature", type=float, default=None)
    parser.add_argument("--lambda-integration", type=float, default=None)

    parser.add_argument("--out", default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=None)
    parser.add_argument("--probe-batch-size", type=int, default=None)
    parser.add_argument("--probe-hidden", type=int, default=None)
    parser.add_argument("--skip-probe", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    if args.no_curvature_bias:
        config.use_curvature_bias = False
    if args.no_heat_bias:
        config.use_heat_bias = False
    if args.distance_bias:
        config.use_distance_bias = True
    train(config)


if __name__ == "__main__":
    main()
