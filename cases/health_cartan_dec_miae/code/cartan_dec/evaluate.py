"""Evaluate a pretrained Cartan-DEC-MiAE checkpoint with a frozen-encoder probe."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import collate_backbones
from .model import CartanDEC, CartanDECConfig
from .probe import extract_features, linear_probe, train_probe
from .train import TrainConfig, _datasets
from .utils import load_checkpoint, resolve_device, save_json, set_seed


def evaluate(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    train_config = TrainConfig(**checkpoint["config"])

    # Override data options from the command line.
    for key in (
        "dataset",
        "samples",
        "test_samples",
        "n_classes",
        "repo_id",
        "train_split",
        "test_repo_id",
        "test_split",
        "max_length",
        "batch_size",
        "num_workers",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(train_config, key, value)

    train_dataset, test_dataset = _datasets(train_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_backbones,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_backbones,
        num_workers=args.num_workers,
    )

    model_fields = {field.name for field in fields(CartanDECConfig)}
    model_kwargs = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in model_fields
    }
    model = CartanDEC(CartanDECConfig(**model_kwargs)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("extracting frozen features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    if args.probe == "linear":
        metrics = linear_probe(train_features, train_labels, test_features, test_labels)
        probe = None
    else:
        probe, metrics = train_probe(
            train_features,
            train_labels,
            test_features,
            test_labels,
            num_classes=train_config.n_classes,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            hidden=args.probe_hidden,
            device=device,
            seed=args.seed,
            verbose=True,
        )

    results = {
        "checkpoint": str(args.checkpoint),
        "probe": args.probe,
        "metrics": metrics,
    }
    if args.out:
        save_json(args.out, results)
    print(metrics)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Cartan-DEC-MiAE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None, choices=["synthetic", "tedbench"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--n-classes", type=int, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--test-repo-id", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-hidden", type=int, default=512)
    parser.add_argument("--out", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
