#!/usr/bin/env python
"""Run a small end-to-end Cartan-DEC-MiAE demo on synthetic backbones.

This is deliberately CPU friendly: a 1-2M parameter model, short chains, and a
handful of epochs.  It verifies the full pretraining + frozen-probe pipeline.

Usage:
    python scripts/quickstart.py
    python scripts/quickstart.py --out runs/my_demo --epochs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cartan_dec.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/quickstart")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = TrainConfig(
        dataset="synthetic",
        samples=args.samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=16,
        d_model=128,
        depth=4,
        n_heads=8,
        decoder_dim=128,
        decoder_depth=2,
        decoder_heads=8,
        min_length=60,
        max_synthetic_length=90,
        max_exact_heat_residues=96,
        probe_epochs=30,
        log_every=4,
        out=args.out,
        device=args.device,
    )
    results = train(config)
    print("\nQuickstart complete.")
    if "probe" in results:
        print(f"Frozen-probe test metrics: {results['probe']}")


if __name__ == "__main__":
    main()
