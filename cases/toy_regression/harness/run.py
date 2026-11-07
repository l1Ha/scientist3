"""A tiny, dependency-free regression harness used by the Scientist3 demo.

The target mixes a periodic term, a quadratic term and an interaction.  A
linear ridge baseline underfits it; random Fourier features and polynomial
features provide controlled, interpretable alternatives for the research loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_data(n: int, seed: int, noise: float = 0.05, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 3))
    x[:, 0] += shift
    y = (
        np.sin(2.2 * x[:, 0])
        + 0.45 * x[:, 1] ** 2
        + 0.30 * x[:, 0] * x[:, 2]
        + rng.normal(scale=noise, size=n)
    )
    return x, y


def ridge(features: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ y)


def predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    return design @ weights


def raw_features(x: np.ndarray) -> np.ndarray:
    return x.copy()


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    columns = [x]
    if degree >= 2:
        columns.extend([x[:, i : i + 1] * x[:, j : j + 1] for i in range(x.shape[1]) for j in range(i, x.shape[1])])
    if degree >= 3:
        columns.extend(
            [
                x[:, i : i + 1] * x[:, j : j + 1] * x[:, k : k + 1]
                for i in range(x.shape[1])
                for j in range(i, x.shape[1])
                for k in range(j, x.shape[1])
            ]
        )
    return np.concatenate(columns, axis=1)


def make_rff_map(
    rng: np.random.Generator,
    n_features: int,
    features: int,
    bandwidth: float,
    phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one random Fourier feature map shared by train/test splits."""
    weights = rng.normal(scale=1.0 / bandwidth, size=(n_features, features))
    offset = rng.uniform(0.0, 2.0 * np.pi, size=features) if phase else np.zeros(features)
    return weights, offset


def rff_features(x: np.ndarray, weights: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.cos(x @ weights + offset)


def fit_predict(args: argparse.Namespace):
    x, y = build_data(args.n, args.seed, noise=args.noise)
    rng = np.random.default_rng(args.seed + 991)
    order = rng.permutation(len(x))
    split = max(1, int(0.75 * len(x)))
    train, test = order[:split], order[split:]
    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]

    method = args.method
    if method == "linear":
        train_features = raw_features(x_train)
        test_features = raw_features(x_test)
    elif method == "poly":
        train_features = polynomial_features(x_train, args.degree)
        test_features = polynomial_features(x_test, args.degree)
    elif method in {"rff", "rff_poly"}:
        weights_rff, offset_rff = make_rff_map(
            rng, x_train.shape[1], args.features, args.bandwidth,
            phase=args.phase == "random",
        )
        train_features = rff_features(x_train, weights_rff, offset_rff)
        test_features = rff_features(x_test, weights_rff, offset_rff)
        if method == "rff_poly":
            train_features = np.concatenate(
                [train_features, polynomial_features(x_train, args.degree)], axis=1
            )
            test_features = np.concatenate(
                [test_features, polynomial_features(x_test, args.degree)], axis=1
            )
        if args.ablate == "rff":
            train_features = raw_features(x_train)
            test_features = raw_features(x_test)
    else:
        raise ValueError(f"unknown method: {method}")

    if args.ablate == "reg":
        alpha = 100.0 if args.alpha < 100.0 else args.alpha
    else:
        alpha = args.alpha

    weights = ridge(train_features, y_train, alpha)

    # A shifted test set is used by supplementary robustness experiments.
    if args.shift:
        x_shift, y_shift = build_data(args.n, args.seed, noise=args.noise, shift=args.shift)
        if method in {"rff", "rff_poly"}:
            shift_features = rff_features(x_shift, weights_rff, offset_rff)
            if method == "rff_poly":
                shift_features = np.concatenate(
                    [shift_features, polynomial_features(x_shift, args.degree)], axis=1
                )
        elif method == "poly":
            shift_features = polynomial_features(x_shift, args.degree)
        else:
            shift_features = raw_features(x_shift)
        if args.ablate == "rff":
            shift_features = raw_features(x_shift)
        x_test, y_test = x_shift, y_shift
        test_features = shift_features

    prediction = predict(weights, test_features)
    rmse = float(np.sqrt(np.mean((prediction - y_test) ** 2)))
    variance = float(np.var(y_test))
    r2 = float(1.0 - np.mean((prediction - y_test) ** 2) / max(variance, 1e-12))
    train_prediction = predict(weights, train_features)
    train_rmse = float(np.sqrt(np.mean((train_prediction - y_train) ** 2)))
    return {
        "rmse": rmse,
        "r2": r2,
        "train_rmse": train_rmse,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": int(train_features.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="linear", choices=["linear", "poly", "rff", "rff_poly"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--phase", default="random", choices=["random", "none"])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--ablate", default="", choices=["", "rff", "phase", "bandwidth", "reg"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.ablate == "phase":
        args.phase = "none"
    if args.ablate == "bandwidth":
        args.bandwidth = 0.05

    metrics = fit_predict(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
