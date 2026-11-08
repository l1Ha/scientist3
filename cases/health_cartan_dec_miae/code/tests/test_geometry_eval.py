import math

import torch
from torch.utils.data import DataLoader

from cartan_dec.data import SyntheticBackboneDataset, collate_backbones
from cartan_dec.geometry_eval import masked_geometry_metrics
from cartan_dec.model import CartanDEC, CartanDECConfig


def _loader():
    dataset = SyntheticBackboneDataset(
        n_samples=4, n_classes=4, min_length=18, max_length=22, seed=0
    )
    return DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collate_backbones
    )


def _model():
    config = CartanDECConfig(
        d_model=48,
        depth=2,
        n_heads=4,
        decoder_dim=48,
        decoder_depth=1,
        decoder_heads=4,
        max_exact_heat_residues=32,
    )
    return CartanDEC(config)


def test_masked_geometry_metrics_keys_and_ranges():
    torch.manual_seed(0)
    model = _model()
    metrics = masked_geometry_metrics(
        model, _loader(), torch.device("cpu"), mask_seed=1234, max_batches=1
    )

    assert set(metrics) == {
        "position_rmse",
        "integrated_position_rmse",
        "rotation_error_deg",
        "masked_residues",
    }
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["position_rmse"] >= 0.0
    assert metrics["integrated_position_rmse"] >= 0.0
    assert 0.0 <= metrics["rotation_error_deg"] <= 180.0
    assert metrics["masked_residues"] > 0
    # The evaluation pass must not leave the model in training mode.
    assert model.training is False


def test_masked_geometry_metrics_is_deterministic():
    torch.manual_seed(0)
    model = _model()
    loader = _loader()
    first = masked_geometry_metrics(model, loader, torch.device("cpu"), mask_seed=7)
    second = masked_geometry_metrics(model, loader, torch.device("cpu"), mask_seed=7)
    assert first == second


def test_masked_geometry_metrics_respects_max_batches():
    torch.manual_seed(0)
    model = _model()
    loader = _loader()  # 4 samples, batch size 2 -> two batches
    both = masked_geometry_metrics(model, loader, torch.device("cpu"), mask_seed=0)
    first_only = masked_geometry_metrics(
        model, loader, torch.device("cpu"), mask_seed=0, max_batches=1
    )
    assert 0 < first_only["masked_residues"] < both["masked_residues"]
