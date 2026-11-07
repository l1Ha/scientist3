import torch

from cartan_dec.data import SyntheticBackboneDataset, collate_backbones
from cartan_dec.model import (
    CartanDEC,
    CartanDECConfig,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _batch():
    dataset = SyntheticBackboneDataset(
        n_samples=4, n_classes=4, min_length=18, max_length=22, seed=0
    )
    return collate_backbones([dataset[i] for i in range(4)])


def _model():
    config = CartanDECConfig(
        d_model=48,
        depth=2,
        n_heads=4,
        decoder_dim=48,
        decoder_depth=1,
        decoder_heads=4,
        max_exact_heat_residues=32,
        mask_strategy="span",
        mask_ratio=0.5,
    )
    return CartanDEC(config)


def test_rotation_6d_roundtrip():
    torch.manual_seed(0)
    rotation = torch.linalg.qr(torch.randn(16, 3, 3)).Q
    rotation = rotation * torch.sign(torch.linalg.det(rotation))[..., None, None]
    vector = matrix_to_rotation_6d(rotation)
    recovered = rotation_6d_to_matrix(vector)
    assert torch.allclose(recovered, rotation, atol=1e-5)


def test_forward_shapes_and_loss():
    batch = _batch()
    model = _model()
    output = model(batch["coords"], batch["valid"], mask_ratio=0.5)
    assert torch.isfinite(output["loss"])
    assert output["predictions"]["omega"].shape == batch["coords"].shape[:2] + (3,)
    assert output["predictions"]["rotation"].shape == batch["coords"].shape[:2] + (6,)
    assert output["masked"].shape == batch["valid"].shape
    assert output["masked"].dtype == torch.bool
    assert not (output["masked"] & ~batch["valid"]).any()


def test_backward_and_features():
    batch = _batch()
    model = _model()
    output = model(batch["coords"], batch["valid"], mask_ratio=0.4)
    output["loss"].backward()
    assert model.encoder.feature_proj[0].weight.grad is not None
    features = model.extract_features(batch["coords"], batch["valid"])
    assert features.shape[0] == 4
    assert torch.isfinite(features).all()


def test_geometry_masking_path():
    batch = _batch()
    model = _model()
    output = model(
        batch["coords"],
        batch["valid"],
        mask_strategy="geometry",
        mask_ratio=0.4,
        return_geometry=True,
    )
    assert output["masked"].any()
    assert torch.isfinite(output["loss"])
