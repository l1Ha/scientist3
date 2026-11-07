import torch

from cartan_dec.data import SyntheticBackboneDataset
from cartan_dec.geometry import (
    connection_forms,
    connection_heat_kernel,
    contact_adjacency,
    frames_from_backbone,
    geometry_targets,
    integrate_sequence_paths,
    nearest_visible_anchor,
    so3_exp,
    so3_log,
    triangle_curvature,
)


def _sample(length: int = 20):
    dataset = SyntheticBackboneDataset(
        n_samples=1, n_classes=1, min_length=length, max_length=length, seed=0
    )
    coords = dataset[0]["coords"].unsqueeze(0)
    valid = torch.ones(1, length, dtype=torch.bool)
    return coords, valid


def test_frames_are_orthonormal():
    coords, valid = _sample()
    frames, translations = frames_from_backbone(coords, valid)
    identity = torch.eye(3).expand_as(frames)
    assert torch.allclose(frames.transpose(-1, -2) @ frames, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(frames), torch.ones(1, 20), atol=1e-5)
    assert torch.allclose(translations, coords[:, :, 1, :])


def test_so3_roundtrip():
    torch.manual_seed(1)
    axis = torch.nn.functional.normalize(torch.randn(256, 3), dim=-1)
    for scale in (1e-6, 1e-3, 0.2, 1.0, 2.8):
        vector = axis * scale
        rotation = so3_exp(vector)
        recovered = so3_log(rotation)
        assert torch.allclose(so3_exp(recovered), rotation, atol=2e-4)


def test_connection_form_matches_relative_rotation():
    coords, valid = _sample()
    frames, translations = frames_from_backbone(coords, valid)
    adj = contact_adjacency(translations, valid, radius=8.0)
    omega = connection_forms(frames, adj)
    relative = frames[:, :-1].transpose(-1, -2) @ frames[:, 1:]
    edge = adj[:, torch.arange(19), torch.arange(1, 20)]
    recovered = so3_exp(omega[:, torch.arange(19), torch.arange(1, 20)][edge])
    assert torch.allclose(recovered, relative[edge], atol=1e-4)


def test_curvature_shapes_and_finiteness():
    coords, valid = _sample()
    frames, translations = frames_from_backbone(coords, valid)
    adj = contact_adjacency(translations, valid, radius=10.0)
    omega = connection_forms(frames, adj)
    curvature, energy = triangle_curvature(frames, adj, omega)
    assert curvature.shape == (1, 20, 3)
    assert energy.shape == (1, 20)
    assert torch.isfinite(curvature).all()
    assert torch.isfinite(energy).all()
    assert (energy >= 0).all()


def test_heat_kernel_shape_and_finiteness():
    coords, valid = _sample()
    frames, translations = frames_from_backbone(coords, valid)
    adj = contact_adjacency(translations, valid, radius=10.0)
    heat = connection_heat_kernel(frames, adj, tau=0.5, max_exact_residues=32)
    assert heat.shape == (1, 20, 20)
    assert torch.isfinite(heat).all()
    assert (heat >= 0).all()


def test_sequence_integration_reproduces_ground_truth():
    coords, valid = _sample(24)
    frames, translations = frames_from_backbone(coords, valid)
    adj = contact_adjacency(translations, valid, radius=8.0)
    omega = connection_forms(frames, adj)
    targets = geometry_targets(frames, translations, adj, omega)

    masked = torch.zeros(1, 24, dtype=torch.bool)
    masked[0, 7:16] = True
    visible = valid & ~masked
    anchors = nearest_visible_anchor(visible, valid)
    integrated_frames, integrated_trans = integrate_sequence_paths(
        targets["omega"], targets["theta"], frames, translations, anchors, masked, valid
    )
    assert torch.allclose(integrated_frames[masked], frames[masked], atol=1e-4)
    assert torch.allclose(integrated_trans[masked], translations[masked], atol=1e-3)
