"""Discrete differential geometry for protein backbones.

This module implements the geometric front-end of Cartan-DEC-MiAE:

* residue frames ``E_i in SO(3)`` built from the N, C-alpha, C backbone atoms;
* a contact graph on C-alpha positions;
* discrete connection 1-forms ``omega_ij = log(E_i^T E_j)``;
* discrete curvature 2-forms on contact triangles;
* a connection Laplacian and its heat kernel;
* curvature/holonomy-like defect scores used by geometry-aware masking.

All functions operate on batched tensors and are differentiable unless marked
otherwise.  Geometry is cheap to cache: see ``scripts/extract_geometry.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Small SO(3) helpers
# ---------------------------------------------------------------------------


def normalize(v: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalise vectors along the last dimension."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def skew(v: Tensor) -> Tensor:
    """Map ``v`` to its skew-symmetric matrix ``[v]_x``.

    Args:
        v: ``[..., 3]`` tensor.

    Returns:
        ``[..., 3, 3]`` tensor.
    """
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(omega: Tensor) -> Tensor:
    """Exponential map from ``so(3)`` vectors to ``SO(3)`` matrices.

    Uses the Rodrigues formula with a Taylor expansion near zero so that it is
    stable for the tiny relative rotations common on protein contact graphs.
    """
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    small = theta < 1e-4

    # sin(t)/t and (1-cos(t))/t^2, with Taylor expansions near zero.
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2,
    )

    k = skew(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(k)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def so3_log(rotation: Tensor) -> Tensor:
    """Logarithm map from ``SO(3)`` matrices to ``so(3)`` vectors.

    The branch near ``pi`` is handled explicitly to avoid the vanishing
    ``sin(theta)`` denominator.  Protein contact edges are almost always far
    from that branch, but robustness is cheap.
    """
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    w = 0.5 * (rotation - rotation.transpose(-1, -2))
    vec = torch.stack([w[..., 2, 1], w[..., 0, 2], w[..., 1, 0]], dim=-1)

    small = theta < 1e-4
    scale = torch.where(small, torch.ones_like(theta), theta / sin_theta.clamp_min(1e-7))
    out = vec * scale[..., None]

    # Near pi: recover the axis from (R + I) / 2 = a a^T.
    near_pi = theta > math.pi - 0.05
    if near_pi.any():
        s = 0.5 * (rotation + torch.eye(3, dtype=rotation.dtype, device=rotation.device))
        diag = s.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        k = diag.argmax(dim=-1)
        row = s.gather(
            -2,
            k[..., None, None].expand(*s.shape[:-2], 1, 3),
        ).squeeze(-2)
        axis = torch.sqrt(diag)
        axis = axis * torch.sign(row)
        # If the chosen row is degenerate, fall back to a canonical basis vector.
        bad = axis.norm(dim=-1, keepdim=True) < 1e-6
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        fallback = torch.nn.functional.one_hot(k, 3).to(rotation.dtype)
        fallback = fallback * eye.sum(dim=0)  # keep shape/broadcast friendly
        axis = torch.where(bad, fallback, axis)
        axis = normalize(axis)
        out = torch.where(near_pi[..., None], axis * theta[..., None], out)

    return out


# ---------------------------------------------------------------------------
# Backbone frames and contact graph
# ---------------------------------------------------------------------------


def frames_from_backbone(
    coords: Tensor,
    valid: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Build right-handed residue frames from N, C-alpha, C coordinates.

    Args:
        coords: ``[B, L, 3, 3]`` with the atom order ``(N, CA, C)``.
        valid: optional ``[B, L]`` boolean mask.

    Returns:
        frames: ``[B, L, 3, 3]`` with columns ``(e1, e2, e3)``.
        translations: ``[B, L, 3]`` C-alpha positions.
    """
    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
    n, ca, c = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]

    e1 = normalize(c - ca, eps)
    v2 = n - ca
    e2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    norm2 = e2.norm(dim=-1, keepdim=True)
    # Fallback for (near-)collinear backbones.
    alt = torch.zeros_like(e1)
    alt[..., 2] = 1.0
    e2_alt = normalize(torch.linalg.cross(e1, alt, dim=-1), eps)
    e2 = torch.where(norm2 < eps, e2_alt, normalize(e2, eps))
    e3 = torch.linalg.cross(e1, e2, dim=-1)

    frames = torch.stack([e1, e2, e3], dim=-1)  # columns

    if valid is not None:
        eye = torch.eye(3, dtype=frames.dtype, device=frames.device).expand_as(frames)
        frames = torch.where(valid[..., None, None], frames, eye)
        ca = torch.where(valid[..., None], ca, torch.zeros_like(ca))

    return frames, ca


def contact_adjacency(
    translations: Tensor,
    valid: Optional[Tensor] = None,
    radius: float = 8.0,
    max_neighbors: Optional[int] = None,
) -> Tensor:
    """Build a symmetric C-alpha contact graph.

    Args:
        translations: ``[B, L, 3]`` C-alpha positions.
        valid: optional ``[B, L]`` mask.
        radius: contact cutoff in Angstrom.
        max_neighbors: optionally keep only the nearest ``k`` neighbours per
            residue (the graph is symmetrised afterwards).

    Returns:
        ``[B, L, L]`` boolean adjacency without self loops.
    """
    dist = torch.cdist(translations, translations)
    adj = dist < radius
    adj.diagonal(dim1=-2, dim2=-1).fill_(False)

    if valid is not None:
        adj = adj & valid[:, :, None] & valid[:, None, :]

    if max_neighbors is not None and max_neighbors > 0:
        k = min(int(max_neighbors), adj.shape[-1] - 1)
        if k > 0:
            masked_dist = dist.masked_fill(~adj, 1e9)
            idx = masked_dist.topk(k, dim=-1, largest=False).indices
            sparse = torch.zeros_like(adj)
            sparse.scatter_(-1, idx, True)
            adj = (sparse | sparse.transpose(-1, -2)) & adj

    return adj


# ---------------------------------------------------------------------------
# Discrete exterior calculus
# ---------------------------------------------------------------------------


def connection_forms(frames: Tensor, adj: Tensor) -> Tensor:
    """Discrete connection 1-forms on graph edges.

    ``omega_ij = log(E_i^T E_j)`` represented as a rotation vector.  Entries
    for non-edges are exactly zero.
    """
    relative = frames.transpose(-1, -2).unsqueeze(-3) @ frames.unsqueeze(-4)
    omega = so3_log(relative)
    return omega * adj[..., None].to(omega.dtype)


def connection_laplacian(frames: Tensor, adj: Tensor) -> Tensor:
    """Build the normalised block connection Laplacian ``L``.

    The matrix is returned as a dense ``[B, 3L, 3L]`` tensor.  It is Hermitian:
    the ``(i, j)`` block is ``E_i^T E_j`` and the ``(j, i)`` block is its
    transpose.  This is the operator used by the connection heat kernel.
    """
    b, length, _, _ = frames.shape
    blocks = frames.transpose(-1, -2).unsqueeze(-3) @ frames.unsqueeze(-4)
    blocks = blocks * adj[..., None, None].to(blocks.dtype)

    degree = adj.sum(dim=-1).to(frames.dtype)
    inv_sqrt = degree.clamp_min(1.0).rsqrt()
    blocks = blocks * (
        inv_sqrt[:, :, None, None] * inv_sqrt[:, None, :, None]
    )[..., None]

    # [B, L, L, 3, 3] -> [B, 3L, 3L]
    block_matrix = blocks.permute(0, 1, 3, 2, 4).reshape(b, 3 * length, 3 * length)
    eye = torch.eye(3 * length, dtype=frames.dtype, device=frames.device)
    return eye.unsqueeze(0) - block_matrix


def connection_heat_kernel(
    frames: Tensor,
    adj: Tensor,
    tau: float = 0.5,
    max_exact_residues: int = 192,
    diffusion_steps: int = 8,
) -> Tensor:
    """Connection heat-transport affinity between residues.

    Computes ``D(i, j) = || exp(-tau L)_{ij} ||_F`` where ``L`` is the block
    connection Laplacian.  For long proteins we use a stable explicit diffusion
    approximation to keep memory bounded; the result is only used as an
    attention bias, so this approximation is sufficient in practice.
    """
    laplacian = connection_laplacian(frames, adj)
    b, n, _ = laplacian.shape
    length = n // 3

    if length <= max_exact_residues:
        heat = torch.matrix_exp(-tau * laplacian)
    else:
        eye = torch.eye(n, dtype=frames.dtype, device=frames.device)
        heat = eye.unsqueeze(0).expand(b, n, n).clone()
        step = tau / max(diffusion_steps, 1)
        for _ in range(diffusion_steps):
            heat = heat - step * (laplacian @ heat)

    heat = heat.reshape(b, length, 3, length, 3)
    # Block (i, j) is a 3x3 matrix; take its Frobenius norm.
    return heat.pow(2).sum(dim=(2, 4)).sqrt()


def triangle_curvature(
    frames: Tensor,
    adj: Tensor,
    omega: Optional[Tensor] = None,
    max_neighbor_slots: int = 24,
) -> tuple[Tensor, Tensor]:
    """Aggregate discrete curvature 2-forms onto residues.

    For a contact triangle ``(i, j, k)`` the curvature is the discrete exterior
    derivative of the connection,

    ``Omega_f = omega_ij + omega_jk + omega_ki + 1/2 [omega_ij, omega_jk]``.

    This implementation vectorises over all vertices.  Neighbour lists are
    padded to at most ``max_neighbor_slots`` entries; each triangle is
    canonicalised with sorted vertices and counted once.  The curvature is
    scattered back to the three vertices and averaged.

    Returns:
        ``curvature``: ``[B, L, 3]`` per-residue curvature vectors.
        ``energy``: ``[B, L]`` Frobenius norms of the curvature vectors.
    """
    if omega is None:
        omega = connection_forms(frames, adj)

    b, length, _, _ = frames.shape
    device, dtype = frames.device, frames.dtype
    curvature = torch.zeros(b, length, 3, dtype=dtype, device=device)
    counts = torch.zeros(b, length, dtype=dtype, device=device)

    # Number of neighbours to consider per residue.  Contact graphs at 8 A
    # typically have 8-12 neighbours; the cap keeps the pair tensor bounded.
    max_degree = int(adj.sum(dim=-1).max().item()) if adj.numel() else 0
    slots = min(max_degree, max_neighbor_slots)
    if slots < 2:
        return curvature, curvature.norm(dim=-1)

    values, neighbours = torch.topk(
        adj.to(torch.uint8), k=slots, dim=-1, largest=True, sorted=False
    )
    neighbour_mask = values.bool()
    p, q = torch.triu_indices(slots, slots, offset=1, device=device)
    n_pairs = p.numel()

    j = neighbours[..., p]  # [B, L, P]
    k = neighbours[..., q]
    valid = neighbour_mask[..., p] & neighbour_mask[..., q] & (j != k)
    batch_index = torch.arange(b, device=device)[:, None, None]
    valid = valid & adj[batch_index, j, k]

    i = torch.arange(length, device=device).view(1, length, 1).expand(b, length, n_pairs)
    vertices = torch.stack([i, j, k], dim=-1)
    a, bb, c = vertices.sort(dim=-1).values.unbind(-1)
    keep = valid & (i == a)

    omega_ab = omega[batch_index, a, bb]
    omega_bc = omega[batch_index, bb, c]
    omega_ca = omega[batch_index, c, a]
    tri_curvature = (
        omega_ab
        + omega_bc
        + omega_ca
        + 0.5 * torch.linalg.cross(omega_ab, omega_bc, dim=-1)
    ) * keep[..., None].to(dtype)

    # Scatter each triangle to its three vertices.
    flat_tri = tri_curvature.reshape(b * length * n_pairs, 3)
    base = (torch.arange(b, device=device)[:, None, None] * length).expand(b, length, n_pairs)
    flat_keep = keep.reshape(-1).to(dtype)
    curvature_flat = curvature.view(b * length, 3)
    counts_flat = counts.view(-1)
    for target in (a, bb, c):
        index = (base + target).reshape(-1)
        curvature_flat.index_add_(0, index, flat_tri)
        counts_flat.index_add_(0, index, flat_keep)

    curvature = curvature / counts.clamp_min(1.0)[..., None]
    energy = curvature.norm(dim=-1)
    return curvature, energy


def holonomy_defect(energy: Tensor) -> Tensor:
    """Per-residue transport-frustration score.

    For frame-exact backbone coordinates the *group* holonomy around a graph
    triangle telescopes to the identity.  The non-abelian part of the discrete
    curvature, however, does not vanish; we expose its norm as the defect used
    by geometry-aware masking.  This matches the paper's practical use of
    ``D_i`` as a localiser of hinges and loop crossovers.
    """
    if energy.numel() == 0:
        return energy
    scale = energy.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return energy / scale


def connection_inner_product(curvature: Tensor) -> Tensor:
    """Pairwise curvature inner product ``Tr(Omega_i Omega_j^T)``.

    Curvature vectors are L2-normalised first, making the bias scale-free and
    numerically well behaved across proteins of different sizes.
    """
    norm = curvature.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    unit = curvature / norm
    return unit @ unit.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Backbone invariant features and integration targets
# ---------------------------------------------------------------------------


def dihedral_angle(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Signed dihedral angle (radians) for four ``[..., 3]`` point tensors."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = normalize(b1)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
    x = (v * w).sum(dim=-1)
    y = (torch.linalg.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def bond_angle(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Angle at ``b`` between ``a-b`` and ``c-b``."""
    u = normalize(a - b)
    v = normalize(c - b)
    return torch.acos((u * v).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def backbone_features(
    coords: Tensor,
    frames: Tensor,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """Rotation/translation-invariant per-residue backbone features.

    Features: sin/cos of phi, psi and omega, three bond lengths, and two bond
    angles.  They are deliberately simple; the geometric signal in the method
    comes from the differential forms, not from hand-crafted features.
    """
    b, length, _, _ = coords.shape
    n, ca, c = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    device, dtype = coords.device, coords.dtype

    def zeros():
        return torch.zeros(b, length, device=device, dtype=dtype)

    phi = zeros()
    psi = zeros()
    omega = zeros()
    if length >= 3:
        phi[:, 1:-1] = dihedral_angle(c[:, :-2], n[:, 1:-1], ca[:, 1:-1], c[:, 1:-1])
    if length >= 2:
        psi[:, :-1] = dihedral_angle(n[:, :-1], ca[:, :-1], c[:, :-1], n[:, 1:])
        omega[:, :-1] = dihedral_angle(ca[:, :-1], c[:, :-1], n[:, 1:], ca[:, 1:])

    n_ca = (n - ca).norm(dim=-1)
    ca_c = (c - ca).norm(dim=-1)
    c_n = torch.zeros_like(n_ca)
    if length >= 2:
        c_n[:, :-1] = (n[:, 1:] - c[:, :-1]).norm(dim=-1)

    angle_n_ca_c = bond_angle(n, ca, c)
    angle_ca_c_n = zeros()
    if length >= 2:
        angle_ca_c_n[:, :-1] = bond_angle(ca[:, :-1], c[:, :-1], n[:, 1:])

    features = torch.stack(
        [
            torch.sin(phi),
            torch.cos(phi),
            torch.sin(psi),
            torch.cos(psi),
            torch.sin(omega),
            torch.cos(omega),
            n_ca / 1.5,
            ca_c / 1.5,
            c_n / 1.3,
            angle_n_ca_c / math.pi,
            angle_ca_c_n / math.pi,
        ],
        dim=-1,
    )
    if valid is not None:
        features = features * valid[..., None].to(features.dtype)
    return features


def geometry_targets(
    frames: Tensor,
    translations: Tensor,
    adj: Tensor,
    omega_contact: Tensor,
    curvature: Optional[Tensor] = None,
    curvature_energy: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Build the per-residue targets used by the Cartan-Darboux decoder."""
    if omega_contact is None:
        omega_contact = connection_forms(frames, adj)

    b, length, _, _ = frames.shape
    device, dtype = frames.device, frames.dtype

    theta = torch.zeros(b, length, 3, device=device, dtype=dtype)
    if length >= 2:
        displacement = translations[:, 1:] - translations[:, :-1]
        theta[:, :-1] = torch.einsum(
            "blij,blj->bli", frames[:, :-1].transpose(-1, -2), displacement
        )

    omega_seq = torch.zeros(b, length, 3, device=device, dtype=dtype)
    if length >= 2:
        omega_seq[:, :-1] = so3_log(
            frames[:, :-1].transpose(-1, -2) @ frames[:, 1:]
        )

    degree = adj.sum(dim=-1).to(dtype).clamp_min(1.0)
    omega_contact_avg = (omega_contact * adj[..., None].to(dtype)).sum(dim=-2) / degree[
        ..., None
    ]

    if curvature is None or curvature_energy is None:
        curvature, curvature_energy = triangle_curvature(frames, adj, omega_contact)

    return {
        "theta": theta,
        "omega": omega_seq,
        "omega_contact": omega_contact_avg,
        "curvature": curvature,
        "curvature_energy": curvature_energy,
    }


def nearest_visible_anchor(visible: Tensor, valid: Tensor) -> Tensor:
    """Index of the nearest visible residue along the sequence.

    Args:
        visible: ``[B, L]`` boolean mask.
        valid: ``[B, L]`` boolean mask.

    Returns:
        ``[B, L]`` long tensor of anchor indices.
    """
    b, length = visible.shape
    anchors = torch.arange(length, device=visible.device).expand(b, length).clone()
    for batch in range(b):
        indices = torch.nonzero(visible[batch] & valid[batch], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        positions = torch.arange(length, device=visible.device)
        insert = torch.searchsorted(indices, positions)
        left = indices[(insert - 1).clamp(0, indices.numel() - 1)]
        right = indices[insert.clamp(max=indices.numel() - 1)]
        choose_left = (positions - left) <= (right - positions)
        anchors[batch] = torch.where(choose_left, left, right)
    return anchors


def integrate_sequence_paths(
    omega: Tensor,
    theta: Tensor,
    reference_frames: Tensor,
    reference_translations: Tensor,
    anchors: Tensor,
    masked: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Integrate predicted Cartan forms from a visible anchor to masked residues.

    The path follows the sequence, which is always connected.  For a masked
    residue ``m`` with anchor ``a``:

    * if ``a < m``: ``E_m = E_a exp(sum_{u=a}^{m-1} omega_u)``;
    * if ``a > m``: traverse backwards, using ``-omega_{u-1}`` for the
      connection and ``-exp(-omega_{u-1}) theta_{u-1}`` for the displacement.

    Args:
        omega: ``[B, L, 3]`` predicted per-residue connection forms (i -> i+1).
        theta: ``[B, L, 3]`` predicted soldering forms (i -> i+1, frame i).
        reference_frames: ``[B, L, 3, 3]`` ground-truth frames at anchors.
        reference_translations: ``[B, L, 3]``.
        anchors: ``[B, L]`` anchor indices from :func:`nearest_visible_anchor`.
        masked: ``[B, L]`` boolean mask of residues to integrate.
        valid: ``[B, L]`` boolean mask.

    Returns:
        Integrated frames ``[B, L, 3, 3]`` and translations ``[B, L, 3]``.
        Entries outside ``masked`` are zero.
    """
    b, length, _ = omega.shape
    device, dtype = omega.device, omega.dtype

    positions = torch.arange(length, device=device).view(1, length).expand(b, length)
    delta = positions - anchors
    steps = delta.abs()
    direction = torch.sign(delta)
    active_all = masked & valid & (steps > 0)

    frame_index = anchors[..., None, None].expand(b, length, 3, 3)
    trans_index = anchors[..., None].expand(b, length, 3)
    current_frames = reference_frames.gather(1, frame_index)
    current_trans = reference_translations.gather(1, trans_index)

    out_frames = torch.zeros_like(reference_frames)
    out_trans = torch.zeros_like(reference_translations)

    zero_step = masked & valid & (steps == 0)
    out_frames = torch.where(zero_step[..., None, None], current_frames, out_frames)
    out_trans = torch.where(zero_step[..., None], current_trans, out_trans)

    max_steps = int(steps.max().item()) if steps.numel() else 0
    for step in range(1, max_steps + 1):
        u = (anchors + direction * (step - 1)).clamp(0, length - 1)
        v = (anchors + direction * step).clamp(0, length - 1)
        forward = direction > 0

        # Connection form for the step u -> v.
        gather_omega = torch.where(forward, u, v)
        omega_step = omega.gather(
            1, gather_omega[..., None].expand(b, length, 3)
        )
        omega_step = torch.where(forward[..., None], omega_step, -omega_step)

        # Soldering form for the step, expressed in the current frame.
        theta_forward = theta.gather(1, u[..., None].expand(b, length, 3))
        omega_v = omega.gather(1, v[..., None].expand(b, length, 3))
        theta_v = theta.gather(1, v[..., None].expand(b, length, 3))
        theta_backward = -torch.einsum(
            "blij,blj->bli", so3_exp(-omega_v), theta_v
        )
        theta_step = torch.where(forward[..., None], theta_forward, theta_backward)

        # Displacement uses the frame at u, before it is advanced.
        current_trans = current_trans + torch.einsum(
            "blij,blj->bli", current_frames, theta_step
        )
        current_frames = current_frames @ so3_exp(omega_step)

        active = active_all & (steps >= step)
        out_frames = torch.where(active[..., None, None], current_frames, out_frames)
        out_trans = torch.where(active[..., None], current_trans, out_trans)

    return out_frames, out_trans
