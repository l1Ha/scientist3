# Method notes

This repository is an independent reimplementation of **Cartan-DEC-MiAE**, the
method described in the ScientistTwo paper *Cartan Moving Frames and Discrete
Exterior Calculus for Self-Supervised Protein Structure Representation*.  The
paper itself is available from the ScientistTwo gallery:

<https://scientist-two.github.io/#applications/Health%20(Cartan-DEC-MiAE)>

The upstream data/model ecosystem is the open-source **TEDBench** project:

<https://github.com/BorgwardtLab/TEDBench>

No code from the paper authors was available; this implementation follows the
paper's equations and the MiAE design exposed by TEDBench.

## 1. Residue frames

For residue `i`, let `N_i`, `CA_i`, `C_i` be backbone atom coordinates.  The
moving frame is

```
e1 = normalize(C_i - CA_i)
e2 = normalize((N_i - CA_i) - ((N_i - CA_i) . e1) e1)
e3 = e1 x e2
E_i = [e1 e2 e3] in SO(3),    t_i = CA_i
```

`E_i` is a right-handed orthonormal frame; `t_i` is the C-alpha position.

## 2. Discrete exterior calculus on the contact graph

Build a graph with an edge `(i, j)` whenever `||t_i - t_j|| < r_c` (default
`r_c = 8 A`).  On an edge the discrete **connection 1-form** is

```
omega_ij = log(E_i^T E_j) in so(3)
```

represented as a rotation vector.  For a contact triangle `(i, j, k)` the
discrete **curvature 2-form** is the exterior derivative of the connection:

```
Omega_f = omega_ij + omega_jk + omega_ki + 1/2 [omega_ij, omega_jk]
```

where `[a, b] = a x b` in the rotation-vector representation.  Curvature is
aggregated onto each vertex (averaged over incident triangles) and exposed as

```
Omega_i in R^3,    curvature_energy_i = ||Omega_i||_2
```

The per-residue **holonomy defect** used for geometry-aware masking is the
normalised curvature energy.  This deserves a note: for frame-exact backbone
coordinates the group holonomy around any closed graph loop telescopes to the
identity (`E_i^T E_j E_j^T E_k ... E_i = I`).  The paper explicitly says the
true group holonomy vanishes on frame-exact coordinates.  What remains is the
non-abelian part of the discrete curvature, which is exactly what the
`Omega_f` formula measures and what the paper uses in practice to localise
hinges and loop crossovers.  The implementation therefore uses `||Omega_i||`
as the defect score rather than pretending that the trivial group holonomy is
informative.

## 3. Connection heat transport

Let `A` be the block adjacency with `A_ij = E_i^T E_j` on edges and `D` the
scalar degree.  The normalised block connection Laplacian is

```
L = I - D^{-1/2} A D^{-1/2}
```

and the heat-transport affinity is

```
D_heat(i, j) = || (exp(-tau L))_ij ||_F
```

For proteins up to 192 residues the matrix exponential is computed exactly
with `torch.matrix_exp`; longer chains use a stable explicit diffusion
approximation.  The affinity is only used as an attention bias, so the
approximation is not on the critical path for the reconstruction targets.

## 4. Sparse, gauge-invariant encoder

Only visible residues are passed to the encoder.  Each visible residue gets
rotation/translation-invariant features (sin/cos of phi, psi, omega, three bond
lengths, two bond angles) plus a sinusoidal positional embedding.  A stack of
pre-LN Transformer blocks applies attention with two additive geometric logit
biases:

```
A_ij = softmax( Q_i K_j^T / sqrt(d)
                + w_curv * <Omega_i, Omega_j>
                + w_heat * MLP(RBF(D_heat(i, j)))
                + [optional] w_dist * MLP(RBF(||t_i - t_j||)) )
```

`<Omega_i, Omega_j>` uses L2-normalised curvature vectors.  The scales are
learnable per attention head, so the model can switch each geometric channel
on or off.  A distance bias is implemented but disabled by default.

## 5. Cartan-Darboux decoder and path integration

The decoder receives the visible encoder latents plus learned mask tokens at
masked positions and predicts, per residue:

* connection 1-form `omega_i` (the form from `i` to `i+1`);
* curvature 2-form `Omega_i`;
* soldering form `theta_i = E_i^T (t_{i+1} - t_i)`;
* anchor-relative position and a 6D rotation.

For a masked residue `m`, let `a(m)` be the nearest visible residue along the
sequence.  The predicted forms are integrated from the anchor:

```
E_m = E_a exp(sum_{u=a}^{m-1} omega_u)
t_m = t_a + sum_{u=a}^{m-1} E_u theta_u
```

with the symmetric backward traversal when `a > m`.  This is the discrete
Cartan structure equation on a sequence path.  A sequence path is used because
it is always connected, including for aggressive span masking; a contact-graph
shortest path is a natural extension.

The integrated `(E_m, t_m)` are compared with ground truth in the anchor frame,
alongside the direct form predictions.  This is the term that forces the
encoder to localise masked geometry from sparse long-range context rather than
interpolating neighbours.

## 6. Mask-centric loss

Every term is evaluated only on masked residues:

```
L = lambda_theta * L_theta
  + lambda_pos   * L_pos
  + lambda_rot   * L_rot
  + lambda_omega * L_omega
  + lambda_curv  * L_Omega
  + lambda_int   * L_integration
```

* `L_theta`, `L_omega`, `L_Omega`: MSE on soldering, connection and curvature
  targets (curvature includes a `0.5 *` energy term).
* `L_pos`: MSE on anchor-relative positions, divided by 10 A for numerical
  balance.
* `L_rot`: `1 - cos(theta)` between predicted and target rotations.
* `L_integration`: the same position/rotation losses after path integration.

## 7. Masking strategies

* `random`: uniform random residues.
* `span`: contiguous sequence spans (the paper's strongest masking strategy at
  scale).
* `geometry`: spans centred on high curvature / holonomy-defect residues, with
  hard top-k or soft Boltzmann sampling.  The paper reports that this does
  **not** beat contiguous span masking; it is included as a controlled
  baseline and for reproducibility of the negative result.

## 8. Scope and deviations

The code is engineered to make the method auditable and runnable, not to claim
a reproduction of the paper's full-scale numbers.  In particular:

* The paper trains a 38M-parameter encoder on the full 369,740-structure
  TEDBench corpus.  This repository supports that dataset through
  `TEDBenchDataset`, but the checked-in demo uses synthetic backbones so it can
  run on a laptop.
* The encoder here is a clean invariant-feature Transformer, not the ESM3
  geometric-attention stack used by MiAE/TEDBench.  The interface is deliberately
  modular: `CartanEncoder` can be swapped for a TEDBench MiAE encoder.
* Path integration is along the sequence rather than a contact-graph shortest
  path.  Both implement the same discrete structure equations; the sequence
  path is simpler and always defined under span masking.
* The exact group holonomy is trivial for frame-exact coordinates; see §2.

These choices are documented so that reported synthetic smoke-test numbers are
never confused with the paper's full-corpus results.  The full-scale protocol
(369,740 pretraining structures, 965-way CATH-T probe, TEDBench test and CATH
4.4 external test) is available through the scripts and data adapters.
