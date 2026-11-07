# Geometry-aware protein self-supervision on synthetic folds

## Problem

Protein self-supervised learning can exploit the differential geometry of the
backbone: residue frames form an `SO(3)` bundle, and the discrete connection
forms that transport a frame from one residue to another are geometric
observables. The Cartan-DEC-MiAE method predicts those connection forms and the
associated curvature under a masked-autoencoding objective, integrates the
predicted forms back into 3D coordinates, and supervises the integrated geometry.

This case asks whether that transport objective improves a frozen representation
of protein backbone geometry. The development benchmark is the bundled synthetic
four-family dataset (helix, sheet, coil, globule). It is a smoke benchmark, not a
substitute for TEDBench: all folds are short, clean and generated from known
families, and a linear probe on the frozen features saturates near 95-100%
accuracy. The primary endpoint is therefore **masked geometry reconstruction**,
which measures the mechanism the method actually adds.

## Evaluation protocol

* Each run trains a Cartan-DEC-MiAE model from scratch on a deterministic
  synthetic split, then evaluates it with a deterministic masked reconstruction
  pass (`cartan_dec.geometry_eval`) and a frozen-feature MLP probe.
* Test data are generated with a disjoint seed (`seed + 10000`); the
  reconstruction mask is fixed across variants (`mask_seed = 1234`), so
  differences come from the learned representation.
* Primary metric: `reconstruction.integrated_position_rmse` (Angstrom, lower is
  better) — the error of geometry obtained by integrating the predicted
  connection forms, relative to an anchor frame. Secondary metrics:
  direct-position RMSE and probe accuracy.
* A candidate is promoted only if its mean primary metric improves by at least
  5% relative and every declared seed moves in the same direction.
* Component ablations remove one declared component at a time — each Cartan
  term, plus the transport loss rebalancing — and at least three quarters of
  them must degrade the primary metric by at least 3% relative.
* The evaluation code, data generator, masking code and probe are protected:
  they are hash-checked at the contract and re-checked by the integrity audit.

## Scope and fidelity

This is a CPU-scale smoke deployment of the case, not a reproduction of the
Cartan-DEC-MiAE paper's reported TEDBench numbers. The published case artifact
(`l1Ha/cartan-dec-miae`) documents the full method and its fidelity limits; this
run exercises the Scientist3 loop on the same codebase at
`--samples 96/160`, `--epochs 8/15`, `--d-model 128 --depth 4`. Conclusions are
valid only for this benchmark and budget. Moving to TEDBench with the documented
training recipe is the next experimental step, not a claim of this artifact.
