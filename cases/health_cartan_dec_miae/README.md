# Health case: Cartan-DEC-MiAE

The real deployment bundled with Scientist3: the Cartan-DEC-MiAE geometry-aware
protein self-supervision code, wrapped in a Scientist3 contract. It runs on CPU
at a smoke-test scale and is fully deterministic.

```powershell
python -m scientist3 run health_cartan_dec_miae --run-id health-demo
python -m scientist3 inspect health-demo
python -m scientist3 show health-demo paper/final.md
```

## Layout

| Path | Role |
|---|---|
| `case.toml` | Scientist3 contract: baseline, ideas, evolved idea, ablations, supplementary experiments, budget, protected paths |
| `problem.md` | Problem statement, endpoints, promotion rule, fidelity limits |
| `code/cartan_dec/` | The Cartan-DEC-MiAE implementation (model, geometry, losses, data, probe, training) |
| `code/cartan_dec/geometry_eval.py` | Deterministic masked-geometry reconstruction endpoint used by the case |
| `code/tests/` | Unit tests for geometry, masking and the model |
| `code/quickstart.py` | Standalone quickstart retained from the case artifact |
| `docs/method.md` | Method notes and fidelity limits from the standalone case |

## What the run does

1. Trains a **vanilla masked autoencoder** baseline (no geometry biases, zero
   connection/curvature/integration losses).
2. Screens three mechanism-level ideas on 96 training chains × 8 epochs × 2 seeds:
   full Cartan-DEC, 4× transport weighting, curvature-guided masking.
3. Promotes the survivors to 160 chains × 15 epochs and evaluates one evolved
   idea (curvature-stabilised transport).
4. Removes the integration, connection-form and curvature terms and reverts the
   transport loss rebalancing, one component at a time.
5. Drafts a manuscript, reviews it (first round below threshold), runs two
   supplementary robustness experiments, reviews again, and seals a four-part
   integrity audit that re-executes the selected command.

## Honest scope

Probe accuracy is saturated on the synthetic families; the scientific signal in
this run is the masked integrated-geometry reconstruction endpoint. The run is
not evidence for TEDBench performance. See `problem.md` and `docs/method.md`.
