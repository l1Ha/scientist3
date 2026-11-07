# Scientist3

An independent, domain-agnostic implementation of the **ScientistTwo** research
workflow: contract → limitations → ideation → screened experiments → ablations →
manuscript → simulated review and rebuttal → meta-review → **re-executable
integrity audit**.

Scientist3 turns a small, declarative *case contract* into a bounded autonomous
research loop, and it keeps one rule absolute:

> **Every promotion, rejection, score and number comes from stored experiment
> observations — never from model text.**

Scripted agents are the default: the whole pipeline, including the health case,
runs offline with no API keys. A live OpenAI-compatible mode exists for the
creative stages only; critics and the audit cannot read model output.

Reference: [ScientistTwo project site](https://scientist-two.github.io/) ·
[`l1Ha/cartan-dec-miae`](https://github.com/l1Ha/cartan-dec-miae) (the
standalone case artifact used by the health case).

## Quickstart (under a minute, NumPy only)

```powershell
pip install -e ".[toy]"          # or: pip install -e ".[dev]" for the test suite
python -m scientist3 list-cases
python -m scientist3 run toy_regression --run-id toy-demo
python -m scientist3 inspect toy-demo
python -m scientist3 show toy-demo paper/final.md
```

Expected shape of a successful run: the wide Fourier-polynomial variant is
selected, all four ablations degrade the primary metric, the first review is
below threshold, two supplementary experiments lift the second review above it,
and the integrity audit passes. Every artifact lands in `runs/toy-demo/`
(git-ignored).

## The real deployment: Cartan-DEC-MiAE

```powershell
pip install -e ".[health]"       # torch + scikit-learn (CPU is fine)
python -m scientist3 run health_cartan_dec_miae --run-id health-demo
python -m scientist3 show health-demo paper/final.md
python -m scientist3 audit health-demo --reexecute
```

The health case wraps the Cartan-DEC-MiAE code (Cartan moving frames, discrete
connection/curvature forms, geometry-aware masking, mask-centric transport
objective) in a Scientist3 contract and runs the full loop on synthetic folds:
baseline vanilla masked autoencoder, three seed ideas, one evolved idea, four
component ablations, two rebuttal supplementaries.

**Honest scope:** this is a CPU smoke deployment. The probe endpoint saturates
on the synthetic families, so the case's primary endpoint is masked integrated
reconstruction error, which measures the transport mechanism directly.
Conclusions hold for this benchmark and budget only — not for TEDBench and not
for the paper's reported numbers.

## How the loop is bounded

| Stage | What it produces | Bounds |
|---|---|---|
| S0 contract | hashed contract, protected evaluation files | — |
| S1 limitations | falsifiable limitation set | verifier gates count/mechanism |
| S2 ideation | seed ideas + novelty report | contract-declared |
| S3 screening | subset runs with engineering rounds | `top_ideas`, `max_screening_rounds` |
| S4 full grid | full runs, evolution, selection by primary metric | `min/max_successful_ideas`, `max_evo_rounds` |
| S5 ablations | component attribution (`clean` at ≥75% degradations) | `ablation_min_gain`, `max_ablation_refinements` |
| S6 drafting | manuscript built from observations | — |
| S7 review + rebuttal | evidence-scored reviews, ≤2 supplementaries per round | `max_review_rounds` |
| S8 meta-review | accept/revise gate | `review_threshold` |
| S9 integrity | re-executed score check, spec compliance, references, method–code alignment | `--no-reexecute-audit` to skip re-execution |

Every stage is resumable (`scientist3 resume <run-id>`), and a completed stage is
loaded from its sealed artifact rather than re-run.

## Repository layout

```text
scientist3/            the Python package (core, runner, agents, paper, audit, pipeline, cli)
cases/
  toy_regression/      dependency-light reference case (frozen NumPy harness)
  health_cartan_dec_miae/  Cartan-DEC-MiAE deployment (code/, tests/, docs/)
tests/                 unit, agent and end-to-end pipeline tests
docs/
  architecture.md      stage graph, trust boundary, resume semantics
  case-guide.md        the complete case.toml schema
  fidelity.md          what is and is not faithful to ScientistTwo
  live-models.md       running creative agents on a live endpoint
```

## Tests

```powershell
pip install -e ".[dev]"
python -m pytest
```

The suite covers the contract schema, command expansion and comparison
semantics, the audit checks, scripted/live agent fallbacks, and an end-to-end
toy run plus an idempotent-resume assertion.

## Status and citation

Pre-1.0 research infrastructure. If you use it, cite the ScientistTwo paper
(see `CITATION.cff`) and note that Scientist3 is an independent implementation,
not the authors' code. Bundled cases carry their own fidelity limits in
`docs/fidelity.md`; quote them when you quote results.

MIT licensed.
