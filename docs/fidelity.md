# Fidelity: what Scientist3 does and does not reproduce

Scientist3 is an independent implementation of the ScientistTwo workflow
(project site: <https://scientist-two.github.io/>). It is **not** the authors'
code, and it makes no claim to reproduce their reported results. This document
states honestly what is faithful, what is approximated, and where the bundled
cases stand relative to their scientific claims.

## Workflow mapping

| ScientistTwo workflow element | Scientist3 implementation | Fidelity |
|---|---|---|
| Problem contract with decoded problem statement | `case.toml` + `problem.md`, hashed protected paths | Faithful in structure; the contract is hand-authored, not extracted from the paper |
| Extract limitations that drive ideation | S1, contract-declared, verifier gates on count/mechanism/shape | Faithful as a contract mechanism; content comes from the case author |
| Generate candidate ideas | S2, contract-declared seeds; live mode can propose | Scripted mode is deterministic by design; novelty is declared, not searched |
| Screening on a bounded subset | S3 with engineering rounds and seed-consistency | Faithful |
| Full-set evaluation and evolution | S4, evolved ideas declared in the contract | Bounded to `max_evo_rounds`; evolution is contract-level, not open-ended hyperparameter search |
| Component ablations | S5 with a cleanliness ratio and own threshold | Faithful |
| Manuscript drafting | S6, deterministic builder from stored observations | Faithful for tables/numbers; prose is templated, not "written" |
| Review and rebuttal loop | S7, evidence-derived scoring; at most two declared supplementaries per round | Simplified: one scripted reviewer (or one live reviewer), no multi-reviewer ensemble or rolling discussion |
| Meta-review gate | S8, threshold + integrity preview | Faithful as a gate |
| Integrity verification | S9, four checks incl. score re-execution | This implementation's own design; ScientistTwo's public site describes an integrity/verification spirit, not this exact gate set |

## Deliberate simplifications

* **Scripted agents by default.** No network, no API keys, fully deterministic.
  Live OpenAI-compatible mode changes only the creative agents
  ([live-models.md](live-models.md)); critics and the audit never read model
  text, so scores and promotions cannot be talked up.
* **No literature search.** Related work comes from contract-declared
  references; the audit checks identifiers resolve (offline shape check, or an
  actual online fetch with `--online-refs`).
* **No unbounded loops.** Every iteration count is a budget field, and each
  stage is resumable from its sealed artifact.
* **One reviewer score function.** The scripted reviewer scores six binary-ish
  evidence facts; this is a workflow signal, not a peer-review simulation.

## Case-level fidelity

### `toy_regression` — workflow demonstration, not a scientific claim

A NumPy-only synthetic regression problem (periodic + quadratic + interaction
terms) evaluated against a frozen harness. Random Fourier features are classical
(Rahimi & Recht, 2007), and the qualitative ordering (linear < polynomial <
Fourier < hybrid) is expected. The case exists so the entire pipeline, audit
included, runs on a laptop in under a minute. Do not cite it as a result.

### `health_cartan_dec_miae` — CPU smoke deployment of a real method

The code is the standalone Cartan-DEC-MiAE case artifact (also published as
[`l1Ha/cartan-dec-miae`](https://github.com/l1Ha/cartan-dec-miae)): Cartan
moving frames, discrete connection and curvature forms, geometry-aware
attention biases, a Cartan–Darboux decoder, and a mask-centric objective.

The Scientist3 run is deliberately small: 96/160 synthetic chains, 8/15 epochs,
a 128-dimensional 4-layer encoder. Limits:

* the probe endpoint saturates on the four synthetic families (≈95–100% for
  every variant), so the run's primary endpoint is **masked integrated
  reconstruction error**, which directly measures the transport objective;
* conclusions apply to the synthetic smoke benchmark only — they are **not**
  evidence for TEDBench performance or for the paper's reported numbers;
* the documented next step is the full TEDBench recipe (large encoder, long
  training), which this artifact does not run.

The `problem.md` and README of the case repeat these limits where a reader will
see them.

## What the integrity audit guarantees

For any run, the audit re-executes the selected command and verifies that
stored observations reproduce within tolerance; checks every observation against
the contract templates; checks reference identifiers; and checks that the
manuscript references every declared ablation and is covered by stored
observations. It does **not** guarantee that the chosen endpoint is the right
scientific question — that is the case author's job, and the case guides say so.
