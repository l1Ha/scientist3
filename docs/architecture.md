# Architecture

Scientist3 is an independent, domain-agnostic implementation of the
ScientistTwo research workflow: a bounded loop that turns a *problem contract*
into experiments, a manuscript, simulated peer review, and a re-executable
integrity audit.

The core design rule is a hard separation between **creative text** and
**trusted evidence**:

> Every promotion, rejection, score and numeric claim is computed from
> experiment observations stored on disk. Model text (scripted or live) may
> shape prose and suggest candidates, but it never decides anything measurable.

## Stage graph

```text
S0 contract      -> contract.json, contract.md, protected hashes
S1 limitations   -> limitations.md
S2 ideation      -> ideas.md (seed ideas + novelty report)
S3 screening     -> artifacts/screening.json (subset runs, engineering rounds)
S4 full grid     -> artifacts/full.json (full runs, evolution, selection)
S5 ablations     -> artifacts/ablations.json (component attribution)
S6 drafting      -> paper/paper_v1.md
S7 review        -> artifacts/review.json, paper/paper_v2.md, paper/reviews.md
S8 meta-review   -> artifacts/meta_review.json
S9 integrity     -> audit/audit.json (four-part audit)
summary          -> summary.json, paper/final.md
```

Each stage is:

* **bounded** — every loop limit lives in the case's `[budget]` table;
* **idempotent** — a completed stage is loaded from its artifact instead of
  re-executed, so `scientist3 resume <run-id>` continues an interrupted run;
* **sealed** — the stage writes one JSON artifact plus human-readable Markdown,
  and records `events.jsonl` entries for every experiment.

## Stages in detail

### S0 Contract
Loads `case.toml`, validates required fields, and captures SHA-256 hashes of the
declared `protected_paths` (evaluation harness, metrics code, problem
statement). The hashes are re-checked by the audit; any modification after the
contract is treated as tampering.

### S1 Limitations
Produces a set of falsifiable limitations. The scripted agent uses the contract;
live mode may propose replacements but falls back to the contract when parsing
fails. A verifier requires at least three well-shaped limitations with at least
two mechanism-level entries.

### S2 Ideation
Returns the seed ideas with a novelty report (delta from closest work,
references). In live mode a model may propose ideas, but each must carry an
executable command that the runner can expand.

### S3 Subset screening
Runs the baseline and each idea on a bounded subset (`subset_samples`,
`subset_epochs`) for every declared seed. A candidate is `accept`ed only if the
primary metric clears `min_relative_gain` **and** every seed moves in the same
direction. Candidates may declare `engineering_commands`: a diagnosed failure
spends one engineering round before acceptance or pruning.

### S4 Full grid + evolution + selection
Re-runs the subset survivors on the full budget with the exact command that
passed screening, then evaluates evolved ideas (contract-declared descendants of
screening feedback) until `min_successful_ideas` is reached. The selector ranks
all successful full-grid entries by the primary metric and stores the winner
together with its command template.

### S5 Ablations
Removes one declared component at a time from the selected method and compares
against the selected run using `ablation_min_gain` (a laxer threshold than
promotion). Attribution is `clean` when at least 75% of drop-direction
ablations show a measurable degradation.

### S6 Drafting
`paper.py` builds the manuscript deterministically: abstract, related work,
method (including the evolved idea), setup, subset table, full table, ablation
table, supplementary section, limitations, and reproducibility notes. Every
number is formatted from stored `ExperimentResult` objects.

### S7 Review, rebuttal, and review again
The reviewer scores the manuscript from the evidence summary only (metrics
count, seeds, significance, ablation cleanliness, references, external
validity, supplementaries). If the score is below `review_threshold`, the
rebuttal planner selects at most two declared supplementary experiments, runs
them, appends their results to the manuscripts, and re-reviews — for at most
`max_review_rounds`.

### S8 Meta-review
Accepts only when the final review score meets the threshold **and** the
integrity preview (protected hashes, observation count) is intact; otherwise the
decision is `revise`.

### S9 Integrity audit
Four independent checks (`audit.py`):

1. **score verification** — re-executes the selected command and compares stored
   vs re-executed metrics within a tolerance;
2. **spec compliance** — re-reads each stored observation and checks the command
   still matches the contract-derived template and that runs completed;
3. **references** — verifies resolvable identifiers (DOI/URL) offline, or
   resolves them online when asked;
4. **method-code alignment** — the paper must name every declared ablation, the
   ablation table must exist, and the observed experiments must cover the
   manuscript's tables.

The audit is itself re-runnable: `scientist3 audit <run-id> --reexecute`.

## Run state and resume

`RunState` rewrites `state.json` atomically (with a retry loop for Windows
file-lock transients) after every transition, and appends to `events.jsonl`.
Stage artifacts are plain JSON under `artifacts/`. `ResearchPipeline.run()`
checks `is_complete(stage)` before every stage, so `resume` never re-runs
completed experiments and produces byte-identical summaries on the second pass
(covered by `tests/test_pipeline.py`).

## Runner and comparison semantics

`runner.py` expands command templates with placeholders
(`{python}`, `{samples}`, `{test_samples}`, `{epochs}`, `{seed}`, `{out}`),
executes them in the case's `code_dir` with no shell, reads the declared metrics
file, resolves dot paths (`"reconstruction.integrated_position_rmse"`) and
stores an `ExperimentResult`.

`compare_results` returns `better` / `worse` / `tie` using:

* mean of the primary metric across seeds vs the reference run;
* a relative threshold — an explicitly passed threshold (promotion or ablation)
  wins over the metric's `min_gain`, which wins over the 2% default;
* seed consistency: with two or more seeds, every pairwise candidate/reference
  difference must point the same way.

## Trust boundary

| Component | May influence | May not influence |
|---|---|---|
| Scripted/live agents | prose, candidate commands, supplementary choice | scores, promotions, thresholds, audit checks |
| Experiment code | its own metrics file | thresholds, selection, review scores |
| Reviewer | score from evidence dict | raw metrics (never reads run logs) |
| Integrity audit | verdict from stored observations and re-execution | paper prose |

## Repository layout

```text
scientist3/            pipeline package
  core.py              case schema, Budget, state, hashing
  runner.py            command expansion, execution, comparison
  agents.py            scripted/live stage agents
  llm.py               LLM backends (scripted default, OpenAI-compatible)
  paper.py             deterministic manuscript + evidence builder
  audit.py             four-part integrity audit
  pipeline.py          stage orchestration and resume
  cli.py               `scientist3` command line
cases/
  toy_regression/      dependency-light reference case (NumPy only)
  health_cartan_dec_miae/  real deployment: Cartan-DEC-MiAE on synthetic folds
tests/                 unit + end-to-end pipeline tests
docs/                  this document and the guides below
```

## Related documents

* [case-guide.md](case-guide.md) — the `case.toml` schema in full.
* [fidelity.md](fidelity.md) — what is and is not faithful to ScientistTwo.
* [live-models.md](live-models.md) — running creative agents on a live model.
