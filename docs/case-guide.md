# Case contract guide

A *case* is one scientific question wrapped in a machine-checkable contract. It
lives in a directory containing `case.toml`, an optional problem statement, the
code to be evaluated, and an optional README. Scientist3 knows nothing about the
domain; everything it needs is declared here.

```text
cases/my_case/
  case.toml        # the contract (required)
  problem.md       # problem statement used in drafting (recommended)
  code/            # experiment code; commands run with this as cwd
  README.md        # human notes (optional)
```

> **TOML gotcha.** Top-level keys (notably `protected_paths`) must appear
> *before* the first `[table]` header. TOML folds keys that follow a table
> header into that table, which fails schema validation with a confusing error.

## Minimal contract

```toml
protected_paths = ["harness/run.py", "problem.md"]

[case]
id = "my_case"
title = "Does method X beat baseline Y on task Z?"
domain = "general"
problem = "problem.md"
code_dir = ""            # "" means the case root; otherwise e.g. "code"
metrics_file = "metrics.json"
keywords = ["..."]

[baseline]
name = "linear ridge"
command = "{python} harness/run.py --method linear --n {samples} --seed {seed} --out {out}"

[[metrics]]
name = "rmse"
direction = "min"
primary = true
path = "rmse"

[budget]
seeds = [0, 1]
```

## `[case]`

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Stable identifier; defaults to the directory name |
| `title` | yes | Human title used in the manuscript |
| `domain` | yes | Free-form domain label |
| `problem` | yes | Path (relative to the case root) of the problem statement |
| `code_dir` | no | Working directory for commands; `""` = case root |
| `metrics_file` | no | File each experiment writes (default `metrics.json`) |
| `keywords` | no | Free-form list |
| `notes` | no | Scope note rendered into the manuscript |

## `[baseline]`

`name` and `command`. The baseline is run on the subset and on the full grid
with the same seeds as the candidates, and is the reference for promotion.

## `[[metrics]]`

Each entry declares one scalar endpoint:

| Key | Meaning |
|---|---|
| `name` | Metric name used in tables, evidence and reviews |
| `direction` | `"min"` or `"max"` |
| `primary` | Exactly one metric may be primary; it drives all decisions |
| `path` | Dot path inside `metrics_file`, e.g. `"reconstruction.integrated_position_rmse"`; empty = top-level key equal to `name` |
| `min_gain` | Optional metric-level promotion threshold (relative). An explicitly passed threshold — promotion or ablation — always wins over it |

Metric values must resolve to finite numbers; a run that produces none of the
declared metrics is an experiment failure, not a zero.

## `[budget]` — bounded loops

| Key | Default | Meaning |
|---|---|---|
| `seeds` | `[0]` | Seeds used for screening and full evaluation |
| `subset_samples` / `subset_epochs` | 256 / 1 | Screening budget |
| `full_samples` / `full_epochs` | 1024 / 2 | Full-grid budget |
| `test_samples` | 128 | Placeholder value passed to commands if referenced |
| `top_ideas` | 2 | Seed ideas screened (in contract order) |
| `max_screening_rounds` | 2 | Screening rounds incl. engineering |
| `max_evo_rounds` | 2 | Evolution rounds available in S4 |
| `min_successful_ideas` / `max_successful_ideas` | 1 / 3 | Full-grid evaluations to run (evolved ideas fill the gap) |
| `max_ablation_refinements` | 1 | Extra ablation rounds if attribution is dirty |
| `max_review_rounds` | 2 | Review/bullet rounds |
| `review_threshold` | 8.0 | Meta-review acceptance bar |
| `max_meta_refinements` | 1 | Meta-review revision budget |
| `timeout_seconds` | 900 | Per-experiment timeout |
| `min_relative_gain` | 0.02 | Promotion threshold (relative) |
| `ablation_min_gain` | 0.02 | Ablation threshold (relative); keep it laxer than promotion |

## `[[limitations]]`, `[[ideas]]`, `[[evolved_ideas]]`

```toml
[[limitations]]
id = "L1"
text = "One falsifiable sentence about a mechanism."
mechanism = true          # false marks an empirical limitation
test = "How the limitation is demonstrated."

[[ideas]]
id = "I1"
name = "short name"
novelty = "low" | "medium" | "high"
targets = ["L1"]
mechanism = "What it does."
delta = "How it differs from closest work."
risk = "What could go wrong."
command = "{python} ... {out}"
subset_command = ""                # optional screening-only variant
engineering_commands = []          # remediation attempts, in order
references = [{ title = "...", doi = "..." }]

[[evolved_ideas]]
id = "I4"
# same fields; evaluated after seed ideas on the full grid
```

Rules enforced at load time: every idea and evolved idea needs a `command`; at
most one primary metric; a baseline command is mandatory.

## `[[ablations]]`

```toml
[[ablations]]
id = "A1"
component = "transport integration loss"
description = "Remove X while keeping everything else."
command = "{python} ... --flag 0 --out {out}"
direction = "drop"    # removing the component should hurt the primary metric
```

Attribution is `clean` when at least 75% of drop-direction ablations show a
degradation of at least `ablation_min_gain` relative to the selected run.

## `[[supplementary]]`

Rebutted experiments, run only when a review scores below the threshold:

```toml
[[supplementary]]
id = "R1"
question = "Does the result survive distribution shift?"
description = "What changes in this run."
command = "{python} ... --shift 0.75 --out {out}"
```

At most two are used per review round, and each at most once per run.

## `[variables]`

Free-form table carrying:

| Key | Meaning |
|---|---|
| `external_validity` | `true` adds the external-validation point to reviewer scores |
| `references` | List of `{ title, url \| doi }` used in related work and audited |
| `refined_command` | Optional template for a post-ablation refinement attempt |

## Command placeholders

Commands are plain strings, tokenized without a shell (quotes are honoured).
Supported placeholders:

| Placeholder | Value |
|---|---|
| `{python}` | Current interpreter (`sys.executable`) |
| `{samples}` | `subset_samples` or `full_samples` depending on the split |
| `{test_samples}` | `test_samples` |
| `{epochs}` | `subset_epochs` or `full_epochs` |
| `{seed}` | Seed of this run |
| `{out}` | Absolute output directory of this run |

Unknown placeholders abort the run before anything executes.

## `protected_paths`

Paths relative to the case root whose hashes are captured at the contract stage
and re-verified by the audit. Protect the data generator, splits, metric
implementation, masking code — anything a candidate could otherwise tamper
with. Changing a protected file invalidates the audit, by design.

## Operating a case

```powershell
scientist3 list-cases
scientist3 validate my_case
scientist3 run my_case --run-id demo-1
scientist3 inspect demo-1
scientist3 show demo-1 paper/final.md
scientist3 audit demo-1 --reexecute
scientist3 resume demo-1

# shrink a device budget for a quick loop
scientist3 run my_case --run-id smoke --set seeds=0 --set subset_samples=64
```

`--set key=value` overrides any `[budget]` field; values are parsed as JSON-ish
scalars, comma lists, or strings.

## Writing a good case

* One primary metric that mechanisms can actually move. If every variant
  saturates an endpoint, switch to one that measures the mechanism (the health
  case uses masked reconstruction instead of saturated probe accuracy).
* Thresholds from a measured noise floor, not taste. Run the baseline twice
  with different seeds first.
* Ablations that remove *one* component at a time; keep several so a single
  weak component cannot make attribution dishonest.
* Commands that are deterministic given `{seed}`. No downloads in the default
  path; keep external data behind an opt-in flag.
* Protect the evaluation code before publishing results.
