# Toy regression case

The fast, dependency-free reference case for Scientist3. It uses a synthetic
regression target and a small NumPy harness, so a full pipeline run takes well
under a minute on a laptop.

```powershell
python -m scientist3 run toy_regression --runs-dir runs --run-id toy-demo
python -m scientist3 inspect toy-demo --runs-dir runs
```

## What it exercises

| Stage | Evidence produced |
|---|---|
| Contract | `contract.json`, `contract.md`, protected-file hashes |
| Limitations | four falsifiable entries, three mechanism-level |
| Ideation | four seed ideas, one engineering refinement, one evolved idea |
| Subset screening | linear vs polynomial vs Fourier vs hybrid, 2 seeds |
| Full grid | 300- and 1200-sample comparisons with a 10% promotion rule |
| Ablations | Fourier map, phase offsets, bandwidth, regularization |
| Drafting | Markdown manuscript with `Table 1`-`Table 3` |
| Review | scripted ICLR-style review, rebuttal supplementaries, second review |
| Meta-review | accept/revise gate over the review score and audit preview |
| Integrity | score re-execution, spec compliance, references, method-code alignment |

The expected outcome is that the wide cubic Fourier-polynomial variant wins, all
four ablations degrade the primary metric, the first review is below threshold,
the two supplementary robustness runs raise it above threshold, and the audit
passes.

## Files

* `case.toml` - full problem contract, commands, budget and protected paths.
* `problem.md` - problem statement and evaluation protocol.
* `harness/run.py` - frozen NumPy harness (data, features, ridge, metrics).

Run artifacts are written to `runs/<run-id>/` and are ignored by git. See
`docs/case-guide.md` for the contract schema in detail.
