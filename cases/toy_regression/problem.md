# Structured regression with random Fourier features

## Problem

Given three real inputs `x1, x2, x3 ~ Uniform(-2, 2)`, predict a scalar target
that mixes a periodic component, a smooth quadratic term and an interaction.

```text
y = sin(2.2 x1) + 0.45 x2^2 + 0.30 x1 x3 + noise,   noise ~ N(0, 0.05^2).
```

The target deliberately mixes a periodic component, a smooth quadratic term and
an interaction. A linear model cannot represent any of them; a low-degree
polynomial can represent the last two; random Fourier features can represent all
three in expectation but depend on the kernel bandwidth and ridge
regularization.

## Evaluation protocol

* The harness (`harness/run.py`) is frozen and protected: it defines the data
  generator, the deterministic 75/25 train/test split, the feature maps, the
  ridge solver and the metric computation.
* A command receives the number of training samples and a seed; the test split is
  generated from the same seed and is never used for fitting.
* Reported metrics are held-out root-mean-square error (`rmse`, lower is better)
  and the coefficient of determination (`r2`, higher is better). `rmse` is the
  primary endpoint.
* A candidate is promoted only when its mean `rmse` improves on the baseline by
  at least 10% relative and every seed moves in the same direction. The full-grid
  evaluation repeats the comparison on a larger sample budget.
* Component ablations remove one mechanism at a time from the selected model and
  must reproduce a measurable degradation.

## Scope

This case is a reproducible smoke test for the Scientist3 loop. It is not a
novel result about kernel methods: random Fourier features are classical
(Rahimi & Recht, 2007), and the expected qualitative ordering is known. Its role
is to exercise every stage of the pipeline with short, deterministic CPU
experiments and a fully traceable evidence graph.
