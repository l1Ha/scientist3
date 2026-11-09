"""Trusted experiment execution, metric extraction and comparison.

Candidate methods never report their own scores to the controller.  A case
declares shell-free command templates; this module expands them, runs them in a
subprocess, reads the declared metrics file, and stores the observation.  All
downstream decisions use these stored observations.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .core import CaseSpec, ExperimentResult, MetricSpec, RunState, resolve_path, write_json


class ExperimentError(RuntimeError):
    """A declared experiment command failed or produced no metrics."""

    def __init__(self, message: str, result: Optional[ExperimentResult] = None) -> None:
        super().__init__(message)
        self.result = result


# ---------------------------------------------------------------------------
# Command expansion
# ---------------------------------------------------------------------------


def _tokenize(template: str) -> list[str]:
    """Split a command template on whitespace, honouring single/double quotes.

    The tokenizer is intentionally small: case commands contain only file
    paths and flags, and everything is executed without a shell.
    """

    tokens: list[str] = []
    current: list[str] = []
    quote: Optional[str] = None
    for char in template.strip():
        if quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
        elif char in {"'", '"'}:
            quote = char
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    if quote:
        raise ValueError(f"unbalanced quote in command template: {template!r}")
    return tokens


def expand_command(template: str, variables: dict[str, Any]) -> list[str]:
    """Expand ``{name}`` placeholders inside a command template.

    A placeholder that occupies a whole token and resolves to a list is spliced
    into the argument vector.  Unknown placeholders raise an error so that a
    malformed contract fails closed.
    """

    argv: list[str] = []
    for token in _tokenize(template):
        if token.startswith("{") and token.endswith("}") and token.count("{") == 1:
            key = token[1:-1]
            if key not in variables:
                raise KeyError(f"unknown command variable {{{key}}} in {template!r}")
            value = variables[key]
            if isinstance(value, (list, tuple)):
                argv.extend(str(item) for item in value)
            elif value is None:
                raise KeyError(f"command variable {{{key}}} is None in {template!r}")
            else:
                argv.append(str(value))
        elif "{" in token:
            expanded = token
            for key, value in variables.items():
                expanded = expanded.replace("{" + key + "}", str(value))
            if "{" in expanded:
                raise KeyError(f"unresolved variable in token {token!r}")
            argv.append(expanded)
        else:
            argv.append(token)
    if not argv:
        raise ValueError(f"empty command template: {template!r}")
    return argv


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_path_name(name: str) -> str:
    """Windows-safe directory component; case names may contain any text."""

    cleaned = _UNSAFE_PATH_CHARS.sub("_", name).strip().rstrip(".")
    return cleaned or "variant"


@dataclass
class RunContext:
    """Everything needed to execute one declared experiment."""

    experiment: str
    variant: str
    split: str
    seed: int
    template: str
    stage: str = "experiments"


class ExperimentRunner:
    def __init__(
        self,
        case: CaseSpec,
        run_dir: Path,
        state: Optional[RunState] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.case = case
        self.run_dir = Path(run_dir)
        self.state = state
        self.timeout_seconds = timeout_seconds or case.budget.timeout_seconds

    # -- context ---------------------------------------------------------

    def variables(self, split: str, out_dir: Path, seed: int) -> dict[str, Any]:
        budget = self.case.budget
        if split == "subset":
            samples = budget.subset_samples
            epochs = budget.subset_epochs
        else:
            samples = budget.full_samples
            epochs = budget.full_epochs
        variables: dict[str, Any] = {
            "python": sys.executable,
            "out": str(out_dir),
            "seed": seed,
            "split": split,
            "samples": samples,
            "epochs": epochs,
            "test_samples": budget.test_samples,
            "case_dir": str(self.case.root),
            "code_dir": str(self.case.code_path),
        }
        variables.update(self.case.variables)
        return variables

    # -- execution -------------------------------------------------------

    def run(self, context: RunContext) -> ExperimentResult:
        variant_dir = _safe_path_name(context.variant)
        out_dir = self.run_dir / "experiments" / context.stage / variant_dir / f"seed{context.seed}"
        # Split-aware directories keep subset and full observations apart.
        if context.split in {"subset", "full"}:
            out_dir = self.run_dir / "experiments" / context.stage / variant_dir / context.split / f"seed{context.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"

        variables = self.variables(context.split, out_dir, context.seed)
        argv = expand_command(context.template, variables)

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["SCIENTIST3_CASE"] = self.case.case_id
        env["SCIENTIST3_VARIANT"] = context.variant
        env["SCIENTIST3_STAGE"] = context.stage
        if self.case.code_dir:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(self.case.code_path) + (
                os.pathsep + existing if existing else ""
            )

        started = time.time()
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"# command: {' '.join(argv)}\n")
            log.write(f"# cwd: {self.case.code_path or self.case.root}\n\n")
            log.flush()
            try:
                completed = subprocess.run(
                    argv,
                    cwd=str(self.case.code_path or self.case.root),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                returncode = -1
                log.write(f"\nTIMEOUT after {self.timeout_seconds}s\n")
                duration = time.time() - started
                result = ExperimentResult(
                    experiment=context.experiment,
                    variant=context.variant,
                    split=context.split,
                    seed=context.seed,
                    metrics={},
                    out_dir=str(out_dir),
                    log_path=str(log_path),
                    command=argv,
                    returncode=returncode,
                    duration_s=duration,
                )
                raise ExperimentError(
                    f"experiment {context.variant} exceeded {self.timeout_seconds}s "
                    f"(see {log_path})",
                    result,
                ) from exc

        duration = time.time() - started
        metrics = self._read_metrics(out_dir)

        result = ExperimentResult(
            experiment=context.experiment,
            variant=context.variant,
            split=context.split,
            seed=context.seed,
            metrics=metrics,
            out_dir=str(out_dir),
            log_path=str(log_path),
            command=argv,
            returncode=returncode,
            duration_s=duration,
        )

        if returncode != 0:
            tail = _tail(log_path)
            raise ExperimentError(
                f"experiment {context.variant} (seed {context.seed}) exited with "
                f"{returncode}; tail of {log_path}:\n{tail}",
                result,
            )
        if not metrics:
            tail = _tail(log_path)
            raise ExperimentError(
                f"experiment {context.variant} produced no declared metrics in "
                f"{self.case.metrics_file}; tail of {log_path}:\n{tail}",
                result,
            )

        write_json(out_dir / "observation.json", result.to_dict())
        if self.state is not None:
            self.state.event(
                "experiment",
                variant=context.variant,
                stage=context.stage,
                split=context.split,
                seed=context.seed,
                metrics=metrics,
                duration_s=round(duration, 3),
            )
        return result

    def _read_metrics(self, out_dir: Path) -> dict[str, float]:
        metrics_path = out_dir / self.case.metrics_file
        if not metrics_path.exists():
            return {}
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        metrics: dict[str, float] = {}
        for spec in self.case.metrics:
            value = resolve_path(payload, spec.path or spec.name)
            if value is None and isinstance(payload, dict):
                value = payload.get(spec.name)
            if value is None:
                nested = payload.get("metrics") if isinstance(payload, dict) else None
                if isinstance(nested, dict):
                    value = nested.get(spec.name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                metrics[spec.name] = numeric
        return metrics


def _tail(path: Path, lines: int = 25) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<no log>"
    return "\n".join(content[-lines:])


# ---------------------------------------------------------------------------
# Aggregation and comparison
# ---------------------------------------------------------------------------


def aggregate(results: Iterable[ExperimentResult], metric: MetricSpec) -> dict[str, float]:
    values = [r.metrics[metric.name] for r in results if metric.name in r.metrics]
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


@dataclass
class Comparison:
    verdict: str  # better | worse | tie
    deltas: dict[str, float]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "deltas": self.deltas, "detail": self.detail}


def compare_results(
    candidate: list[ExperimentResult],
    baseline: list[ExperimentResult],
    metrics: list[MetricSpec],
    min_relative_gain: Optional[float] = None,
    label: str = "candidate",
    baseline_label: str = "baseline",
) -> Comparison:
    """Compare candidate and baseline observations using the primary metric.

    A candidate is *better* only when the mean improvement clears the declared
    relative-gain threshold and, with two or more seeds, every observed seed
    moves in the expected direction.  Ties inside the noise band are reported as
    ``tie`` so that the pipeline can decide whether to engineer or prune.

    Threshold precedence: an explicitly passed ``min_relative_gain`` always
    wins.  When it is omitted, the primary metric's own ``min_gain`` applies,
    and only then the 2% default.  Ablations pass their own (usually lower)
    threshold, so a metric-level promotion gate cannot silently raise it.
    """

    primary = next((m for m in metrics if m.primary), metrics[0])
    cand = aggregate(candidate, primary)
    base = aggregate(baseline, primary)
    if cand["n"] == 0 or base["n"] == 0:
        return Comparison("tie", {}, "missing observations")

    sign = 1.0 if primary.direction == "max" else -1.0
    delta = sign * (cand["mean"] - base["mean"])
    scale = max(abs(base["mean"]), 1e-9)
    relative = delta / scale
    if min_relative_gain is not None:
        threshold = min_relative_gain
    elif primary.min_gain is not None:
        threshold = primary.min_gain
    else:
        threshold = 0.02

    deltas = {primary.name: cand["mean"] - base["mean"]}
    for metric in metrics:
        if metric.name == primary.name:
            continue
        c = aggregate(candidate, metric)
        b = aggregate(baseline, metric)
        if c["n"] and b["n"]:
            deltas[metric.name] = c["mean"] - b["mean"]

    consistent = True
    if cand["n"] > 1 and base["n"] > 1:
        pairwise = []
        for c in candidate:
            for b in baseline:
                if primary.name in c.metrics and primary.name in b.metrics:
                    pairwise.append(sign * (c.metrics[primary.name] - b.metrics[primary.name]))
        if pairwise:
            consistent = all(value > 0 for value in pairwise)
    elif cand["n"] > 1:
        values = [sign * v for v in cand["values"]]
        consistent = all(value > 0 for value in values)

    detail = (
        f"{label} {primary.name}={cand['mean']:.4f}±{cand['std']:.4f} vs "
        f"{baseline_label} {base['mean']:.4f}±{base['std']:.4f}; "
        f"relative gain {relative * 100:+.2f}% (threshold {threshold * 100:.2f}%)"
    )

    if delta > 0 and relative >= threshold and consistent:
        return Comparison("better", deltas, detail)
    if delta < 0 and abs(relative) >= threshold:
        return Comparison("worse", deltas, detail)
    return Comparison("tie", deltas, detail)
