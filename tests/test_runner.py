from __future__ import annotations

from scientist3.core import ExperimentResult, MetricSpec
from scientist3.runner import (
    _safe_path_name,
    aggregate,
    compare_results,
    expand_command,
)


def _result(seed: int, rmse: float, r2: float = 0.0) -> ExperimentResult:
    return ExperimentResult(
        experiment="test",
        variant="v",
        split="full",
        seed=seed,
        metrics={"rmse": rmse, "r2": r2},
        out_dir="",
        log_path="",
        command=[],
        returncode=0,
        duration_s=0.0,
    )


def test_expand_command_placeholders() -> None:
    argv = expand_command(
        "{python} run.py --n {samples} --seed {seed} --out {out}",
        {
            "python": "python",
            "samples": 12,
            "seed": 3,
            "out": "runs/x",
        },
    )
    assert argv == ["python", "run.py", "--n", "12", "--seed", "3", "--out", "runs/x"]


def test_expand_command_splices_lists_and_quotes() -> None:
    argv = expand_command(
        "{python} -m mod {flags} --out '{out dir}'",
        {
            "python": "python",
            "flags": ["--a", "1", "--b", "2"],
            "out dir": "runs/a b",
        },
    )
    assert argv == ["python", "-m", "mod", "--a", "1", "--b", "2", "--out", "runs/a b"]


def test_expand_command_rejects_unknown_placeholder() -> None:
    try:
        expand_command("{python} --seed {missing}", {"python": "python"})
    except KeyError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_compare_min_metric_better_tie_worse() -> None:
    metrics = [
        MetricSpec("rmse", direction="min", primary=True),
        MetricSpec("r2", direction="max"),
    ]
    baseline = [_result(0, 1.0, 0.5), _result(1, 1.0, 0.5)]

    better = compare_results(
        [_result(0, 0.8, 0.6), _result(1, 0.85, 0.6)], baseline, metrics, 0.10
    )
    assert better.verdict == "better"
    assert better.deltas["rmse"] < 0

    tie = compare_results(
        [_result(0, 0.95, 0.5), _result(1, 0.96, 0.5)], baseline, metrics, 0.10
    )
    assert tie.verdict == "tie"

    worse = compare_results(
        [_result(0, 1.3, 0.2), _result(1, 1.2, 0.2)], baseline, metrics, 0.10
    )
    assert worse.verdict == "worse"


def test_compare_requires_seed_consistency() -> None:
    metrics = [MetricSpec("rmse", direction="min", primary=True)]
    baseline = [_result(0, 1.0), _result(1, 1.0)]
    inconsistent = compare_results(
        [_result(0, 0.80), _result(1, 1.01)], baseline, metrics, 0.05
    )
    assert inconsistent.verdict == "tie"

    consistent = compare_results(
        [_result(0, 0.80), _result(1, 0.90)], baseline, metrics, 0.05
    )
    assert consistent.verdict == "better"


def test_safe_path_name_sanitizes_windows_chars() -> None:
    assert _safe_path_name("naive l<=12 (a/b)") == "naive l_=12 (a_b)"
    assert _safe_path_name('weird:"name?"') == "weird__name_"
    assert _safe_path_name("...") == "variant"
    assert _safe_path_name("plain name 2") == "plain name 2"


def test_aggregate_handles_missing_metric() -> None:
    metrics = MetricSpec("rmse", direction="min", primary=True)
    stats = aggregate([_result(0, 0.5), _result(1, 0.7)], metrics)
    assert stats["n"] == 2
    assert abs(stats["mean"] - 0.6) < 1e-12
    assert abs(stats["std"] - 0.1) < 1e-12
