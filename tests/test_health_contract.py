from __future__ import annotations

import re
from pathlib import Path

from scientist3.core import load_case, resolve_path
from scientist3.runner import expand_command

CASE = Path(__file__).resolve().parent.parent / "cases" / "health_cartan_dec_miae"
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def _case():
    return load_case(CASE)


def test_health_case_shape() -> None:
    case = _case()
    assert case.case_id == "health_cartan_dec_miae"
    assert case.primary_metric.name == "recon_integrated_rmse"
    assert case.primary_metric.direction == "min"
    assert case.primary_metric.path == "reconstruction.integrated_position_rmse"
    assert [idea.id for idea in case.ideas] == ["I1", "I2", "I3"]
    assert [idea.id for idea in case.evolved_ideas] == ["I4"]
    assert [a.id for a in case.ablations] == ["A1", "A2", "A3", "A4"]
    assert [s.id for s in case.supplementary] == ["R1", "R2"]


def test_health_budget_invariants() -> None:
    budget = _case().budget
    assert budget.seeds == [0, 1]
    # The 4th full evaluation is the evolved idea, so these must be equal.
    assert budget.min_successful_ideas == budget.max_successful_ideas == 4
    assert budget.top_ideas == 3
    # Ablations use a laxer threshold than promotions; the runner must honour it.
    assert budget.ablation_min_gain < budget.min_relative_gain
    assert budget.ablation_min_gain == 0.03
    assert budget.review_threshold == 9.0


def test_health_commands_are_fully_expandable() -> None:
    case = _case()
    values = {
        "python": "python",
        "samples": 16,
        "test_samples": 8,
        "epochs": 1,
        "seed": 0,
        "out": "runs/x",
    }
    templates = (
        [case.baseline_command]
        + [idea.command for idea in case.ideas + case.evolved_ideas]
        + [a.command for a in case.ablations]
        + [s.command for s in case.supplementary]
    )
    for template in templates:
        missing = set(PLACEHOLDER.findall(template)) - set(values)
        assert not missing, f"unknown placeholders {missing} in {template}"
        argv = expand_command(template, values)
        assert "{samples}" not in argv and "{seed}" not in argv


def test_health_ablation_wiring() -> None:
    case = _case()
    by_id = {a.id: a.command for a in case.ablations}
    # A1 removes only the integration loss; transport weighting stays on.
    assert "--lambda-integration 0" in by_id["A1"]
    assert "--lambda-omega 4" in by_id["A1"]
    # A2 removes only the connection-form loss.
    assert "--lambda-omega 0" in by_id["A2"]
    assert "--lambda-integration 4" in by_id["A2"]
    # A3 removes only the curvature loss.
    assert "--lambda-curvature 0" in by_id["A3"]
    assert "--lambda-integration 4" in by_id["A3"] and "--lambda-omega 4" in by_id["A3"]
    # A4 reverts the loss rebalancing to equal weights.
    assert "--lambda-integration 1" in by_id["A4"]
    assert "--lambda-omega 1" in by_id["A4"]
    assert "--lambda-curvature 1" in by_id["A4"]

    selected = next(i for i in case.evolved_ideas if i.id == "I4")
    assert "--lambda-integration 4" in selected.command
    assert "--lambda-omega 4" in selected.command
    assert "--lambda-curvature 2" in selected.command


def test_health_supplementary_wiring() -> None:
    case = _case()
    by_id = {s.id: s.command for s in case.supplementary}
    assert "--min-length 80" in by_id["R1"]
    assert "--max-synthetic-length 140" in by_id["R1"]
    assert "--mask-strategy random" in by_id["R2"]
    assert "--lambda-integration 4 --lambda-omega 4" in by_id["R1"]


def test_health_protected_paths_exist() -> None:
    case = _case()
    assert "code/cartan_dec/geometry_eval.py" in case.protected_paths
    assert "code/cartan_dec/data.py" in case.protected_paths
    for relative in case.protected_paths:
        assert (case.root / relative).exists(), relative
    assert case.code_path == (case.root / "code").resolve()


def test_health_metric_path_resolves() -> None:
    payload = {
        "reconstruction": {
            "position_rmse": 12.0,
            "integrated_position_rmse": 8.3,
            "rotation_error_deg": 80.0,
        },
        "probe": {"accuracy": 0.95},
    }
    assert resolve_path(payload, "reconstruction.integrated_position_rmse") == 8.3
    assert resolve_path(payload, "probe.accuracy") == 0.95
