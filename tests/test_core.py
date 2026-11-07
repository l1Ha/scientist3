from __future__ import annotations

from pathlib import Path

import pytest

from scientist3.core import (
    Budget,
    RunState,
    load_case,
    read_json,
    resolve_path,
    write_json,
)

CASE = Path(__file__).resolve().parent.parent / "cases" / "toy_regression"


def test_load_toy_case() -> None:
    case = load_case(CASE)
    assert case.case_id == "toy_regression"
    assert case.primary_metric.name == "rmse"
    assert case.primary_metric.direction == "min"
    assert len(case.ideas) == 4
    assert len(case.evolved_ideas) == 1
    assert len(case.ablations) == 4
    assert case.protected_paths
    assert case.code_path == CASE
    assert "random Fourier features" in case.title


def test_budget_defaults_and_override() -> None:
    import dataclasses

    budget = Budget()
    changed = dataclasses.replace(budget, seeds=[3, 4], subset_samples=12)
    assert budget.seeds == [0]
    assert changed.seeds == [3, 4]
    assert changed.subset_samples == 12
    assert changed.primary_seed == 3


def test_resolve_path() -> None:
    payload = {"probe": {"accuracy": 0.9}, "rmse": 1.0}
    assert resolve_path(payload, "probe.accuracy") == 0.9
    assert resolve_path(payload, "rmse") == 1.0
    assert resolve_path(payload, "missing.key") is None


def test_run_state_roundtrip(tmp_path: Path) -> None:
    state = RunState(tmp_path / "run", "case", "run-1")
    state.start_stage("contract")
    state.finish_stage("contract", artifact="contract.md", passed=True)
    state.event("note", value=1)

    loaded = RunState.load(tmp_path / "run")
    assert loaded.case_id == "case"
    assert loaded.is_complete("contract")
    assert loaded.stages["contract"]["detail"]["passed"] is True
    assert (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").strip()


def test_json_helpers(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    write_json(target, {"a": [1, 2]})
    assert read_json(target) == {"a": [1, 2]}
    assert read_json(tmp_path / "absent.json", default={"ok": True}) == {"ok": True}
