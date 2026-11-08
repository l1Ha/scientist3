from __future__ import annotations

from pathlib import Path

from scientist3.core import ExperimentResult, Review, load_case
from scientist3.paper import _problem_summary, build_paper, format_value, results_table

CASE = Path(__file__).resolve().parent.parent / "cases" / "toy_regression"


def _result(seed: int, rmse: float, r2: float = 0.9) -> ExperimentResult:
    return ExperimentResult(
        experiment="idea_full",
        variant="I5_full",
        split="full",
        seed=seed,
        metrics={"rmse": rmse, "r2": r2},
        out_dir="",
        log_path="",
        command=[],
        returncode=0,
        duration_s=0.0,
    )


def _records(case) -> dict:
    selected = [_result(0, 0.057), _result(1, 0.059)]
    baseline = [_result(0, 0.92), _result(1, 1.03)]
    return {
        "limitations": case.limitations,
        "verifier": {"sufficient": True},
        "ideas": case.ideas,
        "novelty": {idea.id: {"delta": idea.delta} for idea in case.ideas},
        "screening": [
            {
                "idea_id": "I1",
                "label": "I1 quadratic polynomial basis",
                "decision": "accept",
                "reason": "clears the subset threshold",
                "results": [_result(0, 0.66)],
            }
        ],
        "baseline_subset": [_result(0, 0.88)],
        "baseline_full": baseline,
        "full": [
            {
                "idea_id": "I5",
                "label": "I5 wide cubic Fourier-polynomial basis",
                "kind": "evolved",
                "results": selected,
                "comparison": {
                    "verdict": "better",
                    "deltas": {"rmse": -0.91},
                    "detail": "I5 rmse=0.0580 vs baseline 0.9750",
                },
            }
        ],
        "selector": {
            "idea_id": "I5",
            "label": "I5 wide cubic Fourier-polynomial basis",
            "rationale": "ranked first by rmse",
        },
        "ablations": [
            {
                "id": "A1",
                "component": "random Fourier features",
                "results": [_result(0, 0.95)],
                "verdict": "component matters",
            }
        ],
        "ablation_verdict": {"cleanliness": "clean", "reason": "1/1 dropped"},
        "supplementary": [
            {"id": "R1", "question": "shift?", "summary": "rmse=0.12"}
        ],
        "reviews": [Review(1, "scripted-reviewer", 9.25, ["s"], ["w"], ["q"])],
        "audit": {
            "passed": True,
            "checks": {"score_verification": {"passed": True}},
        },
        "references": [{"title": "A paper", "url": "https://example.com"}],
    }


def test_problem_summary_joins_first_paragraph() -> None:
    case = load_case(CASE)
    summary = _problem_summary(case)
    assert summary.startswith("Given three real inputs")
    assert summary.endswith("interaction.")
    assert "\n" not in summary


def test_format_value_and_table_header() -> None:
    case = load_case(CASE)
    one = format_value(case, [_result(0, 0.5)], "rmse")
    two = format_value(case, [_result(0, 0.5), _result(1, 0.7)], "rmse")
    assert one == "0.5000"
    assert two == "0.6000 ± 0.1000"
    assert format_value(case, [], "rmse") == "n/a"

    table = results_table(case, [("x", [_result(0, 0.5)])])
    assert "`rmse`" in table and "`r2`" in table
    assert "| x |" in table


def test_build_paper_from_stored_observations() -> None:
    case = load_case(CASE)
    text, evidence = build_paper(case, _records(case))

    # Abstract: joined problem summary, backticked primary metric, evolved idea.
    assert "We study the problem: Given three real inputs" in text
    assert "`rmse`" in text
    assert "wide cubic Fourier-polynomial basis" in text

    # Tables are labelled and the ablation table lists the declared component.
    for token in ("Table 1", "Table 2", "Table 3"):
        assert token in text
    assert "random Fourier features" in text
    assert "Integrity audit: PASS" in text

    assert evidence["backticked_metrics"] is True
    assert evidence["tables_traceable"] is True
    assert evidence["significance"] is True
    assert evidence["ablation_clean"] is True
    assert evidence["n_ablations"] == 1
    assert evidence["n_supplementary"] == 1
    assert evidence["references_ok"] is True
    assert evidence["reviews"][0]["score"] == 9.25


def test_build_paper_handles_missing_audit() -> None:
    case = load_case(CASE)
    records = _records(case)
    records["audit"] = None
    text, evidence = build_paper(case, records)
    assert "Integrity audit" not in text
    assert evidence["n_references"] == 1
