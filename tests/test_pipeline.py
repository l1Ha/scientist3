from __future__ import annotations

import json
from pathlib import Path

from scientist3.core import load_case
from scientist3.pipeline import ResearchPipeline, RunOptions

CASE = Path(__file__).resolve().parent.parent / "cases" / "toy_regression"


def _options(tmp_path: Path) -> RunOptions:
    return RunOptions(
        runs_dir=tmp_path / "runs",
        run_id="tiny",
        reexecute_audit=True,
        overrides={
            "seeds": [0],
            "subset_samples": 80,
            "full_samples": 200,
            "top_ideas": 1,
            "max_screening_rounds": 1,
            "max_evo_rounds": 0,
            "min_successful_ideas": 1,
            "max_successful_ideas": 1,
            "max_ablation_refinements": 0,
            "max_review_rounds": 1,
            "review_threshold": 0.0,
            "min_relative_gain": 0.10,
            "ablation_min_gain": 0.10,
            "timeout_seconds": 120,
        },
    )


def test_toy_pipeline_end_to_end(tmp_path: Path) -> None:
    case = load_case(CASE)
    pipeline = ResearchPipeline(case, _options(tmp_path))
    summary = pipeline.run()

    assert summary["audit_passed"] is True
    assert summary["selected_idea"] == "I1"
    assert summary["meta_decision"] == "accept"
    assert summary["review_score"] >= 0.0

    run_dir = Path(summary["run_dir"])
    for relative in (
        "contract.md",
        "state.json",
        "summary.json",
        "paper/final.md",
        "audit/audit.json",
    ):
        assert (run_dir / relative).exists(), relative

    paper = (run_dir / "paper" / "final.md").read_text(encoding="utf-8")
    assert "Table 1" in paper and "Table 2" in paper and "Table 3" in paper
    assert "Integrity audit: PASS" in paper

    audit = json.loads((run_dir / "audit" / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"]
    assert audit["checks"]["score_verification"]["reexecuted"]
    for check in audit["checks"].values():
        assert check["passed"]


def test_toy_pipeline_resume_is_idempotent(tmp_path: Path) -> None:
    case = load_case(CASE)
    first = ResearchPipeline(case, _options(tmp_path))
    first_summary = first.run()

    second = ResearchPipeline(case, _options(tmp_path))
    second_summary = second.run()
    assert second_summary == first_summary
    assert second.state.is_complete("summary")
