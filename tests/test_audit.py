from __future__ import annotations

import shutil
from pathlib import Path

from scientist3.audit import (
    capture_protected_hashes,
    check_method_code_alignment,
    check_references,
    verify_protected,
)
from scientist3.core import load_case

CASE = Path(__file__).resolve().parent.parent / "cases" / "toy_regression"


def test_references_offline_shape_check() -> None:
    report = check_references(
        [
            {"title": "A paper", "doi": "10.1000/xyz"},
            {"title": "A website", "url": "https://example.com/paper"},
        ],
        online=False,
    )
    assert report["passed"]
    assert report["with_identifier"] == 2
    assert report["entries"][0]["status"] == "doi-declared"
    assert report["entries"][1]["status"] == "unchecked-offline"


def test_references_missing_identifier_fails() -> None:
    report = check_references([{"title": "No identifier"}], online=False)
    assert not report["passed"]
    assert report["unresolved"]


def test_protected_hashes_detect_changes(tmp_path: Path) -> None:
    copied = tmp_path / "toy"
    shutil.copytree(CASE, copied, ignore=shutil.ignore_patterns("runs", "__pycache__"))
    case = load_case(copied)
    hashes = capture_protected_hashes(case)
    assert hashes
    assert verify_protected(case, hashes)["passed"]

    (copied / "problem.md").write_text("changed\n", encoding="utf-8")
    report = verify_protected(case, hashes)
    assert not report["passed"]
    assert "problem.md" in report["changed"]


def test_method_code_alignment_requires_ablation_ids_and_tables() -> None:
    case = load_case(CASE)
    paper = "Table 1 Table 2 Table 3\n" + "\n".join(
        ablation.id for ablation in case.ablations
    )
    observations = [{"stage": "ablations", "variant": "x", "seed": 0, "metrics": {}}]
    report = check_method_code_alignment(case, paper, observations)
    assert report["passed"]
    assert report["ablation_observations"] == 1

    failing = check_method_code_alignment(case, "Table 1", observations)
    assert not failing["passed"]
    assert failing["issues"]
