from __future__ import annotations

from pathlib import Path

import pytest

from scientist3.agents import Agents
from scientist3.core import Review, load_case
from scientist3.llm import (
    LLMConfig,
    LLMError,
    OpenAICompatLLM,
    ScriptedLLM,
    build_llm,
    extract_json,
)

CASE = Path(__file__).resolve().parent.parent / "cases" / "toy_regression"

STRONG_EVIDENCE = {
    "n_metrics": 2,
    "n_seeds": 2,
    "significance": True,
    "n_ablations": 4,
    "ablation_clean": True,
    "references_ok": True,
    "tables_traceable": True,
    "external": False,
    "n_supplementary": 0,
}


def test_extract_json_plain_fenced_and_embedded() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here is the result:\n[{"id": "I1"}]\nThanks!') == [{"id": "I1"}]
    with pytest.raises(LLMError):
        extract_json("no json at all")


def test_build_llm_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(build_llm(LLMConfig(mode="scripted")), ScriptedLLM)
    assert isinstance(build_llm(LLMConfig(mode="offline")), ScriptedLLM)
    monkeypatch.setenv("SCIENTIST3_API_KEY", "test-key")
    live = build_llm(LLMConfig(mode="openai", api_key="test-key"))
    assert isinstance(live, OpenAICompatLLM)
    with pytest.raises(LLMError):
        build_llm(LLMConfig(mode="openai", api_key=""))
    with pytest.raises(LLMError):
        build_llm(LLMConfig(mode="anthropic", api_key="k"))
    with pytest.raises(LLMError):
        build_llm(LLMConfig(mode="nonsense"))


def test_llm_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCIENTIST3_MODEL", "local-model")
    monkeypatch.setenv("SCIENTIST3_API_KEY", "secret")
    monkeypatch.setenv("SCIENTIST3_BASE_URL", "http://localhost:11434/v1")
    config = LLMConfig.from_env("openai")
    assert config.mode == "openai"
    assert config.model == "local-model"
    assert config.api_key == "secret"
    assert config.base_url == "http://localhost:11434/v1"


def test_scripted_llm_refuses_completion() -> None:
    with pytest.raises(LLMError):
        ScriptedLLM().complete("system", "user")


def test_scripted_limitations_and_novelty() -> None:
    case = load_case(CASE)
    agents = Agents(case, ScriptedLLM(), live=False)

    limitations, verifier = agents.limitations()
    assert len(limitations) == 4
    assert verifier["sufficient"] is True
    assert verifier["mechanism_count"] >= 2

    novelty = agents.novelty_check(case.ideas)
    assert novelty["I2"]["verdict"] == "pass"
    assert novelty["I1"]["verdict"] == "weak"  # declared low novelty
    assert novelty["I2"]["references"]


def test_scripted_review_scores_from_evidence_only() -> None:
    case = load_case(CASE)
    agents = Agents(case, ScriptedLLM(), live=False)

    strong = agents.review("paper", dict(STRONG_EVIDENCE), round_number=1)
    weak = agents.review(
        "paper",
        {
            **STRONG_EVIDENCE,
            "n_seeds": 1,
            "significance": False,
            "ablation_clean": False,
            "n_ablations": 1,
        },
        round_number=1,
    )
    assert strong.score > weak.score
    assert strong.score == pytest.approx(8.75, abs=1e-9)  # no supplementary yet
    assert strong.decision == "refine"  # threshold is 9.0 in the toy contract
    assert any("ablation" in item.lower() for item in weak.weaknesses)

    with_supp = agents.review(
        "paper", {**STRONG_EVIDENCE, "n_supplementary": 2}, round_number=2
    )
    assert with_supp.score == pytest.approx(9.25, abs=1e-9)
    assert with_supp.decision == "accept"

    # scores are floats derived from the evidence dict, never from LLM text
    assert isinstance(strong.score, float)


def test_rebuttal_plan_budget_and_reuse() -> None:
    case = load_case(CASE)
    agents = Agents(case, ScriptedLLM(), live=False)
    review = Review(1, "scripted-reviewer", 8.0, [], ["weakness"], [])
    planned = agents.rebuttal_plan(review, used=set())
    assert 1 <= len(planned) <= 2
    used = {item.id for item in planned}
    planned_again = agents.rebuttal_plan(review, used=used)
    assert all(item.id not in used for item in planned_again)


def test_meta_review_gate() -> None:
    case = load_case(CASE)
    agents = Agents(case, ScriptedLLM(), live=False)

    accept = agents.meta_review(
        "paper", Review(1, "r", 9.5, [], [], []), {"passed": True}, 1
    )
    assert accept["decision"] == "accept"

    revise = agents.meta_review(
        "paper", Review(1, "r", 8.0, [], [], []), {"passed": True}, 1
    )
    assert revise["decision"] == "revise"

    broken = agents.meta_review(
        "paper", Review(1, "r", 9.5, [], [], []), {"passed": False}, 1
    )
    assert broken["decision"] == "revise"


class _BrokenLiveLLM:
    """Raises or returns unparseable text, exercising the scripted fallback."""

    def __init__(self, payload: str = "not json") -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        return self.payload


def test_live_mode_falls_back_to_contract_on_bad_json() -> None:
    case = load_case(CASE)
    llm = _BrokenLiveLLM()
    agents = Agents(case, llm, live=True)

    limitations, verifier = agents.limitations()
    assert llm.calls == 1
    assert len(limitations) == len(case.limitations)
    assert "fallback" in verifier["source"]

    review = agents.review("paper", dict(STRONG_EVIDENCE), round_number=1)
    assert review.reviewer == "scripted-reviewer"

    meta = agents.meta_review(
        "paper", Review(1, "r", 9.5, [], [], []), {"passed": True}, 1
    )
    assert meta["decision"] == "accept"
