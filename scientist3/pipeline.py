"""The Scientist3 research loop.

Stages follow the ScientistTwo paper (sections 3.1 - 3.6) plus the integrity
audit:

``contract -> limitations -> ideation -> screening -> full -> ablations ->
drafting -> review -> meta_review -> integrity``

Every stage is bounded by :class:`~scientist3.core.Budget`, persists its
artifact under ``runs/<run-id>/``, and can be resumed.  Promotion decisions use
:func:`~scientist3.runner.compare_results` on trusted observations only.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .agents import Agents
from .audit import capture_protected_hashes, run_audit
from .core import (
    Ablation,
    Budget,
    CaseSpec,
    ExperimentResult,
    Idea,
    Limitation,
    Review,
    RunState,
    Supplementary,
    read_json,
    write_json,
)
from .llm import LLMConfig, build_llm
from .paper import build_paper
from .runner import (
    Comparison,
    ExperimentError,
    ExperimentRunner,
    RunContext,
    aggregate,
    compare_results,
)


@dataclass
class RunOptions:
    runs_dir: Path = Path("runs")
    run_id: str = ""
    live: bool = False
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    online_refs: bool = False
    allow_offline_refs: bool = True
    reexecute_audit: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)

    def apply_overrides(self, budget: Budget) -> Budget:
        allowed = {f.name for f in dataclasses.fields(Budget)}
        unknown = set(self.overrides) - allowed
        if unknown:
            raise KeyError(f"unknown budget override(s): {sorted(unknown)}")
        return dataclasses.replace(budget, **self.overrides)


class ResearchPipeline:
    def __init__(
        self,
        case: CaseSpec,
        options: Optional[RunOptions] = None,
        llm: Any = None,
    ) -> None:
        self.case = dataclasses.replace(
            case, budget=options.apply_overrides(case.budget) if options else case.budget
        )
        self.options = options or RunOptions()
        if self.options.run_id:
            run_id = self.options.run_id
        else:
            run_id = time.strftime(f"{self.case.case_id}-%Y%m%d-%H%M%S")
        # Resolve once: experiment subprocesses run in the case directory, so a
        # relative run directory would place their outputs somewhere else.
        self.run_dir = (Path(self.options.runs_dir) / run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        state_path = self.run_dir / "state.json"
        if state_path.exists():
            self.state = RunState.load(self.run_dir)
        else:
            self.state = RunState(self.run_dir, self.case.case_id, run_id)
            self.state.metadata["budget"] = dataclasses.asdict(self.case.budget)
            self.state.save()

        self.llm_config = LLMConfig(
            mode="openai" if self.options.live else "scripted",
            model=self.options.model,
            base_url=self.options.base_url,
            api_key=self.options.api_key,
        )
        self.llm = llm or build_llm(self.llm_config)
        self.agents = Agents(self.case, self.llm, live=self.options.live)
        self.runner = ExperimentRunner(self.case, self.run_dir, self.state)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def artifact_path(self, stage: str) -> Path:
        return self.run_dir / "artifacts" / f"{stage}.json"

    def load_artifact(self, stage: str, default: Any = None) -> Any:
        return read_json(self.artifact_path(stage), default)

    def save_artifact(self, stage: str, payload: Any) -> None:
        write_json(self.artifact_path(stage), payload)

    @staticmethod
    def _dump(items: list[Any]) -> list[dict[str, Any]]:
        return [dataclasses.asdict(item) for item in items]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        print(f"Scientist3 run {self.state.run_id}  case={self.case.case_id}")
        print(f"  run directory: {self.run_dir}")
        self.contract()
        limitations, verifier = self.limitations()
        ideas, novelty = self.ideation(limitations)
        screening = self.screening(ideas)
        full = self.full_set(screening, ideas)
        ablations, ablation_verdict = self.ablation_studies(ideas, full)
        paper_text, evidence = self.drafting(ideas, full, ablations, ablation_verdict)
        reviews, paper_text, evidence = self.review_loop(
            ideas, full, ablations, ablation_verdict, paper_text, evidence
        )
        meta = self.meta_review(paper_text, reviews, evidence)
        audit = self.integrity(paper_text)
        # The review-stage manuscript predates the audit; rebuild it so the
        # sealed `paper/final.md` carries the audit verdict.
        paper_text = self._paper_with_audit(reviews, paper_text)
        summary = self.finalize(paper_text, reviews, meta, audit, evidence)
        return summary

    def _paper_with_audit(self, reviews: list[Review], paper_text: str) -> str:
        review_payload = self.load_artifact("review", {})
        ablations_payload = self.load_artifact("ablations", {})
        if not review_payload or not ablations_payload:
            return paper_text
        ideas, _ = self.ideation([])
        records = self._paper_records(
            ideas,
            self.load_artifact("full", {}),
            ablations_payload.get("entries", []),
            ablations_payload.get("verdict", {}),
            review_payload.get("supplementary", []),
            reviews,
        )
        final_text, _ = build_paper(self.case, records)
        return final_text

    # ------------------------------------------------------------------
    # G0 contract
    # ------------------------------------------------------------------

    def contract(self) -> dict[str, Any]:
        stage = "contract"
        if self.state.is_complete(stage):
            return self.load_artifact(stage, {})
        self.state.start_stage(stage)
        try:
            hashes = capture_protected_hashes(self.case)
            self.state.metadata["protected_hashes"] = hashes
            contract = {
                "case_id": self.case.case_id,
                "title": self.case.title,
                "domain": self.case.domain,
                "baseline": self.case.baseline_name,
                "baseline_command": self.case.baseline_command,
                "metrics": [dataclasses.asdict(m) for m in self.case.metrics],
                "budget": dataclasses.asdict(self.case.budget),
                "protected_paths": self.case.protected_paths,
                "protected_hashes": hashes,
                "problem": self.case.problem_text,
                "notes": self.case.notes,
            }
            write_json(self.run_dir / "contract.json", contract)
            (self.run_dir / "contract.md").write_text(
                _contract_markdown(self.case, contract), encoding="utf-8"
            )
            self.save_artifact(stage, contract)
            self.state.finish_stage(stage, artifact=str(self.run_dir / "contract.md"))
            return contract
        except Exception as exc:  # noqa: BLE001 - stage boundary
            self.state.fail_stage(stage, str(exc))
            raise

    # ------------------------------------------------------------------
    # S1 limitations
    # ------------------------------------------------------------------

    def limitations(self) -> tuple[list[Limitation], dict[str, Any]]:
        stage = "limitations"
        if self.state.is_complete(stage):
            payload = self.load_artifact(stage, {})
            return (
                [Limitation(**item) for item in payload.get("limitations", [])],
                payload.get("verifier", {}),
            )
        self.state.start_stage(stage)
        try:
            limitations, verifier = self.agents.limitations()
            payload = {
                "limitations": self._dump(limitations),
                "verifier": verifier,
            }
            self.save_artifact(stage, payload)
            (self.run_dir / "limitations.md").write_text(
                _limitations_markdown(limitations, verifier), encoding="utf-8"
            )
            self.state.finish_stage(stage, artifact=str(self.run_dir / "limitations.md"))
            print(
                f"[S1] limitations: {len(limitations)} "
                f"(mechanism={verifier.get('mechanism_count')}, "
                f"sufficient={verifier.get('sufficient')})"
            )
            return limitations, verifier
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    # ------------------------------------------------------------------
    # S2 ideation
    # ------------------------------------------------------------------

    def ideation(
        self, limitations: list[Limitation]
    ) -> tuple[list[Idea], dict[str, dict[str, Any]]]:
        stage = "ideation"
        if self.state.is_complete(stage):
            payload = self.load_artifact(stage, {})
            return (
                [Idea(**item) for item in payload.get("ideas", [])],
                payload.get("novelty", {}),
            )
        self.state.start_stage(stage)
        try:
            ideas = self.agents.ideate(limitations)
            novelty = self.agents.novelty_check(ideas)
            payload = {"ideas": self._dump(ideas), "novelty": novelty}
            self.save_artifact(stage, payload)
            (self.run_dir / "ideas.md").write_text(
                _ideas_markdown(ideas, novelty), encoding="utf-8"
            )
            self.state.finish_stage(stage, artifact=str(self.run_dir / "ideas.md"))
            print(f"[S2] seed ideas: {len(ideas)}")
            return ideas, novelty
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    # ------------------------------------------------------------------
    # S3 subset screening
    # ------------------------------------------------------------------

    def screening(self, ideas: list[Idea]) -> dict[str, Any]:
        stage = "screening"
        if self.state.is_complete(stage):
            return self.load_artifact(stage, {})
        self.state.start_stage(stage)
        try:
            budget = self.case.budget
            baseline = self._run_variants(
                stage="screening",
                experiment="baseline_subset",
                variant=f"{self.case.baseline_name}",
                template=self.case.baseline_command,
                split="subset",
                seeds=budget.seeds,
            )
            entries = []
            candidates = ideas[: budget.top_ideas]
            for idea in candidates:
                entry = self._screen_idea(idea, baseline)
                entries.append(entry)
            payload = {
                "baseline_subset": [r.to_dict() for r in baseline],
                "entries": entries,
            }
            self.save_artifact(stage, payload)
            self.state.finish_stage(
                stage,
                artifact=str(self.artifact_path(stage)),
                accepted=sum(1 for e in entries if e["decision"] == "accept"),
                pruned=sum(1 for e in entries if e["decision"] in {"reject", "pruned"}),
            )
            for entry in entries:
                print(
                    f"[S3] {entry['idea_id']} {entry['decision'].upper()}: "
                    f"{entry['reason']}"
                )
            return payload
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    def _screen_idea(self, idea: Idea, baseline: list[ExperimentResult]) -> dict[str, Any]:
        budget = self.case.budget
        template = idea.subset_command or idea.command
        history: list[dict[str, Any]] = []
        decision = "pruned"
        reason = "no screening round completed"
        results: list[ExperimentResult] = []

        rounds = 1 + min(len(idea.engineering_commands), budget.max_screening_rounds)
        for round_index in range(rounds):
            variant = f"{idea.id}_subset_r{round_index}"
            results = self._run_variants(
                stage="screening",
                experiment="idea_subset",
                variant=variant,
                template=template,
                split="subset",
                seeds=budget.seeds,
            )
            comparison = compare_results(
                results, baseline, self.case.metrics, budget.min_relative_gain, label=idea.id
            )
            history.append({"round": round_index, "comparison": comparison.to_dict()})
            if comparison.verdict == "better":
                decision = "accept"
                reason = comparison.detail
                break
            if comparison.verdict == "worse" and round_index == 0 and not idea.engineering_commands:
                decision = "reject"
                reason = comparison.detail
                break
            if round_index < len(idea.engineering_commands):
                template = idea.engineering_commands[round_index]
                reason = f"engineer round {round_index + 1}: {comparison.detail}"
                continue
            if comparison.verdict == "worse":
                decision = "reject"
                reason = comparison.detail
            else:
                decision = "pruned"
                reason = f"engineer budget exhausted: {comparison.detail}"
        return {
            "idea_id": idea.id,
            "name": idea.name,
            "label": f"{idea.id} {idea.name}",
            "decision": decision,
            "reason": reason,
            "history": history,
            "template": template,
            "results": [r.to_dict() for r in results],
        }

    # ------------------------------------------------------------------
    # S4 full grid + evolution + selection
    # ------------------------------------------------------------------

    def full_set(self, screening: dict[str, Any], ideas: list[Idea]) -> dict[str, Any]:
        stage = "full"
        if self.state.is_complete(stage):
            return self.load_artifact(stage, {})
        self.state.start_stage(stage)
        try:
            budget = self.case.budget
            baseline = self._run_variants(
                stage="full",
                experiment="baseline_full",
                variant=self.case.baseline_name,
                template=self.case.baseline_command,
                split="full",
                seeds=budget.seeds,
            )
            accepted = [
                entry for entry in screening.get("entries", []) if entry["decision"] == "accept"
            ]
            by_id = {idea.id: idea for idea in ideas}
            entries: list[dict[str, Any]] = []
            evaluated: set[str] = set()

            def evaluate(idea: Idea, label: str, template: str = "") -> None:
                evaluated.add(idea.id)
                template = template or idea.command
                results = self._run_variants(
                    stage="full",
                    experiment="idea_full",
                    variant=f"{idea.id}_full",
                    template=template,
                    split="full",
                    seeds=budget.seeds,
                )
                comparison = compare_results(
                    results, baseline, self.case.metrics, budget.min_relative_gain, label=idea.id
                )
                entries.append(
                    {
                        "idea_id": idea.id,
                        "label": label,
                        "kind": idea.kind,
                        "template": template,
                        "comparison": comparison.to_dict(),
                        "results": [r.to_dict() for r in results],
                    }
                )

            for entry in accepted:
                idea = by_id.get(entry["idea_id"])
                if idea is not None:
                    # A candidate promoted after a subset engineering round must
                    # be evaluated on the command that actually passed screening.
                    evaluate(idea, entry["label"], entry.get("template", ""))

            successes = [
                entry for entry in entries if entry["comparison"].get("verdict") == "better"
            ]
            evolved_log: list[dict[str, Any]] = []
            rounds = 0
            pool = list(self.case.evolved_ideas)
            while (
                len(successes) < budget.min_successful_ideas
                and pool
                and rounds < budget.max_evo_rounds
                and len(evaluated) < budget.max_successful_ideas + budget.top_ideas
            ):
                rounds += 1
                next_idea = pool.pop(0)
                if next_idea.id in evaluated:
                    continue
                evolved_log.append(
                    {"round": rounds, "idea_id": next_idea.id, "source": "evolution"}
                )
                evaluate(next_idea, f"{next_idea.id} {next_idea.name} (evolved r{rounds})")
                successes = [
                    entry
                    for entry in entries
                    if entry["comparison"].get("verdict") == "better"
                ]

            selection = self._select(entries, successes)
            payload = {
                "baseline_full": [r.to_dict() for r in baseline],
                "entries": entries,
                "successes": [entry["idea_id"] for entry in successes],
                "evolutions": evolved_log,
                "selection": selection,
            }
            self.save_artifact(stage, payload)
            self.state.finish_stage(
                stage,
                artifact=str(self.artifact_path(stage)),
                successes=len(successes),
                selected=selection.get("idea_id"),
            )
            print(
                f"[S4] full evaluations: {len(entries)}, successes: {len(successes)}, "
                f"selected: {selection.get('label', selection.get('idea_id'))}"
            )
            return payload
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    def _select(
        self, entries: list[dict[str, Any]], successes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not successes:
            return {"idea_id": "", "label": "baseline", "rationale": "no idea beat the baseline", "template": ""}
        primary = self.case.primary_metric
        sign = 1.0 if primary.direction == "max" else -1.0

        def rank(entry: dict[str, Any]) -> tuple[float, float]:
            results = [ExperimentResult.from_dict(item) for item in entry["results"]]
            best = sign * aggregate(results, primary)["mean"]
            secondary_scores = 0.0
            for metric in self.case.metrics:
                if metric.name == primary.name:
                    continue
                direction = 1.0 if metric.direction == "max" else -1.0
                secondary_scores += direction * aggregate(results, metric)["mean"]
            return best, secondary_scores

        best_entry = max(successes, key=rank)
        rationale = (
            f"ranked first by `{primary.name}` then by the remaining declared metrics "
            f"across {len(self.case.budget.seeds)} seed(s)"
        )
        return {
            "idea_id": best_entry["idea_id"],
            "label": best_entry["label"],
            "template": best_entry["template"],
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # S5 ablations
    # ------------------------------------------------------------------

    def ablation_studies(
        self, ideas: list[Idea], full: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        stage = "ablations"
        if self.state.is_complete(stage):
            payload = self.load_artifact(stage, {})
            return payload.get("entries", []), payload.get("verdict", {})
        self.state.start_stage(stage)
        try:
            budget = self.case.budget
            plan = self.agents.ablation_plan(self._selected_idea(ideas, full))
            if not plan:
                payload = {"entries": [], "verdict": {"cleanliness": "not_applicable", "reason": "no ablations declared"}}
                self.save_artifact(stage, payload)
                self.state.finish_stage(stage, artifact=str(self.artifact_path(stage)))
                return payload["entries"], payload["verdict"]

            selected_results = self._selected_results(full)
            entries, verdict = self._run_ablation_round(plan, selected_results, round_index=0)

            refinements = 0
            while (
                verdict["cleanliness"] == "dirty"
                and refinements < budget.max_ablation_refinements
                and self.case.variables.get("refined_command")
            ):
                refinements += 1
                refined = self._run_refined(plan, selected_results, refinements)
                if refined is not None:
                    selected_results = refined["results"]
                    full["selection"] = refined["selection"]
                    self.save_artifact("full", full)
                    entries, verdict = self._run_ablation_round(
                        plan, selected_results, round_index=refinements
                    )
                    verdict["refinement_rounds"] = refinements
                else:
                    break

            payload = {"entries": entries, "verdict": verdict}
            self.save_artifact(stage, payload)
            self.state.finish_stage(
                stage,
                artifact=str(self.artifact_path(stage)),
                cleanliness=verdict["cleanliness"],
            )
            print(f"[S5] ablation attribution: {verdict['cleanliness']}")
            return entries, verdict
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    def _run_ablation_round(
        self,
        plan: list[Ablation],
        selected_results: list[ExperimentResult],
        round_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        seed = self.case.budget.primary_seed
        entries = []
        for ablation in plan:
            variant = f"ablation_{ablation.id}_r{round_index}"
            results = self._run_variants(
                stage="ablations",
                experiment=f"ablation_{ablation.id}",
                variant=variant,
                template=ablation.command,
                split="full",
                seeds=[seed],
            )
            comparison = compare_results(
                results,
                selected_results,
                self.case.metrics,
                self.case.budget.ablation_min_gain,
                label=ablation.id,
                baseline_label="selected",
            )
            if ablation.direction == "drop":
                verdict = "component matters" if comparison.verdict == "worse" else "no measurable drop"
            else:
                verdict = "component raises cost" if comparison.verdict == "better" else "inconclusive"
            entries.append(
                {
                    "id": ablation.id,
                    "component": ablation.component,
                    "description": ablation.description,
                    "direction": ablation.direction,
                    "comparison": comparison.to_dict(),
                    "verdict": verdict,
                    "results": [r.to_dict() for r in results],
                    "round": round_index,
                }
            )
        drops = [entry for entry in entries if entry["direction"] == "drop"]
        matters = [entry for entry in drops if entry["verdict"] == "component matters"]
        ratio = len(matters) / len(drops) if drops else 1.0
        clean = ratio >= 0.75
        verdict = {
            "cleanliness": "clean" if clean else "dirty",
            "components_mattering": len(matters),
            "components_checked": len(drops),
            "reason": (
                f"{len(matters)}/{len(drops)} removed components measurably drop the "
                f"primary metric"
            ),
        }
        return entries, verdict

    def _run_refined(
        self,
        plan: list[Ablation],
        selected_results: list[ExperimentResult],
        round_index: int,
    ) -> Optional[dict[str, Any]]:
        template = self.case.variables.get("refined_command")
        if not template:
            return None
        results = self._run_variants(
            stage="ablations",
            experiment="refined_method",
            variant=f"refined_r{round_index}",
            template=template,
            split="full",
            seeds=self.case.budget.seeds,
        )
        comparison = compare_results(
            results,
            selected_results,
            self.case.metrics,
            self.case.budget.min_relative_gain,
            label="refined",
            baseline_label="selected",
        )
        if comparison.verdict != "better":
            return None
        return {
            "results": results,
            "selection": {
                "idea_id": "refined",
                "label": "refined method",
                "template": template,
                "rationale": comparison.detail,
            },
        }

    # ------------------------------------------------------------------
    # S6 drafting
    # ------------------------------------------------------------------

    def drafting(
        self,
        ideas: list[Idea],
        full: dict[str, Any],
        ablations: list[dict[str, Any]],
        ablation_verdict: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        stage = "drafting"
        if self.state.is_complete(stage):
            payload = self.load_artifact(stage, {})
            return payload["paper"], payload["evidence"]
        self.state.start_stage(stage)
        try:
            records = self._paper_records(ideas, full, ablations, ablation_verdict)
            paper_text, evidence = build_paper(self.case, records)
            payload = {"paper": paper_text, "evidence": evidence}
            self.save_artifact(stage, payload)
            paper_dir = self.run_dir / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "paper_v1.md").write_text(paper_text, encoding="utf-8")
            self.state.finish_stage(stage, artifact=str(paper_dir / "paper_v1.md"))
            print("[S6] initial manuscript drafted")
            return paper_text, evidence
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    # ------------------------------------------------------------------
    # S7 review and rebuttal
    # ------------------------------------------------------------------

    def review_loop(
        self,
        ideas: list[Idea],
        full: dict[str, Any],
        ablations: list[dict[str, Any]],
        ablation_verdict: dict[str, Any],
        paper_text: str,
        evidence: dict[str, Any],
    ) -> tuple[list[Review], str, dict[str, Any]]:
        stage = "review"
        if self.state.is_complete(stage):
            payload = self.load_artifact(stage, {})
            return (
                [Review(**item) for item in payload.get("reviews", [])],
                payload.get("paper", paper_text),
                payload.get("evidence", evidence),
            )
        self.state.start_stage(stage)
        try:
            budget = self.case.budget
            reviews: list[Review] = []
            used_supplementary: set[str] = set()
            supplementary_records: list[dict[str, Any]] = []

            for round_number in range(1, budget.max_review_rounds + 1):
                review = self.agents.review(paper_text, evidence, round_number)
                reviews.append(review)
                print(f"[S7] review r{round_number}: {review.score} ({review.decision})")
                if review.score >= budget.review_threshold:
                    review.decision = "accept"
                    break

                planned = self.agents.rebuttal_plan(review, used_supplementary)
                if not planned:
                    review.decision = "best_effort"
                    break
                review.supplementary_used = [item.id for item in planned]
                for item in planned:
                    used_supplementary.add(item.id)
                    record = self._run_supplementary(item)
                    supplementary_records.append(record)
                records = self._paper_records(
                    ideas,
                    full,
                    ablations,
                    ablation_verdict,
                    supplementary_records,
                    reviews,
                )
                paper_text, evidence = build_paper(self.case, records)

            # Rebuild once more so the sealed evidence summary includes the
            # final review round (the manuscript itself is unchanged).
            records = self._paper_records(
                ideas,
                full,
                ablations,
                ablation_verdict,
                supplementary_records,
                reviews,
            )
            paper_text, evidence = build_paper(self.case, records)

            payload = {
                "reviews": [review.to_dict() for review in reviews],
                "supplementary": supplementary_records,
                "paper": paper_text,
                "evidence": evidence,
            }
            self.save_artifact(stage, payload)
            paper_dir = self.run_dir / "paper"
            (paper_dir / "paper_v2.md").write_text(paper_text, encoding="utf-8")
            (paper_dir / "reviews.md").write_text(
                _reviews_markdown(reviews, supplementary_records), encoding="utf-8"
            )
            self.state.finish_stage(
                stage,
                artifact=str(paper_dir / "paper_v2.md"),
                final_score=reviews[-1].score if reviews else 0.0,
            )
            return reviews, paper_text, evidence
        except ExperimentError as exc:
            self.state.fail_stage(stage, str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    def _run_supplementary(self, item: Supplementary) -> dict[str, Any]:
        results = self._run_variants(
            stage="supplementary",
            experiment=f"supplementary_{item.id}",
            variant=f"supplementary_{item.id}",
            template=item.command,
            split="full",
            seeds=[self.case.budget.primary_seed],
        )
        metric_names = [metric.name for metric in self.case.metrics]
        summary = "; ".join(
            f"{name}={result.metrics.get(name, float('nan')):.4f}"
            for result in results
            for name in metric_names
        )
        return {
            "id": item.id,
            "question": item.question,
            "description": item.description,
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }

    # ------------------------------------------------------------------
    # S8 meta review
    # ------------------------------------------------------------------

    def meta_review(
        self,
        paper_text: str,
        reviews: list[Review],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        stage = "meta_review"
        if self.state.is_complete(stage):
            return self.load_artifact(stage, {})
        self.state.start_stage(stage)
        try:
            review = reviews[-1] if reviews else Review(0, "none", 0.0, [], [], [])
            audit_preview = self._audit_preview(paper_text)
            meta = self.agents.meta_review(paper_text, review, audit_preview, 1)
            meta["audit_preview"] = audit_preview
            self.save_artifact(stage, meta)
            self.state.finish_stage(stage, artifact=str(self.artifact_path(stage)), decision=meta["decision"])
            print(f"[S8] meta-review: {meta['decision']} ({meta.get('critique', '')})")
            return meta
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    # ------------------------------------------------------------------
    # S9 integrity
    # ------------------------------------------------------------------

    def integrity(self, paper_text: str) -> dict[str, Any]:
        stage = "integrity"
        if self.state.is_complete(stage):
            return self.load_artifact(stage, {})
        self.state.start_stage(stage)
        try:
            full = self.load_artifact("full", {})
            selection = full.get("selection", {})
            audit = run_audit(
                case=self.case,
                run_dir=self.run_dir,
                state=self.state,
                observations=self._observations(),
                paper_text=paper_text,
                references=self._references(),
                selected_template=selection.get("template", ""),
                selected_variant=self._selected_variant(full),
                reexecute=self.options.reexecute_audit,
                online=self.options.online_refs,
                allow_offline_refs=self.options.allow_offline_refs,
            )
            self.save_artifact(stage, audit)
            self.state.finish_stage(stage, artifact=str(self.run_dir / "audit" / "audit.json"), passed=audit["passed"])
            print(f"[S9] integrity audit: {'PASS' if audit['passed'] else 'FAIL'}")
            return audit
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    def _audit_preview(self, paper_text: str) -> dict[str, Any]:
        hashes = self.state.metadata.get("protected_hashes", {})
        from .audit import verify_protected

        protected = verify_protected(self.case, hashes) if hashes else {"passed": True}
        return {
            "passed": protected.get("passed", True),
            "protected": protected,
            "observations": len(self._observations()),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def finalize(
        self,
        paper_text: str,
        reviews: list[Review],
        meta: dict[str, Any],
        audit: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        stage = "summary"
        if self.state.is_complete(stage):
            return self.load_artifact(stage, {})
        self.state.start_stage(stage)
        try:
            full = self.load_artifact("full", {})
            selection = full.get("selection", {})
            summary = {
                "run_id": self.state.run_id,
                "case": self.case.case_id,
                "title": self.case.title,
                "selected": selection.get("label", self.case.baseline_name),
                "selected_idea": selection.get("idea_id", ""),
                "review_score": reviews[-1].score if reviews else 0.0,
                "meta_decision": meta.get("decision"),
                "audit_passed": audit.get("passed", False),
                "evidence": evidence,
                "seeds": self.case.budget.seeds,
                "run_dir": str(self.run_dir),
            }
            write_json(self.run_dir / "summary.json", summary)
            paper_dir = self.run_dir / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "final.md").write_text(paper_text, encoding="utf-8")
            self.save_artifact(stage, summary)
            self.state.finish_stage(stage, artifact=str(self.run_dir / "summary.json"))
            print(f"[done] selected: {summary['selected']}, audit: {'PASS' if summary['audit_passed'] else 'FAIL'}")
            return summary
        except Exception as exc:  # noqa: BLE001
            self.state.fail_stage(stage, str(exc))
            raise

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _run_variants(
        self,
        stage: str,
        experiment: str,
        variant: str,
        template: str,
        split: str,
        seeds: list[int],
    ) -> list[ExperimentResult]:
        results = []
        for seed in seeds:
            context = RunContext(
                experiment=experiment,
                variant=variant,
                split=split,
                seed=seed,
                template=template,
                stage=stage,
            )
            results.append(self.runner.run(context))
        return results

    def _selected_idea(self, ideas: list[Idea], full: dict[str, Any]) -> Idea:
        selection = full.get("selection", {})
        idea_id = selection.get("idea_id", "")
        for idea in ideas + self.case.evolved_ideas:
            if idea.id == idea_id:
                return idea
        return ideas[0] if ideas else Idea("", "", "low", [], "", "", "")

    def _selected_results(self, full: dict[str, Any]) -> list[ExperimentResult]:
        selection = full.get("selection", {})
        idea_id = selection.get("idea_id", "")
        for entry in full.get("entries", []):
            if entry["idea_id"] == idea_id:
                return [ExperimentResult.from_dict(item) for item in entry["results"]]
        baseline = full.get("baseline_full", [])
        return [ExperimentResult.from_dict(item) for item in baseline]

    def _selected_variant(self, full: dict[str, Any]) -> str:
        selection = full.get("selection", {})
        return f"{selection.get('idea_id', 'baseline')}_full"

    def _results_from_full(self, key: str) -> list[ExperimentResult]:
        payload = self.load_artifact("full", {})
        return [ExperimentResult.from_dict(item) for item in payload.get(key, [])]

    def _paper_records(
        self,
        ideas: list[Idea],
        full: dict[str, Any],
        ablations: list[dict[str, Any]],
        ablation_verdict: dict[str, Any],
        supplementary: Optional[list[dict[str, Any]]] = None,
        reviews: Optional[list[Review]] = None,
    ) -> dict[str, Any]:
        limitations_payload = self.load_artifact("limitations", {})
        ideation_payload = self.load_artifact("ideation", {})
        screening_payload = self.load_artifact("screening", {})
        return {
            "limitations": [Limitation(**item) for item in limitations_payload.get("limitations", [])],
            "verifier": limitations_payload.get("verifier", {}),
            "ideas": ideas,
            "novelty": ideation_payload.get("novelty", {}),
            "screening": [
                {
                    **entry,
                    "results": [ExperimentResult.from_dict(item) for item in entry.get("results", [])],
                }
                for entry in screening_payload.get("entries", [])
            ],
            "baseline_subset": [
                ExperimentResult.from_dict(item) for item in screening_payload.get("baseline_subset", [])
            ],
            "baseline_full": self._results_from_full("baseline_full"),
            "full": [
                {
                    **entry,
                    "results": [ExperimentResult.from_dict(item) for item in entry.get("results", [])],
                }
                for entry in full.get("entries", [])
            ],
            "selector": full.get("selection", {}),
            "ablations": [
                {
                    **entry,
                    "results": [
                        ExperimentResult.from_dict(item)
                        for item in entry.get("results", [])
                    ],
                }
                for entry in ablations
            ],
            "ablation_verdict": ablation_verdict,
            "supplementary": supplementary or [],
            "reviews": reviews or [],
            "audit": self.load_artifact("integrity"),
            "references": self._references(),
        }

    def _observations(self) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for stage_key, key in (
            ("screening", "baseline_subset"),
            ("screening", "entries"),
            ("full", "baseline_full"),
            ("full", "entries"),
            ("ablations", "entries"),
            ("review", "supplementary"),
        ):
            payload = self.load_artifact(stage_key, {})
            if key == "entries":
                for entry in payload.get("entries", []):
                    for result in entry.get("results", []):
                        observations.append({**result, "stage": stage_key})
            elif key == "supplementary":
                for entry in payload.get("supplementary", []):
                    for result in entry.get("results", []):
                        observations.append({**result, "stage": stage_key})
            else:
                for result in payload.get(key, []):
                    observations.append({**result, "stage": stage_key})
        return observations

    def _references(self) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        ideation = self.load_artifact("ideation", {})
        for report in ideation.get("novelty", {}).values():
            for reference in report.get("references", []):
                if reference not in references:
                    references.append(reference)
        for reference in self.case.variables.get("references", []) or []:
            if reference not in references:
                references.append(reference)
        return references


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _contract_markdown(case: CaseSpec, contract: dict[str, Any]) -> str:
    lines = [
        f"# Problem contract: {case.title}",
        "",
        f"- **Case id:** `{case.case_id}`",
        f"- **Domain:** {case.domain}",
        f"- **Baseline:** {case.baseline_name}",
        f"- **Metrics:** " + ", ".join(
            f"`{m.name}` ({m.direction}{', primary' if m.primary else ''})"
            for m in case.metrics
        ),
        f"- **Seeds:** {case.budget.seeds}",
        f"- **Budget:** subset={case.budget.subset_samples}, full={case.budget.full_samples}, "
        f"review rounds={case.budget.max_review_rounds}, threshold={case.budget.review_threshold}",
        f"- **Protected paths:** {', '.join(case.protected_paths) or '(none declared)'}",
        "",
        "## Problem statement",
        "",
        case.problem_text.strip(),
        "",
        "## Baseline command",
        "",
        "```text",
        case.baseline_command,
        "```",
        "",
    ]
    if case.notes:
        lines.extend(["## Scope note", "", case.notes, ""])
    return "\n".join(lines)


def _limitations_markdown(limitations: list[Limitation], verifier: dict[str, Any]) -> str:
    lines = ["# Limitations", ""]
    for item in limitations:
        kind = "mechanism" if item.mechanism else "empirical"
        lines.append(f"## {item.id} ({kind})")
        lines.append(item.text)
        if item.test:
            lines.append(f"")
            lines.append(f"**Test:** {item.test}")
        lines.append("")
    lines.extend(
        [
            "## Verifier",
            "",
            f"- sufficient: **{verifier.get('sufficient')}**",
            f"- count: {verifier.get('count')}",
            f"- mechanism-level: {verifier.get('mechanism_count')}",
            f"- source: {verifier.get('source')}",
            "",
            verifier.get("reason", ""),
        ]
    )
    return "\n".join(lines)


def _ideas_markdown(ideas: list[Idea], novelty: dict[str, dict[str, Any]]) -> str:
    lines = ["# Seed ideas", ""]
    for idea in ideas:
        report = novelty.get(idea.id, {})
        lines.append(f"## {idea.id} - {idea.name}")
        lines.append("")
        lines.append(f"- Novelty: **{idea.novelty}**")
        lines.append(f"- Targets: {', '.join(idea.targets) or 'n/a'}")
        lines.append(f"- Mechanism: {idea.mechanism}")
        lines.append(f"- Delta: {report.get('delta', idea.delta)}")
        lines.append(f"- Risk: {idea.risk}")
        if idea.references:
            lines.append("- References:")
            for reference in idea.references:
                lines.append(f"  - {reference.get('title', '')} {reference.get('url', '')}")
        lines.append("")
    return "\n".join(lines)


def _reviews_markdown(
    reviews: list[Review], supplementary: list[dict[str, Any]]
) -> str:
    lines = ["# Simulated peer review", ""]
    for review in reviews:
        lines.append(f"## Round {review.round} - {review.reviewer} - score {review.score}/10")
        lines.append("")
        lines.append("**Strengths**")
        lines.extend(f"- {item}" for item in review.strengths)
        lines.append("")
        lines.append("**Weaknesses**")
        lines.extend(f"- {item}" for item in review.weaknesses)
        lines.append("")
        if review.questions:
            lines.append("**Questions**")
            lines.extend(f"- {item}" for item in review.questions)
            lines.append("")
    if supplementary:
        lines.append("## Supplementary experiments")
        lines.append("")
        for item in supplementary:
            lines.append(f"- **{item['id']}**: {item.get('summary', '')}")
        lines.append("")
    return "\n".join(lines)
