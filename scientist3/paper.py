"""Deterministic manuscript builder.

The offline pipeline writes the paper from trusted observations.  Live mode may
ask a model to polish the prose, but the tables, numbers, and traceability
identifiers always come from this module.  Every claim is attached to a table
or section label so the integrity audit can resolve it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .core import CaseSpec, ExperimentResult
from .runner import aggregate


def _problem_summary(case: CaseSpec, limit: int = 240) -> str:
    """Join the first paragraph of the problem statement into one summary line."""

    collected: list[str] = []
    for line in case.problem_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if collected:
                break
            continue
        if stripped.startswith("#"):
            continue
        collected.append(stripped)
    summary = " ".join(collected)
    if not summary:
        return case.title
    if len(summary) > limit:
        sentence_end = summary.rfind(". ", 0, limit)
        if sentence_end > 0:
            summary = summary[: sentence_end + 1]
        else:
            summary = summary[:limit].rsplit(" ", 1)[0] + "…"
    return summary


def _format_number(value: float) -> str:
    """Fixed 4 decimals for normal values, scientific for near-zero ones."""

    if value != 0.0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_value(case: CaseSpec, result: list[ExperimentResult], metric_name: str) -> str:
    spec = next(m for m in case.metrics if m.name == metric_name)
    stats = aggregate(result, spec)
    if stats["n"] == 0:
        return "n/a"
    if stats["n"] == 1:
        return _format_number(stats["mean"])
    return f"{_format_number(stats['mean'])} ± {_format_number(stats['std'])}"


def results_table(case: CaseSpec, groups: list[tuple[str, list[ExperimentResult]]]) -> str:
    header = "| Variant | " + " | ".join(f"`{m.name}`" for m in case.metrics) + " | Seeds |"
    divider = "|---" * (len(case.metrics) + 2) + "|"
    lines = [header, divider]
    for label, results in groups:
        cells = [format_value(case, results, m.name) for m in case.metrics]
        seeds = sorted({r.seed for r in results})
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {len(seeds)} |")
    return "\n".join(lines)


def build_paper(case: CaseSpec, records: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the manuscript and the reviewer evidence summary."""

    primary = case.primary_metric
    limitations = records.get("limitations", [])
    verifier = records.get("verifier", {})
    ideas = records.get("ideas", [])
    novelty = records.get("novelty", {})
    screening = records.get("screening", [])
    full = records.get("full", [])
    selector = records.get("selector", {})
    ablations = records.get("ablations", [])
    ablation_verdict = records.get("ablation_verdict", {})
    supplementary = records.get("supplementary", [])
    reviews = records.get("reviews", [])
    audit = records.get("audit")
    references = records.get("references", [])

    best_id = selector.get("idea_id") or (full[0]["idea_id"] if full else "")
    all_ideas = list(ideas) + list(case.evolved_ideas)
    best_idea = next((i for i in all_ideas if getattr(i, "id", None) == best_id), None)

    selected_groups: list[tuple[str, list[ExperimentResult]]] = []
    baseline_results = records.get("baseline_full", [])
    if baseline_results:
        selected_groups.append((f"{case.baseline_name} (baseline)", baseline_results))
    for entry in full:
        selected_groups.append((entry.get("label", entry["idea_id"]), entry.get("results", [])))

    screening_groups = [
        (f"{case.baseline_name} (subset baseline)", records.get("baseline_subset", []))
    ] + [
        (entry.get("label", entry["idea_id"]), entry.get("results", []))
        for entry in screening
    ]

    lines: list[str] = []
    lines.append(f"# {case.title}")
    lines.append("")
    lines.append(
        "**Scientist3 research artifact.** This manuscript was assembled by the "
        "Scientist3 pipeline from trusted experiment observations. It is a "
        "workflow artifact, not a claim of peer-reviewed novelty."
    )
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    summary = _problem_summary(case)
    if summary and summary[-1] not in ".!?…":
        summary += "."
    abstract = (
        f"We study the problem: {summary} "
        f"Following the ScientistTwo-style loop, we extract {len(limitations)} limitations, "
        f"screen {len(ideas)} candidate ideas on a bounded subset, and promote the best "
        f"variant ({best_idea.name if best_idea else 'n/a'}) to the full declared grid. "
        f"On the primary metric `{primary.name}` the selected method reaches "
        f"{format_value(case, _results_for(full, best_id), primary.name)}. "
        f"{len(ablations)} component ablations and {len(supplementary)} supplementary "
        "experiments are reported; an executable integrity audit accompanies the artifact."
    )
    lines.append(abstract)
    lines.append("")
    lines.append("## 1. Introduction")
    lines.append("")
    lines.append(case.problem_text.strip() if case.problem_text else case.title)
    lines.append("")
    if case.notes:
        lines.append(f"**Scope note.** {case.notes}")
        lines.append("")
    lines.append("## 2. Related work")
    lines.append("")
    if references:
        for index, reference in enumerate(references, start=1):
            title = reference.get("title") or reference.get("name") or f"Reference {index}"
            url = reference.get("url") or reference.get("doi") or ""
            note = reference.get("note", "")
            lines.append(f"[{index}] {title}. {note} {url}".strip())
    else:
        lines.append("No external references were recorded for this run.")
    lines.append("")
    lines.append("## 3. Method")
    lines.append("")
    if best_idea:
        lines.append(f"**{best_idea.name}** ({best_id}).")
        lines.append("")
        lines.append(f"- Mechanism: {best_idea.mechanism}")
        lines.append(f"- Difference from closest work: {best_idea.delta}")
        lines.append(f"- Targeted limitations: {', '.join(best_idea.targets) or 'n/a'}")
        lines.append(f"- Risk: {best_idea.risk}")
    else:
        lines.append("No candidate was promoted; the baseline remains the reference.")
    lines.append("")
    lines.append("## 4. Experimental setup")
    lines.append("")
    lines.append(
        f"| Item | Value |\n|---|---|\n"
        f"| Domain | {case.domain} |\n"
        f"| Baseline | {case.baseline_name} |\n"
        f"| Seeds | {', '.join(str(s) for s in case.budget.seeds)} |\n"
        f"| Subset / full samples | {case.budget.subset_samples} / {case.budget.full_samples} |\n"
        f"| Promotion rule | primary metric `{primary.name}` must improve by at least "
        f"{case.budget.min_relative_gain * 100:.1f}% with seed-consistent direction |"
    )
    lines.append("")
    lines.append("### 4.1 Limitations used to drive ideation")
    lines.append("")
    for limitation in limitations:
        lines.append(f"- **{limitation.id}** ({'mechanism' if limitation.mechanism else 'empirical'}): {limitation.text} _{limitation.test}_")
    lines.append("")
    lines.append("### 4.2 Seed ideas and novelty")
    lines.append("")
    lines.append("| Idea | Novelty | Mechanism delta |")
    lines.append("|---|---|---|")
    for idea in ideas:
        entry = novelty.get(getattr(idea, "id", ""), {})
        lines.append(
            f"| {idea.id} {idea.name} | {idea.novelty} | {entry.get('delta', idea.delta)} |"
        )
    lines.append("")
    lines.append("## 5. Results")
    lines.append("")
    lines.append("### 5.1 Subset screening (`Table 1`)")
    lines.append("")
    lines.append(results_table(case, screening_groups))
    lines.append("")
    for entry in screening:
        lines.append(
            f"- **{entry['idea_id']}**: {entry['decision'].upper()} — {entry.get('reason', '')}"
        )
    lines.append("")
    lines.append("### 5.2 Full-grid evaluation (`Table 2`)")
    lines.append("")
    lines.append(results_table(case, selected_groups))
    lines.append("")
    if selector:
        lines.append(
            f"**Selected variant:** {selector.get('label', best_id)}. "
            f"{selector.get('rationale', '')}"
        )
    lines.append("")
    lines.append("## 6. Ablations")
    lines.append("")
    lines.append("`Table 3` removes one component at a time from the selected method.")
    lines.append("")
    lines.append("| Ablation | Component | Primary metric | Verdict |")
    lines.append("|---|---|---|---|")
    for entry in ablations:
        lines.append(
            f"| {entry['id']} | {entry['component']} | "
            f"{format_value(case, entry.get('results', []), primary.name)} | "
            f"{entry.get('verdict', 'n/a')} |"
        )
    lines.append("")
    if ablation_verdict:
        lines.append(
            f"**Attribution:** component breakdown is "
            f"**{ablation_verdict.get('cleanliness', 'unknown')}**. "
            f"{ablation_verdict.get('reason', '')}"
        )
        lines.append("")
    if supplementary:
        lines.append("## 7. Supplementary experiments (rebuttal)")
        lines.append("")
        for entry in supplementary:
            lines.append(f"- **{entry['id']}** {entry.get('question', '')}: {entry.get('summary', '')}")
        lines.append("")
    lines.append("## 8. Limitations and integrity")
    lines.append("")
    lines.append(
        "- The artifact is assembled from the declared, bounded experimental grid; "
        "it does not imply results beyond that grid."
    )
    lines.append(
        "- Reviewer scores in this artifact come from the configured reviewer "
        "implementation and are workflow signals, not external peer review."
    )
    if audit:
        lines.append(
            f"- Integrity audit: {'PASS' if audit.get('passed') else 'FAIL'} "
            f"({', '.join(name for name, check in audit.get('checks', {}).items() if check.get('passed')) or 'no checks'} passed)."
        )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "Every number in this manuscript is derived from stored observations under "
        "`experiments/`. The audit stage re-executes the selected command and "
        "checks the artefact graph against the paper."
    )
    lines.append("")

    text = "\n".join(lines)

    selected_results = _results_for(full, best_id)
    primary_comparison = None
    for entry in full:
        if entry["idea_id"] == best_id:
            primary_comparison = entry.get("comparison", {})
    references_ok = bool(references) and all(
        reference.get("url") or reference.get("doi") for reference in references
    )
    evidence = {
        "n_metrics": len(case.metrics),
        "n_seeds": len(case.budget.seeds),
        "significance": bool(primary_comparison and primary_comparison.get("verdict") == "better"),
        "primary_delta": (primary_comparison or {}).get("deltas", {}).get(primary.name, 0.0),
        "n_ablations": len(ablations),
        "ablation_clean": ablation_verdict.get("cleanliness") == "clean",
        "references_ok": references_ok,
        "n_references": len(references),
        "n_supplementary": len(supplementary),
        "tables_traceable": all(
            token in text for token in ("Table 1", "Table 2", "Table 3")
        ),
        "external": bool(case.variables.get("external_validity", False)),
        "baseline_name": case.baseline_name,
        "selected": getattr(best_idea, "name", best_id),
        "reviews": [review.to_dict() for review in reviews],
        "backticked_metrics": all(f"`{m.name}`" in text for m in case.metrics),
    }
    return text, evidence


def _results_for(full: list[dict[str, Any]], idea_id: str) -> list[ExperimentResult]:
    for entry in full:
        if entry["idea_id"] == idea_id:
            return entry.get("results", [])
    return []
