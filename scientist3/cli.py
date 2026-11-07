"""Command line interface for Scientist3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .core import load_case, read_json
from .pipeline import ResearchPipeline, RunOptions
from .runner import ExperimentError


def _default_cases_dir() -> Path:
    candidates = [Path("cases"), Path(__file__).resolve().parent.parent / "cases"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _find_case(name: str, cases_dir: Path) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    located = cases_dir / name
    if located.exists():
        return located
    for path in sorted(cases_dir.glob("*/case.toml")):
        if path.parent.name == name:
            return path.parent
    raise FileNotFoundError(f"case not found: {name} (searched {cases_dir})")


def _parse_set(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key.strip()] = _parse_value(raw.strip())
    return overrides


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if "," in raw:
        return [_parse_value(part) for part in raw.split(",") if part != ""]
    if raw.startswith("[") and raw.endswith("]"):
        return [_parse_value(part) for part in raw[1:-1].split(",") if part.strip()]
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def _run_dir(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / run_id


def cmd_list_cases(args: argparse.Namespace) -> int:
    cases_dir = Path(args.cases_dir or _default_cases_dir())
    found = False
    for path in sorted(cases_dir.glob("*/case.toml")):
        try:
            case = load_case(path)
        except Exception as exc:  # noqa: BLE001
            print(f"- {path.parent.name}: INVALID ({exc})")
            continue
        found = True
        print(f"- {case.case_id}: {case.title} [{case.domain}]")
        print(f"  baseline={case.baseline_name} metrics={[m.name for m in case.metrics]}")
    if not found:
        print(f"no cases found under {cases_dir}")
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    case = load_case(_find_case(args.case, Path(args.cases_dir or _default_cases_dir())))
    print(f"case        : {case.case_id} - {case.title}")
    print(f"domain      : {case.domain}")
    print(f"root        : {case.root}")
    print(f"code dir    : {case.code_path}")
    print(f"baseline    : {case.baseline_name}")
    print(f"metrics     : " + ", ".join(f"{m.name}({m.direction})" for m in case.metrics))
    print(f"seeds       : {case.budget.seeds}")
    print(f"limitations : {len(case.limitations)}")
    print(f"ideas       : {[idea.id for idea in case.ideas]}")
    print(f"evolved     : {[idea.id for idea in case.evolved_ideas]}")
    print(f"ablations   : {[item.id for item in case.ablations]}")
    print(f"protected   : {case.protected_paths}")
    return 0


def _build_pipeline(args: argparse.Namespace) -> ResearchPipeline:
    cases_dir = Path(getattr(args, "cases_dir", None) or _default_cases_dir())
    case_path = _find_case(args.case, cases_dir)
    case = load_case(case_path)
    options = RunOptions(
        runs_dir=Path(args.runs_dir),
        run_id=args.run_id or "",
        live=args.live,
        model=args.model or "",
        base_url=args.base_url or "",
        api_key=args.api_key or "",
        online_refs=args.online_refs,
        allow_offline_refs=not args.strict_refs,
        reexecute_audit=not args.no_reexecute_audit,
        overrides=_parse_set(args.set or []),
    )
    pipeline = ResearchPipeline(case, options)
    pipeline.state.metadata["case_path"] = str(case_path)
    pipeline.state.metadata["case_id"] = case.case_id
    pipeline.state.save()
    return pipeline


def cmd_run(args: argparse.Namespace) -> int:
    try:
        pipeline = _build_pipeline(args)
        summary = pipeline.run()
    except ExperimentError as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        return 2
    print("\nsummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("audit_passed", False) else 3


def _load_pipeline_for_run(args: argparse.Namespace) -> ResearchPipeline:
    runs_dir = Path(args.runs_dir)
    run_dir = _run_dir(runs_dir, args.run_id)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"run not found: {run_dir}")
    payload = read_json(state_path, {})
    case_path = payload.get("metadata", {}).get("case_path")
    if not case_path:
        raise FileNotFoundError(
            f"{run_dir} does not record its case path; run it with this version of Scientist3"
        )
    case = load_case(case_path)
    options = RunOptions(
        runs_dir=runs_dir,
        run_id=args.run_id,
        live=getattr(args, "live", False),
        model=getattr(args, "model", None) or "",
        base_url=getattr(args, "base_url", None) or "",
        api_key=getattr(args, "api_key", None) or "",
        online_refs=getattr(args, "online_refs", False),
        allow_offline_refs=not getattr(args, "strict_refs", False),
        reexecute_audit=not getattr(args, "no_reexecute_audit", False),
    )
    return ResearchPipeline(case, options)


def cmd_resume(args: argparse.Namespace) -> int:
    try:
        pipeline = _load_pipeline_for_run(args)
        summary = pipeline.run()
    except (ExperimentError, FileNotFoundError) as exc:
        print(f"resume failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"run {summary['run_id']}: selected={summary['selected']} audit={'PASS' if summary['audit_passed'] else 'FAIL'}")
    return 0 if summary.get("audit_passed", False) else 3


def cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = _run_dir(Path(args.runs_dir), args.run_id)
    state = read_json(run_dir / "state.json")
    if not state:
        print(f"run not found: {run_dir}", file=sys.stderr)
        return 2
    print(f"run      : {state['run_id']} (case {state['case_id']})")
    print(f"directory: {run_dir}")
    print("stages")
    for name, record in state.get("stages", {}).items():
        status = record.get("status")
        detail = record.get("detail") or {}
        extra = ", ".join(f"{k}={v}" for k, v in detail.items() if k != "artifact")
        print(f"  - {name:12s} {status:8s} {extra}")
    summary = read_json(run_dir / "summary.json")
    if summary:
        print("summary")
        for key in (
            "selected",
            "review_score",
            "meta_decision",
            "audit_passed",
        ):
            print(f"  {key}: {summary.get(key)}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    run_dir = _run_dir(Path(args.runs_dir), args.run_id)
    if args.reexecute:
        try:
            pipeline = _load_pipeline_for_run(args)
            pipeline.state.stages.pop("integrity", None)
            pipeline.state.save()
            paper_path = run_dir / "paper" / "final.md"
            if not paper_path.exists():
                paper_path = run_dir / "paper" / "paper_v2.md"
            if not paper_path.exists():
                paper_path = run_dir / "paper" / "paper_v1.md"
            paper_text = paper_path.read_text(encoding="utf-8")
            audit = pipeline.integrity(paper_text)
        except (ExperimentError, FileNotFoundError) as exc:
            print(f"audit failed: {exc}", file=sys.stderr)
            return 2
    else:
        audit = read_json(run_dir / "audit" / "audit.json")
        if not audit:
            print(f"no audit record in {run_dir}", file=sys.stderr)
            return 2
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit.get("passed") else 3


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = _run_dir(Path(args.runs_dir), args.run_id)
    path = run_dir / args.artifact
    if not path.exists():
        print(f"artifact not found: {path}", file=sys.stderr)
        return 2
    print(path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scientist3",
        description="General-purpose autonomous research pipeline (ScientistTwo-style).",
    )
    parser.add_argument("--version", action="version", version=f"scientist3 {__version__}")
    parser.add_argument(
        "--cases-dir", default=None, help="directory containing case folders (default: ./cases)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-cases", help="list available cases")
    p.set_defaults(func=cmd_list_cases)

    p = sub.add_parser("validate", help="validate a case contract")
    p.add_argument("case")
    p.set_defaults(func=cmd_validate)

    def add_run_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--runs-dir", default="runs")
        p.add_argument("--run-id", default="")
        p.add_argument("--live", action="store_true", help="use a live OpenAI-compatible model")
        p.add_argument("--model", default="")
        p.add_argument("--base-url", default="")
        p.add_argument("--api-key", default="")
        p.add_argument("--online-refs", action="store_true", help="fetch reference identifiers")
        p.add_argument(
            "--strict-refs",
            action="store_true",
            help="fail reference verification when identifiers are not fetched",
        )
        p.add_argument("--no-reexecute-audit", action="store_true")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            help="override a budget field, e.g. --set seeds=0,1 --set subset_samples=64",
        )

    p = sub.add_parser("run", help="run the full research pipeline for a case")
    p.add_argument("case")
    add_run_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="resume an interrupted run")
    p.add_argument("run_id")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--live", action="store_true")
    p.add_argument("--model", default="")
    p.add_argument("--base-url", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--online-refs", action="store_true")
    p.add_argument("--strict-refs", action="store_true")
    p.add_argument("--no-reexecute-audit", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("inspect", help="show the stage table of a run")
    p.add_argument("run_id")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("audit", help="show or re-execute the integrity audit")
    p.add_argument("run_id")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--reexecute", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--model", default="")
    p.add_argument("--base-url", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--online-refs", action="store_true")
    p.add_argument("--strict-refs", action="store_true")
    p.add_argument("--no-reexecute-audit", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("show", help="print a run artifact")
    p.add_argument("run_id")
    p.add_argument("artifact", help="path relative to the run directory")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "cases_dir"):
        args.cases_dir = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
