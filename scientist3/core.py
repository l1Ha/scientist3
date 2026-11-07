"""Core data models, case loading, run state and artifact helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Case specification
# ---------------------------------------------------------------------------


@dataclass
class MetricSpec:
    """A metric declared by the case contract.

    ``direction`` is ``"max"`` or ``"min"``.  ``path`` is a dot path inside the
    experiment's metrics JSON file (for example ``"probe.accuracy"``).  When it
    is empty the metric name itself is used as a top-level key.
    """

    name: str
    direction: str = "max"
    primary: bool = False
    path: str = ""
    min_gain: Optional[float] = None


@dataclass
class Budget:
    """Bounded loops.  Every limit from the paper is represented here."""

    seeds: list[int] = field(default_factory=lambda: [0])
    subset_samples: int = 256
    full_samples: int = 1024
    test_samples: int = 128
    subset_epochs: int = 1
    full_epochs: int = 2
    top_ideas: int = 2
    max_screening_rounds: int = 2
    max_evo_rounds: int = 2
    min_successful_ideas: int = 1
    max_successful_ideas: int = 3
    max_ablation_refinements: int = 1
    max_review_rounds: int = 2
    review_threshold: float = 8.0
    max_meta_refinements: int = 1
    timeout_seconds: int = 900
    min_relative_gain: float = 0.02
    ablation_min_gain: float = 0.02

    @property
    def primary_seed(self) -> int:
        return self.seeds[0] if self.seeds else 0


@dataclass
class Limitation:
    id: str
    text: str
    mechanism: bool = True
    test: str = ""


@dataclass
class Idea:
    id: str
    name: str
    novelty: str
    targets: list[str]
    mechanism: str
    delta: str
    risk: str
    command: str = ""
    subset_command: str = ""
    engineering_commands: list[str] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    kind: str = "seed"  # seed | evolved


@dataclass
class Ablation:
    id: str
    component: str
    description: str
    command: str
    direction: str = "drop"  # removing the component should make the primary metric drop


@dataclass
class Supplementary:
    id: str
    question: str
    command: str
    description: str = ""


@dataclass
class CaseSpec:
    case_id: str
    title: str
    domain: str
    problem_path: str
    keywords: list[str]
    baseline_name: str
    baseline_command: str
    metrics: list[MetricSpec]
    budget: Budget
    limitations: list[Limitation]
    ideas: list[Idea]
    ablations: list[Ablation]
    supplementary: list[Supplementary]
    evolved_ideas: list[Idea]
    variables: dict[str, Any]
    protected_paths: list[str]
    code_dir: str
    metrics_file: str
    notes: str
    root: Path
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def problem_text(self) -> str:
        path = self.root / self.problem_path
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self.raw.get("case", {}).get("problem", "")

    @property
    def primary_metric(self) -> MetricSpec:
        for metric in self.metrics:
            if metric.primary:
                return metric
        return self.metrics[0]

    @property
    def code_path(self) -> Path:
        return (self.root / self.code_dir).resolve() if self.code_dir else self.root


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_case(path: str | Path) -> CaseSpec:
    """Load a ``case.toml`` file (or a directory containing one)."""

    path = Path(path)
    if path.is_dir():
        path = path / "case.toml"
    if not path.exists():
        raise FileNotFoundError(f"case file not found: {path}")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    case = raw.get("case", {})
    budget_raw = raw.get("budget", {})
    budget_fields = {f.name for f in dataclasses.fields(Budget)}
    budget = Budget(**{k: v for k, v in budget_raw.items() if k in budget_fields})

    metrics = [
        MetricSpec(**{k: v for k, v in item.items()})
        for item in _as_list(raw.get("metrics"))
    ]
    if not metrics:
        raise ValueError(f"{path}: at least one [[metrics]] entry is required")

    limitations = [
        Limitation(**{k: v for k, v in item.items()})
        for item in _as_list(raw.get("limitations"))
    ]
    ideas = [_idea(item) for item in _as_list(raw.get("ideas"))]
    evolved = [_idea(item, kind="evolved") for item in _as_list(raw.get("evolved_ideas"))]
    ablations = [
        Ablation(**{k: v for k, v in item.items()})
        for item in _as_list(raw.get("ablations"))
    ]
    supplementary = [
        Supplementary(**{k: v for k, v in item.items()})
        for item in _as_list(raw.get("supplementary"))
    ]

    baseline = raw.get("baseline", {})
    spec = CaseSpec(
        case_id=case.get("id", path.parent.name),
        title=case.get("title", path.parent.name),
        domain=case.get("domain", "general"),
        problem_path=case.get("problem", "problem.md"),
        keywords=list(case.get("keywords", [])),
        baseline_name=baseline.get("name", "baseline"),
        baseline_command=baseline.get("command", ""),
        metrics=metrics,
        budget=budget,
        limitations=limitations,
        ideas=ideas,
        ablations=ablations,
        supplementary=supplementary,
        evolved_ideas=evolved,
        variables=dict(raw.get("variables", {})),
        protected_paths=list(raw.get("protected_paths", [])),
        code_dir=raw.get("case", {}).get("code_dir", ""),
        metrics_file=raw.get("case", {}).get("metrics_file", "metrics.json"),
        notes=case.get("notes", ""),
        root=path.parent.resolve(),
        raw=raw,
    )
    _validate_case(spec, path)
    return spec


def _idea(item: dict, kind: str = "seed") -> Idea:
    fields = {f.name for f in dataclasses.fields(Idea)}
    data = {k: v for k, v in item.items() if k in fields}
    data["kind"] = kind
    return Idea(**data)


def _validate_case(spec: CaseSpec, path: Path) -> None:
    if not spec.baseline_command:
        raise ValueError(f"{path}: [baseline].command is required")
    for idea in spec.ideas + spec.evolved_ideas:
        if not idea.command:
            raise ValueError(f"{path}: idea {idea.id} has no command")
    primary = [m for m in spec.metrics if m.primary]
    if len(primary) > 1:
        raise ValueError(f"{path}: only one metric may be primary")


# ---------------------------------------------------------------------------
# Experiment records
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    experiment: str
    variant: str
    split: str
    seed: int
    metrics: dict[str, float]
    out_dir: str
    log_path: str
    command: list[str]
    returncode: int
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class Review:
    round: int
    reviewer: str
    score: float
    strengths: list[str]
    weaknesses: list[str]
    questions: list[str]
    decision: str = "refine"
    supplementary_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Run state and artifacts
# ---------------------------------------------------------------------------


class RunState:
    """Persistent stage state for one research run.

    The state file is rewritten atomically after every transition so that an
    interrupted pipeline can be resumed with ``scientist3 resume``.
    """

    def __init__(self, run_dir: Path, case_id: str, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.case_id = case_id
        self.run_id = run_id
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.stages: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    def save(self) -> None:
        self.updated_at = time.time()
        payload = {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "stages": self.stages,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Windows occasionally holds a transient lock on a file that was just
        # written (indexing, antivirus).  Atomic replace is still the goal, so
        # retry briefly instead of falling back to an unsafe in-place write.
        last_error: Optional[Exception] = None
        for _ in range(20):
            try:
                tmp.replace(self.state_path)
                return
            except PermissionError as exc:  # pragma: no cover - platform timing
                last_error = exc
                time.sleep(0.05)
        assert last_error is not None
        raise last_error

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        run_dir = Path(run_dir)
        payload = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        state = cls(run_dir, payload["case_id"], payload["run_id"])
        state.created_at = payload.get("created_at", state.created_at)
        state.updated_at = payload.get("updated_at", state.updated_at)
        state.metadata = payload.get("metadata", {})
        state.stages = payload.get("stages", {})
        return state

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def start_stage(self, name: str) -> None:
        self.stages[name] = {
            "status": "running",
            "started": time.time(),
            "ended": None,
            "artifact": None,
            "detail": {},
        }
        self.save()
        self.event("stage_start", stage=name)

    def finish_stage(self, name: str, artifact: Optional[str] = None, **detail: Any) -> None:
        record = self.stages.setdefault(name, {})
        record.update(
            status="complete",
            ended=time.time(),
            artifact=artifact,
            detail=detail,
        )
        self.save()
        self.event("stage_complete", stage=name, artifact=artifact, **detail)

    def fail_stage(self, name: str, error: str) -> None:
        record = self.stages.setdefault(name, {})
        record.update(status="failed", ended=time.time(), error=error)
        self.save()
        self.event("stage_failed", stage=name, error=error)

    def is_complete(self, name: str) -> bool:
        return self.stages.get(name, {}).get("status") == "complete"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(payload: Any, path: str) -> Any:
    """Resolve a dot path such as ``probe.accuracy`` inside nested dicts."""

    current = payload
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current
