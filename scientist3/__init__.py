"""Scientist3: a general-purpose autonomous research pipeline.

Scientist3 is an independent, domain-agnostic implementation of the research
workflow described in *ScientistTwo: Pioneering the Human Knowledge Frontier
with Autonomous AI* (Nam et al., 2026, arXiv:2609.19644).

It takes a declarative problem contract (a case), runs bounded
limitation -> idea -> experiment -> ablation -> writing -> review -> meta-review
loops, and seals the result together with a four-part integrity audit.
"""

from .core import (
    Ablation,
    Budget,
    CaseSpec,
    ExperimentResult,
    Idea,
    Limitation,
    MetricSpec,
    RunState,
    Supplementary,
    load_case,
)
from .runner import ExperimentError, ExperimentRunner, compare_results, expand_command
from .pipeline import ResearchPipeline, RunOptions

__version__ = "0.1.0"

__all__ = [
    "Ablation",
    "Budget",
    "CaseSpec",
    "ExperimentError",
    "ExperimentResult",
    "ExperimentRunner",
    "Idea",
    "Limitation",
    "MetricSpec",
    "ResearchPipeline",
    "RunOptions",
    "RunState",
    "Supplementary",
    "compare_results",
    "expand_command",
    "load_case",
    "__version__",
]
