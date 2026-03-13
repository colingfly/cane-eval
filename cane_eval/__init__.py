"""
cane-eval -- LLM-as-Judge evaluation for AI agents.

Open-source eval toolkit: YAML test suites, Claude-powered judging,
regression diffs, failure mining, and training data export.
"""

__version__ = "0.1.0"

from cane_eval.suite import TestSuite, TestCase
from cane_eval.judge import Judge, JudgeResult, CriteriaScore
from cane_eval.engine import EvalRunner, EvalResult, RunSummary
from cane_eval.export import Exporter
from cane_eval.mining import FailureMiner

__all__ = [
    "TestSuite",
    "TestCase",
    "Judge",
    "JudgeResult",
    "CriteriaScore",
    "EvalRunner",
    "EvalResult",
    "RunSummary",
    "Exporter",
    "FailureMiner",
]
