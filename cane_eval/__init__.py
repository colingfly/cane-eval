"""
cane-eval -- LLM-as-Judge evaluation for AI agents.

Open-source eval toolkit: YAML test suites, Claude-powered judging,
regression diffs, failure mining, root cause analysis, and training data export.
"""

__version__ = "0.3.0"

from cane_eval.providers import get_provider, AnthropicProvider, OpenAIProvider, GeminiProvider, OpenAICompatibleProvider
from cane_eval.suite import TestSuite, TestCase
from cane_eval.judge import Judge, JudgeResult, CriteriaScore
from cane_eval.engine import EvalRunner, EvalResult, RunSummary
from cane_eval.export import Exporter
from cane_eval.mining import FailureMiner
from cane_eval.rca import RootCauseAnalyzer, RCAResult, TargetedRCAResult

# Integrations (lazy-loaded to avoid import errors if frameworks not installed)
from cane_eval.integrations import (
    evaluate_langchain,
    evaluate_llamaindex,
    evaluate_openai,
    evaluate_fastapi,
)

__all__ = [
    # Providers
    "get_provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    # Core
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
    "RootCauseAnalyzer",
    "RCAResult",
    "TargetedRCAResult",
    # Integrations
    "evaluate_langchain",
    "evaluate_llamaindex",
    "evaluate_openai",
    "evaluate_fastapi",
]
