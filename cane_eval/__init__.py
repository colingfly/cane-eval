"""
cane-eval -- AI system reliability infrastructure.

Evaluate any AI system's reliability across correctness, structure, and performance.
Extensible criteria, configurable weights, parallel execution, and domain integrations.
"""

__version__ = "1.0.0"

from cane_eval.providers import get_provider, AnthropicProvider, OpenAIProvider, GeminiProvider, OpenAICompatibleProvider
from cane_eval.suite import ReliabilitySuite, ReliabilityCase, TestSuite, TestCase
from cane_eval.judge import Judge, JudgeResult, CriteriaScore
from cane_eval.engine import (
    ReliabilityRunner, ReliabilityResult, ReliabilitySummary,
    EvalRunner, EvalResult, RunSummary,
    LatencyStats, SchemaResult,
)
from cane_eval.export import Exporter
from cane_eval.mining import FailureMiner
from cane_eval.rca import RootCauseAnalyzer, RCAResult, TargetedRCAResult
from cane_eval.criteria import CriterionPlugin, CriteriaRegistry, EvalContext, CriterionType
from cane_eval.reliability import ReliabilityConfig

# Integrations (lazy-loaded to avoid import errors if frameworks not installed)
from cane_eval.integrations import (
    evaluate_langchain,
    evaluate_llamaindex,
    evaluate_openai,
    evaluate_fastapi,
    evaluate_phlote,
)

__all__ = [
    # Providers
    "get_provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    # Core (new primary names)
    "ReliabilityRunner",
    "ReliabilitySuite",
    "ReliabilityCase",
    "ReliabilityResult",
    "ReliabilitySummary",
    # Core (deprecated aliases, backward compat)
    "EvalRunner",
    "TestSuite",
    "TestCase",
    "EvalResult",
    "RunSummary",
    # Criteria system
    "CriterionPlugin",
    "CriteriaRegistry",
    "EvalContext",
    "CriterionType",
    # Config
    "ReliabilityConfig",
    # Judge
    "Judge",
    "JudgeResult",
    "CriteriaScore",
    # Other
    "LatencyStats",
    "SchemaResult",
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
    "evaluate_phlote",
]
