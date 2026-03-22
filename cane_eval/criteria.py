"""
criteria.py -- Extensible criteria plugin system.

Provides a registry for pluggable evaluation criteria, enabling domain-specific
scoring dimensions beyond the built-in LLM judge criteria.

Built-in criteria wrap existing scoring logic from engine.py and judge.py.
Custom criteria can be registered for domain-specific evaluation (e.g., Phlote
music metadata validation, rights clearance checks).
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class CriterionType(Enum):
    """How a criterion computes its score."""
    LLM = "llm"                    # Scored by LLM judge
    DETERMINISTIC = "deterministic"  # Scored by code/rules
    HYBRID = "hybrid"              # Code + LLM combination


@dataclass
class EvalContext:
    """All data available to a criterion for scoring."""
    question: str = ""
    expected_answer: str = ""
    agent_answer: str = ""
    response_time_ms: int = 0
    schema_result: Optional[Any] = None  # SchemaResult from engine
    judge_result: Optional[Any] = None   # JudgeResult from judge
    metadata: dict = field(default_factory=dict)
    context: Optional[str] = None        # Source documents


class CriterionPlugin(ABC):
    """
    Base class for evaluation criteria.

    Subclass this to create custom criteria. Each criterion produces a score
    from 0-100 for a single evaluation dimension.

    Example:
        class MyCriterion(CriterionPlugin):
            name = "my_criterion"
            criterion_type = CriterionType.DETERMINISTIC
            weight = 1.0
            description = "Checks something specific"

            def score(self, ctx: EvalContext) -> float:
                if "required_field" in ctx.agent_answer:
                    return 100.0
                return 0.0
    """

    name: str = ""
    criterion_type: CriterionType = CriterionType.DETERMINISTIC
    weight: float = 1.0
    description: str = ""

    @abstractmethod
    def score(self, ctx: EvalContext) -> float:
        """
        Score the evaluation context on this criterion.

        Args:
            ctx: All available evaluation data.

        Returns:
            Score from 0.0 to 100.0.
        """
        ...


class SchemaValidityCriterion(CriterionPlugin):
    """Wraps schema validation from engine.py."""

    name = "schema_validity"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Response conforms to expected JSON schema"

    def score(self, ctx: EvalContext) -> float:
        if ctx.schema_result is None:
            return 100.0  # No schema configured, skip
        return 100.0 if ctx.schema_result.valid else 0.0


class LatencyPerformanceCriterion(CriterionPlugin):
    """Wraps latency scoring from engine.py _compute_reliability."""

    name = "latency_performance"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Response latency within target threshold"

    def __init__(self, target_ms: int = 5000):
        self.target_ms = target_ms

    def score(self, ctx: EvalContext) -> float:
        if ctx.response_time_ms <= 0:
            return 100.0  # No latency data, skip
        if ctx.response_time_ms <= self.target_ms:
            return 100.0
        # Linear degradation: at 2x target = 0
        return max(0.0, 100.0 - ((ctx.response_time_ms - self.target_ms) / self.target_ms) * 100)


class AccuracyCriterion(CriterionPlugin):
    """Extracts accuracy score from JudgeResult."""

    name = "accuracy"
    criterion_type = CriterionType.LLM
    weight = 1.0
    description = "Factual correctness vs expected answer"

    def score(self, ctx: EvalContext) -> float:
        return _extract_judge_score(ctx, "accuracy")


class CompletenessCriterion(CriterionPlugin):
    """Extracts completeness score from JudgeResult."""

    name = "completeness"
    criterion_type = CriterionType.LLM
    weight = 1.0
    description = "Covers all key points from expected answer"

    def score(self, ctx: EvalContext) -> float:
        return _extract_judge_score(ctx, "completeness")


class HallucinationCriterion(CriterionPlugin):
    """Extracts hallucination score from JudgeResult."""

    name = "hallucination"
    criterion_type = CriterionType.LLM
    weight = 1.0
    description = "No fabricated or unsupported information"

    def score(self, ctx: EvalContext) -> float:
        return _extract_judge_score(ctx, "hallucination")


def _extract_judge_score(ctx: EvalContext, key: str) -> float:
    """Extract a specific criterion score from JudgeResult."""
    if ctx.judge_result is None:
        return 50.0
    scores = ctx.judge_result.score_dict() if hasattr(ctx.judge_result, "score_dict") else {}
    return float(scores.get(key, 50.0))


class CriteriaRegistry:
    """
    Registry for evaluation criteria plugins.

    Manages criterion registration and lookup. Supports filtering by type
    for selective scoring (e.g., only deterministic criteria for fast checks).

    Usage:
        registry = CriteriaRegistry()
        registry.register(SchemaValidityCriterion())
        registry.register(MyCustomCriterion())

        for criterion in registry.all():
            score = criterion.score(ctx)
    """

    def __init__(self):
        self._criteria: dict[str, CriterionPlugin] = {}

    def register(self, criterion: CriterionPlugin) -> None:
        """Register a criterion plugin."""
        if not criterion.name:
            raise ValueError("Criterion must have a non-empty name")
        self._criteria[criterion.name] = criterion

    def unregister(self, name: str) -> None:
        """Remove a criterion by name."""
        self._criteria.pop(name, None)

    def get(self, name: str) -> Optional[CriterionPlugin]:
        """Get a criterion by name."""
        return self._criteria.get(name)

    def all(self) -> list[CriterionPlugin]:
        """Return all registered criteria."""
        return list(self._criteria.values())

    def by_type(self, criterion_type: CriterionType) -> list[CriterionPlugin]:
        """Return criteria of a specific type."""
        return [c for c in self._criteria.values() if c.criterion_type == criterion_type]

    def names(self) -> list[str]:
        """Return all registered criterion names."""
        return list(self._criteria.keys())

    def __len__(self) -> int:
        return len(self._criteria)

    def __contains__(self, name: str) -> bool:
        return name in self._criteria


def default_registry() -> CriteriaRegistry:
    """Create a registry with the built-in criteria."""
    registry = CriteriaRegistry()
    registry.register(SchemaValidityCriterion())
    registry.register(LatencyPerformanceCriterion())
    registry.register(AccuracyCriterion())
    registry.register(CompletenessCriterion())
    registry.register(HallucinationCriterion())
    return registry
