"""
reliability.py -- Reliability scoring computation.

Extracted from engine.py to support configurable weights and
integration with the criteria plugin system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReliabilityConfig:
    """
    Configuration for reliability score computation.

    Weights control the balance between the three pillars.
    When a pillar has no data (e.g., no schema configured), weights
    are redistributed proportionally among available pillars.
    """
    correctness_weight: float = 0.50
    structural_weight: float = 0.25
    performance_weight: float = 0.25

    def __post_init__(self):
        total = self.correctness_weight + self.structural_weight + self.performance_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Reliability weights must sum to 1.0, got {total:.2f} "
                f"({self.correctness_weight}:{self.structural_weight}:{self.performance_weight})"
            )


def compute_reliability(
    accuracy_score: float,
    schema_score: Optional[float],
    latency_score: Optional[float],
    config: Optional[ReliabilityConfig] = None,
) -> tuple[float, str]:
    """
    Compute Agent Reliability Score.

    Three pillars:
    - Correctness (accuracy_score from judge, 0-100)
    - Structural (schema adherence, 0 or 100 per test)
    - Performance (latency vs target)

    When a pillar has no data, its weight is redistributed proportionally
    among the remaining pillars.

    Args:
        accuracy_score: Overall judge score (0-100).
        schema_score: Schema pass percentage (0-100), or None if no schema.
        latency_score: Latency score (0-100), or None if no latency data.
        config: Weight configuration. Uses defaults if not provided.

    Returns:
        (score, grade) tuple where score is 0-100 and grade is A-F.
    """
    if config is None:
        config = ReliabilityConfig()

    has_schema = schema_score is not None
    has_latency = latency_score is not None

    if has_schema and has_latency:
        reliability = (
            accuracy_score * config.correctness_weight
            + schema_score * config.structural_weight
            + latency_score * config.performance_weight
        )
    elif has_schema:
        # Redistribute performance weight proportionally
        total = config.correctness_weight + config.structural_weight
        c_w = config.correctness_weight / total
        s_w = config.structural_weight / total
        reliability = accuracy_score * c_w + schema_score * s_w
    elif has_latency:
        # Redistribute structural weight proportionally
        total = config.correctness_weight + config.performance_weight
        c_w = config.correctness_weight / total
        p_w = config.performance_weight / total
        reliability = accuracy_score * c_w + latency_score * p_w
    else:
        reliability = accuracy_score

    reliability = round(reliability, 1)

    # Grade
    if reliability >= 90:
        grade = "A"
    elif reliability >= 75:
        grade = "B"
    elif reliability >= 60:
        grade = "C"
    elif reliability >= 40:
        grade = "D"
    else:
        grade = "F"

    return reliability, grade
